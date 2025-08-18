"""
RefineTracklets API - Tracklab VideoLevelModule wrapper

This module wraps the refine_tracklets functionality from sn_gamestate.gta_link.refine_tracklets
to work as a tracklab VideoLevelModule that can be integrated into the video processing pipeline.
"""

import os
import logging
from typing import Any, Dict, Optional
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
import copy
import time
import threading
from queue import Queue, Empty
import multiprocessing as mp  # For CPU count only
import time

from tracklab.utils.coordinates import ltrb_to_ltwh
from tracklab.pipeline.videolevel_module import VideoLevelModule
from tracker.gta_link.utils.Tracklet import Tracklet
# Import refinement functions with fallbacks
from tracker.gta_link.utils.refine_tracklets import (
    find_consecutive_segments,
    query_subtracks,
    get_distance_matrix,
    detect_id_switch,
    get_spatial_constraints
)

from tracker.gta_link.utils.refine_tracklets_batched import (
    split_tracklets,
    merge_tracklets,
    merge_tracklets_batched,
    merge_tracklets_batched_parallel_processes
)
from tracker.utils.pipeline_base import MessageType, PipelineMessage


log = logging.getLogger(__name__)


class TrackletsRefiner():
    """
    VideoLevelModule for refining and merging tracklets using advanced algorithms.
    
    This module takes a dictionary of Tracklet objects (typically from GenerateTracklets)
    and applies splitting and merging algorithms to improve tracking consistency.
    """
    def __init__(
        self,
        cfg,
        device,
        batch_size,
        **kwargs
    ):
        super().__init__()        
        self.use_split = cfg.use_split
        self.eps = cfg.eps
        self.min_samples = cfg.min_samples
        self.max_k = cfg.max_k
        self.min_len = cfg.min_len
        self.merge_dist_thres = cfg.merge_dist_thres
        self.spatial_factor = cfg.spatial_factor
        self.batch_size = batch_size
        self.device = device
        self.use_batched_merge = cfg.use_batched_merge
        self.mapping_strategy = cfg.mapping_strategy
        self.return_refined_detections = cfg.return_refined_detections
        self.max_wait_time = cfg.max_wait_time
        self.wait_interval = cfg.wait_interval
        self.save_tracklets = cfg.save_tracklets

        video_name = "refined_tracklets.pkl"
        self.save_dir = os.path.join("outputs", video_name)

        self.seq_tracks = {}

        # Optimized threading approach - better for this use case
        self.batch_queue = Queue()  # Simple threading queue
        self.batch_counter = 0
        self.processing_thread = None
        self.accumulated_tracklets = {}  # Simple dict (no process sharing needed)
        self.final_merge_done = False
        self.pending_batches = 0  # Track number of batches being processed
        self._batch_lock = threading.Lock()  # Lock for thread-safe counter updates

        log.info(f"RefineTracklets initialized with use_split={self.use_split}, "
                f"eps={self.eps}, min_samples={self.min_samples}, merge_dist_thres={self.merge_dist_thres}, "
                f"mapping_strategy={self.mapping_strategy}, return_refined_detections={self.return_refined_detections}")
        log.info(f"OPTIMIZED THREADING approach enabled with batch_size={self.batch_size} (CPU work delegated to existing parallel functions)")


    def process(self, input: PipelineMessage) -> PipelineMessage:
        frame_id = input.metadata.get("frame_id", -1)
        tracklets = input.data.get("tracklets", np.array([]))
        features = input.data.get("features", np.array([]))
        self._update_tracklets(frame_id, tracklets, features)

        # Check if current batch is ready for processing
        if len(self.seq_tracks) >= self.batch_size:
            self._queue_current_batch_for_processing()

        out_pipeline = PipelineMessage(
            msg_type=MessageType.DATA,
            data={
                'frame': input.data['frame'],
                "tracklets": tracklets,  # Return original tracklets for real-time display
            },
            metadata=input.metadata,
            timestamp=input.timestamp,
        )
        return out_pipeline

    def _queue_current_batch_for_processing(self):
        """Queue current batch for background processing"""
        if self.seq_tracks:
            batch_data = {
                'batch_id': self.batch_counter,
                'tracklets': copy.deepcopy(self.seq_tracks),
                'timestamp': time.time()
            }
            
            # Increment pending counter before queuing
            with self._batch_lock:
                self.pending_batches += 1
            
            self.batch_queue.put(batch_data)
            log.info(f"Queued batch {self.batch_counter} with {len(self.seq_tracks)} tracklets (pending: {self.pending_batches})")
            
            # Clear current batch
            self.seq_tracks = {}
            self.batch_counter += 1
            
            # Start processing thread if not running
            if self.processing_thread is None or not self.processing_thread.is_alive():
                self._start_batch_processing_thread()

    def _start_batch_processing_thread(self):
        """Start background thread for batch coordination (CPU work delegated to parallel functions)"""
        self.processing_thread = threading.Thread(
            target=self._batch_processing_worker,
            daemon=True
        )
        self.processing_thread.start()
        log.info("Started optimized batch processing thread")

    def _batch_processing_worker(self):
        """Background worker that coordinates batches (delegates CPU work to parallel functions)"""
        while True:
            try:
                # Get next batch (blocking)
                batch_data = self.batch_queue.get(timeout=1.0)
                
                if batch_data is None:  # Shutdown signal
                    log.info("Received shutdown signal, stopping batch processing worker")
                    break
                
                log.info(f"Processing batch {batch_data['batch_id']} with {len(batch_data['tracklets'])} tracklets")
                
                # Process batch using EXISTING PARALLEL FUNCTION (this is where true parallelism happens)
                processed_batch = self._process_batch_parallel(batch_data['tracklets'])
                
                # Store processed result and merge incrementally
                result = {
                    'batch_id': batch_data['batch_id'],
                    'processed_tracklets': processed_batch,
                    'timestamp': batch_data['timestamp'],
                    'processing_time': time.time() - batch_data['timestamp']
                }
                
                # Incremental merge with accumulated results
                self._merge_batch_incrementally(result)
                
                # Decrement pending counter after processing
                with self._batch_lock:
                    self.pending_batches -= 1
                
                log.info(f"Completed batch {batch_data['batch_id']}: {len(processed_batch)} tracklets "
                        f"(took {result['processing_time']:.2f}s, pending: {self.pending_batches})")
                
            except Empty:
                continue
            except Exception as e:
                log.error(f"Error processing batch: {e}")
                # Still decrement counter on error
                with self._batch_lock:
                    self.pending_batches = max(0, self.pending_batches - 1)

    def _process_batch_parallel(self, tracklets):
        """Process batch using existing well-tested parallel functions"""
        try:
            # Get spatial constraints
            max_x_range, max_y_range = self._get_spatial_constraints(tracklets)
            
            # Split if enabled
            if self.use_split:
                split_tracklets = self._split_tracklets(tracklets)
            else:
                split_tracklets = tracklets
            
            # Use existing PROVEN parallel merge function - THIS is where true CPU parallelism happens
            merged_tracklets = merge_tracklets_batched_parallel_processes(
                split_tracklets,
                seq2Dist={},
                batch_size=min(self.batch_size // 4, 20),  # Smaller sub-batches for parallel processing
                max_x_range=max_x_range,
                max_y_range=max_y_range,
                merge_dist_thres=self.merge_dist_thres,
                max_workers=min(mp.cpu_count(), 4)
            )
            
            return merged_tracklets
            
        except Exception as e:
            log.error(f"Error in batch processing: {e}")
            return tracklets  # Return original on error

    def _merge_batch_incrementally(self, batch_result):
        """Incrementally merge new batch results with accumulated tracklets"""
        new_tracklets = batch_result['processed_tracklets']
        
        for track_id, tracklet in new_tracklets.items():            
            if track_id in self.accumulated_tracklets:
                # Merge with existing tracklet
                self.accumulated_tracklets[track_id] = self.accumulated_tracklets[track_id].update_sequential(
                    tracklet
                )
                log.debug(f"Merged tracklet {track_id} from batch {batch_result['batch_id']}")
            else:
                # New tracklet
                self.accumulated_tracklets[track_id] = tracklet
        
        log.info(f"Accumulated {len(self.accumulated_tracklets)} unique tracklets")

    def finalize_and_get_results(self):
        """Call this when tracking is finished to get final merged results"""
        if self.final_merge_done:
            return getattr(self, 'final_tracklets', {})
        
        log.info("Starting finalization process...")
        
        # Queue any remaining tracklets and wait for processing
        if self.seq_tracks:
            log.info(f"Queuing final batch with {len(self.seq_tracks)} remaining tracklets")
            self._queue_current_batch_for_processing()
            
            # Wait a bit to ensure the batch is picked up by the worker thread
            import time
            time.sleep(0.5)
        
        # Wait for all pending batches to be processed BEFORE signaling shutdown
        self._wait_for_queue_completion()
        
        # Now signal processing thread to stop
        log.info("Signaling worker thread to shutdown...")
        self.batch_queue.put(None)
        
        # Wait for thread to complete
        if self.processing_thread and self.processing_thread.is_alive():
            log.info("Waiting for batch processing thread to complete...")
            self.processing_thread.join(timeout=60)  # Increased timeout
            if self.processing_thread.is_alive():
                log.warning("Batch processing thread did not complete within timeout")
            else:
                log.info("Batch processing thread completed successfully")
        
        # Perform final merge of accumulated tracklets
        log.info(f"Performing final merge of {len(self.accumulated_tracklets)} accumulated tracklets")
        self.final_tracklets = self._final_merge_accumulated_tracklets()
        self.final_merge_done = True
        
        return self.final_tracklets

    def _wait_for_queue_completion(self):
        """Wait for all queued batches to be processed"""       

        elapsed_time = 0
        
        log.info("Waiting for all queued batches to be processed...")
        
        while elapsed_time < self.max_wait_time:
            with self._batch_lock:
                pending_count = self.pending_batches
            
            # Check if all batches are processed
            if pending_count == 0 and self.batch_queue.empty():
                log.info("All batches processed successfully")
                return
            
            time.sleep(self.wait_interval)
            elapsed_time += self.wait_interval

            if elapsed_time % 10 == 0:  # Log every 10 seconds
                log.info(f"Still waiting for batch completion... ({pending_count} pending, {elapsed_time}s elapsed)")
        
        # Final check with lock
        with self._batch_lock:
            final_pending = self.pending_batches
        
        if final_pending > 0:
            log.warning(f"Timeout waiting for batch completion after {self.max_wait_time}s. {final_pending} batches still pending.")
        else:
            log.info("All batches completed just before timeout")

    def get_processing_status(self):
        """Get current processing status for debugging"""
        with self._batch_lock:
            pending = self.pending_batches
        
        return {
            'pending_batches': pending,
            'queue_size': self.batch_queue.qsize(),
            'accumulated_tracklets': len(self.accumulated_tracklets),
            'thread_alive': self.processing_thread.is_alive() if self.processing_thread else False,
            'final_merge_done': self.final_merge_done
        }

    def _final_merge_accumulated_tracklets(self):
        """Final merge pass on accumulated tracklets"""
        if not self.accumulated_tracklets:
            return {}
        
        log.info(f"Final merge: processing {len(self.accumulated_tracklets)} accumulated tracklets")
        
        # Validate and fix all tracklets before final merge
        validated_tracklets = self.accumulated_tracklets
        
        # Perform final merge using existing function for any remaining similar tracklets
        max_x_range, max_y_range = self._get_spatial_constraints(validated_tracklets)
        final_merged = merge_tracklets(
            validated_tracklets,
            merge_dist_thres=self.merge_dist_thres,
            max_x_range=max_x_range,
            max_y_range=max_y_range
        )
        
        # Update tracklet IDs to be sequential
        final_tracklets = self._update_tracklet_ids(final_merged)
        
        log.info(f"Final result: {len(final_tracklets)} tracklets after final merge")
        return final_tracklets

    def _update_tracklets(self, frame_id, tracklets: np.ndarray, features: np.ndarray):
        """Add tracklets to the internal dictionary."""
        #  tracklet: M X (x, y, x, y, id, conf, cls, ind)        
        for i, tracklet in enumerate(tracklets):
            l, t, w, h = ltrb_to_ltwh(tracklet[:4])
            track_id = tracklet[4]
            bbox_conf = tracklet[5]
            # Update tracklet with detection info
            if track_id not in self.seq_tracks:
                self.seq_tracks[track_id] = Tracklet(track_id, frame_id, bbox_conf, [l, t, w, h])
            else:
                self.seq_tracks[track_id].append_det(frame_id, bbox_conf, [l, t, w, h])
            self.seq_tracks[track_id].append_feat(np.squeeze(features[i, :]))        

    def _refine_tracklets(self, tracklets: Dict[int, Tracklet]):
        try:
            # Step 1: Get spatial constraints
            max_x_range, max_y_range = self._get_spatial_constraints(tracklets)

            # Step 2: Split tracklets if enabled
            if self.use_split:
                log.info(f"Splitting tracklets - before: {len(tracklets)}")
                split_tracklets = self._split_tracklets(tracklets)
                log.info(f"After splitting: {len(split_tracklets)}")
            else:
                split_tracklets = tracklets
            # Step 3: Merge tracklets
            log.info(f"Merging tracklets - before: {len(split_tracklets)}")
            if self.use_batched_merge:
                refined_tracklets = self._merge_tracklets_batched(
                    split_tracklets, max_x_range, max_y_range
                )
            else:
                refined_tracklets = self._merge_tracklets(
                    split_tracklets, max_x_range, max_y_range
                )
            log.info(f"After merging: {len(refined_tracklets)}")             
            # Step 5: Update tracklet ids with sequential IDs
            refined_tracklets = self._update_tracklet_ids(refined_tracklets)            
        except Exception as e:
            log.error(f"Error during tracklet refinement: {e}")
            return {}
        return refined_tracklets

    def _get_spatial_constraints(self, tracklets_dict: Dict) -> tuple:
        """Get spatial constraints for merging."""
        try:
            max_x_range, max_y_range = get_spatial_constraints(tracklets_dict, self.spatial_factor)
            log.debug(f"Spatial constraints: x_range={max_x_range}, y_range={max_y_range}")
            return max_x_range, max_y_range
        except Exception as e:
            log.warning(f"Failed to compute spatial constraints: {e}")
            return 1000.0, 1000.0  # Default values
    
    def _split_tracklets(self, tracklets_dict: Dict) -> Dict:
        """Split tracklets using DBSCAN clustering."""
        try:
            return split_tracklets(
                tracklets_dict,
                eps=self.eps,
                max_k=self.max_k,
                min_samples=self.min_samples,
                len_thres=self.min_len
            )
        except Exception as e:
            log.error(f"Error during tracklet splitting: {e}")
            return tracklets_dict

    def _merge_tracklets_batched(self, tracklets_dict: Dict, max_x_range: float, max_y_range: float) -> Dict:
        """Merge tracklets in batches."""
        try:
            # # Use the batched merging function from refine_tracklets_batched
            # return merge_tracklets_batched(
            #     tracklets_dict,
            #     seq2Dist={},  # Empty dict as we're not using it for visualization
            #     batch_size=self.batch_size,
            #     max_x_range=max_x_range,
            #     max_y_range=max_y_range,
            #     merge_dist_thres=self.merge_dist_thres
            # )
        
            # Use the batched merging function from refine_tracklets_batched
            return merge_tracklets_batched_parallel_processes(
                tracklets_dict,
                seq2Dist={},  # Empty dict as we're not using it for visualization
                batch_size=self.batch_size,
                max_x_range=max_x_range,
                max_y_range=max_y_range,
                merge_dist_thres=self.merge_dist_thres
            )            
            
        except Exception as e:
            log.error(f"Error during tracklet merging: {e}")
            return tracklets_dict
    
    def _merge_tracklets(self, tracklets_dict: Dict, max_x_range: float, max_y_range: float) -> Dict:
        """Merge tracklets using distance threshold."""
        try:
            # Use the simple merge function from refine_tracklets_batched
            return merge_tracklets(
                tracklets_dict,
                merge_dist_thres=self.merge_dist_thres,
                max_x_range=max_x_range,
                max_y_range=max_y_range
            )
        except Exception as e:
            log.error(f"Error in merge_tracklets: {e}")
            return tracklets_dict
    
    def _create_sequential_track_id_mapping(self, original_tracklets: Dict, refined_tracklets: Dict) -> Dict:
        """
        Create mapping from original track IDs to sequential IDs (1.0, 2.0, 3.0, ...).
        
        This is a simple mapping strategy that assigns sequential floating-point IDs
        starting from 1.0 to all original tracklets, regardless of the refined tracklets.
        
        Args:
            original_tracklets (Dict): Dictionary of original tracklets
            refined_tracklets (Dict): Dictionary of refined tracklets (not used in this strategy)
            
        Returns:
            Dict: Mapping from original track IDs to sequential IDs
        """
        mapping = {}
        
        try:
            # Sort original tracklet IDs for consistent ordering
            sorted_orig_ids = sorted(refined_tracklets.keys())
            
            # Assign sequential IDs starting from 1.0
            for i, orig_id in enumerate(sorted_orig_ids, start=1):
                mapping[orig_id] = float(i)
            
            log.debug(f"Created sequential mapping: {len(mapping)} tracklets -> IDs 1.0 to {len(mapping)}.0")
            
        except Exception as e:
            log.error(f"Error creating sequential track ID mapping: {e}")
            # Fallback: identity mapping
            mapping = {orig_id: orig_id for orig_id in original_tracklets.keys()}
        
        return mapping
    
    def _create_track_id_mapping(self, original_tracklets: Dict, refined_tracklets: Dict) -> Dict:
        """
        Create mapping from original track IDs to refined track IDs.
        
        Uses a more sophisticated approach considering:
        1. Temporal overlap
        2. Spatial similarity (bbox overlap)
        3. Feature similarity (if available)
        4. One-to-one assignment with conflict resolution
        """
        mapping = {}
        
        try:
            # Calculate similarity matrix between original and refined tracklets
            similarity_matrix = self._calculate_tracklet_similarity_matrix(
                original_tracklets, refined_tracklets
            )
            
            # Use Hungarian algorithm or greedy assignment with conflict resolution
            mapping = self._assign_tracklets_with_conflicts(
                original_tracklets, refined_tracklets, similarity_matrix
            )
            
            log.debug(f"Created tracklet mapping with {len(mapping)} assignments")
            
        except Exception as e:
            log.error(f"Error creating track ID mapping: {e}")
            # Fallback: identity mapping
            mapping = {orig_id: orig_id for orig_id in original_tracklets.keys()}
        
        return mapping
    
    def _calculate_tracklet_similarity_matrix(self, original_tracklets: Dict, refined_tracklets: Dict) -> Dict:
        """
        Calculate similarity matrix between original and refined tracklets.
        
        Returns:
            Dict with structure: {(orig_id, refined_id): similarity_score}
        """
        similarity_matrix = {}
        
        for orig_id, orig_tracklet in original_tracklets.items():
            for refined_id, refined_tracklet in refined_tracklets.items():
                similarity = self._calculate_tracklet_similarity(orig_tracklet, refined_tracklet)
                similarity_matrix[(orig_id, refined_id)] = similarity
        
        return similarity_matrix
    
    def _calculate_tracklet_similarity(self, tracklet1, tracklet2) -> float:
        """
        Calculate similarity between two tracklets using multiple factors.
        
        Returns:
            float: Similarity score between 0 and 1
        """
        # 1. Temporal overlap (most important)
        times1 = set(tracklet1.times) if hasattr(tracklet1, 'times') else set()
        times2 = set(tracklet2.times) if hasattr(tracklet2, 'times') else set()
        
        if not times1 or not times2:
            return 0.0
        
        temporal_overlap = len(times1.intersection(times2))
        temporal_union = len(times1.union(times2))
        temporal_score = temporal_overlap / temporal_union if temporal_union > 0 else 0.0
        
        # If no temporal overlap, similarity is 0
        if temporal_overlap == 0:
            return 0.0
        
        # 2. Spatial similarity (bbox overlap in overlapping frames)
        spatial_score = self._calculate_spatial_similarity(tracklet1, tracklet2, times1.intersection(times2))
        
        # 3. Feature similarity (if features are available)
        feature_score = self._calculate_feature_similarity(tracklet1, tracklet2)
        
        # Weighted combination
        weights = {
            'temporal': 0.5,
            'spatial': 0.3,
            'feature': 0.2
        }
        
        total_score = (
            weights['temporal'] * temporal_score +
            weights['spatial'] * spatial_score +
            weights['feature'] * feature_score
        )
        
        return total_score
    
    def _calculate_spatial_similarity(self, tracklet1, tracklet2, common_times) -> float:
        """Calculate spatial similarity based on bbox overlap in common frames."""
        if not common_times:
            return 0.0
        
        try:
            bbox_overlaps = []
            
            for time in common_times:
                # Find bboxes for this time in both tracklets
                bbox1 = self._get_bbox_at_time(tracklet1, time)
                bbox2 = self._get_bbox_at_time(tracklet2, time)
                
                if bbox1 is not None and bbox2 is not None:
                    overlap = self._calculate_bbox_iou(bbox1, bbox2)
                    bbox_overlaps.append(overlap)
            
            return np.mean(bbox_overlaps) if bbox_overlaps else 0.0
            
        except Exception as e:
            log.debug(f"Error calculating spatial similarity: {e}")
            return 0.0
    
    def _get_bbox_at_time(self, tracklet, time):
        """Get bbox for tracklet at specific time."""
        try:
            if hasattr(tracklet, 'times') and hasattr(tracklet, 'bboxes'):
                if time in tracklet.times:
                    idx = tracklet.times.index(time)
                    if idx < len(tracklet.bboxes):
                        return tracklet.bboxes[idx]
        except:
            pass
        return None
    
    def _calculate_bbox_iou(self, bbox1, bbox2) -> float:
        """Calculate IoU between two bboxes in [l, t, w, h] format."""
        try:
            # Convert to [x1, y1, x2, y2]
            l1, t1, w1, h1 = bbox1[:4]
            l2, t2, w2, h2 = bbox2[:4]
            
            x1_1, y1_1, x2_1, y2_1 = l1, t1, l1 + w1, t1 + h1
            x1_2, y1_2, x2_2, y2_2 = l2, t2, l2 + w2, t2 + h2
            
            # Calculate intersection
            xi1 = max(x1_1, x1_2)
            yi1 = max(y1_1, y1_2)
            xi2 = min(x2_1, x2_2)
            yi2 = min(y2_1, y2_2)
            
            if xi2 <= xi1 or yi2 <= yi1:
                return 0.0
            
            intersection = (xi2 - xi1) * (yi2 - yi1)
            
            # Calculate union
            area1 = w1 * h1
            area2 = w2 * h2
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
            
        except:
            return 0.0
    
    def _calculate_feature_similarity(self, tracklet1, tracklet2) -> float:
        """Calculate feature similarity using cosine similarity."""
        try:
            if hasattr(tracklet1, 'features') and hasattr(tracklet2, 'features'):
                features1 = tracklet1.features
                features2 = tracklet2.features
                
                if features1 and features2:
                    # Use average feature vectors
                    avg_feat1 = np.mean(features1, axis=0)
                    avg_feat2 = np.mean(features2, axis=0)
                    
                    # Cosine similarity
                    dot_product = np.dot(avg_feat1, avg_feat2)
                    norm1 = np.linalg.norm(avg_feat1)
                    norm2 = np.linalg.norm(avg_feat2)
                    
                    if norm1 > 0 and norm2 > 0:
                        return dot_product / (norm1 * norm2)
            
            return 0.0
            
        except Exception as e:
            log.debug(f"Error calculating feature similarity: {e}")
            return 0.0
    
    def _assign_tracklets_with_conflicts(self, original_tracklets: Dict, refined_tracklets: Dict, 
                                       similarity_matrix: Dict) -> Dict:
        """
        Assign tracklets resolving conflicts using a greedy approach with thresholding.
        
        Alternative approaches:
        1. Hungarian algorithm for optimal assignment
        2. Greedy with conflict resolution (implemented here)
        3. Many-to-one mapping for merged tracklets
        """
        mapping = {}
        used_refined_ids = set()
        min_similarity_threshold = 0.1  # Minimum similarity to create mapping
        
        # Sort original tracklets by their best similarity score (descending)
        orig_scores = {}
        for orig_id in original_tracklets.keys():
            best_score = max(
                similarity_matrix.get((orig_id, ref_id), 0.0) 
                for ref_id in refined_tracklets.keys()
            )
            orig_scores[orig_id] = best_score
        
        sorted_orig_ids = sorted(orig_scores.keys(), key=lambda x: orig_scores[x], reverse=True)
        
        # Assign each original tracklet to best available refined tracklet
        for orig_id in sorted_orig_ids:
            best_refined_id = None
            best_similarity = min_similarity_threshold
            
            for refined_id in refined_tracklets.keys():
                similarity = similarity_matrix.get((orig_id, refined_id), 0.0)
                
                # For one-to-one mapping, skip already used refined tracklets
                # Comment out next line to allow many-to-one mapping
                if refined_id in used_refined_ids:
                    continue
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_refined_id = refined_id
            
            if best_refined_id is not None:
                mapping[orig_id] = best_refined_id
                used_refined_ids.add(best_refined_id)
            else:
                # No good match found, keep original ID
                mapping[orig_id] = orig_id
        
        return mapping
    

    def _update_tracklet_ids(self, tracklets: Dict) -> Dict:
        """Update tracklets with refined track IDs."""
        new_tracklets = defaultdict(list)
        for seq_idx, (track_id, tracklet) in enumerate(tracklets.items()):
            if isinstance(tracklet, Tracklet):
                # Assign new track ID as sequential index
                new_track_id = float(seq_idx+1)
                new_tracklets[new_track_id] = tracklet
        return new_tracklets
    
    
    def save_refined_tracklets(self, refined_tracklets: Dict, output_path: str):
        """
        Save refined tracklets dictionary to pickle file.
        
        Args:
            refined_tracklets: Dictionary of refined Tracklet objects
            output_path: Path to save the pickle file
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                pickle.dump(refined_tracklets, f)
            log.info(f"Refined tracklets saved to {output_path}")
        except Exception as e:
            log.error(f"Failed to save refined tracklets to {output_path}: {e}")
