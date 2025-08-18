"""
GPU-accelerated tracklet refinement using CuPy
Optimized version of refine_tracklets_batched.py with significant performance improvements
"""

import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm

try:
    import cupy as cp
    import cupyx.scipy.spatial.distance as cupy_distance
    CUPY_AVAILABLE = True
    print("✅ CuPy available - GPU acceleration enabled")
except ImportError:
    import numpy as cp
    print("⚠️  CuPy not available - falling back to CPU (install with: pip install cupy-cuda12x)")
    CUPY_AVAILABLE = False

from .Tracklet import Tracklet

log = logging.getLogger(__name__)


class GPUTrackletRefiner:
    """
    GPU-accelerated tracklet refinement using CuPy for matrix operations.
    Provides significant speedup for large tracklet datasets.
    """
    
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        if self.use_gpu:
            log.info("🚀 GPU acceleration enabled with CuPy")
        else:
            log.info("🐌 Using CPU fallback (install CuPy for GPU acceleration)")
    
    def extract_features_batch(self, tracklets: Dict) -> cp.ndarray:
        """
        Extract all features from tracklets and create a batch tensor on GPU.
        
        Returns:
            features_gpu: Shape (total_detections, feature_dim) on GPU
            tracklet_indices: Map each detection back to its tracklet
            detection_counts: Number of detections per tracklet
        """
        all_features = []
        tracklet_indices = []
        detection_counts = []
        
        for tid, tracklet in tracklets.items():
            features = np.stack(tracklet.features)  # (n_detections, feature_dim)
            all_features.append(features)
            tracklet_indices.extend([tid] * len(features))
            detection_counts.append(len(features))
        
        # Stack all features and move to GPU
        features_batch = np.vstack(all_features)  # (total_detections, feature_dim)
        if self.use_gpu:
            features_gpu = cp.asarray(features_batch)
        else:
            features_gpu = features_batch
            
        return features_gpu, tracklet_indices, detection_counts
    
    def compute_pairwise_cosine_distances_gpu(self, features1: cp.ndarray, features2: cp.ndarray) -> cp.ndarray:
        """
        Compute pairwise cosine distances between two feature sets on GPU.
        
        Args:
            features1: Shape (n1, feature_dim)
            features2: Shape (n2, feature_dim)
            
        Returns:
            distances: Shape (n1, n2) cosine distances
        """
        # Normalize features for cosine similarity
        features1_norm = features1 / cp.linalg.norm(features1, axis=1, keepdims=True)
        features2_norm = features2 / cp.linalg.norm(features2, axis=1, keepdims=True)
        
        # Cosine similarity = dot product of normalized vectors
        similarity = cp.dot(features1_norm, features2_norm.T)
        
        # Convert to cosine distance
        distances = 1 - similarity
        
        return distances
    
    def get_distance_matrix_gpu_optimized(self, tracklets: Dict) -> cp.ndarray:
        """
        Compute distance matrix using GPU acceleration with optimized memory usage.
        
        Key optimizations:
        1. Batch feature extraction
        2. GPU-accelerated cosine distance computation
        3. Vectorized temporal overlap checking
        4. Memory-efficient processing
        """
        n_tracklets = len(tracklets)
        tracklet_ids = list(tracklets.keys())
        
        # Extract features for all tracklets at once
        all_features, _, detection_counts = self.extract_features_batch(tracklets)
        
        # Initialize distance matrix on GPU
        if self.use_gpu:
            distance_matrix = cp.zeros((n_tracklets, n_tracklets), dtype=cp.float32)
        else:
            distance_matrix = np.zeros((n_tracklets, n_tracklets), dtype=np.float32)
        
        # Precompute time sets for overlap checking (keep on CPU for set operations)
        time_sets = [set(tracklets[tid].times) for tid in tracklet_ids]
        
        # Process features in chunks to manage memory
        feature_start_idx = 0
        
        for i, tid1 in enumerate(tracklet_ids):
            tracklet1 = tracklets[tid1]
            n_features1 = detection_counts[i]
            features1 = all_features[feature_start_idx:feature_start_idx + n_features1]
            
            feature_start_idx2 = 0
            for j, tid2 in enumerate(tracklet_ids):
                if j <= i:  # Use symmetry
                    if i == j:
                        distance_matrix[i, j] = 0  # Distance to self is 0
                    else:
                        distance_matrix[i, j] = distance_matrix[j, i]
                    if j < len(detection_counts):
                        feature_start_idx2 += detection_counts[j]
                    continue
                
                tracklet2 = tracklets[tid2]
                n_features2 = detection_counts[j]
                
                # Check temporal overlap (faster on CPU)
                if tid1 != tid2 and time_sets[i] & time_sets[j]:
                    distance_matrix[i, j] = 1.0  # Maximum distance for overlapping tracks
                else:
                    # Compute feature-based distance on GPU
                    features2 = all_features[feature_start_idx2:feature_start_idx2 + n_features2]
                    
                    pairwise_distances = self.compute_pairwise_cosine_distances_gpu(features1, features2)
                    
                    # Average distance across all feature pairs
                    mean_distance = cp.mean(pairwise_distances)
                    distance_matrix[i, j] = mean_distance
                
                feature_start_idx2 += n_features2
            
            feature_start_idx += n_features1
        
        return distance_matrix
    
    def find_closest_pair_gpu(self, distance_matrix: cp.ndarray, threshold: float) -> Optional[Tuple[int, int, float]]:
        """
        Find the closest pair of tracklets below threshold using GPU operations.
        
        Returns:
            (idx1, idx2, distance) or None if no pair below threshold
        """
        # Mask diagonal (self-distances)
        n = distance_matrix.shape[0]
        if self.use_gpu:
            mask = cp.eye(n, dtype=bool)
        else:
            mask = np.eye(n, dtype=bool)
        
        masked_distances = distance_matrix.copy()
        masked_distances[mask] = float('inf')  # Ignore self-distances
        
        # Find minimum distance
        min_distance = cp.min(masked_distances)
        
        if min_distance < threshold:
            # Find indices of minimum
            min_indices = cp.unravel_index(cp.argmin(masked_distances), masked_distances.shape)
            idx1, idx2 = int(min_indices[0]), int(min_indices[1])
            return idx1, idx2, float(min_distance)
        
        return None
    
    def check_spatial_constraints_vectorized(self, tracklet1, tracklet2, max_x_range: float, max_y_range: float) -> bool:
        """
        Vectorized spatial constraint checking using GPU.
        """
        if not tracklet1.bboxes or not tracklet2.bboxes:
            return False
        
        # Convert to arrays
        bboxes1 = np.array(tracklet1.bboxes)  # (n1, 4)
        bboxes2 = np.array(tracklet2.bboxes)  # (n2, 4)
        
        if self.use_gpu:
            bboxes1_gpu = cp.asarray(bboxes1)
            bboxes2_gpu = cp.asarray(bboxes2)
        else:
            bboxes1_gpu = bboxes1
            bboxes2_gpu = bboxes2
        
        # Get end of tracklet1 and start of tracklet2
        end_bbox1 = bboxes1_gpu[-1]  # Last bbox of tracklet1
        start_bbox2 = bboxes2_gpu[0]  # First bbox of tracklet2
        
        # Compute center positions
        center1 = end_bbox1[:2] + end_bbox1[2:] / 2  # [cx, cy]
        center2 = start_bbox2[:2] + start_bbox2[2:] / 2  # [cx, cy]
        
        # Check spatial constraints
        dx = abs(center2[0] - center1[0])
        dy = abs(center2[1] - center1[1])
        
        return float(dx) <= max_x_range and float(dy) <= max_y_range
    
    def merge_tracklets_gpu_optimized(self, tracklets: Dict, merge_dist_thres: float, 
                                    max_x_range: float, max_y_range: float) -> Dict:
        """
        GPU-optimized tracklet merging with smart matrix updates.
        
        Key optimizations:
        1. Initial distance matrix computation on GPU
        2. Incremental updates instead of full recomputation
        3. Early termination when no more merges possible
        4. Vectorized spatial constraint checking
        """
        if len(tracklets) <= 1:
            return tracklets
        
        log.info(f"🔥 GPU merging {len(tracklets)} tracklets (threshold: {merge_dist_thres})")
        start_time = time.time()
        
        # Convert to list for indexing
        tracklet_items = list(tracklets.items())
        active_indices = set(range(len(tracklet_items)))
        
        # Compute initial distance matrix on GPU
        distance_matrix = self.get_distance_matrix_gpu_optimized(tracklets)
        
        merge_count = 0
        max_merges = len(tracklets) * 2  # Safety limit
        
        while merge_count < max_merges:
            # Find closest pair among active tracklets
            active_matrix = distance_matrix[list(active_indices)][:, list(active_indices)]
            active_list = list(active_indices)
            
            closest_pair = self.find_closest_pair_gpu(active_matrix, merge_dist_thres)
            
            if closest_pair is None:
                break  # No more merges possible
            
            local_idx1, local_idx2, min_distance = closest_pair
            global_idx1 = active_list[local_idx1]
            global_idx2 = active_list[local_idx2]
            
            # Get actual tracklets
            tid1, tracklet1 = tracklet_items[global_idx1]
            tid2, tracklet2 = tracklet_items[global_idx2]
            
            # Check temporal overlap
            if set(tracklet1.times) & set(tracklet2.times):
                # Overlapping tracks - set distance to infinity to avoid future merges
                distance_matrix[global_idx1, global_idx2] = float('inf')
                distance_matrix[global_idx2, global_idx1] = float('inf')
                continue
            
            # Check spatial constraints
            if not self.check_spatial_constraints_vectorized(tracklet1, tracklet2, max_x_range, max_y_range):
                # Failed spatial constraints - mark as unmergeable
                distance_matrix[global_idx1, global_idx2] = float('inf')
                distance_matrix[global_idx2, global_idx1] = float('inf')
                continue
            
            # Perform merge
            log.debug(f"Merging tracklets {tid1} + {tid2} (distance: {min_distance:.4f})")
            
            # Merge tracklet2 into tracklet1
            tracklet1.features.extend(tracklet2.features)
            tracklet1.times.extend(tracklet2.times)
            tracklet1.bboxes.extend(tracklet2.bboxes)
            tracklet1.scores.extend(tracklet2.scores)
            
            # Update tracklet_items
            tracklet_items[global_idx1] = (tid1, tracklet1)
            
            # Remove merged tracklet from active set
            active_indices.remove(global_idx2)
            
            # Update distance matrix for the merged tracklet (only recompute affected row/column)
            self._update_distance_matrix_row(distance_matrix, global_idx1, tracklet_items, active_indices)
            
            merge_count += 1
        
        # Build final result from active tracklets
        final_tracklets = {}
        for idx in active_indices:
            tid, tracklet = tracklet_items[idx]
            final_tracklets[tid] = tracklet
        
        elapsed = time.time() - start_time
        log.info(f"✅ GPU merging complete: {len(tracklets)} → {len(final_tracklets)} tracklets "
                f"({merge_count} merges in {elapsed:.2f}s)")
        
        return final_tracklets
    
    def _update_distance_matrix_row(self, distance_matrix: cp.ndarray, updated_idx: int, 
                                  tracklet_items: List, active_indices: set):
        """
        Update only the row/column for a merged tracklet instead of recomputing entire matrix.
        """
        updated_tid, updated_tracklet = tracklet_items[updated_idx]
        
        if self.use_gpu:
            updated_features = cp.asarray(np.stack(updated_tracklet.features))
        else:
            updated_features = np.stack(updated_tracklet.features)
        
        updated_times = set(updated_tracklet.times)
        
        for other_idx in active_indices:
            if other_idx == updated_idx:
                continue
            
            other_tid, other_tracklet = tracklet_items[other_idx]
            other_times = set(other_tracklet.times)
            
            # Check temporal overlap
            if updated_times & other_times:
                distance = 1.0  # Maximum distance
            else:
                # Compute feature distance
                if self.use_gpu:
                    other_features = cp.asarray(np.stack(other_tracklet.features))
                else:
                    other_features = np.stack(other_tracklet.features)
                
                pairwise_distances = self.compute_pairwise_cosine_distances_gpu(
                    updated_features, other_features
                )
                distance = float(cp.mean(pairwise_distances))
            
            # Update matrix symmetrically
            distance_matrix[updated_idx, other_idx] = distance
            distance_matrix[other_idx, updated_idx] = distance


# High-level API functions that match the original interface
def merge_tracklets_gpu(tracklets: Dict, merge_dist_thres: float, max_x_range: float, max_y_range: float) -> Dict:
    """
    GPU-accelerated version of merge_tracklets with the same interface.
    Drop-in replacement for the original function.
    """
    refiner = GPUTrackletRefiner()
    return refiner.merge_tracklets_gpu_optimized(tracklets, merge_dist_thres, max_x_range, max_y_range)


def merge_tracklets_batched_gpu(tracklets: Dict, seq2Dist: Dict, batch_size: int = 50, 
                               seq_name: str = None, max_x_range: float = None, 
                               max_y_range: float = None, merge_dist_thres: float = None) -> Dict:
    """
    GPU-accelerated batched merging with smart batch sizing based on GPU memory.
    """
    refiner = GPUTrackletRefiner()
    
    # Estimate optimal batch size based on GPU memory
    if refiner.use_gpu:
        try:
            gpu_memory = cp.cuda.Device().mem_info[0]  # Available memory in bytes
            # Rough estimate: each tracklet uses ~1MB for features, batch size should fit in memory
            estimated_optimal_batch = min(batch_size, int(gpu_memory / (1024 * 1024 * 50)))  # Conservative estimate
            batch_size = max(10, estimated_optimal_batch)  # Minimum batch size of 10
            log.info(f"🎯 Auto-tuned batch size: {batch_size} (based on GPU memory)")
        except:
            log.warning("Could not estimate GPU memory, using default batch size")
    
    temp_tracklets = {}
    tracklet_items = list(tracklets.items())
    
    log.info(f"🔥 Processing {len(tracklet_items)} tracklets in batches of {batch_size}")
    
    # Process in batches
    for i in tqdm(range(0, len(tracklet_items), batch_size), desc="GPU batch processing"):
        batch_tracklets = dict(tracklet_items[i:i+batch_size])
        log.debug(f"Processing batch {i//batch_size + 1}: {len(batch_tracklets)} tracklets")
        
        merged_batch = refiner.merge_tracklets_gpu_optimized(
            batch_tracklets, merge_dist_thres, max_x_range, max_y_range
        )
        
        temp_tracklets.update(merged_batch)
        log.debug(f"Batch result: {len(merged_batch)} tracklets remaining")
    
    # Final merge across batches
    log.info(f"🔗 Final cross-batch merging of {len(temp_tracklets)} tracklets")
    final_result = refiner.merge_tracklets_gpu_optimized(
        temp_tracklets, merge_dist_thres, max_x_range, max_y_range
    )
    
    return final_result


# Performance benchmarking
def benchmark_gpu_vs_cpu(tracklets: Dict, merge_dist_thres: float, max_x_range: float, max_y_range: float):
    """
    Benchmark GPU vs CPU performance for tracklet merging.
    """
    from .refine_tracklets_batched import merge_tracklets as cpu_merge_tracklets
    
    print("🏁 Performance Benchmark: GPU vs CPU")
    print(f"   Tracklets: {len(tracklets)}")
    print(f"   Threshold: {merge_dist_thres}")
    
    # CPU benchmark
    print("\n⏱️  CPU Implementation...")
    start_time = time.time()
    cpu_result = cpu_merge_tracklets(tracklets.copy(), merge_dist_thres, max_x_range, max_y_range)
    cpu_time = time.time() - start_time
    
    # GPU benchmark
    print("\n🚀 GPU Implementation...")
    start_time = time.time()
    gpu_result = merge_tracklets_gpu(tracklets.copy(), merge_dist_thres, max_x_range, max_y_range)
    gpu_time = time.time() - start_time
    
    # Results
    speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
    print(f"\n📊 Results:")
    print(f"   CPU Time: {cpu_time:.2f}s → {len(cpu_result)} tracklets")
    print(f"   GPU Time: {gpu_time:.2f}s → {len(gpu_result)} tracklets")
    print(f"   Speedup: {speedup:.1f}x")
    
    return {'cpu_time': cpu_time, 'gpu_time': gpu_time, 'speedup': speedup}
