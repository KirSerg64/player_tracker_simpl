"""
PyTorch-based tracklet refinement module.

This module provides GPU-accelerated tracklet merging and refinement using PyTorch
instead of CuPy for better memory management and error handling. Follows the same API
as the original refine_tracklets.py for drop-in replacement.
"""

import argparse
import copy
import os
import pickle
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

from tracker.gta_link.utils.Tracklet import Tracklet
import torch

# Configure PyTorch for optimal performance
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed

# Set up logging using the same format as the original
log = logging.getLogger(__name__)


def find_consecutive_segments(track_times):
    """
    Identifies and returns the start and end indices of consecutive segments in a list of times.

    Args:
        track_times (list): A list of frame times (integers) representing when a tracklet was detected.

    Returns:
        list of tuples: Each tuple contains two integers (start_index, end_index) representing the start and end of a consecutive segment.
    """
    segments = []
    start_index = 0
    end_index = 0
    for i in range(1, len(track_times)):
        if track_times[i] == track_times[end_index] + 1:
            end_index = i
        else:
            segments.append((start_index, end_index))
            start_index = i
            end_index = i
    segments.append((start_index, end_index))
    return segments


def query_subtracks(seg1, seg2, track1, track2):
    """
    Processes and pairs up segments from two different tracks to form valid subtracks based on their temporal alignment.

    Args:
        seg1 (list of tuples): List of segments from the first track where each segment is a tuple of start and end indices.
        seg2 (list of tuples): List of segments from the second track similar to seg1.
        track1 (Tracklet): First track object containing times and bounding boxes.
        track2 (Tracklet): Second track object similar to track1.

    Returns:
        list: Returns a list of subtracks which are either segments of track1 or track2 sorted by time.
    """
    subtracks = []  # List to store valid subtracks
    while seg1 and seg2:  # Continue until seg1 or seg1 is empty
        s1_start, s1_end = seg1[0]  # Get the start and end indices of the first segment in seg1
        s2_start, s2_end = seg2[0]  # Get the start and end indices of the first segment in seg2

        subtrack_1 = track1.extract(s1_start, s1_end)
        subtrack_2 = track2.extract(s2_start, s2_end)

        s1_startFrame = track1.times[s1_start]  # Get the starting frame of subtrack 1
        s2_startFrame = track2.times[s2_start]  # Get the starting frame of subtrack 2

        if s1_startFrame < s2_startFrame:  # Compare the starting frames of the two subtracks
            assert track1.times[s1_end] <= s2_startFrame, "Overlapping: track1.times[s1_end] <= s2_startFrame"
            subtracks.append(subtrack_1)
            subtracks.append(subtrack_2)
        else:
            assert s1_startFrame >= track2.times[s2_end], "Overlapping: s1_startFrame >= track2.times[s2_end]"
            subtracks.append(subtrack_2)
            subtracks.append(subtrack_1)
        seg1.pop(0)
        seg2.pop(0)
    
    seg_remain = seg1 if seg1 else seg2
    track_remain = track1 if seg1 else track2
    while seg_remain:
        s_start, s_end = seg_remain[0]
        if(s_end - s_start) < 30:
            seg_remain.pop(0)
            continue
        subtracks.append(track_remain.extract(s_start, s_end))
        seg_remain.pop(0)
    
    return subtracks  # Return the list of valid subtracks sorted ascending temporally


def get_subtrack(track, s_start, s_end):
    """
    Extracts a subtrack from a given track.

    Args:
    track (STrack): The original track object from which the subtrack is to be extracted.
    s_start (int): The starting index of the subtrack.
    s_end (int): The ending index of the subtrack.

    Returns:
    STrack: A subtrack object extracted from the original track object, containing the specified time intervals
            and bounding boxes. The parent track ID is also assigned to the subtrack.
    """
    subtrack = Tracklet()
    subtrack.times = track.times[s_start : s_end + 1]
    subtrack.bboxes = track.bboxes[s_start : s_end + 1]
    subtrack.parent_id = track.track_id

    return subtrack


def get_spatial_constraints(tid2track, factor):
    """
    Calculates and returns the maximal spatial constraints for bounding boxes across all tracks.

    Args:
        tid2track (dict): Dictionary mapping track IDs to their respective track objects.
        factor (float): Factor by which to scale the calculated x and y ranges.

    Returns:
        tuple: Maximal x and y range scaled by the given factor.
    """

    min_x = float('inf')
    max_x = -float('inf')
    min_y = float('inf')
    max_y = -float('inf')

    for track in tid2track.values():
        for bbox in track.bboxes:
            assert len(bbox) == 4
            x, y, w, h = bbox[0:4]  # x, y is coordinate of top-left point of bounding box
            x += w / 2  # get center point
            y += h / 2  # get center point
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

    x_range = abs(max_x - min_x) * factor
    y_range = abs(max_y - min_y) * factor

    return x_range, y_range


def display_Dist(Dist, seq_name=None, isMerged=False, isSplit=False):
    """
    Displays a heatmap for the distances between tracklets for one or more sequences.

    Args:
        Dist (array): Distance matrix to display.
        seq_name (str, optional): Specific sequence name to display the heatmap for.
        isMerged (bool): Flag indicating whether the distances are post-merge.
        isSplit (bool): Flag indicating whether the distances are post-split.
    """
    split_info = " After Split" if isSplit else " Before Split"
    merge_info = " After Merge" if isMerged else " Before Merge"
    info = split_info + merge_info
    
    plt.figure(figsize=(10, 8))  # Optional: adjust the size of the heatmap

    # Plot the heatmap
    sns.heatmap(Dist, cmap='Blues')

    plt.title(f"{seq_name}{info}")
    plt.show()


def get_distance_matrix(tid2track):
    """
    Constructs and returns a distance matrix between all tracklets based on overlapping times and feature similarities.
    Uses PyTorch for efficient batch computation when available.

    Args:
        tid2track (dict): Dictionary mapping track IDs to their respective track objects.

    Returns:
        ndarray: A square matrix where each element (i, j) represents the calculated distance between track i and track j.
    """
    n_tracks = len(tid2track)
    Dist = np.zeros((n_tracks, n_tracks))
    
    # Check if we can use PyTorch batch processing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_batch_processing = False #torch.cuda.is_available() and n_tracks > 10
    
    if use_batch_processing:
        logger.info(f"Using PyTorch batch processing for {n_tracks} tracklets on {device}")
        
        # Collect all tracklets and features
        track_list = list(tid2track.items())
        
        # Precompute temporal overlaps efficiently
        temporal_overlaps = np.zeros((n_tracks, n_tracks), dtype=bool)
        for i, (_, track1) in enumerate(track_list):
            times1_set = set(track1.times)
            for j, (_, track2) in enumerate(track_list):
                if i != j:
                    times2_set = set(track2.times)
                    temporal_overlaps[i, j] = bool(times1_set & times2_set)
        
        # Batch process features when possible
        try:
            # Group tracklets by feature dimensions for efficient batching
            feature_groups = {}
            for i, (tid, track) in enumerate(track_list):
                if track.features:
                    feat_shape = np.array(track.features[0]).shape
                    if feat_shape not in feature_groups:
                        feature_groups[feat_shape] = []
                    feature_groups[feat_shape].append((i, tid, track))
            
            # Process each feature group in batches
            for feat_shape, group_tracks in feature_groups.items():
                if len(group_tracks) > 1:
                    # Extract features for this group
                    group_features = []
                    group_indices = []
                    
                    for i, tid, track in group_tracks:
                        if len(track.features) > 0:
                            group_features.append(torch.tensor(np.stack(track.features), dtype=torch.float32))
                            group_indices.append(i)
                    
                    if len(group_features) > 1:
                        # Move to GPU if available
                        group_features = [f.to(device) for f in group_features]
                        
                        # Compute pairwise distances within this group
                        for idx1, (i, feat1) in enumerate(zip(group_indices, group_features)):
                            for idx2, (j, feat2) in enumerate(zip(group_indices, group_features)):
                                if i <= j:
                                    if i == j:
                                        Dist[i, j] = 0.0
                                    elif temporal_overlaps[i, j]:
                                        Dist[i, j] = Dist[j, i] = 1.0
                                    else:
                                        # Batch cosine distance computation
                                        with torch.no_grad():
                                            # Normalize features
                                            feat1_norm = torch.nn.functional.normalize(feat1, p=2, dim=1)
                                            feat2_norm = torch.nn.functional.normalize(feat2, p=2, dim=1)
                                            
                                            # Compute cosine similarity matrix
                                            cos_sim = torch.mm(feat1_norm, feat2_norm.t())
                                            cos_dist = 1 - cos_sim
                                            
                                            # Average distance
                                            distance = float(torch.mean(cos_dist))
                                            Dist[i, j] = Dist[j, i] = distance
            
            # Fill in distances for tracks not processed in batches
            for i, (track1_id, track1) in enumerate(track_list):
                for j, (track2_id, track2) in enumerate(track_list):
                    if Dist[i, j] == 0.0 and i != j:  # Not computed yet
                        if temporal_overlaps[i, j]:
                            Dist[i, j] = Dist[j, i] = 1.0
                        else:
                            # Fall back to individual computation
                            Dist[i, j] = get_distance(track1_id, track2_id, track1, track2)
                            Dist[j, i] = Dist[i, j]
                            
        except Exception as e:
            logger.warning(f"Batch processing failed: {e}, falling back to individual computation")
            use_batch_processing = False
    
    if not use_batch_processing:
        # Original implementation
        for i, (track1_id, track1) in enumerate(tid2track.items()):
            assert len(track1.times) == len(track1.bboxes)
            for j, (track2_id, track2) in enumerate(tid2track.items()):
                if j < i:
                    Dist[i][j] = Dist[j][i]
                else:
                    Dist[i][j] = get_distance(track1_id, track2_id, track1, track2)
    
    return Dist


def get_distance(track1_id, track2_id, track1, track2):
    """
    Calculates the cosine distance between two tracks using PyTorch for efficient computation.
    Enhanced with better memory management and error handling.

    Args:
        track1_id (int): ID of the first track.
        track2_id (int): ID of the second track.
        track1 (Tracklet): First track object.
        track2 (Tracklet): Second track object.

    Returns:
        float: Cosine distance between the two tracks.
    """
    assert track1_id == track1.track_id and track2_id == track2.track_id   # debug line
    
    # Check temporal overlap first (fastest check)
    if track1_id != track2_id:
        doesOverlap = bool(set(track1.times) & set(track2.times))
        if doesOverlap:
            return 1.0  # Maximum distance for overlapping tracks
    
    # Check if both tracks have features
    if not track1.features or not track2.features:
        return 1.0
    
    try:
        # Use PyTorch for efficient cosine distance computation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        with torch.no_grad():  # Disable gradient computation for inference
            # Convert features to tensors
            track1_features = torch.tensor(np.stack(track1.features), dtype=torch.float32, device=device)
            track2_features = torch.tensor(np.stack(track2.features), dtype=torch.float32, device=device)
            
            # Normalize features for cosine similarity
            track1_normalized = torch.nn.functional.normalize(track1_features, p=2, dim=1)
            track2_normalized = torch.nn.functional.normalize(track2_features, p=2, dim=1)
            
            # Compute cosine similarity matrix efficiently
            cos_sim_matrix = torch.mm(track1_normalized, track2_normalized.t())
            
            # Convert to cosine distance
            cos_dist_matrix = 1 - cos_sim_matrix
            
            # Compute mean distance
            mean_distance = torch.mean(cos_dist_matrix)
            
            return float(mean_distance)
            
    except Exception as e:
        logger.warning(f"PyTorch distance computation failed for tracks {track1_id}, {track2_id}: {e}")
        
        # Fallback to CPU numpy computation
        try:
            track1_features_np = np.stack(track1.features)
            track2_features_np = np.stack(track2.features)
            
            # Normalize features
            track1_norm = track1_features_np / np.linalg.norm(track1_features_np, axis=1, keepdims=True)
            track2_norm = track2_features_np / np.linalg.norm(track2_features_np, axis=1, keepdims=True)
            
            # Compute cosine similarity
            cos_sim = np.dot(track1_norm, track2_norm.T)
            cos_dist = 1 - cos_sim
            
            return float(np.mean(cos_dist))
            
        except Exception as e2:
            logger.error(f"Both PyTorch and numpy distance computation failed: {e2}")
            return 1.0  # Return maximum distance on failure


def get_distance_voting(track1_id, track2_id, track1, track2):
    """
    Calculates the cosine distance between two tracks using PyTorch for efficient computation.
    Enhanced with better memory management and error handling.

    Args:
        track1_id (int): ID of the first track.
        track2_id (int): ID of the second track.
        track1 (Tracklet): First track object.
        track2 (Tracklet): Second track object.

    Returns:
        float: Cosine distance between the two tracks.
    """
    assert track1_id == track1.track_id and track2_id == track2.track_id   # debug line
    
    # Check temporal overlap first (fastest check)
    if track1_id != track2_id:
        doesOverlap = bool(set(track1.times) & set(track2.times))
        if doesOverlap:
            return 1.0  # Maximum distance for overlapping tracks
    
    # Check if both tracks have features
    if not track1.features or not track2.features:
        return 1.0
    
    try:
        # Use PyTorch for efficient cosine distance computation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        with torch.no_grad():  # Disable gradient computation for inference
            # Convert features to tensors
            track1_features = torch.tensor(np.stack(track1.features), dtype=torch.float32, device=device)
            track2_features = torch.tensor(np.stack(track2.features), dtype=torch.float32, device=device)
            
            # Normalize features for cosine similarity
            track1_normalized = torch.nn.functional.normalize(track1_features, p=2, dim=1)
            track2_normalized = torch.nn.functional.normalize(track2_features, p=2, dim=1)
            
            # Compute cosine similarity matrix efficiently
            cos_sim_matrix = torch.mm(track1_normalized, track2_normalized.t())
            
            # Convert to cosine distance
            cos_dist_matrix = 1 - cos_sim_matrix
            
            # Compute mean distance
            mean_distance = torch.mean(cos_dist_matrix)
            
            return float(mean_distance)
            
    except Exception as e:
        logger.warning(f"PyTorch distance computation failed for tracks {track1_id}, {track2_id}: {e}")
        
        # Fallback to CPU numpy computation
        try:
            track1_features_np = np.stack(track1.features)
            track2_features_np = np.stack(track2.features)
            
            # Normalize features
            track1_norm = track1_features_np / np.linalg.norm(track1_features_np, axis=1, keepdims=True)
            track2_norm = track2_features_np / np.linalg.norm(track2_features_np, axis=1, keepdims=True)
            
            # Compute cosine similarity
            cos_sim = np.dot(track1_norm, track2_norm.T)
            cos_dist = 1 - cos_sim
            
            return float(np.mean(cos_dist))
            
        except Exception as e2:
            logger.error(f"Both PyTorch and numpy distance computation failed: {e2}")
            return 1.0  # Return maximum distance on failure


def check_spatial_constraints(trk_1, trk_2, max_x_range, max_y_range):
    """
    Checks if two tracklets meet spatial constraints for potential merging.

    Args:
        trk_1 (Tracklet): The first tracklet object containing times and bounding boxes.
        trk_2 (Tracklet): The second tracklet object containing times and bounding boxes, to be evaluated
                        against trk_1 for merging possibility.
        max_x_range (float): The maximum allowed distance in the x-coordinate between the end of trk_1 and
                             the start of trk_2 for them to be considered for merging.
        max_y_range (float): The maximum allowed distance in the y-coordinate under the same conditions as
                             the x-coordinate.

    Returns:
        bool: True if the spatial constraints are met (the tracklets are close enough to consider merging),
              False otherwise.
    """
    inSpatialRange = True
    seg_1 = find_consecutive_segments(trk_1.times)
    seg_2 = find_consecutive_segments(trk_2.times)
    
    subtracks = query_subtracks(seg_1, seg_2, trk_1, trk_2)
    assert(len(subtracks) > 1), "No common subtracks"
    subtrack_1st = subtracks.pop(0)
    
    while subtracks:
        subtrack_2nd = subtracks.pop(0)
        if subtrack_1st.parent_id == subtrack_2nd.parent_id:
            subtrack_1st = subtrack_2nd
            continue
        x_1, y_1, w_1, h_1 = subtrack_1st.bboxes[-1][0 : 4]
        x_2, y_2, w_2, h_2 = subtrack_2nd.bboxes[0][0 : 4]
        x_1 += w_1 / 2
        y_1 += h_1 / 2
        x_2 += w_2 / 2
        y_2 += h_2 / 2
        dx = abs(x_1 - x_2)
        dy = abs(y_1 - y_2)
        
        # check the distance between exit location of track_1 and enter location of track_2
        if dx > max_x_range or dy > max_y_range:
            inSpatialRange = False
            break
        else:
            subtrack_1st = subtrack_2nd
    
    return inSpatialRange


def merge_tracklets(tracklets, max_x_range=None, max_y_range=None, merge_dist_thres=None):
    """
    Main tracklet merging function that follows the original CPU implementation logic.
    Uses PyTorch optimizations for distance matrix operations while maintaining the original algorithm.
    """
    logger.info(f"===========Using PyTorch-optimized implementation=============")
    
    # calculate tracklets distance matrix with PyTorch optimizations
    Dist = get_distance_matrix(tracklets)    

    idx2tid = {idx: tid for idx, tid in enumerate(tracklets.keys())}
    
    # Convert distance matrix to PyTorch tensor for efficient operations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_gpu_matrix = torch.cuda.is_available() and len(tracklets) > 50
    
    if use_gpu_matrix:
        logger.info(f"Using GPU tensor operations for distance matrix on {device}")
        Dist_tensor = torch.tensor(Dist, dtype=torch.float32, device=device)
    else:
        Dist_tensor = None
    
    # Hierarchical Clustering
    # While there are still values (exclude diagonal) in distance matrix lower than merging distance threshold
    diagonal_mask = np.eye(Dist.shape[0], dtype=bool)
    non_diagonal_mask = ~diagonal_mask
    
    merges_performed = 0
    
    while (np.any(Dist[non_diagonal_mask] < merge_dist_thres)):
        # Use PyTorch for finding minimum if using GPU tensor
        if use_gpu_matrix and Dist_tensor is not None:
            try:
                with torch.no_grad():
                    # Create mask for diagonal elements
                    diag_mask_tensor = torch.eye(Dist_tensor.shape[0], device=device, dtype=torch.bool)
                    
                    # Set diagonal to infinity to exclude from minimum finding
                    masked_dist = Dist_tensor.clone()
                    masked_dist[diag_mask_tensor] = float('inf')
                    
                    # Find minimum value and indices
                    min_value_tensor = torch.min(masked_dist)
                    min_value = float(min_value_tensor)
                    
                    if min_value >= merge_dist_thres:
                        break
                    
                    # Find indices of minimum value
                    min_indices = torch.where(masked_dist == min_value_tensor)
                    track1_idx = int(min_indices[0][0])
                    track2_idx = int(min_indices[1][0])
                    
                    # Update numpy matrix to keep in sync
                    Dist = Dist_tensor.cpu().numpy()
                    
            except Exception as e:
                logger.warning(f"GPU matrix operations failed: {e}, falling back to CPU")
                use_gpu_matrix = False
                Dist_tensor = None
        
        if not use_gpu_matrix:
            # Original CPU implementation for finding minimum
            min_index = np.argmin(Dist[non_diagonal_mask])
            min_value = np.min(Dist[non_diagonal_mask])
            # Translate this index to the original array's indices
            masked_indices = np.where(non_diagonal_mask)
            track1_idx, track2_idx = masked_indices[0][min_index], masked_indices[1][min_index]
        
        logger.debug("Tracks idx to merge:", track1_idx, track2_idx)
        logger.debug(f"Minimum value in masked Dist: {min_value}")

        assert min_value == Dist[track1_idx, track2_idx] == Dist[track2_idx, track1_idx], "Values should match!"

        track1 = tracklets[idx2tid[track1_idx]]
        track2 = tracklets[idx2tid[track2_idx]]

        inSpatialRange = check_spatial_constraints(track1, track2, max_x_range, max_y_range)
        logger.debug("In spatial range:", inSpatialRange)
        
        if inSpatialRange:
            # Perform the merge
            track1.features += track2.features      # Note: currently we merge track 2 to track 1 without creating a new track
            track1.times += track2.times
            track1.bboxes += track2.bboxes
            track1.scores += track2.scores
            # update tracklets dictionary
            tracklets[idx2tid[track1_idx]] = track1
            tracklets.pop(idx2tid[track2_idx])

            # Remove the merged tracklet (track2) from the distance matrix
            Dist = np.delete(Dist, track2_idx, axis=0)  # Remove row for track2
            Dist = np.delete(Dist, track2_idx, axis=1)  # Remove column for track2
            
            # Update GPU tensor if using it
            if use_gpu_matrix and Dist_tensor is not None:
                try:
                    # Remove row and column from GPU tensor
                    mask = torch.ones(Dist_tensor.shape[0], dtype=torch.bool, device=device)
                    mask[track2_idx] = False
                    Dist_tensor = Dist_tensor[mask][:, mask]
                except Exception as e:
                    logger.warning(f"GPU tensor update failed: {e}, falling back to CPU")
                    use_gpu_matrix = False
                    Dist_tensor = None
            
            # update idx2tid
            idx2tid = {idx: tid for idx, tid in enumerate(tracklets.keys())}
            
            # Update distance matrix only for the merged tracklet's row and column
            # Use vectorized operations where possible
            if track1_idx < len(idx2tid):  # Ensure index is still valid after deletion
                merged_tracklet = tracklets[idx2tid[track1_idx]]
                
                # Batch update distances for efficiency
                for idx in range(Dist.shape[0]):
                    if idx != track1_idx:
                        distance = get_distance(idx2tid[track1_idx], idx2tid[idx], merged_tracklet, tracklets[idx2tid[idx]])
                        Dist[track1_idx, idx] = distance
                        Dist[idx, track1_idx] = distance  # Ensure symmetry
                
                # Update GPU tensor if using it
                if use_gpu_matrix and Dist_tensor is not None:
                    try:
                        Dist_tensor = torch.tensor(Dist, dtype=torch.float32, device=device)
                    except Exception as e:
                        logger.warning(f"GPU tensor sync failed: {e}")
                        use_gpu_matrix = False
                        Dist_tensor = None
            
            # update mask
            diagonal_mask = np.eye(Dist.shape[0], dtype=bool)
            non_diagonal_mask = ~diagonal_mask
            
            merges_performed += 1
            
            # Periodic GPU memory cleanup
            if use_gpu_matrix and merges_performed % 10 == 0:
                torch.cuda.empty_cache()
                
        else:
            # change distance between track pair to threshold
            Dist[track1_idx, track2_idx], Dist[track2_idx, track1_idx] = merge_dist_thres, merge_dist_thres
            
            # Update GPU tensor if using it
            if use_gpu_matrix and Dist_tensor is not None:
                try:
                    Dist_tensor[track1_idx, track2_idx] = merge_dist_thres
                    Dist_tensor[track2_idx, track1_idx] = merge_dist_thres
                except Exception as e:
                    logger.warning(f"GPU tensor update failed: {e}")
                    use_gpu_matrix = False
                    Dist_tensor = None
    
    logger.info(f"Merging completed. Performed {merges_performed} merges using {'GPU' if use_gpu_matrix else 'CPU'} optimizations.")
    return tracklets


def merge_tracklets_batched(
    tracklets, 
    seq2Dist,
    batch_size=50,
    seq_name=None,
    max_x_range=None,
    max_y_range=None,
    merge_dist_thres=None
):
    """
    Merges tracklets in batches based on a distance threshold.
    
    Parameters:
    tracklets (dict): A dictionary of tracklets where keys are tracklet IDs and values are tracklet objects.
    seq2Dist (dict): A dictionary to store distance matrices for sequences.
    batch_size (int): The size of the batches to process at a time.
    seq_name (str): The name of the sequence being processed.
    max_x_range (float): Maximum allowed distance in the x direction for merging.
    max_y_range (float): Maximum allowed distance in the y direction for merging.
    merge_dist_thres (float): Distance threshold below which tracklets should be merged.
    
    Returns:
    dict: The merged tracklets.
    """
    temp_tracklets = {}
    tracklet_items = list(tracklets.items())

    print(f"Batch size: {batch_size}")
    for i in range(0, len(tracklet_items), batch_size):
        batch_tracklets = dict(tracklet_items[i:i+batch_size])
        print(f"Processing batch from index {i} to {min(i+batch_size - 1, len(tracklet_items) - 1)}")
        
        merged_batch_tracklets= merge_tracklets(batch_tracklets, merge_dist_thres, max_x_range, max_y_range)
        print(f"{len(merged_batch_tracklets)} of {batch_size} tracklets left after merging.")
        temp_tracklets.update(merged_batch_tracklets)
    print(f"Merging {len(temp_tracklets)} tracklets after batched processing.")
    print()
    merged_tracklets = merge_tracklets(temp_tracklets, merge_dist_thres, max_x_range, max_y_range)
    
    return merged_tracklets


def detect_id_switch(embs, eps=None, min_samples=None, max_clusters=None):
    """
    Detects identity switches within a tracklet using clustering.

    Args:
        embs (list of numpy arrays): A list where each element is a numpy array representing an embedding.
                                     Each embedding has the same dimensionality.
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
        bool: True if an identity switch is detected, otherwise False.
    """
    embs = np.stack(embs)
    
    # Standardize the embeddings
    scaler = StandardScaler()
    embs_scaled = scaler.fit_transform(embs)

    # Apply DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(embs_scaled)
    labels = db.labels_

    # Count the number of clusters (excluding noise)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]

    if -1 in labels and len(unique_labels) > 1:
        # Find the cluster centers
        cluster_centers = np.array([embs_scaled[labels == label].mean(axis=0) for label in unique_labels])
        
        # Assign noise points to the nearest cluster
        noise_indices = np.where(labels == -1)[0]
        for idx in noise_indices:
            distances = cdist([embs_scaled[idx]], cluster_centers, metric='cosine')
            nearest_cluster = np.argmin(distances)
            labels[idx] = list(unique_labels)[nearest_cluster]
    
    n_clusters = len(unique_labels)

    if max_clusters and n_clusters > max_clusters:
        # Merge clusters to ensure the number of clusters does not exceed max_clusters
        while n_clusters > max_clusters:
            cluster_centers = np.array([embs_scaled[labels == label].mean(axis=0) for label in unique_labels])
            distance_matrix = cdist(cluster_centers, cluster_centers, metric='cosine')
            np.fill_diagonal(distance_matrix, np.inf)  # Ignore self-distances
            
            # Find the closest pair of clusters
            min_dist_idx = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
            cluster_to_merge_1, cluster_to_merge_2 = unique_labels[min_dist_idx[0]], unique_labels[min_dist_idx[1]]

            # Merge the clusters
            labels[labels == cluster_to_merge_2] = cluster_to_merge_1
            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels != -1]
            n_clusters = len(unique_labels)

    return n_clusters > 1, labels


def split_tracklets(tmp_trklets, eps=None, max_k=None, min_samples=None, len_thres=None):
    """
    Splits each tracklet into multiple tracklets based on an internal distance threshold.

    Args:
        tmp_trklets (dict): Dictionary of tracklets to be processed.
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        len_thres (int): Length threshold to filter out short tracklets.
        max_k (int): Maximum number of clusters to consider.

    Returns:
        dict: New dictionary of tracklets after splitting.
    """
    new_id = max(tmp_trklets.keys()) + 1
    tracklets = defaultdict()
    # Splitting algorithm to process every tracklet in a sequence
    for tid in tqdm(sorted(list(tmp_trklets.keys())), total=len(tmp_trklets), desc="Splitting tracklets"):
        trklet = tmp_trklets[tid]
        if len(trklet.times) < len_thres:  # NOTE: Set tracklet length threshold to filter out short ones
            tracklets[tid] = trklet
        else:
            embs = np.stack(trklet.features)
            frames = np.array(trklet.times)
            bboxes = np.stack(trklet.bboxes)
            scores = np.array(trklet.scores)
            # Perform DBSCAN clustering
            id_switch_detected, clusters = detect_id_switch(embs, eps=eps, min_samples=min_samples, max_clusters=max_k)
            if not id_switch_detected:
                tracklets[tid] = trklet
            else:
                unique_labels = set(clusters)

                for label in unique_labels:
                    if label == -1:
                        continue  # Skip noise points
                    tmp_embs = embs[clusters == label]
                    tmp_frames = frames[clusters == label]
                    tmp_bboxes = bboxes[clusters == label]
                    tmp_scores = scores[clusters == label]
                    assert new_id not in tmp_trklets
                    
                    tracklets[new_id] = Tracklet(new_id, tmp_frames.tolist(), tmp_scores.tolist(), tmp_bboxes.tolist(), feats=tmp_embs.tolist())
                    new_id += 1

    assert len(tracklets) >= len(tmp_trklets)
    return tracklets


def save_results(sct_output_path, tracklets):
    """
    Saves the final tracklet results into a specified path.

    Args:
        sct_output_path (str): Path where the results will be saved.
        tracklets (dict): Dictionary of tracklets containing their final states.

    """
    results = []

    for i, tid in enumerate(sorted(tracklets.keys())): # add each track to results
        track = tracklets[tid]
        tid = i + 1
        for instance_idx, frame_id in enumerate(track.times):
            bbox = track.bboxes[instance_idx]
            
            results.append(
                [frame_id, tid, bbox[0], bbox[1], bbox[2], bbox[3], 1, -1, -1, -1]
            )
    results = sorted(results, key=lambda x: x[0])
    txt_results = []
    for line in results:
        txt_results.append(
            f"{line[0]},{line[1]},{line[2]:.2f},{line[3]:.2f},{line[4]:.2f},{line[5]:.2f},{line[6]},{line[7]},{line[8]},{line[9]}\n"
            )
    
    # NOTE: uncomment to save results
    with open(sct_output_path, 'w') as f:
        f.writelines(txt_results)
    logger.info(f"save SCT results to {sct_output_path}")


def save_results_pkl(sct_output_path, tracklets):
    """
    Saves the final tracklet results as Tracklet objects into a pickle file.
    Format matches the output of generate_tracklets.py - a dictionary where 
    keys are track IDs and values are Tracklet objects.
    Preserves original track IDs from the input tracklets.

    Args:
        sct_output_path (str): Path where the pickle file will be saved.
        tracklets (dict): Dictionary of tracklets containing their final states.
    """
    # Save as pickle file (ensure the path has .pkl extension)
    if not sct_output_path.endswith('.pkl'):
        sct_output_path = sct_output_path.replace('.txt', '.pkl')
    
    with open(sct_output_path, 'wb') as f:
        pickle.dump(tracklets, f)
    
    logger.info(f"Saved {len(tracklets)} refined tracklets to {sct_output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Global tracklet association with splitting and connecting.")
    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        help='Dataset name (e.g., SportsMOT, SoccerNet).')
    
    parser.add_argument('--tracker',
                        type=str,
                        required=True,
                        help='Tracker name.')
    
    parser.add_argument('--track_src',
                        type=str,
                        default=r"C:\Users\Ciel Sun\OneDrive - UW\EE 599\SoccerNet\SORT_results\SORT_Tracklets_test",
                        required=True,
                        help='Source directory of tracklet pkl files.'
                        )
    
    parser.add_argument('--use_split',
                        action='store_true',
                        help='If using split component.')
    
    parser.add_argument('--min_len',
                        type=int,
                        default=100,
                        help='Minimum length for a tracklet required for splitting.')
    
    parser.add_argument('--eps',
                        type=float,
                        default=0.7,
                        help='For DBSCAN clustering, the maximum distance between two samples for one to be considered as in the neighborhood of the other.')
    
    parser.add_argument('--min_samples',
                        type=int,
                        default=10,
                        help='The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.')
    
    parser.add_argument('--max_k',
                        type=int,
                        default=3,
                        help='Maximum number of clusters/subtracklets to be output by splitting component.')
    
    parser.add_argument('--use_connect',
                        action='store_true',
                        help='If using connecting component.')
    
    parser.add_argument('--spatial_factor',
                        type=float,
                        default=1,
                        help='Factor to adjust spatial distances.')
    
    parser.add_argument('--merge_dist_thres',
                        type=float,
                        default=0.4,
                        help='Minimum cosine distance between two tracklets for merging.')
    return parser.parse_args()


def main():
    args = parse_args()
    # Determine the process based on the flags
    if args.use_split and args.use_connect:
        process = "Split+Connect"
    elif args.use_split:
        process = "Split"
    elif args.use_connect:
        process = "Connect"
    else:
        raise ValueError("Both use_split and use_connect are false, must at least use connect.")

    # Log PyTorch capabilities
    if torch.cuda.is_available():
        logger.info(f"🚀 PyTorch GPU acceleration available: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.info("⚠️  PyTorch GPU not available, using CPU optimizations")

    seq_tracks_dir = args.track_src
    data_path = seq_tracks_dir # os.path.dirname(seq_tracks_dir)
    seqs_tracks = [f for f in os.listdir(seq_tracks_dir) if os.path.isfile(os.path.join(seq_tracks_dir, f))]
    
    tracker = args.tracker
    dataset = args.dataset

    seqs_tracks.sort()
    seq2Dist = dict()

    process_limit = 10000                          # debug line, delete later
    total_start_time = time.time()
    
    for seq_idx, seq in enumerate(seqs_tracks):
        if seq_idx >= process_limit:            # debug line, delete later
            break                               # debug line, delete later
        
        seq_name = seq.split('.')[0]
        seq_start_time = time.time()
        logger.info(f"Processing seq {seq_idx+1} / {len(seqs_tracks)}: {seq_name}")
        
        with open(os.path.join(seq_tracks_dir, seq), 'rb') as pkl_f:
            tmp_trklets = pickle.load(pkl_f)     # dict(key:track id, value:tracklet)

        logger.info(f"Loaded {len(tmp_trklets)} tracklets")
        max_x_range, max_y_range = get_spatial_constraints(tmp_trklets, args.spatial_factor)
        
        # Distance matrix computation with timing
        dist_start_time = time.time()
        Dist = get_distance_matrix(tmp_trklets)
        dist_time = time.time() - dist_start_time
        logger.info(f"Distance matrix computed in {dist_time:.2f}s")
        
        seq2Dist[seq_name] = Dist                                              # save all seqs distance matrix, debug line, delete later
        # display_Dist(Dist, seq_name, isMerged=False, isSplit=False)         # used to display Dist, debug line, delete later

        if args.use_split:
            split_start_time = time.time()
            print(f"----------------Number of tracklets before splitting: {len(tmp_trklets)}----------------")
            splitTracklets = split_tracklets(tmp_trklets, eps=args.eps, max_k=args.max_k, min_samples=args.min_samples, len_thres=args.min_len)
            split_time = time.time() - split_start_time
            logger.info(f"Splitting completed in {split_time:.2f}s")
        else:
            splitTracklets = tmp_trklets
        
        # Recompute distance matrix after splitting
        if args.use_split:
            dist_start_time = time.time()
            Dist = get_distance_matrix(splitTracklets)
            dist_time = time.time() - dist_start_time
            logger.info(f"Post-split distance matrix computed in {dist_time:.2f}s")
        
        # display_Dist(Dist, seq_name, isMerged=False, isSplit=True)
        print(f"----------------Number of tracklets before merging: {len(splitTracklets)}----------------")
        
        merge_start_time = time.time()
        mergedTracklets = merge_tracklets(splitTracklets, max_x_range=max_x_range, max_y_range=max_y_range, merge_dist_thres=args.merge_dist_thres)
        merge_time = time.time() - merge_start_time
        logger.info(f"Merging completed in {merge_time:.2f}s")
        
        Dist = get_distance_matrix(mergedTracklets)
        # display_Dist(Dist, seq_name, isMerged=True, isSplit=True)
        print(f"----------------Number of tracklets after merging: {len(mergedTracklets)}----------------")

        # sct_name = f'{tracker}_{dataset}_{process}_eps{args.eps}_minSamples{args.min_samples}_K{args.max_k}_mergeDist{args.merge_dist_thres}_spatial{args.spatial_factor}'
        sct_name = f"Refined"
        os.makedirs(os.path.join(data_path, sct_name), exist_ok=True)
        # new_sct_output_path = os.path.join(data_path, sct_name, '{}.txt'.format(seq_name))
        new_sct_output_path = os.path.join(data_path, sct_name, '{}_refined.pkl'.format(seq_name))
        save_results_pkl(new_sct_output_path, mergedTracklets)
        
        seq_time = time.time() - seq_start_time
        logger.info(f"Sequence {seq_name} completed in {seq_time:.2f}s")
        
        # Clean up GPU memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    total_time = time.time() - total_start_time
    logger.info(f"🎉 All sequences processed in {total_time:.2f}s using PyTorch optimizations")


if __name__ == "__main__":
    main()
