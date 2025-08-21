import copy
from fnmatch import fnmatch
from typing import Dict, List
import numpy as np
import os
import torch
import pickle

from collections import defaultdict

from loguru import logger

from tracker.gta_link.utils.Tracklet import Tracklet

import argparse


def get_distance_matrix(tid2track):
    """
    Constructs and returns a distance matrix between all tracklets based on overlapping times and feature similarities.

    Args:
        tid2track (dict): Dictionary mapping track IDs to their respective track objects.

    Returns:
        ndarray: A square matrix where each element (i, j) represents the calculated distance between track i and track j.
    """
    # print("number of tracks:", len(tid2track))
    Dist = np.zeros((len(tid2track), len(tid2track)))

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

    Args:
        track1_id (int): ID of the first track.
        track2_id (int): ID of the second track.
        track1 (Tracklet): First track object.
        track2 (Tracklet): Second track object.

    Returns:
        float: Cosine distance between the two tracks.
    """
    assert track1_id == track1.track_id and track2_id == track2.track_id   # debug line
    doesOverlap = False
    if (track1_id != track2_id):
        doesOverlap = set(track1.times) & set(track2.times)
    if doesOverlap:
        return 1                # make the cosine distance between two tracks maximum, max = 1
    else:
        # calculate cosine distance between two tracks based on features
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        track1_features_tensor = torch.tensor(np.stack(track1.features), dtype=torch.float32).to(device)
        track2_features_tensor = torch.tensor(np.stack(track2.features), dtype=torch.float32).to(device)
        count1 = len(track1_features_tensor)
        count2 = len(track2_features_tensor)

        cos_sim_Numerator = torch.matmul(track1_features_tensor, track2_features_tensor.T)
        track1_features_dist = torch.norm(track1_features_tensor, p=2, dim=1, keepdim=True)
        track2_features_dist = torch.norm(track2_features_tensor, p=2, dim=1, keepdim=True)
        cos_sim_Denominator = torch.matmul(track1_features_dist, track2_features_dist.T)
        cos_Dist = 1 - cos_sim_Numerator / cos_sim_Denominator
        
        total_cos_Dist = cos_Dist.sum()
        result = total_cos_Dist / (count1 * count2)
        return result


def calculate_bbox_distance(bbox1: List[float], bbox2: List[float]) -> float:
    """Calculate Euclidean distance between two bounding box centers"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Calculate centers
    center1_x = x1 + w1 / 2
    center1_y = y1 + h1 / 2
    center2_x = x2 + w2 / 2
    center2_y = y2 + h2 / 2
    
    return np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)


def calculate_bbox_overlap(bbox1: List[float], bbox2: List[float]) -> float:
    """Calculate IoU (Intersection over Union) between two bounding boxes"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Calculate intersection
    left = max(x1, x2)
    top = max(y1, y2)
    right = min(x1 + w1, x2 + w2)
    bottom = min(y1 + h1, y2 + h2)
    
    if left < right and top < bottom:
        intersection = (right - left) * (bottom - top)
    else:
        intersection = 0
    
    # Calculate union
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def get_closest_tracklets_full_matrix(
        start_tracklets, 
        end_tracklets,
        start_window=1, 
        end_window=1,
    ):
    overlap = np.zeros((len(start_tracklets['tracklets']), len(end_tracklets['tracklets'])))

    start_bboxes = {}
    start_delta = start_tracklets['time_delta'] - start_window    
    for track_id, tracklet in start_tracklets['tracklets'].items():
        ids = np.nonzero(np.asarray(tracklet.times) > start_delta)[0]
        if ids.shape[0] > 0:
            if track_id not in start_bboxes:
                start_bboxes[track_id] = np.asarray(tracklet.bboxes)[ids]
            else:
                start_bboxes[track_id] = np.vstack((start_bboxes[track_id], np.asarray(tracklet.bboxes)[ids]))    

    end_bboxes = {}
    end_delta = np.float64(end_window)
    for track_id, tracklet in end_tracklets['tracklets'].items():
        ids = np.nonzero(tracklet.times < end_delta)[0][-1:]
        if ids.shape[0] > 0:
            if track_id not in end_bboxes:
                end_bboxes[track_id] = np.asarray(tracklet.bboxes)[ids]
            else:
                end_bboxes[track_id] = np.vstack((end_bboxes[track_id], np.asarray(tracklet.bboxes)[ids]))  
    
    overlap = np.zeros((len(start_bboxes), len(end_bboxes)))

    # Get sorted lists of keys for consistent indexing
    start_keys = sorted(start_bboxes.keys())
    end_keys = sorted(end_bboxes.keys())

    # Create mapping from key to index
    start_tracklet2id = {key: i for i, key in enumerate(start_keys)}
    end_tracklet2id = {key: i for i, key in enumerate(end_keys)}

    for start_id, bbox0 in start_bboxes.items():
        for end_id, bbox1 in end_bboxes.items():
            i = start_tracklet2id[start_id]
            j = end_tracklet2id[end_id]
            overlap[i,j] = calculate_bbox_overlap(bbox0[0], bbox1[0])    # Take first bbox if multiple

    return (
        overlap, 
        start_bboxes, 
        end_bboxes, 
        start_tracklet2id, 
        end_tracklet2id,
    )


def find_closest_tracklets_mapping(
    start_tracklets, 
    end_tracklets, 
    max_start_window=5, 
    max_end_window=5,
    max_overlap_threshold=0.9,
    ):

    best_mapping = {}
    best_overlaps = {}
    best_start_window = 1
    best_end_window = 1
    # Initialize with defaultdicts
    best_start_bboxes = defaultdict(list)
    best_end_bboxes = defaultdict(list)
    
    # Try different window sizes to find best matches
    for start_window in range(1, max_start_window + 1):
        for end_window in range(1, max_end_window + 1):
            # Get closest tracklets with current windows
            (overlap_matrix, 
             start_bboxes, 
             end_bboxes, 
             start_tracklet2id, 
             end_tracklet2id
            ) = get_closest_tracklets_full_matrix(
                start_tracklets, 
                end_tracklets,
                start_window=start_window, 
                end_window=end_window,
            )
            start_id2tracklet = {i: track_id for track_id, i in start_tracklet2id.items()}
            end_id2tracklet = {i: track_id for track_id, i in end_tracklet2id.items()}

            # Update best mappings if we find better overlaps
            for i in range(overlap_matrix.shape[0]):
                start_id = start_id2tracklet[i]
                max_overlap = np.max(overlap_matrix[i])
                
                # Only consider mappings with significant overlap
                if max_overlap > max_overlap_threshold:  # You can adjust this threshold
                    if start_id not in best_overlaps or max_overlap > best_overlaps[start_id]:                        
                        # Get the track ID from end_tracklets that corresponds to this match
                        matched_idx = np.argmax(overlap_matrix[i])
                        end_track_id = end_id2tracklet[matched_idx]
                        # save mapping
                        if end_track_id in best_mapping.values():
                            continue
                        best_overlaps[start_id] = max_overlap
                        best_mapping[start_id] = end_track_id
                        # window sizes with maximum intersection
                        best_start_window = start_window
                        best_end_window = end_window
                        # append new boxes to the best bboxes
                        best_start_bboxes[start_id].append(start_bboxes[start_id])
                        best_end_bboxes[end_track_id].append(end_bboxes[end_track_id])
    # After the loop, convert to arrays
    best_start_bboxes = {
        track_id: np.concatenate(bbox_list, axis=0) 
        for track_id, bbox_list in best_start_bboxes.items()
    }
    best_end_bboxes = {
        track_id: np.concatenate(bbox_list, axis=0) 
        for track_id, bbox_list in best_end_bboxes.items()
    }
    return best_mapping, best_overlaps, best_start_window, best_end_window, best_start_bboxes, best_end_bboxes


def replace_tracklet_keys(tracklets, mapping):
    # Create a new dictionary with the same structure
    new_tracklets = {}    
    # Get all existing track IDs
    existing_ids = set(tracklets.keys())
    # Mapped tracklets
    new_mapped_ids = set(mapping.keys())
    # non-conflict ids
    old_mapped_ids = set(mapping.values())    
    
    reverse_mapping = {v: k for k, v in mapping.items()}
    old_new_intersection = old_mapped_ids.intersection(new_mapped_ids)
    rewritten_tracklets = []

    for old_id in old_mapped_ids:
        new_id = reverse_mapping[old_id]
        if (new_id != old_id) and (old_id not in old_new_intersection):
            old_tracklet = copy.deepcopy(tracklets[old_id])
            old_tracklet.track_id = None
            rewritten_tracklets.append(old_tracklet)
        new_tracklet = copy.deepcopy(tracklets[old_id])
        new_tracklet.track_id = new_id
        new_tracklets[new_id] = new_tracklet   
    
    # Copy non-mapped tracklets from original dict
    old_new_union = old_mapped_ids.union(new_mapped_ids)
    for track_id, tracklet in tracklets.items():
        if track_id not in old_new_union:
            new_tracklets[track_id] = copy.deepcopy(tracklet)
            new_tracklets[track_id].track_id = track_id
    
    not_used_ids = list(existing_ids.difference(new_tracklets.keys()))
    assert len(not_used_ids) == len(rewritten_tracklets), \
        f"Not enough elements in not_used_ids: {len(not_used_ids)} to assign"
     
    for tracklet in rewritten_tracklets:
        track_id = not_used_ids.pop(0)
        new_tracklets[track_id] = copy.deepcopy(tracklet)
        new_tracklets[track_id].track_id = track_id
        # logger.info(f"Assigned tracklet {tracklet.track_id} to new track ID {track_id}")

    return new_tracklets

def save_refined_tracklets(refined_tracklets: Dict, output_path: str):
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
        logger.info(f"Refined tracklets saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save refined tracklets to {output_path}: {e}")


def list_dir_tree(root_path, template = "video_segment_*.pkl"):
    """
    List all files in a directory tree, excluding .jpg files.

    Args:
        root_path (str): The root directory to start the search.

    Returns:
        list: A list of all files (excluding .jpg) in the directory tree.
    """
    all_files = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            if fnmatch(filename, template):
                all_files.append(os.path.join(dirpath, filename))
    return all_files


def parse_args():
    parser = argparse.ArgumentParser(description="Global tracklet association with splitting and connecting.")
    
    parser.add_argument('--track_base_dir',
                        type=str,
                        default=r"",
                        required=True,
                        help='Source directory of tracklet pkl files.'
                        )

    # parser.add_argument('--save_path',
    #                     type=str,
    #                     default=r"",
    #                     required=True,
    #                     help='Output directory for tracklets and video.'
    #                     )
    
    parser.add_argument('--max_start_window',
                        type=int,
                        default=15,
                        required=False,
                        help='Maximum start window for tracklet matching.'
                        )

    parser.add_argument('--max_end_window',
                        type=int,
                        default=15,
                        required=False,
                        help='Maximum end window for tracklet matching.'
                        )


    return parser.parse_args()


def main():
    args = parse_args()

    seq_tracks_dir = args.track_base_dir
    # data_path = os.path.dirname(seq_tracks_dir)
    # save_path = args.track_base_dir #args.save_path
    seqs_tracks = list_dir_tree(seq_tracks_dir)

    seqs_tracks.sort()

    tracklets_spatial = {0:None, 1:None}  # always only previous and current track
    for seq_idx, seq in enumerate(seqs_tracks):
        seq_idx = int(seq_idx)
        save_path = os.path.dirname(seq)
        seq_name = os.path.splitext(os.path.basename(seq))[0]
        logger.info(f"Processing seq {seq_idx+1} / {len(seqs_tracks)}")
        with open(os.path.join(seq_tracks_dir, seq), 'rb') as pkl_f:
            tmp_trklets = pickle.load(pkl_f)     # dict(key:track id, value:tracklet)

        if seq_idx == 0:
            tracklets_spatial[0] = {'tracklets': tmp_trklets, 
                                    "time_delta":max([max(tracklet.times) for _,tracklet in tmp_trklets.items()])}
            save_refined_tracklets(tracklets_spatial[0], os.path.join(save_path, f"merged_{seq_name}.pkl"))
            continue
        else:
            tracklets_spatial[1] = {'tracklets': tmp_trklets, 
                                    "time_delta":max([max(tracklet.times) for _,tracklet in tmp_trklets.items()])}
        # Find the best matches
        mapping, overlaps, _, _, _, _ = find_closest_tracklets_mapping(
            tracklets_spatial[0], 
            tracklets_spatial[1], 
            max_start_window=args.max_start_window, 
            max_end_window=args.max_end_window
        )
        # Print results
        for start_id, end_id in mapping.items():
            logger.info(f"Track {start_id} matches with track {end_id} (overlap: {overlaps[start_id]:.3f})")

        new_tracklets_dict = replace_tracklet_keys(tracklets_spatial[1]['tracklets'], mapping)

        new_tracklets_dict = {
            "tracklets": new_tracklets_dict, 
            "time_delta":max([max(tracklet.times) for _,tracklet in new_tracklets_dict.items()])
        }
        save_refined_tracklets(new_tracklets_dict, os.path.join(save_path, f"merged_{seq_name}.pkl"))
        tracklets_spatial[0] = new_tracklets_dict

    return

if __name__ == "__main__":
    main()