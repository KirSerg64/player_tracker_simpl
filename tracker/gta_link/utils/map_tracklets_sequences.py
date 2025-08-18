from pathlib import Path
import numpy as np
import os
import torch
import pickle

from collections import defaultdict

from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from loguru import logger
from tqdm import tqdm

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

from sn_gamestate.gta_link.Tracklet import Tracklet

import argparse
from copy import copy
import shutil
import cv2 as cv
from collections import defaultdict
import random
import cv2


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


def reid_mapped_tracklets(tracklets, mapping):
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
            old_tracklet = copy(tracklets[new_id])
            old_tracklet.track_id = None
            rewritten_tracklets.append(old_tracklet)
        new_tracklet = copy(tracklets[old_id])
        new_tracklet.track_id = new_id
        new_tracklets[new_id] = new_tracklet   
    
    # Copy non-mapped tracklets from original dict
    old_new_union = old_mapped_ids.union(new_mapped_ids)
    for track_id, tracklet in tracklets.items():
        if track_id not in old_new_union:
            new_tracklets[track_id] = copy(tracklet)
            new_tracklets[track_id].track_id = track_id
    
    not_used_ids = list(existing_ids.difference(new_tracklets.keys()))
    assert len(not_used_ids) == len(rewritten_tracklets), \
        f"Not enough elements in not_used_ids: {len(not_used_ids)} to assign"
    
    for tracklet in rewritten_tracklets:
        track_id = not_used_ids.pop(0)
        new_tracklets[track_id] = copy(tracklet)
        new_tracklets[track_id].track_id = track_id

    return new_tracklets


def load_tracklets(seq_tracks_dir: str) -> Dict[int, Dict[int, Tracklet]]:
    
    seqs_tracks = sorted(os.listdir(seq_tracks_dir))

    tracklets_spartial = {}
    for seq_idx, seq in enumerate(seqs_tracks):
        seq_idx = int(seq_idx)

        seq_name = seq.split('.')[0] 
        logger.info(f"Processing seq {seq_idx+1} / {len(seqs_tracks)}")
        with open(os.path.join(seq_tracks_dir, seq), 'rb') as pkl_f:
            tmp_trklets = pickle.load(pkl_f)     # dict(key:track id, value:tracklet)

        tracklets_spartial[seq_idx] = {
            "tracklets": tmp_trklets, 
            "time_delta":max([max(tracklet.times) for _,tracklet in tmp_trklets.items()]),
            "max_trackid": max(tmp_trklets.keys()),
        }
    return tracklets_spartial


def draw_mot_tracklets_to_video(
    tracklets,
    img_folder,
    output_video_path,
    box_color_fn=None,
    thickness=1,
    font_scale=0.5,
    frame_rate=15
):
    # Create frame_bboxes dictionary
    frame_bboxes = defaultdict(list)
    for track_id, track in tracklets.items():
        for instance_idx, frame_id in enumerate(track.times):
            bbox = track.bboxes[instance_idx]            
            frame_bboxes[frame_id].append(
                [track_id, bbox[0], bbox[1], bbox[2], bbox[3]]
            )
    
    # Get sorted list of image files from directory
    img_files = sorted([
        f for f in os.listdir(img_folder) 
        if f.endswith(('.jpg', '.png'))
    ])

    # Read first frame to get size
    first_frame = cv2.imread(os.path.join(img_folder, img_files[0]))
    if first_frame is None:
        raise ValueError("Could not read first image.")
    height, width = first_frame.shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    # Process each frame
    for img_name in img_files:
        frame_id = float(os.path.splitext(img_name)[0])
        img_path = os.path.join(img_folder, img_name)
        frame = cv2.imread(img_path)
        
        if frame is None:
            logger.warning(f"Could not read {img_path}")
            continue

        # Draw boxes for this frame
        if frame_id in frame_bboxes:
            for track_id, x, y, w, h in frame_bboxes[frame_id]:
                color = box_color_fn(track_id) if box_color_fn else (0, 255, 0)
                x, y, w, h = int(x), int(y), int(w), int(h)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
                cv2.putText(frame, f'ID {int(track_id)}', (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

        video_writer.write(frame)

    video_writer.release()
    logger.info(f"Video saved to: {output_video_path}")


# def parse_args():   
#     parser = argparse.ArgumentParser(description="Merge tracklets from multiple sequences")
#     parser.add_argument('--seq_tracks_dir', type=str, required=True, help='Directory containing sequence tracklets')
#     parser.add_argument('--output_dir', type=str, required=True, help='Output directory for merged tracklets')
#     return parser.parse_args()


def main(args=None):
    seq_tracks_dir = Path(args.seq_tracks_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tracklets_spartial = load_tracklets(seq_tracks_dir)

    new_tracklets_spartial = {}

    for seq, tracklets in tracklets_spartial.items():
        if seq == 0:
            new_tracklets_spartial[seq] = tracklets
        else:
            # Find the best matches
            (mapping, 
            overlaps, 
            best_start_window, 
            best_end_window, 
            best_start_bboxes, 
            best_end_bboxes
            ) = find_closest_tracklets_mapping(
                new_tracklets_spartial[seq-1], 
                tracklets, 
                max_start_window=args.max_start_window, 
                max_end_window=args.max_end_window,
                max_overlap_threshold=args.max_overlap_threshold
            )
            # Print mapping
            for start_id, end_id in mapping.items():
                logger.info(f"Track {start_id} from previous sequence matches with track {end_id} of next "
                            f"(overlap: {overlaps[start_id]:.3f})")

            new_tracklets_dict = reid_mapped_tracklets(tracklets['tracklets'], mapping)

            new_tracklets_dict = {
                "tracklets": new_tracklets_dict, 
                "time_delta":max([max(tracklet.times) for _,tracklet in new_tracklets_dict.items()]),
                "max_trackid": max(new_tracklets_dict.keys()),
            }
            new_tracklets_spartial[seq] = new_tracklets_dict

    # Save new tracklets
    for seq, tracklets in new_tracklets_spartial.items():
        draw_mot_tracklets_to_video(
            tracklets['tracklets'],
            img_folder=seq_tracks_dir / f'seq_{seq}',
            output_video_path=output_dir / f'seq_{seq}_tracklets.mp4',
            # box_color_fn=lambda track_id: (0, 255, 0),  # Green for all boxes
            thickness=2,
            font_scale=0.5,
            frame_rate=15
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge tracklets from multiple sequences")
    parser.add_argument('--seq_tracks_dir', type=str, required=True, help='Directory containing sequence tracklets')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for merged tracklets')
    parser.add_argument('--max_start_window', type=int, default=15, help='Maximum start window for tracklet matching')
    parser.add_argument('--max_end_window', type=int, default=15, help='Maximum end window for tracklet matching')
    parser.add_argument('--max_overlap_threshold', type=float, default=0.9, help='Maximum overlap threshold for tracklet matching')
    args = parser.parse_args()

    main(args)    
 