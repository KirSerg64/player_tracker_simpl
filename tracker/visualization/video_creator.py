"""
Video creation utilities for tracklet visualization.

This module contains functions for creating videos with refined tracklets,
providing visual analysis of the tracking results.
"""

import cv2
import os
import logging
from typing import Dict
from tqdm import tqdm

from tracker.visualization.players_drawer import EllipseDetection

log = logging.getLogger(__name__)


def create_final_tracklet_video(video_path: str, final_tracklets: dict, output_path: str, show_trajectories: bool = False):
    """
    Create a video showing the final refined tracklets.
    
    Args:
        video_path: Path to original video
        final_tracklets: Dictionary of final refined tracklets
        output_path: Path for output video
        show_trajectories: Whether to show trajectory trails
    """
    log.info(f"Creating final tracklet video: {output_path}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Open original video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error(f"Failed to open video: {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        log.error(f"Failed to create video writer for: {output_path}")
        cap.release()
        return
    
    # Create visualizer
    visualizer = EllipseDetection()
    
    # Progress bar for final video creation
    final_progress = tqdm(
        total=total_frames,
        desc="Creating final tracklet video",
        unit="frames",
        ncols=100
    )
    
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw final tracklets for current frame
        visualizer.draw_final_tracklets(frame, final_tracklets, frame_id)
        
        # Optionally draw trajectories
        if show_trajectories:
            visualizer.draw_tracklet_trajectories(frame, final_tracklets, frame_id)
        
        # Add watermark/info
        info_text = f"Final Tracklets: {len(final_tracklets)} | Frame: {frame_id}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
        frame_id += 1
        final_progress.update(1)
    
    # Cleanup
    cap.release()
    out.release()
    final_progress.close()
    
    log.info(f"Final tracklet video saved: {output_path}")


def create_comparison_video(video_path: str, real_time_tracklets: list, final_tracklets: dict, output_path: str):
    """
    Create a side-by-side comparison video showing real-time vs final tracklets.
    
    Args:
        video_path: Path to original video
        real_time_tracklets: List of real-time tracklets per frame
        final_tracklets: Dictionary of final refined tracklets
        output_path: Path for output video
    """
    log.info(f"Creating comparison video: {output_path}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Open original video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error(f"Failed to open video: {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer for side-by-side comparison (double width)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
    
    if not out.isOpened():
        log.error(f"Failed to create video writer for: {output_path}")
        cap.release()
        return
    
    # Create visualizer
    visualizer = EllipseDetection()
    
    # Progress bar
    progress = tqdm(
        total=total_frames,
        desc="Creating comparison video",
        unit="frames",
        ncols=100
    )
    
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create two copies of the frame
        frame_real_time = frame.copy()
        frame_final = frame.copy()
        
        # Draw real-time tracklets (if available)
        if frame_id < len(real_time_tracklets):
            visualizer.draw_detection(frame_real_time, real_time_tracklets[frame_id])
        
        # Draw final tracklets
        visualizer.draw_final_tracklets(frame_final, final_tracklets, frame_id)
        
        # Add labels
        cv2.putText(frame_real_time, "Real-time Tracking", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame_final, "Final Refined Tracking", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Combine frames side by side
        combined_frame = cv2.hconcat([frame_real_time, frame_final])
        
        out.write(combined_frame)
        frame_id += 1
        progress.update(1)
    
    # Cleanup
    cap.release()
    out.release()
    progress.close()
    
    log.info(f"Comparison video saved: {output_path}")
