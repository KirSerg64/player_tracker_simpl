"""
Statistics utilities for tracklet analysis.

This module contains functions for analyzing and saving statistics about
tracklet refinement results and tracking performance.
"""

import os
import logging
from typing import Dict, List, Any
import json
import csv

log = logging.getLogger(__name__)


def save_tracklet_statistics(final_tracklets: dict, output_path: str):
    """
    Save statistics about the final tracklets to a text file.
    
    Args:
        final_tracklets: Dictionary of final refined tracklets
        output_path: Path for statistics file
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    stats_path = output_path.replace('.txt', '_tracklet_stats.txt') if output_path.endswith('.txt') else f"{output_path}_tracklet_stats.txt"
    
    with open(stats_path, 'w') as f:
        f.write("=== FINAL TRACKLET STATISTICS ===\n\n")
        f.write(f"Total number of tracklets: {len(final_tracklets)}\n\n")
        
        # Per tracklet statistics
        f.write("Per-tracklet details:\n")
        f.write("-" * 60 + "\n")
        
        total_detections = 0
        tracklet_durations = []
        tracklet_scores = []
        
        for track_id, tracklet in final_tracklets.items():
            duration = len(tracklet.times)
            tracklet_durations.append(duration)
            
            if tracklet.times:
                start_frame = min(tracklet.times)
                end_frame = max(tracklet.times)
                avg_score = sum(tracklet.scores) / len(tracklet.scores) if tracklet.scores else 0
                tracklet_scores.append(avg_score)
            else:
                start_frame = end_frame = avg_score = 0
                tracklet_scores.append(0)
            
            f.write(f"Track ID {int(track_id):3d}: {duration:4d} detections | "
                   f"Frames {start_frame:5d}-{end_frame:5d} | "
                   f"Avg Score: {avg_score:.3f}\n")
            total_detections += duration
        
        f.write("-" * 60 + "\n")
        f.write(f"Total detections across all tracklets: {total_detections}\n")
        
        if final_tracklets:
            avg_detections_per_track = total_detections / len(final_tracklets)
            avg_duration = sum(tracklet_durations) / len(tracklet_durations)
            avg_score = sum(tracklet_scores) / len(tracklet_scores)
            
            f.write(f"Average detections per tracklet: {avg_detections_per_track:.1f}\n")
            f.write(f"Average tracklet duration: {avg_duration:.1f} frames\n")
            f.write(f"Average detection score: {avg_score:.3f}\n")
            f.write(f"Shortest tracklet: {min(tracklet_durations)} frames\n")
            f.write(f"Longest tracklet: {max(tracklet_durations)} frames\n")
    
    log.info(f"Tracklet statistics saved: {stats_path}")


def save_tracklet_statistics_json(final_tracklets: dict, output_path: str):
    """
    Save tracklet statistics in JSON format for programmatic analysis.
    
    Args:
        final_tracklets: Dictionary of final refined tracklets
        output_path: Path for JSON statistics file
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    json_path = output_path.replace('.json', '_tracklet_stats.json') if output_path.endswith('.json') else f"{output_path}_tracklet_stats.json"
    
    stats = {
        "summary": {
            "total_tracklets": len(final_tracklets),
            "total_detections": 0
        },
        "tracklets": {}
    }
    
    tracklet_durations = []
    tracklet_scores = []
    
    for track_id, tracklet in final_tracklets.items():
        duration = len(tracklet.times)
        tracklet_durations.append(duration)
        
        if tracklet.times:
            start_frame = min(tracklet.times)
            end_frame = max(tracklet.times)
            avg_score = sum(tracklet.scores) / len(tracklet.scores) if tracklet.scores else 0
            tracklet_scores.append(avg_score)
        else:
            start_frame = end_frame = avg_score = 0
            tracklet_scores.append(0)
        
        stats["tracklets"][str(track_id)] = {
            "duration": duration,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "average_score": avg_score,
            "total_detections": duration
        }
        
        stats["summary"]["total_detections"] += duration
    
    if final_tracklets:
        stats["summary"].update({
            "average_detections_per_tracklet": stats["summary"]["total_detections"] / len(final_tracklets),
            "average_duration": sum(tracklet_durations) / len(tracklet_durations),
            "average_score": sum(tracklet_scores) / len(tracklet_scores),
            "min_duration": min(tracklet_durations),
            "max_duration": max(tracklet_durations),
            "min_score": min(tracklet_scores),
            "max_score": max(tracklet_scores)
        })
    
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    log.info(f"Tracklet statistics (JSON) saved: {json_path}")


def save_tracklet_statistics_csv(final_tracklets: dict, output_path: str):
    """
    Save tracklet statistics in CSV format for spreadsheet analysis.
    
    Args:
        final_tracklets: Dictionary of final refined tracklets
        output_path: Path for CSV statistics file
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    csv_path = output_path.replace('.csv', '_tracklet_stats.csv') if output_path.endswith('.csv') else f"{output_path}_tracklet_stats.csv"
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['Track_ID', 'Duration', 'Start_Frame', 'End_Frame', 'Average_Score', 'Total_Detections'])
        
        # Data rows
        for track_id, tracklet in final_tracklets.items():
            duration = len(tracklet.times)
            
            if tracklet.times:
                start_frame = min(tracklet.times)
                end_frame = max(tracklet.times)
                avg_score = sum(tracklet.scores) / len(tracklet.scores) if tracklet.scores else 0
            else:
                start_frame = end_frame = avg_score = 0
            
            writer.writerow([
                int(track_id),
                duration,
                start_frame,
                end_frame,
                round(avg_score, 3),
                duration
            ])
    
    log.info(f"Tracklet statistics (CSV) saved: {csv_path}")


def analyze_tracking_performance(final_tracklets: dict, video_fps: float = 30.0) -> Dict[str, Any]:
    """
    Analyze tracking performance metrics.
    
    Args:
        final_tracklets: Dictionary of final refined tracklets
        video_fps: Video frame rate for time calculations
        
    Returns:
        Dictionary containing performance metrics
    """
    if not final_tracklets:
        return {"error": "No tracklets to analyze"}
    
    # Basic statistics
    total_tracklets = len(final_tracklets)
    total_detections = sum(len(tracklet.times) for tracklet in final_tracklets.values())
    
    # Duration analysis
    durations = [len(tracklet.times) for tracklet in final_tracklets.values()]
    avg_duration_frames = sum(durations) / len(durations)
    avg_duration_seconds = avg_duration_frames / video_fps
    
    # Score analysis
    all_scores = []
    for tracklet in final_tracklets.values():
        all_scores.extend(tracklet.scores)
    
    avg_detection_score = sum(all_scores) / len(all_scores) if all_scores else 0
    
    # Temporal coverage analysis
    all_frames = set()
    for tracklet in final_tracklets.values():
        all_frames.update(tracklet.times)
    
    if all_frames:
        video_span = max(all_frames) - min(all_frames) + 1
        coverage_ratio = len(all_frames) / video_span
    else:
        video_span = coverage_ratio = 0
    
    # ID consistency metrics
    short_tracklets = sum(1 for d in durations if d < 30)  # Less than 1 second at 30fps
    long_tracklets = sum(1 for d in durations if d > 300)  # More than 10 seconds at 30fps
    
    return {
        "total_tracklets": total_tracklets,
        "total_detections": total_detections,
        "average_duration_frames": avg_duration_frames,
        "average_duration_seconds": avg_duration_seconds,
        "average_detection_score": avg_detection_score,
        "video_span_frames": video_span,
        "temporal_coverage_ratio": coverage_ratio,
        "short_tracklets": short_tracklets,
        "long_tracklets": long_tracklets,
        "short_tracklet_ratio": short_tracklets / total_tracklets,
        "long_tracklet_ratio": long_tracklets / total_tracklets
    }


def save_all_statistics(final_tracklets: dict, output_dir: str, video_fps: float = 30.0):
    """
    Save tracklet statistics in all formats (TXT, JSON, CSV) and performance analysis.
    
    Args:
        final_tracklets: Dictionary of final refined tracklets
        output_dir: Directory for output files
        video_fps: Video frame rate for time calculations
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    base_path = os.path.join(output_dir, "tracklet_analysis")
    
    # Save in all formats
    save_tracklet_statistics(final_tracklets, f"{base_path}.txt")
    save_tracklet_statistics_json(final_tracklets, f"{base_path}.json")
    save_tracklet_statistics_csv(final_tracklets, f"{base_path}.csv")
    
    # Save performance analysis
    performance = analyze_tracking_performance(final_tracklets, video_fps)
    performance_path = os.path.join(output_dir, "performance_analysis.json")
    
    with open(performance_path, 'w') as f:
        json.dump(performance, f, indent=2)
    
    log.info(f"Complete tracklet analysis saved to: {output_dir}")
    
    return performance
