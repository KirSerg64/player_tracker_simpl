import argparse
from collections import defaultdict
import os
import pickle
import cv2
from tqdm import tqdm
import glob
from pathlib import Path
import re
import json
from aws_utils import upload_file_to_bucket
from datetime import datetime
import numpy as np
import logging

from tracker.visualization.video_creator import create_final_tracklet_video
from tracker.visualization.players_drawer import EllipseDetection
from tracklab.utils.cv2 import draw_text

from tracker.utils.video_writer import FFmpegVideoWriter

log = logging.getLogger(__name__)


def load_tracklets_from_pickle(tracklets_path):    
    """Load tracklets from pickle file."""
    with open(tracklets_path, 'rb') as pkl_f:
        tmp_trklets = pickle.load(pkl_f)     # dict(key:track id, value:tracklet)
    return tmp_trklets


def validate_segment_structure(base_dir):
    """
    Validate the segment directory structure for multi-segment processing.
    
    Args:
        base_dir: Path to directory containing segments
        
    Returns:
        dict: Validation results and recommendations
    """
    base_path = Path(base_dir)
    segments_dir = base_path / "segments"
    
    results = {
        'valid': False,
        'segments_found': 0,
        'video_files_found': 0,
        'tracklet_files_found': 0,
        'missing_components': [],
        'recommendations': []
    }
    
    if not segments_dir.exists():
        results['missing_components'].append('segments directory')
        results['recommendations'].append(f"Create segments directory at {segments_dir}")
        return results
    
    # Find segment directories
    segment_dirs = sorted([
        d for d in segments_dir.iterdir() 
        if d.is_dir() and d.name.startswith('segment_')
    ], key=lambda x: int(re.findall(r'segment_(\d+)', x.name)[0]))
    
    results['segments_found'] = len(segment_dirs)
    
    if not segment_dirs:
        results['missing_components'].append('segment directories')
        results['recommendations'].append("Create segment_* directories in segments folder")
        return results
    
    # Check each segment
    for segment_dir in segment_dirs:
        # Check for video files
        video_files = list(segment_dir.glob("*.mp4"))
        if video_files:
            results['video_files_found'] += 1
        else:
            results['missing_components'].append(f'video file in {segment_dir.name}')
        
        # Check for tracklet files
        tracklet_files = list(segment_dir.glob("merged*.pkl"))
        if tracklet_files:
            results['tracklet_files_found'] += 1
        else:
            results['missing_components'].append(f'merged*.pkl file in {segment_dir.name}')
    
    # Check metadata
    metadata_file = segments_dir / "segments_metadata.json"
    if metadata_file.exists():
        results['has_metadata'] = True
    else:
        results['recommendations'].append("Consider adding segments_metadata.json for overlap information")
    
    # Determine if valid
    results['valid'] = (
        results['segments_found'] > 0 and 
        results['video_files_found'] == results['segments_found'] and
        results['tracklet_files_found'] == results['segments_found']
    )
    
    if results['valid']:
        results['recommendations'].append("✅ Structure is valid for multi-segment processing")
    else:
        results['recommendations'].append("❌ Fix missing components before processing")
    
    return results


# Legacy function for backward compatibility with image-based processing
def draw_mot_tracklets_to_video(
    tracklets,
    img_folder,
    output_video_path,
    box_color_fn=None,
    thickness=1,
    font_scale=0.7,
    frame_rate=15,
    frame_offset=0  # Not used for frame numbering, kept for compatibility
):
    """
    Legacy function to create frame_bboxes dictionary from tracklets.
    Kept for backward compatibility with image-based processing.
    """
    # Create frame_bboxes dictionary - use original frame numbering from tracklet
    frame_bboxes = defaultdict(list)
    for track_id, track in tracklets.items():
        for instance_idx, frame_id in enumerate(track.times):
            bbox = track.bboxes[instance_idx]            
            # Use original frame_id from tracklet (no offset)
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
        raise ValueError(f"Could not read first image from {img_folder}.")
    height, width = first_frame.shape[:2]

    return frame_bboxes, img_files, (width, height)


def create_combined_video_from_segments(
    base_dir,
    output_video_path=None,
    frame_rate=15,
    show_trajectories=False
):
    """
    Create a single video from all segments in sequential order, avoiding duplicated overlapping frames.
    Uses the create_final_tracklet_video function from video_creator.py for consistent visualization.
    
    Args:
        base_dir: Path to directory containing segments (e.g., results_parallel)
        output_video_path: Output path for combined video (default: base_dir/combined_tracklets_video.mp4)
        frame_rate: Output video frame rate
        show_trajectories: Whether to show trajectory trails
    """
    base_path = Path(base_dir)
    segments_dir = base_path / "segments"
    
    if not segments_dir.exists():
        raise ValueError(f"Segments directory not found: {segments_dir}")
    
    # Set default output path to the parent directory of segments (base_dir)
    if output_video_path is None:
        output_video_path = str(base_path / "combined_video.mp4")
    
    # Load segments metadata to understand overlap structure
    metadata_file = segments_dir / "segments_metadata.json"
    segments_metadata = None
    overlap_duration = 0
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            segments_metadata = json.load(f)
            overlap_duration = segments_metadata.get('overlap_duration', 0)
    
    # Find all segment directories and sort them
    segment_dirs = sorted([
        d for d in segments_dir.iterdir() 
        if d.is_dir() and d.name.startswith('segment_')
    ], key=lambda x: int(re.findall(r'segment_(\d+)', x.name)[0]))
    
    if not segment_dirs:
        raise ValueError(f"No segment directories found in {segments_dir}")
    
    print(f"Found {len(segment_dirs)} segments to process")
    if overlap_duration > 0:
        print(f"Detected overlap duration: {overlap_duration}s")
    
    print(f"Output video will be saved to: {output_video_path}")
    
    # Initialize video writer
    video_writer = None
    total_frames_written = 0
    
    for segment_idx, segment_dir in enumerate(segment_dirs):
        print(f"\nProcessing {segment_dir.name}...")
        
        # Find tracklet file (merged*.pkl)
        outputs_dir = segment_dir
        if not outputs_dir.exists():
            print(f"Warning: No outputs directory in {segment_dir}")
            continue
            
        tracklet_files = list(outputs_dir.glob("merged*.pkl"))
        if not tracklet_files:
            print(f"Warning: No merged*.pkl files found in {outputs_dir}")
            continue
        
        tracklet_file = tracklet_files[0]  # Use first found file
        print(f"Using tracklet file: {tracklet_file}")
        
        # Find video file in the segment directory
        video_files = list(outputs_dir.glob("*.mp4"))
        if not video_files:
            print(f"Warning: No video files found in {outputs_dir}")
            continue
            
        video_file = video_files[0]  # Use first found video file
        print(f"Using video file: {video_file}")
        
        # Load tracklets
        tracklets = load_tracklets_from_pickle(str(tracklet_file))
        final_tracklets = tracklets.get('tracklets', tracklets)
        
        # Calculate frames to skip for overlap (skip at beginning of segment if not first)
        frames_to_skip = 0
        if segment_idx > 0 and overlap_duration > 0:
            frames_to_skip = int(overlap_duration * frame_rate)
            print(f"Skipping {frames_to_skip} frames at start of {segment_dir.name} (overlap: {overlap_duration}s)")
        
        # Create segment video with proper frame skipping
        segment_output_path = str(outputs_dir / f"segment_{segment_idx}_tracklets.mp4")
        create_segment_video_with_skip(
            str(video_file), 
            final_tracklets, 
            segment_output_path, 
            frames_to_skip=frames_to_skip,
            frame_rate=frame_rate,
            show_trajectories=show_trajectories
        )
        
        # If this is the first segment, initialize the combined video writer
        if video_writer is None:
            # Get video properties from first segment
            cap = cv2.VideoCapture(str(video_file))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            # Use FFmpeg writer instead of cv2.VideoWriter
            video_writer = FFmpegVideoWriter(
                output_path=output_video_path,
                fps=frame_rate,
                frame_size=(width, height),
                codec='libx264',
                crf=23,  # Good quality
                preset='medium'  # Balance of speed and compression
            )
            print(f"Initialized FFmpeg video writer: {width}x{height} at {frame_rate} FPS")
        
        # Read the segment video and append to combined video
        segment_cap = cv2.VideoCapture(segment_output_path)
        segment_frames_written = 0
        
        while True:
            ret, frame = segment_cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (width, height))
            video_writer.write(frame)
            segment_frames_written += 1
            total_frames_written += 1
        
        segment_cap.release()
        print(f"Completed {segment_dir.name}. Frames written: {segment_frames_written}")
        
        # Clean up temporary segment video
        os.remove(segment_output_path)
    
    # Finalize video
    if video_writer:
        video_writer.release()
        print(f"\n✅ Combined video saved to: {output_video_path}")
        print(f"📊 Total frames written: {total_frames_written}")
        print(f"🎬 Video duration: {total_frames_written / frame_rate:.2f} seconds")
    else:
        print("❌ Error: No video writer was initialized. No valid segments found.")

    # # Upload to S3
    # try:
    #     # Get current date and time
    #     date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     save_name = f"full_result_video_{date_time}.mp4"
    #     upload_file_to_bucket(output_video_path, save_name)
    # except Exception as e:
    #     print(f"Warning: Failed to upload to S3: {e}")


def create_segment_video_with_skip(video_path: str, final_tracklets: dict, output_path: str, 
                                 frames_to_skip: int = 0, frame_rate: int = 15, 
                                 show_trajectories: bool = False):
    """
    Create a video for a single segment with frame skipping support.
    Based on create_final_tracklet_video but with frame skipping capability.
    
    Args:
        video_path: Path to segment video
        final_tracklets: Dictionary of final refined tracklets
        output_path: Path for output video
        frames_to_skip: Number of frames to skip at the beginning
        frame_rate: Output video frame rate
        show_trajectories: Whether to show trajectory trails
    """
    log.info(f"Creating segment tracklet video: {output_path} (skipping {frames_to_skip} frames)")
    
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
    
    # Create FFmpeg video writer
    out = FFmpegVideoWriter(
        output_path=output_path,
        fps=frame_rate,
        frame_size=(width, height),
        codec='libx264',
        crf=23,
        preset='fast'  # Faster encoding for segment processing
    )
    
    if not out.isOpened():
        log.error(f"Failed to create FFmpeg writer for: {output_path}")
        cap.release()
        return
    
    # Create visualizer
    visualizer = EllipseDetection(font_algorithm="smart")

    # Skip frames at the beginning
    for _ in range(frames_to_skip):
        ret, _ = cap.read()
        if not ret:
            break
    
    # Progress bar for segment video creation
    frames_to_process = total_frames - frames_to_skip
    progress = tqdm(
        total=frames_to_process,
        desc=f"Creating segment video (skip {frames_to_skip})",
        unit="frames",
        ncols=100
    )
    
    frame_id = frames_to_skip  # Start frame ID accounting for skipped frames
    frames_written = 0
    
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
        info_text = f"Tracklets: {len(final_tracklets)} | Frame: {frame_id}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
        frame_id += 1
        frames_written += 1
        progress.update(1)
    
    # Cleanup
    cap.release()
    out.release()
    progress.close()
    
    log.info(f"Segment tracklet video saved: {output_path} ({frames_written} frames)")


def create_single_video_from_tracklets(video_path: str, tracklet_file: str, output_path: str, 
                                     frame_rate: int = 15, show_trajectories: bool = False):
    """
    Create a video from tracklets for a single video file.
    Uses the create_final_tracklet_video function from video_creator.py.
    
    Args:
        video_path: Path to input video
        tracklet_file: Path to tracklet pickle file
        output_path: Path for output video
        frame_rate: Output video frame rate (optional, uses source video fps if not specified)
        show_trajectories: Whether to show trajectory trails
    """
    print(f"Creating single video from tracklets...")
    print(f"Video: {video_path}")
    print(f"Tracklets: {tracklet_file}")
    print(f"Output: {output_path}")
    
    # Load tracklets
    tracklets = load_tracklets_from_pickle(tracklet_file)
    final_tracklets = tracklets.get('tracklets', tracklets)
    
    # Use the create_final_tracklet_video function
    create_final_tracklet_video(
        video_path=video_path,
        final_tracklets=final_tracklets,
        output_path=output_path,
        show_trajectories=show_trajectories
    )
    
    print(f"✅ Single video created: {output_path}")


# Legacy function for backward compatibility
def draw_mot_tracklets_to_video_single(
    tracklets,
    img_folder,
    output_video_path,
    box_color_fn=None,
    thickness=1,
    font_scale=0.3,
    frame_rate=15
):
    """
    Legacy function for single segment processing.
    Kept for backward compatibility but now uses the new visualization approach.
    """
    print("⚠️ Using legacy function. Consider migrating to create_single_video_from_tracklets.")
    
    frame_bboxes, img_files, (width, height) = draw_mot_tracklets_to_video(
        tracklets, img_folder, "", box_color_fn, thickness, font_scale, frame_rate
    )
    
    # Create FFmpeg video writer
    video_writer = FFmpegVideoWriter(
        output_path=output_video_path,
        fps=frame_rate,
        frame_size=(width, height),
        codec='libx264',
        crf=23,
        preset='medium'
    )

    # Create visualizer
    visualizer = EllipseDetection()

    # Process each frame
    for img_name in tqdm(img_files, desc='Drawing Tracklets'):
        frame_id = float(os.path.splitext(img_name)[0])
        img_path = os.path.join(img_folder, img_name)
        frame = cv2.imread(img_path)
        
        if frame is None:
            print(f"Warning: Could not read {img_path}")
            continue

        # Draw boxes for this frame
        if frame_id in frame_bboxes:
            for track_id, x, y, w, h in frame_bboxes[frame_id]:
                color = box_color_fn(track_id) if box_color_fn else (0, 255, 0)
                x1, y1, w, h = int(x), int(y), int(w), int(h)
                x2, y2 = x1 + w, y1 + h
                center = (int((x1 + x2) / 2), int(y2))
                
                cv2.ellipse(
                    frame,
                    center=center,
                    axes=(int(w), int(0.35 * w)),
                    angle=0.0,
                    startAngle=-45.0,
                    endAngle=235.0,
                    color=(0,255,0),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )                
                draw_text(
                    frame,
                    f"{track_id}",
                    center,
                    fontFace=1,
                    fontScale=1.5,
                    thickness=1,
                    alignH="c",
                    alignV="c",
                    color_bg=(0,255,0),
                    color_txt=(0,0,0),
                    alpha_bg=1,
                )

        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to: {output_video_path}")


def load_tracklets_from_pickle(tracklets_path):    
    """Load tracklets from pickle file."""
    with open(tracklets_path, 'rb') as pkl_f:
        tmp_trklets = pickle.load(pkl_f)     # dict(key:track id, value:tracklet)
    return tmp_trklets


def parse_args():
    parser = argparse.ArgumentParser(description="Create videos from tracklets using video_creator.py functions.")
    
    parser.add_argument('--tracklet_file',
                        type=str,
                        default=r"",
                        help='Path to the tracklet pkl file (for single video mode).'
                        )
    parser.add_argument('--video_path',
                        type=str,
                        default=r"",
                        help='Path to the input video file (for single video mode).'
                        )
    parser.add_argument('--output_video_path',
                        type=str,
                        default=r"",
                        help='Path to the output video file (optional for multi-segment, auto-saved to base_dir).'
                        )
    parser.add_argument('--base_dir',
                        type=str,
                        default=r"",
                        help='Path to the base directory containing segments folder (e.g., results_parallel, my_output, etc.).'
                        )
    parser.add_argument('--mode',
                        type=str,
                        choices=['single', 'multi'],
                        default='single',
                        help='Processing mode: single video or multi-segment.'
                        )
    parser.add_argument('--frame_rate',
                        type=int,
                        default=15,
                        help='Output video frame rate.'
                        )
    parser.add_argument('--show_trajectories',
                        action='store_true',
                        help='Show trajectory trails in the output video.'
                        )
    
    # FFmpeg-specific arguments
    parser.add_argument('--codec',
                        type=str,
                        default='libx264',
                        choices=['libx264', 'libx265', 'h264_nvenc', 'h264_qsv'],
                        help='Video codec for encoding (default: libx264 for H.264).'
                        )
    parser.add_argument('--crf',
                        type=int,
                        default=23,
                        help='Constant Rate Factor for quality (18-28, lower=better quality, default: 23).'
                        )
    parser.add_argument('--preset',
                        type=str,
                        default='medium',
                        choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'],
                        help='Encoding preset (speed vs compression, default: medium).'
                        )
    
    # Legacy arguments (kept for backward compatibility)
    parser.add_argument('--image_path',
                        type=str,
                        default=r"",
                        help='[LEGACY] Path to the image directory (for legacy single segment processing).'
                        )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == 'multi':
        # Multi-segment processing
        if not args.base_dir:
            raise ValueError("--base_dir is required for multi-segment processing")
        
        # Use provided output path or default to base_dir
        output_path = args.output_video_path if args.output_video_path else None
        
        create_combined_video_from_segments(
            args.base_dir,
            output_path,
            frame_rate=args.frame_rate,
            show_trajectories=args.show_trajectories
        )
    else:
        # Single video processing
        if args.video_path and args.tracklet_file:
            # New method: using video file directly
            if not args.output_video_path:
                raise ValueError("--output_video_path is required for single video processing")
            
            create_single_video_from_tracklets(
                args.video_path,
                args.tracklet_file,
                args.output_video_path,
                frame_rate=args.frame_rate,
                show_trajectories=args.show_trajectories
            )
        elif args.tracklet_file and args.image_path:
            # Legacy method: using image directory
            print("⚠️ Using legacy image-based processing. Consider using --video_path instead.")
            if not args.output_video_path:
                raise ValueError("--output_video_path is required for single segment processing")
            
            tracklets = load_tracklets_from_pickle(args.tracklet_file)
            
            draw_mot_tracklets_to_video_single(
                tracklets['tracklets'],
                args.image_path,
                args.output_video_path,
                frame_rate=args.frame_rate
            )
        else:
            raise ValueError("For single mode, provide either (--video_path and --tracklet_file) or (--tracklet_file and --image_path)")


if __name__ == "__main__":
    main()