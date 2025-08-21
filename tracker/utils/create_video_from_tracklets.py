
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
from aws_utils import load_file_to_bucket
from datetime import datetime
import numpy as np
from tracker.visualization.players_drawer import EllipseDetection
from tracklab.utils.cv2 import draw_text


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
    box_color_fn=None,
    thickness=1,
    font_scale=0.5,
    frame_rate=15
):
    """
    Create a single video from all segments in sequential order, avoiding duplicated overlapping frames.
    
    Args:
        base_dir: Path to directory containing segments (e.g., results_parallel)
        output_video_path: Output path for combined video (default: base_dir/combined_tracklets_video.mp4)
        box_color_fn: Function to determine box color by track_id
        thickness: Line thickness for drawing
        font_scale: Font scale for track ID text
        frame_rate: Output video frame rate
    """
    base_path = Path(base_dir)
    segments_dir = base_path / "segments"
    
    if not segments_dir.exists():
        raise ValueError(f"Segments directory not found: {segments_dir}")
    
    # Set default output path to the parent directory of segments (base_dir)
    if output_video_path is None:
        output_video_path = str(base_path / "combined_tracklets_video.mp4")
    
    # Load segments metadata to understand overlap structure
    metadata_file = segments_dir / "segments_metadata.json"
    segments_metadata = None
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            segments_metadata = json.load(f)
    
    # Find all segment directories and sort them
    segment_dirs = sorted([
        d for d in segments_dir.iterdir() 
        if d.is_dir() and d.name.startswith('segment_')
    ], key=lambda x: int(re.findall(r'segment_(\d+)', x.name)[0]))
    
    if not segment_dirs:
        raise ValueError(f"No segment directories found in {segments_dir}")
    
    print(f"Found {len(segment_dirs)} segments to process")
    if segments_metadata:
        overlap_duration = segments_metadata.get('overlap_duration', 0)
        print(f"Detected overlap duration: {overlap_duration}s")
    
    print(f"Output video will be saved to: {output_video_path}")
    
    #create visualizer
    visualizer = EllipseDetection()
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
        
        # Find image directory
        img_dir = outputs_dir / "seq_0" / "img1"
        if not img_dir.exists():
            print(f"Warning: Image directory not found: {img_dir}")
            continue
            
        # Load tracklets
        tracklets = load_tracklets_from_pickle(str(tracklet_file))
        
        # Process this segment
        frame_bboxes, img_files, (width, height) = draw_mot_tracklets_to_video(
            tracklets['tracklets'],
            str(img_dir),
            "",  # No output file yet
            box_color_fn=box_color_fn,
            thickness=thickness,
            font_scale=font_scale,
            frame_rate=frame_rate
        )
        
        # Initialize video writer with first segment dimensions
        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
            video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
            print(f"Initialized video writer: {width}x{height} at {frame_rate} FPS")
        
        # Calculate frames to skip for overlap (skip at beginning of segment if not first)
        frames_to_skip = 0
        if segment_idx > 0 and overlap_duration > 0:
            frames_to_skip = int(overlap_duration * frame_rate)
            print(f"Skipping {frames_to_skip} frames at start of {segment_dir.name} (overlap: {overlap_duration}s)")
        
        # Process frames for this segment
        frames_to_process = len(img_files) - frames_to_skip
        print(f"Processing {frames_to_process}/{len(img_files)} frames from {segment_dir.name}")
        
        for img_idx, img_name in enumerate(tqdm(img_files, desc=f'Segment {segment_idx:03d}')):
            # Skip overlapping frames at the beginning (except for first segment)
            if img_idx < frames_to_skip:
                continue
                
            # Use original frame numbering from image filename
            original_frame_id = float(os.path.splitext(img_name)[0])
            img_path = img_dir / img_name
            frame = cv2.imread(str(img_path))
            
            if frame is None:
                print(f"Warning: Could not read {img_path}")
                continue
            
            # Resize frame if dimensions don't match (safety check)
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            
            # Draw boxes for this frame using original frame numbering
            if original_frame_id in frame_bboxes:
                for track_id, x, y, w, h in frame_bboxes[original_frame_id]:
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
                        fontScale=0.7,
                        thickness=1,
                        alignH="c",
                        alignV="c",
                        color_bg=(0,255,0),
                        color_txt=(0,0,0),
                        alpha_bg=1,
                    )

                    # cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
                    
                    # # Keep original track ID without segment prefix
                    # label = f'ID {int(track_id)}'
                    # cv2.putText(frame, label, (x, y - 5), 
                    #            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
                    # visualizer.draw_detection(frame, np.asarray([x, y, x + w, y + h, track_id]))

            
            video_writer.write(frame)
            total_frames_written += 1
        
        print(f"Completed {segment_dir.name}. Frames processed: {len(img_files) - frames_to_skip}/{len(img_files)}")
    
    # Finalize video
    if video_writer:
        video_writer.release()
        print(f"\n✅ Combined video saved to: {output_video_path}")
        print(f"📊 Total frames written: {total_frames_written}")
        print(f"🎬 Video duration: {total_frames_written / frame_rate:.2f} seconds")
    else:
        print("❌ Error: No video writer was initialized. No valid segments found.")

    # load to 
    # Get current date and time
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"full_result_video_{date_time}.mp4"
    load_file_to_bucket(output_video_path, save_name)
    


def draw_mot_tracklets_to_video_single(
    tracklets,
    img_folder,
    output_video_path,
    box_color_fn=None,
    thickness=1,
    font_scale=0.3,
    frame_rate=15
):
    """Original function for single segment processing"""
    frame_bboxes, img_files, (width, height) = draw_mot_tracklets_to_video(
        tracklets, img_folder, "", box_color_fn, thickness, font_scale, frame_rate
    )
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

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
                x, y, w, h = int(x), int(y), int(w), int(h)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
                cv2.putText(frame, f'ID {int(track_id)}', (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness+1)

        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to: {output_video_path}")


def load_tracklets_from_pickle(tracklets_path):    
    with open(tracklets_path, 'rb') as pkl_f:
        tmp_trklets = pickle.load(pkl_f)     # dict(key:track id, value:tracklet)
    return tmp_trklets


def parse_args():
    parser = argparse.ArgumentParser(description="Draw tracklets to video.")
    
    parser.add_argument('--tracklet_file',
                        type=str,
                        default=r"",
                        help='Path to the tracklet pkl file (for single segment).'
                        )
    parser.add_argument('--image_path',
                        type=str,
                        default=r"",
                        help='Path to the image directory (for single segment).'
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
                        help='Processing mode: single segment or multi-segment.'
                        )
    parser.add_argument('--frame_rate',
                        type=int,
                        default=15,
                        help='Output video frame rate.'
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
            frame_rate=args.frame_rate
        )
    else:
        # Single segment processing (original functionality)
        if not args.tracklet_file or not args.image_path:
            raise ValueError("--tracklet_file and --image_path are required for single segment processing")
        if not args.output_video_path:
            raise ValueError("--output_video_path is required for single segment processing")
        
        tracklets = load_tracklets_from_pickle(args.tracklet_file)
        
        draw_mot_tracklets_to_video_single(
            tracklets['tracklets'],
            args.image_path,
            args.output_video_path,
            frame_rate=args.frame_rate
        )


if __name__ == "__main__":
    main()