#!/usr/bin/env python3
"""
Video Splitter - Standalone Script

Splits input video into overlapping segments using FFmpeg with optional FPS conversion.

Usage:
    python video_splitter.py input_video.mp4 ./segments --segment_duration 600 --overlap 1.0
    python video_splitter.py input_video.mp4 ./segments --segment_duration 30 --overlap 2.0 --output_fps 15
    
Example:
    python video_splitter.py match.mp4 ./output --segment_duration 300 --overlap 2.0 --output_fps 15 --debug
    
The FPS conversion ensures that when segments are merged (accounting for overlaps), 
the resulting video maintains temporal consistency with the target FPS.
"""

import subprocess
import logging
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import math

def setup_logging(debug: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

logger = logging.getLogger(__name__)

class VideoSplitter:
    """Handles video segmentation into overlapping chunks"""
    
    def __init__(self, input_video: str, output_dir: str, segment_duration: int = 600, 
                 overlap: float = 1.0, video_codec: str = 'libx264', audio_codec: str = 'aac', 
                 quality: int = 23, preset: str = 'medium', target_fps: Optional[float] = None, 
                 debug: bool = False):
        self.input_video = Path(input_video)
        self.output_dir = Path(output_dir)
        self.segment_duration = segment_duration
        self.overlap = overlap
        self.video_codec = video_codec
        self.audio_codec = audio_codec
        self.quality = quality
        self.preset = preset
        self.target_fps = target_fps
        self.debug = debug
        
        # Validate input video exists
        if not self.input_video.exists():
            raise FileNotFoundError(f"Input video not found: {self.input_video}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def split_video(self) -> List[Path]:
        """Split video into overlapping segments"""
        logger.info(f"Starting video segmentation for: {self.input_video}")
        
        # Get video information
        video_info = self._get_video_info(self.input_video)
        original_fps = video_info['fps']
        logger.info(f"Video duration: {video_info['duration']:.1f} seconds")
        logger.info(f"Original FPS: {original_fps:.2f}" if original_fps else "Original FPS: unknown")
        
        # Determine target FPS
        if self.target_fps is None:
            effective_fps = original_fps
            logger.info("No target FPS specified - using original FPS")
        else:
            effective_fps = self.target_fps
            logger.info(f"Target FPS: {self.target_fps:.2f}")
        
        # Calculate segments
        segment_info = self._calculate_segments(video_info['duration'])
        logger.info(f"Will create {len(segment_info)} segments")
        
        # Create segments
        segment_paths = []
        metadata = {
            'original_video': str(self.input_video),
            'original_duration': video_info['duration'],
            'original_fps': original_fps,
            'target_fps': self.target_fps,
            'effective_fps': effective_fps,
            'segment_count': len(segment_info),
            'segment_duration': self.segment_duration,
            'overlap_duration': self.overlap,
            'segments': []
        }
        
        import time
        start_time = time.time()
        
        for i, (start_time_seg, duration) in enumerate(segment_info):
            segment_start = time.time()
            segment_path = self._create_segment(self.input_video, i, start_time_seg, duration, effective_fps)
            segment_paths.append(segment_path)
            segment_elapsed = time.time() - segment_start
            
            # Add segment metadata
            metadata['segments'].append({
                'segment_id': i,
                'start_time': start_time_seg,
                'duration': duration,
                'fps': effective_fps,
                'path': str(segment_path),
                'processing_time': segment_elapsed
            })
            
            # Progress and time estimation
            elapsed_total = time.time() - start_time
            avg_time_per_segment = elapsed_total / (i + 1)
            remaining_segments = len(segment_info) - (i + 1)
            eta_seconds = avg_time_per_segment * remaining_segments
            
            logger.info(f"Created segment {i+1}/{len(segment_info)}: {segment_path.name} "
                       f"({segment_elapsed:.1f}s, ETA: {eta_seconds/60:.1f}min)")
        
        total_elapsed = time.time() - start_time
        logger.info(f"Total processing time: {total_elapsed/60:.1f} minutes")
        
        # Save segment metadata
        metadata_file = self.output_dir / 'segments_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Validation
        self._validate_segments(segment_paths)
        
        result = {
            'output_paths': segment_paths + [str(metadata_file)],
            'metadata': metadata,
            'segment_count': len(segment_paths)
        }
        
        logger.info(f"Video segmentation completed. Created {len(segment_paths)} segments")
        return segment_paths
    
    def _get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Get video information using FFprobe"""
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', 
            '-show_format', '-show_streams',
            str(video_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            info = json.loads(result.stdout)
            duration = float(info['format']['duration'])
            
            # Extract video stream information (FPS, resolution, etc.)
            video_stream = None
            for stream in info['streams']:
                if stream['codec_type'] == 'video':
                    video_stream = stream
                    break
            
            # Parse frame rate
            fps = None
            if video_stream and 'r_frame_rate' in video_stream:
                fps_str = video_stream['r_frame_rate']
                if '/' in fps_str:
                    num, den = fps_str.split('/')
                    fps = float(num) / float(den) if float(den) != 0 else None
                else:
                    fps = float(fps_str)
            
            return {
                'duration': duration,
                'size': int(info['format']['size']),
                'fps': fps,
                'video_stream': video_stream,
                'format': info['format']
            }
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to get video info: {e}")
        except (KeyError, ValueError) as e:
            raise RuntimeError(f"Failed to parse video info: {e}")
    
    def _calculate_segments(self, video_duration: float) -> List[tuple]:
        """Calculate segment start times and durations"""        
        segments = []
        current_start = 0
        max_segments = 1000  # Safety limit to prevent infinite loops
        
        while current_start < video_duration and len(segments) < max_segments:
            # Calculate end time for this segment
            segment_end = min(current_start + self.segment_duration, video_duration)
            actual_duration = segment_end - current_start
            
            # Skip segments that are too short (less than 1 second)
            if actual_duration >= 1.0:
                segments.append((current_start, actual_duration))
            
            # Move to next segment (with overlap)
            current_start += self.segment_duration - self.overlap
            
            # If the remaining time is less than overlap, make it the final segment
            if video_duration - current_start <= self.overlap:
                break
        
        if len(segments) >= max_segments:
            logger.warning(f"Hit maximum segment limit ({max_segments}). Video might be too long or settings incorrect.")
        
        return segments
    
    def _create_segment(self, video_path: Path, segment_id: int, start_time: float, duration: float, target_fps: Optional[float] = None) -> Path:
        """Create a video segment using FFmpeg"""
        output_path = self.output_dir / f'segment_{segment_id:03d}' / f'video_segment_{segment_id:03d}.mp4'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            'ffmpeg', '-y',  # Overwrite output files
            '-i', str(video_path),
            '-ss', str(start_time),
            '-t', str(duration),
            '-c:v', self.video_codec,
            '-c:a', self.audio_codec,
            '-crf', str(self.quality),
            '-preset', self.preset,
        ]
        
        # Add FPS conversion if target FPS is specified
        if target_fps is not None:
            cmd.extend(['-r', str(target_fps)])
            logger.debug(f"Setting FPS to {target_fps} for segment {segment_id}")
        
        cmd.append(str(output_path))
        
        try:
            if self.debug:
                logger.debug(f"FFmpeg command: {' '.join(cmd)}")
                # For debug mode, capture output but limit it and add timeout
                result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)  # 5 min timeout
                # Save only first/last parts to avoid memory issues
                stderr_lines = result.stderr.split('\n')
                if len(stderr_lines) > 100:
                    truncated_stderr = '\n'.join(stderr_lines[:50] + ['... (truncated) ...'] + stderr_lines[-50:])
                else:
                    truncated_stderr = result.stderr
                    
                # Save FFmpeg output for debugging
                debug_file = self.output_dir / f'ffmpeg_segment_{segment_id:03d}.log'
                with open(debug_file, 'w') as f:
                    f.write(f"Command: {' '.join(cmd)}\n\n")
                    f.write(f"STDERR:\n{truncated_stderr}\n")
            else:
                # For production, don't capture output - redirect to null device with timeout
                result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=600)  # 10 min timeout per segment
            
            return output_path
            
        except subprocess.TimeoutExpired as e:
            error_msg = f"FFmpeg timeout for segment {segment_id} (command took too long)"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        except subprocess.CalledProcessError as e:
            error_msg = f"FFmpeg failed for segment {segment_id}: {e.stderr}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _validate_segments(self, segment_paths: List[str]):
        """Validate created segments"""
        logger.info("Validating created segments...")
        
        for segment_path in segment_paths:
            if segment_path.parts[-1] == '.json':
                continue  # Skip metadata file
                
            path = Path(segment_path)
            if not path.exists():
                raise RuntimeError(f"Segment file not created: {path}")
            
            if path.stat().st_size == 0:
                raise RuntimeError(f"Segment file is empty: {path}")
            
            # Quick FFprobe check
            try:
                self._get_video_info(path)
            except Exception as e:
                raise RuntimeError(f"Invalid segment created {path}: {e}")
        
        logger.info("All segments validated successfully")
    
    def validate_outputs(self, output_paths: List[str]) -> Dict[str, Any]:
        """Validate the outputs of this step"""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'segment_count': 0,
            'total_duration': 0
        }
        
        segment_files = [p for p in output_paths if not p.endswith('.json')]
        validation_result['segment_count'] = len(segment_files)
        
        for segment_path in segment_files:
            path = Path(segment_path)
            try:
                if not path.exists():
                    validation_result['errors'].append(f"Missing segment: {path}")
                    validation_result['valid'] = False
                    continue
                
                # Get segment info
                info = self._get_video_info(path)
                validation_result['total_duration'] += info['duration']
                
                if info['duration'] < 1:
                    validation_result['warnings'].append(f"Very short segment: {path} ({info['duration']:.1f}s)")
                
            except Exception as e:
                validation_result['errors'].append(f"Cannot validate {path}: {e}")
                validation_result['valid'] = False
        
        return validation_result
    
    def _generate_debug_info(self, segment_paths: List[str], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate debug information"""
        debug_info = {
            'ffmpeg_version': self._get_ffmpeg_version(),
            'segment_sizes': {},
            'segment_durations': {},
            'total_segments': len([p for p in segment_paths if not p.endswith('.json')])
        }
        
        for segment_path in segment_paths:
            if segment_path.endswith('.json'):
                continue
                
            path = Path(segment_path)
            if path.exists():
                debug_info['segment_sizes'][path.name] = path.stat().st_size
                try:
                    info = self._get_video_info(path)
                    debug_info['segment_durations'][path.name] = info['duration']
                except:
                    debug_info['segment_durations'][path.name] = 'unknown'
        
        return debug_info
    
    def _get_ffmpeg_version(self) -> str:
        """Get FFmpeg version for debugging"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
            first_line = result.stdout.split('\n')[0]
            return first_line
        except:
            return 'unknown'


def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Split video into overlapping segments')
    parser.add_argument('input_video', type=str, help='Path to input video file')
    parser.add_argument('output_dir', type=str, help='Output directory for segments')
    parser.add_argument('--segment_duration', type=int, default=600, help='Segment duration in seconds (default: 600)')
    parser.add_argument('--overlap', type=float, default=1.0, help='Overlap between segments in seconds (default: 1.0)')
    parser.add_argument('--output_format', type=str, default='mp4', help='Output format (default: mp4)')
    parser.add_argument('--output_fps', type=float, default=None, help='Target FPS for output segments (default: same as input)')
    parser.add_argument('--video_codec', type=str, default='libx264', help='Video codec (default: libx264)')
    parser.add_argument('--audio_codec', type=str, default='aac', help='Audio codec (default: aac)')
    parser.add_argument('--quality', type=int, default=23, help='Video quality CRF value (default: 23)')
    parser.add_argument('--preset', type=str, default='medium', help='FFmpeg preset (default: medium)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.debug)
    
    # Create video splitter instance
    splitter = VideoSplitter(
        input_video=args.input_video,
        output_dir=args.output_dir,
        segment_duration=args.segment_duration,
        overlap=args.overlap,
        video_codec=args.video_codec,
        audio_codec=args.audio_codec,
        quality=args.quality,
        preset=args.preset,
        target_fps=args.output_fps,
        debug=args.debug
    )
    
    try:
        # Split the video
        segments = splitter.split_video()
        
        print(f"Successfully created {len(segments)} segments:")
        for i, segment in enumerate(segments):
            print(f"  Segment {i+1}: {segment}")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
