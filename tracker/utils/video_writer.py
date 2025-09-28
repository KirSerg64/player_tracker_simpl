import ffmpeg
import subprocess
import tempfile
import logging
import cv2

log = logging.getLogger(__name__)


class FFmpegVideoWriter:
    """
    A replacement for cv2.VideoWriter using ffmpeg-python.
    Provides better codec control and H.264 encoding by default.
    """
    
    def __init__(self, output_path, fourcc=None, fps=30, frame_size=(640, 480), codec='libx264', crf=23, preset='medium'):
        """
        Initialize FFmpeg video writer.
        
        Args:
            output_path: Output video file path
            fourcc: Ignored (kept for cv2.VideoWriter compatibility)
            fps: Frames per second
            frame_size: (width, height) tuple
            codec: Video codec (default: libx264 for H.264)
            crf: Constant Rate Factor for quality (18-28, lower=better quality)
            preset: Encoding preset (ultrafast, fast, medium, slow, veryslow)
        """
        self.output_path = output_path
        self.fps = fps
        self.width, self.height = frame_size
        self.codec = codec
        self.crf = crf
        self.preset = preset
        self.process = None
        self.frames_written = 0
        self.is_opened = False
        
        # Create temporary directory for frame storage
        self.temp_dir = tempfile.mkdtemp(prefix='ffmpeg_frames_')
        
        # Initialize ffmpeg process
        self._init_ffmpeg_process()
    
    def _init_ffmpeg_process(self):
        """Initialize the ffmpeg process with pipe input."""
        try:
            # Create ffmpeg process with pipe input
            self.process = (
                ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{self.width}x{self.height}', r=self.fps)
                .output(
                    self.output_path,
                    vcodec=self.codec,
                    crf=self.crf,
                    preset=self.preset,
                    pix_fmt='yuv420p',  # Ensures compatibility
                    movflags='+faststart'  # Optimize for streaming
                )
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )
            self.is_opened = True
            log.info(f"Initialized FFmpeg writer: {self.output_path} ({self.width}x{self.height} @ {self.fps}fps)")
        except Exception as e:
            log.error(f"Failed to initialize FFmpeg process: {e}")
            self.is_opened = False
    
    def write(self, frame):
        """Write a frame to the video."""
        if not self.is_opened or self.process is None:
            log.error("FFmpeg writer is not properly initialized")
            return False
        
        try:
            # Ensure frame is the correct size
            if frame.shape[:2] != (self.height, self.width):
                frame = cv2.resize(frame, (self.width, self.height))
            
            # Write frame to ffmpeg stdin
            self.process.stdin.write(frame.tobytes())
            self.frames_written += 1
            return True
        except Exception as e:
            log.error(f"Error writing frame {self.frames_written}: {e}")
            return False
    
    def release(self):
        """Close the video writer and finalize the video."""
        if self.process is not None:
            try:
                self.process.stdin.close()
                self.process.wait()
                log.info(f"FFmpeg writer released. Total frames: {self.frames_written}")
            except Exception as e:
                log.error(f"Error releasing FFmpeg writer: {e}")
            finally:
                self.process = None
                self.is_opened = False
        
        # Clean up temporary directory
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass
    
    def isOpened(self):
        """Check if the writer is opened (cv2.VideoWriter compatibility)."""
        return self.is_opened
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
