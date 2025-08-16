import multiprocessing as mp
import queue
import time
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pickle
import traceback
import cv2
import numpy as np
import pandas as pd
from tracker.utils.pipeline_base import PipelineProcess, ProcessConfig, PipelineMessage, MessageType


log = logging.getLogger(__name__)



class VideoReaderProcess(PipelineProcess):
    """Process for reading video frames"""
    
    def __init__(self, config: ProcessConfig, video_path: str, target_fps: int):
        super().__init__(config)
        self.video_path = video_path
        self.target_fps = target_fps
        self.video_cap = None
        self.fps = None
        self.frame_modulo = None
        
    def _setup(self):
        """Setup video capture"""
        video_filename = int(self.video_path) if str(self.video_path).isnumeric() else str(self.video_path)
        self.video_cap = cv2.VideoCapture(video_filename)
        
        if not self.video_cap.isOpened():
            raise RuntimeError(f"Error opening video stream or file {video_filename}")
            
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        self.frame_modulo = max(1, int(self.fps // self.target_fps))
        
        video_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        log.info(f"Video {video_filename} opened: {self.fps} FPS, "
                f"modulo: {self.frame_modulo}, resolution: {video_width}x{video_height}")
    
    def _run(self):
        """Custom run for video reading - doesn't wait for input"""
        self.start_time = time.time()
        
        try:
            self._setup()
            frame_idx = -1
            
            while not self.stop_event.is_set():
                frame_idx += 1
                ret, frame = self.video_cap.read()
                
                if not ret or frame is None:
                    # End of video
                    end_msg = PipelineMessage(
                        msg_type=MessageType.END_OF_STREAM,
                        source_process=self.config.name
                    )
                    self._send_output(end_msg)
                    log.info(f"Video reading completed. Total frames: {self.processed_count}")
                    break
                    
                if frame_idx % self.frame_modulo != 0:
                    continue
                    
                # Convert to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Create frame data
                frame_data = {
                    "frame_idx": frame_idx,
                    "frame": frame,
                    "image": image,
                    "video_path": self.video_path
                }
                
                metadata = {
                    "frame_idx": frame_idx,
                    "video_id": self.video_path,
                    "timestamp": time.time()
                }
                
                # Send frame
                message = PipelineMessage(
                    msg_type=MessageType.DATA,
                    data=frame_data,
                    metadata=metadata,
                    source_process=self.config.name,
                    sequence_id=frame_idx
                )
                
                # Non-blocking send - if queue full, wait briefly
                sent = False
                for attempt in range(1000):
                    if self._try_send_output(message, timeout=0.1):
                        sent = True
                        break
                    time.sleep(0.01)  # Brief pause before retry
                
                if sent:
                    self.processed_count += 1
                    if self.config.enable_stats and self.processed_count % 50 == 0:
                        self._send_stats()
                else:
                    log.warning(f"Dropped frame {frame_idx} - output queue full")
                        
        except Exception as e:
            log.error(f"Fatal error in video reader: {e}")
        finally:
            self._cleanup()
            
    def _try_send_output(self, message: PipelineMessage, timeout: float = 0.1) -> bool:
        """Try to send output with timeout"""
        try:
            self.output_queue.put(message, timeout=timeout)
            return True
        except queue.Full:
            return False
            
    def _cleanup(self):
        """Cleanup video capture"""
        if self.video_cap:
            self.video_cap.release()