# import multiprocessing as mp
# import queue
# import time
# import logging
# import threading
# from typing import Dict, List, Optional, Any, Tuple
# from dataclasses import dataclass, field
# from enum import Enum
# import pickle
# import traceback
# import cv2
# import numpy as np
# import pandas as pd
# from tracker.utils.pipeline_base import PipelineProcess, ProcessConfig, PipelineMessage, MessageType
# from tracker.utils.video_reader import VideoReaderProcess

# log = logging.getLogger(__name__)


# class MultiprocessingPipeline:
#     """Main multiprocessing pipeline coordinator"""
    
#     def __init__(self, video_path: str, target_fps: int, modules: List, 
#                  video_buffer_size: int = 50, module_buffer_size: int = 20):
#         self.video_path = video_path
#         self.target_fps = target_fps
#         self.modules = modules
#         self.processes: List[PipelineProcess] = []
#         self.stats_monitor = None
#         self.running = False
        
#         # Create video reader process
#         video_config = ProcessConfig(
#             name="video_reader",
#             input_queue_size=1,  # No input needed
#             output_queue_size=video_buffer_size
#         )
#         self.video_reader = VideoReaderProcess(video_config, video_path, target_fps)
#         self.processes.append(self.video_reader)
        
#         # Create module processes
#         for i, module in enumerate(modules):
#             module_config = ProcessConfig(
#                 name=f"module_{i}_{module.name}",
#                 input_queue_size=module_buffer_size,
#                 output_queue_size=module_buffer_size
#             )
#             module_process = ModuleProcess(module_config, module)
#             self.processes.append(module_process)
            
#         log.info(f"Pipeline created with {len(self.processes)} processes")
        
#     def start(self):
#         """Start all processes"""
#         self.running = True
        
#         # Start all processes
#         for process in self.processes:
#             process.start()
            
#         # Start stats monitor
#         self.stats_monitor = threading.Thread(target=self._monitor_stats, daemon=True)
#         self.stats_monitor.start()
        
#         log.info("Multiprocessing pipeline started")
        
#     def stop(self):
#         """Stop all processes"""
#         self.running = False
        
#         # Stop all processes
#         for process in self.processes:
#             process.stop()
            
#         log.info("Multiprocessing pipeline stopped")
        
#     def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
#         """Run the complete pipeline"""
#         try:
#             self.start()            
 
#             # Start data flow
#             threading.Thread(target=self._coordinate_data_flow, daemon=True).start()
            
#             # Collect results
#             results = self._collect_results()
            
#             return results
            
#         finally:
#             self.stop()
            
#     def _coordinate_data_flow(self):
#         """Coordinate data flow between processes"""
#         while self.running:
#             # Get output from video reader
#             video_output = self.video_reader.get_output(timeout=0.1)
#             if video_output is None:
#                 time.sleep(0.001)
#                 continue
                
#             if video_output.msg_type == MessageType.END_OF_STREAM:
#                 # Propagate end of stream through all modules
#                 for process in self.processes[1:]:  # Skip video reader
#                     process.put_input(video_output, timeout=1.0)
#                 break
                
#             if video_output.msg_type == MessageType.DATA:
#                 # Send to first module
#                 if len(self.processes) > 1:
#                     self.processes[1].put_input(video_output, timeout=0.1)
                    
#             # Flow data through modules
#             for i in range(1, len(self.processes) - 1):
#                 current_process = self.processes[i]
#                 next_process = self.processes[i + 1]
                
#                 output = current_process.get_output(timeout=0.01)
#                 if output and output.msg_type == MessageType.DATA:
#                     next_process.put_input(output, timeout=0.1)


#     def _collect_results(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
#         """Collect final results from last process"""
#         detections = pd.DataFrame()
#         image_preds = pd.DataFrame()
        
#         if not self.processes:
#             return detections, image_preds
            
#         last_process = self.processes[-1]
#         frames_processed = 0
        
#         while self.running:
#             result = last_process.get_output(timeout=0.1)
#             if result is None:
#                 continue
                
#             if result.msg_type == MessageType.END_OF_STREAM:
#                 log.info(f"Pipeline completed. Processed {frames_processed} frames")
#                 break
                
#             if result.msg_type == MessageType.DATA:
#                 frames_processed += 1
                
#                 # Extract detections if available
#                 if result.data and isinstance(result.data, dict):
#                     batch_detections = result.data.get("detections", pd.DataFrame())
#                     detections = merge_dataframes(detections, batch_detections)
#                     batch_metadata = result.data.get("metadata", pd.DataFrame())
#                     image_preds = merge_dataframes(image_preds, batch_metadata)

#                 if frames_processed % 50 == 0:
#                     log.info(f"Collected results for {frames_processed} frames")
                    
#         return detections, image_preds
        
#     def _monitor_stats(self):
#         """Monitor process statistics"""
#         while self.running:
#             try:
#                 for process in self.processes:
#                     try:
#                         stats = process.stats_queue.get_nowait()
#                         log.info(f"Stats - {stats['process_name']}: "
#                                 f"{stats['processed_count']} processed, "
#                                 f"{stats['fps']:.1f} FPS, "
#                                 f"queues: {stats['input_queue_size']}/{stats['output_queue_size']}")
#                     except queue.Empty:
#                         pass
                        
#                 time.sleep(5.0)  # Stats every 5 seconds
                
#             except Exception as e:
#                 log.error(f"Stats monitor error: {e}")
                
#     def get_pipeline_stats(self) -> Dict:
#         """Get current pipeline statistics"""
#         stats = {}
#         for process in self.processes:
#             stats[process.config.name] = {
#                 "alive": process.is_alive(),
#                 "input_queue": process.input_queue.qsize() if process.input_queue else 0,
#                 "output_queue": process.output_queue.qsize() if process.output_queue else 0,
#             }
#         return stats