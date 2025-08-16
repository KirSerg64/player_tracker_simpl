import multiprocessing as mp
import queue
import time
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field
import traceback
import logging


log = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages passed between processes"""
    DATA = "data"
    END_OF_STREAM = "end_of_stream"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    STATS = "stats"


@dataclass
class PipelineMessage:
    """Message passed between pipeline processes"""
    msg_type: MessageType
    data: Any = None
    metadata: Dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source_process: str = ""
    sequence_id: int = 0


@dataclass
class ProcessConfig:
    """Configuration for a pipeline process"""
    name: str
    input_queue_size: int = 100
    output_queue_size: int = 100
    process_timeout: float = 30.0
    max_retries: int = 3
    enable_stats: bool = True


class PipelineProcess:
    """Base class for pipeline processes"""
    
    def __init__(self, config: ProcessConfig):
        self.config = config
        self.input_queue = mp.Queue(maxsize=config.input_queue_size)
        self.output_queue = mp.Queue(maxsize=config.output_queue_size)
        self.process = None
        self.stats_queue = mp.Queue()
        self.stop_event = mp.Event()
        
        # Statistics
        self.processed_count = 0
        self.error_count = 0
        self.start_time = None
        
    def start(self):
        """Start the process"""
        self.process = mp.Process(target=self._run, daemon=True)
        self.process.start()
        log.info(f"Started process {self.config.name} (PID: {self.process.pid})")
        
    def stop(self, timeout: float = 5.0):
        """Stop the process gracefully"""
        self.stop_event.set()
        if self.process and self.process.is_alive():
            self.process.join(timeout=timeout)
            if self.process.is_alive():
                log.warning(f"Force terminating process {self.config.name}")
                self.process.terminate()
                self.process.join(timeout=2.0)
        self.process = None
                
    def is_alive(self) -> bool:
        """Check if process is alive"""
        return self.process is not None and self.process.is_alive()
        
    def put_input(self, message: PipelineMessage, timeout: float = 1.0) -> bool:
        """Put message into input queue (non-blocking)"""
        try:
            self.input_queue.put(message, timeout=timeout)
            return True
        except queue.Full:
            log.warning(f"Input queue full for {self.config.name}")
            return False
            
    def get_output(self, timeout: float = 0.1) -> Optional[PipelineMessage]:
        """Get message from output queue (non-blocking)"""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def _run(self):
        """Main process loop - override in subclasses"""
        self.start_time = time.time()
        
        try:
            self._setup()
            
            while not self.stop_event.is_set():
                try:
                    # Get input with timeout
                    message = self.input_queue.get(timeout=0.1)
                    
                    if message.msg_type == MessageType.END_OF_STREAM:
                        log.info(f"Process {self.config.name} received end of stream")
                        # Pass end of stream to output
                        self._send_output(message)
                        break
                        
                    elif message.msg_type == MessageType.DATA:
                        # Process the data
                        result = self._process_data(message.data, message.metadata)
                        
                        # Send result
                        output_msg = PipelineMessage(
                            msg_type=MessageType.DATA,
                            data=result,
                            metadata=message.metadata,
                            source_process=self.config.name,
                            sequence_id=message.sequence_id
                        )
                        self._send_output(output_msg)
                        
                        self.processed_count += 1
                        
                        # Send stats periodically
                        if self.config.enable_stats and self.processed_count % 10 == 0:
                            self._send_stats()
                            
                except queue.Empty:
                    continue
                except Exception as e:
                    self.error_count += 1
                    error_msg = PipelineMessage(
                        msg_type=MessageType.ERROR,
                        data=str(e),
                        metadata={"traceback": traceback.format_exc()},
                        source_process=self.config.name
                    )
                    self._send_output(error_msg)
                    log.error(f"Error in {self.config.name}: {e}")
                    
        except Exception as e:
            log.error(f"Fatal error in {self.config.name}: {e}")
        finally:
            self._cleanup()
            
    def _setup(self):
        """Setup process - override in subclasses"""
        pass
        
    def _process_data(self, data: Any, metadata: Dict) -> Any:
        """Process data - override in subclasses"""
        return data
        
    def _cleanup(self):
        """Cleanup process - override in subclasses"""
        pass
        
    def _send_output(self, message: PipelineMessage):
        """Send message to output queue"""
        try:
            self.output_queue.put(message, timeout=1.0)
        except queue.Full:
            log.warning(f"Output queue full for {self.config.name}, dropping message")
            
    def _send_stats(self):
        """Send statistics"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        fps = self.processed_count / elapsed if elapsed > 0 else 0
        
        stats = {
            "process_name": self.config.name,
            "processed_count": self.processed_count,
            "error_count": self.error_count,
            "fps": fps,
            "uptime": elapsed,
            "input_queue_size": self.input_queue.qsize(),
            "output_queue_size": self.output_queue.qsize()
        }
        
        try:
            self.stats_queue.put(stats, timeout=0.1)
        except queue.Full:
            pass  # Drop stats if queue full
