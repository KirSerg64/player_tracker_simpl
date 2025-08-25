from __future__ import absolute_import
from typing import Dict
import numpy as np
import cv2
import os
import logging
import time

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    trt = None
    cuda = None

from tracker.utils.pipeline_base import MessageType, ProcessConfig, PipelineMessage

log = logging.getLogger(__name__)


class TensorRTFeatureExtractor(object):
    """TensorRT-optimized feature extractor for high-performance inference."""

    def __init__(
        self, cfg, device, batch_size, **kwargs
    ):
        if not TENSORRT_AVAILABLE:
            raise ImportError("TensorRT not available. Install tensorrt and pycuda packages.")
        
        # Build model
        self.model_name = cfg.model_name
        self.model_path = cfg.model_path
        self.image_size = cfg.image_size
        self.pixel_norm = cfg.pixel_norm
        self.pixel_mean = np.array(cfg.pixel_mean, dtype=np.float32).reshape(1, 1, 3)
        self.pixel_std = np.array(cfg.pixel_std, dtype=np.float32).reshape(1, 1, 3)
        self.device = device
        self.verbose = cfg.verbose
        self.batch_size = batch_size
        self.enable_batch_padding = cfg.enable_batch_padding
        
        # TensorRT specific attributes
        self.engine = None
        self.context = None
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = None
        
        # Performance monitoring
        self.inference_times = []
        self.warmup_done = False
        
        # Load TensorRT engine
        self._load_engine()
        
        # Scale factor for normalization
        self.scale = 1.0 / 255.0

    def _load_engine(self):
        """Load TensorRT engine from file."""
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"TensorRT engine not found: {self.model_path}")
        
        if not self.model_path.endswith('.engine') and not self.model_path.endswith('.plan'):
            log.warning(f"Model path doesn't appear to be a TensorRT engine: {self.model_path}")
        
        # Initialize TensorRT logger
        trt_logger = trt.Logger(trt.Logger.WARNING)
        
        # Load engine
        with open(self.model_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(trt_logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {self.model_path}")
        
        # Create execution context
        self.context = self.engine.create_execution_context()
        
        # Create CUDA stream
        self.stream = cuda.Stream()
        
        # Allocate memory for inputs and outputs
        self._allocate_memory()
        
        log.info(f"TensorRT engine loaded successfully: {self.model_path}")
        log.info(f"Engine max batch size: {self.engine.max_batch_size}")

    def _allocate_memory(self):
        """Allocate GPU and CPU memory for inputs and outputs."""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for i in range(self.engine.num_bindings):
            binding_name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = self.engine.get_binding_shape(i)
            
            # Handle dynamic shapes
            if -1 in shape:
                # For dynamic batch size, use the configured batch size
                shape = list(shape)
                batch_idx = shape.index(-1)
                shape[batch_idx] = self.batch_size
                shape = tuple(shape)
            
            size = trt.volume(shape)
            
            # Allocate memory
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(i):
                self.inputs.append({
                    'name': binding_name,
                    'host': host_mem,
                    'device': device_mem,
                    'shape': shape,
                    'dtype': dtype
                })
                log.info(f"Input {binding_name}: shape={shape}, dtype={dtype}")
            else:
                self.outputs.append({
                    'name': binding_name,
                    'host': host_mem,
                    'device': device_mem,
                    'shape': shape,
                    'dtype': dtype
                })
                log.info(f"Output {binding_name}: shape={shape}, dtype={dtype}")

    def _preprocess(
        self,
        image: np.ndarray, 
        detections: list[Dict],
    ) -> np.ndarray:
        """
        Crop preprocessing optimized for TensorRT inference.

        Args:
            image: Input BGR image (H, W, C) uint8
            detections: List of detection dictionaries
            
        Returns:
            Batch tensor (N, C, H, W) float32
        """
        h_out, w_out = self.image_size
        n_crops = len(detections)
        
        # Handle dynamic batch size by padding if necessary
        if self.enable_batch_padding and n_crops < self.batch_size:
            effective_batch_size = self.batch_size
        else:
            effective_batch_size = n_crops
        
        # Pre-allocate output (faster than appending)
        batch = np.zeros((effective_batch_size, 3, h_out, w_out), dtype=np.float32)

        for i, detection in enumerate(detections):       
            # Step 1: Crop with slicing (view only)
            (l, t, r, b) = detection[:4].astype(int).tolist()  # (x1, y1, x2, y2)
            crop = image[t:b, l:r]
            
            # Step 2: Resize directly to target
            # Use INTER_LINEAR for best speed/quality tradeoff
            resized = cv2.resize(
                crop, (w_out, h_out), 
                interpolation=cv2.INTER_LINEAR
            )         
            
            # Step 3: Vectorized normalization
            # Use in-place operations to reduce memory
            normalized = resized.astype(np.float32, copy=False)
            np.multiply(normalized, self.scale, out=normalized)  # /= 255.0
            np.subtract(normalized, self.pixel_mean, out=normalized)
            np.divide(normalized, self.pixel_std, out=normalized)
            
            # Step 4: CHW conversion (no copy)
            batch[i] = normalized.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
        
        # If padding was applied, duplicate the last valid sample
        if self.enable_batch_padding and n_crops < self.batch_size:
            for i in range(n_crops, self.batch_size):
                batch[i] = batch[n_crops - 1]
           
        return batch, n_crops

    def _warmup(self):
        """Perform warmup inference to optimize GPU kernels."""
        if self.warmup_done:
            return
        
        log.info("Performing TensorRT warmup...")
        
        # Create dummy input
        dummy_input = np.random.rand(*self.inputs[0]['shape']).astype(self.inputs[0]['dtype'])
        
        # Perform several warmup inferences
        for _ in range(3):
            self._inference(dummy_input)
        
        self.warmup_done = True
        log.info("TensorRT warmup completed")

    def _inference(self, input_batch: np.ndarray) -> np.ndarray:
        """
        Run TensorRT inference.
        
        Args:
            input_batch: Input batch (N, C, H, W)
            
        Returns:
            Output features
        """
        start_time = time.time()
        
        # Handle dynamic batch size
        if hasattr(self.context, 'set_binding_shape'):
            # For engines with dynamic shapes
            actual_batch_size = input_batch.shape[0]
            input_shape = (actual_batch_size,) + input_batch.shape[1:]
            self.context.set_binding_shape(0, input_shape)
        
        # Copy input to GPU
        np.copyto(self.inputs[0]['host'], input_batch.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # Run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Copy output from GPU
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        
        # Synchronize stream
        self.stream.synchronize()
        
        # Reshape output
        output_shape = self.outputs[0]['shape']
        if hasattr(self.context, 'get_binding_shape'):
            # For dynamic shapes, get actual output shape
            output_shape = self.context.get_binding_shape(1)
        
        output = self.outputs[0]['host'][:np.prod(output_shape)].reshape(output_shape)
        
        # Record performance
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # Keep only last 100 measurements for rolling average
        if len(self.inference_times) > 100:
            self.inference_times.pop(0)
        
        if self.verbose and len(self.inference_times) % 50 == 0:
            avg_time = np.mean(self.inference_times)
            log.info(f"TensorRT inference avg time: {avg_time*1000:.2f}ms")
        
        return output

    def process(self, input: PipelineMessage) -> PipelineMessage:
        """
        Process input through TensorRT feature extractor.
        
        Args:
            input: Pipeline message containing frame and detections
            
        Returns:
            Pipeline message with extracted features
        """
        # Warmup on first inference
        if not self.warmup_done:
            self._warmup()
        
        # Preprocess input
        input_batch, n_valid = self._preprocess(input.data['frame'], input.data['detections'])
        
        # Run TensorRT inference
        features = self._inference(input_batch)
        
        # Extract only valid features (remove padding if applied)
        if self.enable_batch_padding and n_valid < self.batch_size:
            features = features[:n_valid]
        
        # Create output message
        out_features = PipelineMessage(
            msg_type=MessageType.DATA,
            data={
                'frame': input.data['frame'],
                'detections': np.stack(input.data['detections']),
                'features': features,
            },
            metadata=input.metadata,
            timestamp=input.timestamp,
        )
        return out_features

    def get_performance_stats(self) -> dict:
        """Get performance statistics."""
        if not self.inference_times:
            return {}
        
        return {
            'avg_inference_time_ms': np.mean(self.inference_times) * 1000,
            'min_inference_time_ms': np.min(self.inference_times) * 1000,
            'max_inference_time_ms': np.max(self.inference_times) * 1000,
            'total_inferences': len(self.inference_times),
            'throughput_fps': 1.0 / np.mean(self.inference_times) if self.inference_times else 0
        }

    def __del__(self):
        """Cleanup resources."""
        try:
            if hasattr(self, 'stream') and self.stream:
                self.stream.synchronize()
            
            # Free GPU memory
            for inp in getattr(self, 'inputs', []):
                if 'device' in inp:
                    inp['device'].free()
            
            for out in getattr(self, 'outputs', []):
                if 'device' in out:
                    out['device'].free()
        except Exception as e:
            log.warning(f"Error during TensorRT cleanup: {e}")


# Fallback to ONNX if TensorRT is not available
class FeatureExtractorTensorRT(object):
    """
    TensorRT feature extractor with automatic fallback to ONNX.
    """
    
    def __new__(cls, cfg, device, batch_size, **kwargs):
        if TENSORRT_AVAILABLE and device == "cuda":
            try:
                return TensorRTFeatureExtractor(cfg, device, batch_size, **kwargs)
            except Exception as e:
                log.warning(f"Failed to initialize TensorRT feature extractor: {e}")
                log.info("Falling back to ONNX feature extractor")
        
        # Fallback to original ONNX implementation
        from tracker.reid.feature_extractor import FeatureExtractor
        return FeatureExtractor(cfg, device, batch_size, **kwargs)
