from __future__ import absolute_import
from typing import Dict
import numpy as np
import cv2
import os
import logging
import time

try:
    import tensorrt as trt
    from cuda import cuda, cudart
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    trt = None
    cuda = None
    cudart = None

from tracker.utils.pipeline_base import MessageType, ProcessConfig, PipelineMessage

log = logging.getLogger(__name__)


def check_cuda_errors(result):
    """Check CUDA errors and raise exception if needed."""
    if isinstance(result, tuple):
        error, *returns = result
        if error != cuda.CUresult.CUDA_SUCCESS:
            _, error_string = cuda.cuGetErrorString(error)
            raise RuntimeError(f"CUDA error: {error_string.decode('utf-8') if isinstance(error_string, bytes) else error_string}")
        return returns[0] if len(returns) == 1 else returns
    elif result != cudart.cudaError_t.cudaSuccess:
        _, error_string = cudart.cudaGetErrorString(result)
        raise RuntimeError(f"CUDA runtime error: {error_string.decode('utf-8') if isinstance(error_string, bytes) else error_string}")


class FeatureExtractorTensorRT(object):
    """TensorRT-optimized feature extractor for high-performance inference."""

    def __init__(
        self, cfg, device, batch_size, **kwargs
    ):
        if not TENSORRT_AVAILABLE:
            raise ImportError("TensorRT not available. Install tensorrt and cuda-python packages.")
        
        # Initialize CUDA
        check_cuda_errors(cuda.cuInit(0))
        
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
        
        # CUDA context and device
        self.cuda_device = None
        self.cuda_context = None
        
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
        self._dtype_warned = False  # Debug flag for dtype warnings
        
        # Setup CUDA context
        self._setup_cuda()
        
        # Load TensorRT engine
        self._load_engine()
        
        # Scale factor for normalization
        self.scale = 1.0 / 255.0

    def _setup_cuda(self):
        """Setup CUDA device and context."""
        # Get device count
        device_count = check_cuda_errors(cuda.cuDeviceGetCount())
        if device_count == 0:
            raise RuntimeError("No CUDA devices available")
        
        # Get device
        self.cuda_device = check_cuda_errors(cuda.cuDeviceGet(0))  # Use device 0
        
        # Try to get existing CUDA context first
        self._created_own_context = False
        try:
            # Check if there's already an active context
            result = cuda.cuCtxGetCurrent()
            if isinstance(result, tuple) and len(result) == 2:
                error, current_context = result
                if error == cuda.CUresult.CUDA_SUCCESS and current_context is not None:
                    self.cuda_context = current_context
                    log.info("Using existing CUDA context")
                else:
                    # No active context, create new one
                    self.cuda_context = check_cuda_errors(
                        cuda.cuCtxCreate(cuda.CUctx_flags.CU_CTX_SCHED_AUTO, self.cuda_device)
                    )
                    self._created_own_context = True
                    log.info("Created new CUDA context on device 0")
            else:
                # Fallback: create new context
                self.cuda_context = check_cuda_errors(
                    cuda.cuCtxCreate(cuda.CUctx_flags.CU_CTX_SCHED_AUTO, self.cuda_device)
                )
                self._created_own_context = True
                log.info("Created fallback CUDA context on device 0")
        except Exception as e:
            # Final fallback: create new context
            log.warning(f"Context detection failed: {e}, creating new context")
            self.cuda_context = check_cuda_errors(
                cuda.cuCtxCreate(cuda.CUctx_flags.CU_CTX_SCHED_AUTO, self.cuda_device)
            )
            self._created_own_context = True
            log.info("Created emergency fallback CUDA context on device 0")

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
        self.stream = check_cuda_errors(cuda.cuStreamCreate(cuda.CUstream_flags.CU_STREAM_DEFAULT))
        
        # Allocate memory for inputs and outputs
        self._allocate_memory()
        
        log.info(f"TensorRT engine loaded successfully: {self.model_path}")

    def _allocate_memory(self):
        """Allocate GPU and CPU memory for inputs and outputs."""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            shape = self.engine.get_tensor_shape(tensor_name)

            # Handle dynamic shapes
            if -1 in shape:
                # For dynamic batch size, use the configured batch size
                shape = list(shape)
                batch_idx = shape.index(-1)
                shape[batch_idx] = self.batch_size
                shape = tuple(shape)
            
            size = trt.volume(shape)
            nbytes = size * np.dtype(dtype).itemsize
            
            # Allocate pinned host memory
            host_mem = check_cuda_errors(
                cudart.cudaMallocHost(nbytes)
            )
            
            # Allocate device memory
            device_mem = check_cuda_errors(
                cuda.cuMemAlloc(nbytes)
            )
            
            self.bindings.append(int(device_mem))
            
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.inputs.append({
                    'name': tensor_name,
                    'host': host_mem,
                    'device': device_mem,
                    'shape': shape,
                    'dtype': dtype,
                    'nbytes': nbytes
                })
                log.info(f"Input {tensor_name}: shape={shape}, dtype={dtype}")
            else:
                self.outputs.append({
                    'name': tensor_name,
                    'host': host_mem,
                    'device': device_mem,
                    'shape': shape,
                    'dtype': dtype,
                    'nbytes': nbytes
                })
                log.info(f"Output {tensor_name}: shape={shape}, dtype={dtype}")

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
        
        # Validate CUDA context before using it
        if self.cuda_context is None:
            raise RuntimeError("CUDA context is None - initialization failed")
        
        # Push our CUDA context to ensure we're using the right one
        try:
            check_cuda_errors(cuda.cuCtxPushCurrent(self.cuda_context))
        except Exception as e:
            log.error(f"Failed to push CUDA context: {e}")
            # Try to recreate context as emergency fallback
            self._setup_cuda()
            check_cuda_errors(cuda.cuCtxPushCurrent(self.cuda_context))
        
        try:
            # Handle dynamic batch size
            if hasattr(self.context, 'set_binding_shape'):
                # For engines with dynamic shapes
                actual_batch_size = input_batch.shape[0]
                input_shape = (actual_batch_size,) + input_batch.shape[1:]
                self.context.set_binding_shape(0, input_shape)
            
            # Copy input data to pinned host memory
            input_flat = input_batch.flatten().astype(self.inputs[0]['dtype'])
            
            # Convert host memory pointer to numpy array with correct dtype and copy data
            host_ptr = self.inputs[0]['host']
            
            # Direct approach: create array with exact byte size needed
            expected_dtype = self.inputs[0]['dtype']
            total_bytes = input_flat.nbytes
            
            # Create byte array from memory, then interpret as correct dtype
            host_bytes = np.ctypeslib.as_array(host_ptr, shape=(total_bytes,)).view(dtype=np.uint8)
            host_array = np.frombuffer(host_bytes, dtype=expected_dtype).reshape(input_flat.shape)
            
            # Ensure we have a writable copy
            if not host_array.flags.writeable:
                host_array = host_array.copy()
            
            # Debug: Verify dtypes match
            if self.verbose and hasattr(self, '_dtype_warned') and not self._dtype_warned:
                log.info(f"Input dtypes - input_flat: {input_flat.dtype}, host_array: {host_array.dtype}, expected: {self.inputs[0]['dtype']}")
                self._dtype_warned = True
            
            # Verify dtypes match before copying
            if input_flat.dtype != host_array.dtype:
                log.warning(f"Dtype mismatch: {input_flat.dtype} != {host_array.dtype}, converting...")
                input_flat = input_flat.astype(host_array.dtype)
            
            # Now copy with matching dtypes
            np.copyto(host_array, input_flat)
            
            # Copy input from host to device asynchronously
            check_cuda_errors(
                cuda.cuMemcpyHtoDAsync(
                    self.inputs[0]['device'],
                    host_ptr,
                    self.inputs[0]['nbytes'],
                    self.stream
                )
            )
            
            # Run inference
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream)
            
            # Copy output from device to host asynchronously
            check_cuda_errors(
                cuda.cuMemcpyDtoHAsync(
                    self.outputs[0]['host'],
                    self.outputs[0]['device'],
                    self.outputs[0]['nbytes'],
                    self.stream
                )
            )
            
            # Synchronize stream
            check_cuda_errors(cuda.cuStreamSynchronize(self.stream))
            
            # Convert output to numpy array
            output_shape = self.outputs[0]['shape']
            if hasattr(self.context, 'get_binding_shape'):
                # For dynamic shapes, get actual output shape
                output_shape = self.context.get_binding_shape(1)
            
            # Convert host memory pointer to numpy array with correct dtype
            output_size = np.prod(output_shape)
            
            # Direct approach: avoid view() issues
            expected_bytes = output_size * np.dtype(self.outputs[0]['dtype']).itemsize
            output_bytes = np.ctypeslib.as_array(self.outputs[0]['host'], shape=(expected_bytes,)).view(dtype=np.uint8)
            output_array = np.frombuffer(output_bytes, dtype=self.outputs[0]['dtype'])
            
            output = output_array[:output_size].reshape(output_shape).copy()
            
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
            
        finally:
            # Always pop the context when done - with error handling
            try:
                check_cuda_errors(cuda.cuCtxPopCurrent())
            except Exception as e:
                log.warning(f"Failed to pop CUDA context: {e}")

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
            # Synchronize stream before cleanup
            if hasattr(self, 'stream') and self.stream:
                check_cuda_errors(cuda.cuStreamSynchronize(self.stream))
                check_cuda_errors(cuda.cuStreamDestroy(self.stream))
            
            # Free device memory
            for inp in getattr(self, 'inputs', []):
                if 'device' in inp and inp['device']:
                    check_cuda_errors(cuda.cuMemFree(inp['device']))
                if 'host' in inp and inp['host']:
                    check_cuda_errors(cudart.cudaFreeHost(inp['host']))
            
            for out in getattr(self, 'outputs', []):
                if 'device' in out and out['device']:
                    check_cuda_errors(cuda.cuMemFree(out['device']))
                if 'host' in out and out['host']:
                    check_cuda_errors(cudart.cudaFreeHost(out['host']))
            
            # Only destroy CUDA context if we created it ourselves
            # Don't destroy shared contexts as they might be used by other components
            if hasattr(self, 'cuda_context') and self.cuda_context and hasattr(self, '_created_own_context'):
                if self._created_own_context:
                    check_cuda_errors(cuda.cuCtxDestroy(self.cuda_context))
                
        except Exception as e:
            log.warning(f"Error during TensorRT cleanup: {e}")


