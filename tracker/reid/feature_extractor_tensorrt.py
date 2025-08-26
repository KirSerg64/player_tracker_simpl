from __future__ import absolute_import
from typing import Dict
import numpy as np
import cv2
import os
import logging
import time
import tensorrt as trt
from cuda.bindings import runtime as cudart
import cuda
import tracker.reid.utils.common as common

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
        # Build model
        self.model_name = cfg.model_name
        self.engine_path = cfg.model_path
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
        # self._setup_cuda()
        
        # Load TensorRT engine
        self._load_engine()
        self._allocate_memory()
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
        if not os.path.isfile(self.engine_path):
            raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")

        if not self.engine_path.endswith('.engine') and not self.engine_path.endswith('.plan'):
            log.warning(f"Model path doesn't appear to be a TensorRT engine: {self.engine_path}")

        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(self.engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            assert runtime
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine, "Failed to load TensorRT engine"
        self.context = self.engine.create_execution_context()
        assert self.context, "Failed to create execution context"
        log.info(f"TensorRT engine loaded successfully: {self.engine_path}")

    def _allocate_memory(self):
        """Allocate GPU and CPU memory for inputs and outputs."""
        self.inputs = []
        self.outputs = []
        self.allocations = []
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = False
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            if is_input:
                if shape[0] == -1: # check for dynamic batch size
                    shape[0] = self.batch_size
                else:
                    self.batch_size = max(self.batch_size, shape[0])
            else: 
                if shape[0] == -1: # check for dynamic batch size
                    shape[0] = self.batch_size
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = common.cuda_call(cudart.cudaMalloc(size))
            binding = {
                "index": i,
                "name": name,
                "dtype": np.dtype(trt.nptype(dtype)),
                "shape": list(shape),
                "allocation": allocation,
                "size": size,
            }
            self.allocations.append(allocation)
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        assert self.batch_size > 0, "Failed to determine batch size"
        assert len(self.inputs) > 0, "Failed to create input bindings"
        assert len(self.outputs) > 0, "Failed to create output bindings"
        assert len(self.allocations) > 0, "Failed to allocate memory"

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

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]["shape"], self.inputs[0]["dtype"]

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for o in self.outputs:
            specs.append((o["shape"], o["dtype"]))
        return specs

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

    def _inference(self, batch: np.ndarray, scales=None, nms_threshold=None) -> np.ndarray:

        # Prepare the output data.
        outputs = []
        for shape, dtype in self.output_spec():
            outputs.append(np.zeros(shape, dtype))

        # Process I/O and execute the network.
        common.memcpy_host_to_device(
            self.inputs[0]["allocation"], np.ascontiguousarray(batch)
        )

        self.context.execute_v2(self.allocations)

        for o in range(len(outputs)):
            common.memcpy_device_to_host(outputs[o], self.outputs[o]["allocation"])

        # Process the results.
        features = outputs[0]

        return features

    def process(self, input: PipelineMessage) -> PipelineMessage:
        """
        Process input through TensorRT feature extractor.
        
        Args:
            input: Pipeline message containing frame and detections
            
        Returns:
            Pipeline message with extracted features
        """
        # Warmup on first inference
        # if not self.warmup_done:
        #     self._warmup()
        
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


