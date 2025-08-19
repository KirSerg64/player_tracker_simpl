from __future__ import absolute_import
from typing import Dict
import numpy as np
import onnxruntime
import cv2
import os
# from torchreid.utils import (
#     check_isfile, load_pretrained_weights, compute_model_complexity
# )
# from torchreid.models import build_model
from tracker.utils.pipeline_base import MessageType, ProcessConfig, PipelineMessage

import logging
log = logging.getLogger(__name__)


class FeatureExtractorPrtReid(object):
    """A simple API for feature extraction."""

    def __init__(
        self, cfg, device, batch_size, **kwargs
    ):
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
        self.enable_batch_padding = cfg.enable_batch_padding  # Enable padding to optimal batch sizes
        self.session = None

        if self.model_path and os.path.isfile(self.model_path):
            providers = ["CPUExecutionProvider"]
            if device == "cuda":
                if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
                    providers.insert(0, "CUDAExecutionProvider")
                else:  # Only log warning if CUDA was requested but unavailable
                    log.warning("Failed to start ONNX Runtime with CUDA. Using CPU...")
            
            # Configure session options for optimal performance
            session_options = onnxruntime.SessionOptions()
            # Use CPU core count for optimal performance
            cpu_count = os.cpu_count() or 1
            session_options.intra_op_num_threads = min(4, cpu_count)  # Max 4 threads for ops
            session_options.inter_op_num_threads = 1  # Keep sequential between operations
            session_options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
            session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            log.info(f"Using ONNX Runtime {providers[0]}")
            self.session = onnxruntime.InferenceSession(
                self.model_path, 
                providers=providers,
                sess_options=session_options
            )            
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

        assert self.session is not None, "Model path is invalid."
        self.scale = 1.0 / 255.0 # Scale factor for normalization

    def _preprocess(
        self,
        image: np.ndarray, 
        detections: list[Dict],
    ) -> np.ndarray:
        """
        Crop preprocessing for ONNX inference (CPU only)

        Args:
            image: Input BGR image (H, W, C) uint8
            bboxes: [(x1, y1, x2, y2), ...]
            target_size: (height, width)
            mean: RGB normalization means
            std: RGB normalization stds
            
        Returns:
            Batch tensor (N, C, H, W) float32
        """
        h_out, w_out = self.image_size
        n_crops = len(detections)
        
        # Pre-allocate output (faster than appending)
        batch = np.zeros((n_crops, 3, h_out, w_out), dtype=np.float32) 

        for i, detection in enumerate(detections):       
            # Step 1: Crop with slicing (view only)
            (l, t, r, b) = detection[:4].astype(int).tolist()  # (x1, y1, x2, y2)
            crop = image[t:b, l:r]
            # Step 2: Resize directly to target
            # - Use INTER_LINEAR for best speed/quality tradeoff
            # - Convert to RGB during resize
            resized = cv2.resize(
                crop, (w_out, h_out), 
                interpolation=cv2.INTER_LINEAR
            )         
            # Step 3: Vectorized normalization
            # - Use in-place operations to reduce memory
            normalized = resized.astype(np.float32, copy=False)
            np.multiply(normalized, self.scale, out=normalized)  # /= 255.0
            np.subtract(normalized, self.pixel_mean, out=normalized)
            np.divide(normalized, self.pixel_std, out=normalized)
            # Step 4: CHW conversion (no copy)
            batch[i] = normalized.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
           
        return batch

    def process(self, input: PipelineMessage) -> PipelineMessage:

        input_batch = self._preprocess(input.data['frame'], input.data['detections'])

        # Run ONNX inference on the padded batch
        features = self.session.run([self.output_name], {self.input_name: input_batch})[0]
  
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
