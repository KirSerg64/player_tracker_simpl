"""
DINOv3 Feature Extractor for Person Re-Identification

This module provides a feature extractor using Meta's DINOv3 vision transformer model.
Supports both ONNX (production) and PyTorch (fallback) inference modes.

Key Features:
- Pad-to-square preprocessing for aspect ratio preservation
- ImageNet normalization (DINOv3 standard)
- 768-dim features from DINOv3-base
- Dynamic batch processing
- BGR to RGB conversion

Usage:
    See tracker/configs/reid/feature_extractor_dinov3.yaml for configuration
"""

from __future__ import absolute_import
from typing import Dict, Optional
import numpy as np
import cv2
import os
import logging

from tracker.utils.pipeline_base import MessageType, PipelineMessage

log = logging.getLogger(__name__)


class FeatureExtractorDINOv3(object):
    """DINOv3-based feature extractor for person ReID."""

    def __init__(
        self, 
        cfg, 
        device: str, 
        batch_size: int, 
        **kwargs
    ):
        """
        Initialize DINOv3 feature extractor.
        
        Args:
            cfg: Configuration object with model settings
            device: 'cuda' or 'cpu'
            batch_size: Maximum batch size for inference
        """
        self.model_name = cfg.model_name
        self.model_path = cfg.model_path
        self.image_size = cfg.image_size
        self.pixel_mean = np.array(cfg.pixel_mean, dtype=np.float32).reshape(1, 1, 3)
        self.pixel_std = np.array(cfg.pixel_std, dtype=np.float32).reshape(1, 1, 3)
        self.device = device
        self.verbose = cfg.verbose
        self.batch_size = batch_size
        self.use_pytorch_fallback = cfg.get('use_pytorch_fallback', False)
        self.pad_to_square = cfg.get('pad_to_square', True)
        self.session = None
        self.torch_model = None
        
        # DINOv3 expects square images
        assert self.image_size[0] == self.image_size[1], \
            f"DINOv3 requires square input size, got {self.image_size}"
        
        self.scale = 1.0 / 255.0  # Scale factor for normalization
        
        # Try loading ONNX model first
        if self.model_path and os.path.isfile(self.model_path):
            self._load_onnx_model()
        
        # Fallback to PyTorch if ONNX not available and fallback enabled
        if self.session is None and self.use_pytorch_fallback:
            self._load_pytorch_model()
        
        if self.session is None and self.torch_model is None:
            raise RuntimeError(
                f"Failed to load model. Check model_path: {self.model_path} "
                f"or enable PyTorch fallback with use_pytorch_fallback=True"
            )
        
        inference_mode = "ONNX" if self.session else "PyTorch"
        log.info(f"DINOv3 Feature Extractor initialized ({inference_mode} mode)")

    def _load_onnx_model(self):
        """Load ONNX model with optimized runtime settings."""
        try:
            import onnxruntime
            
            providers = ["CPUExecutionProvider"]
            if self.device == "cuda":
                if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
                    providers.insert(0, "CUDAExecutionProvider")
                else:
                    log.warning("CUDA requested but not available in ONNX Runtime. Using CPU...")
            
            # Configure session options for optimal performance
            session_options = onnxruntime.SessionOptions()
            cpu_count = os.cpu_count() or 1
            session_options.intra_op_num_threads = min(4, cpu_count)
            session_options.inter_op_num_threads = 1
            session_options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
            session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            log.info(f"Loading ONNX model from: {self.model_path}")
            log.info(f"Using ONNX Runtime provider: {providers[0]}")
            
            self.session = onnxruntime.InferenceSession(
                self.model_path, 
                providers=providers,
                sess_options=session_options
            )
            
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            # Log model info
            input_shape = self.session.get_inputs()[0].shape
            output_shape = self.session.get_outputs()[0].shape
            log.info(f"ONNX Model loaded - Input: {input_shape}, Output: {output_shape}")
            
        except Exception as e:
            log.warning(f"Failed to load ONNX model: {e}")
            self.session = None

    def _load_pytorch_model(self):
        """Load PyTorch model as fallback."""
        try:
            import torch
            from transformers import AutoModel, AutoImageProcessor
            
            log.info(f"Loading PyTorch DINOv3 model: {self.model_name}")
            
            self.torch_model = AutoModel.from_pretrained(self.model_name)
            self.torch_model.eval()
            
            if self.device == "cuda" and torch.cuda.is_available():
                self.torch_model = self.torch_model.cuda()
                log.info("PyTorch model moved to CUDA")
            else:
                log.info("PyTorch model using CPU")
            
            log.info("PyTorch DINOv3 model loaded successfully")
            
        except Exception as e:
            log.error(f"Failed to load PyTorch model: {e}")
            self.torch_model = None

    def _pad_to_square(self, crop: np.ndarray) -> np.ndarray:
        """
        Pad image crop to square shape while preserving aspect ratio.
        
        Args:
            crop: Input image (H, W, C) in RGB format
            
        Returns:
            Square padded image
        """
        h, w = crop.shape[:2]
        
        if h == w:
            return crop
        
        # Calculate padding for shorter dimension
        max_dim = max(h, w)
        pad_h_top = (max_dim - h) // 2
        pad_h_bottom = max_dim - h - pad_h_top
        pad_w_left = (max_dim - w) // 2
        pad_w_right = max_dim - w - pad_w_left
        
        # Use mean color for padding (better than black/zero padding)
        mean_color = crop.mean(axis=(0, 1)).astype(np.uint8)
        
        padded = cv2.copyMakeBorder(
            crop,
            pad_h_top, pad_h_bottom,
            pad_w_left, pad_w_right,
            cv2.BORDER_CONSTANT,
            value=mean_color.tolist()
        )
        
        return padded

    def _preprocess(
        self,
        image: np.ndarray, 
        detections: list,
    ) -> np.ndarray:
        """
        Preprocess crops for DINOv3 inference.
        
        Process:
        1. Crop bounding boxes from image
        2. Convert BGR to RGB
        3. Pad to square (if enabled)
        4. Resize to target size (224x224)
        5. Normalize with ImageNet stats
        6. Convert to NCHW format
        
        Args:
            image: Input BGR image (H, W, C) uint8
            detections: List of detections with bbox coordinates
            
        Returns:
            Preprocessed batch (N, C, H, W) float32
        """
        h_out, w_out = self.image_size
        n_crops = len(detections)
        
        # Pre-allocate output batch
        batch = np.zeros((n_crops, 3, h_out, w_out), dtype=np.float32)
        
        for i, detection in enumerate(detections):
            # Extract bounding box coordinates
            (l, t, r, b) = detection[:4].astype(int).tolist()
            
            # Ensure coordinates are within image bounds
            h_img, w_img = image.shape[:2]
            l = max(0, min(l, w_img - 1))
            t = max(0, min(t, h_img - 1))
            r = max(l + 1, min(r, w_img))
            b = max(t + 1, min(b, h_img))
            
            # Crop from original image
            crop = image[t:b, l:r]
            
            # Convert BGR (OpenCV) to RGB (DINOv3 expects RGB)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            # Pad to square if enabled (preserves aspect ratio)
            if self.pad_to_square:
                crop_rgb = self._pad_to_square(crop_rgb)
            
            # Resize to target size (224x224 for DINOv3-base)
            resized = cv2.resize(
                crop_rgb, 
                (w_out, h_out), 
                interpolation=cv2.INTER_LINEAR
            )
            
            # Normalize: scale to [0, 1], then apply ImageNet stats
            normalized = resized.astype(np.float32, copy=False)
            np.multiply(normalized, self.scale, out=normalized)  # /= 255.0
            np.subtract(normalized, self.pixel_mean, out=normalized)
            np.divide(normalized, self.pixel_std, out=normalized)
            
            # Convert HWC to CHW
            batch[i] = normalized.transpose(2, 0, 1)
        
        return batch

    def _inference_onnx(self, batch: np.ndarray) -> np.ndarray:
        """Run ONNX inference."""
        features = self.session.run(
            [self.output_name], 
            {self.input_name: batch}
        )[0]
        return features

    def _inference_pytorch(self, batch: np.ndarray) -> np.ndarray:
        """Run PyTorch inference."""
        import torch
        
        # Convert numpy to torch tensor
        tensor = torch.from_numpy(batch)
        
        if self.device == "cuda" and torch.cuda.is_available():
            tensor = tensor.cuda()
        
        # Run inference
        with torch.no_grad():
            outputs = self.torch_model(pixel_values=tensor)
            # Extract pooler_output (CLS token) or use last_hidden_state[:, 0, :]
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                features = outputs.pooler_output
            else:
                features = outputs.last_hidden_state[:, 0, :]
        
        # Convert back to numpy
        features = features.cpu().numpy()
        return features

    def process(self, input: PipelineMessage) -> PipelineMessage:
        """
        Process pipeline message with person detections.
        
        Args:
            input: PipelineMessage containing frame and detections
            
        Returns:
            PipelineMessage with extracted and normalized features
        """
        # Preprocess crops
        input_batch = self._preprocess(
            input.data['frame'], 
            input.data['detections']
        )
        
        # Run inference (ONNX or PyTorch)
        if self.session is not None:
            features = self._inference_onnx(input_batch)
        else:
            features = self._inference_pytorch(input_batch)
        
        # L2 normalize features (standard for ReID)
        normed_features = features / np.linalg.norm(features, axis=1, keepdims=True)
        
        # Prepare output message
        out_features = PipelineMessage(
            msg_type=MessageType.DATA,
            data={
                'frame': input.data['frame'],
                'detections': np.array(input.data['detections']) if not isinstance(
                    input.data['detections'], np.ndarray
                ) else input.data['detections'],
                'features': normed_features,
            },
            metadata=input.metadata,
            timestamp=input.timestamp,
        )
        
        return out_features
