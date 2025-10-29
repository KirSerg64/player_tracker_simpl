"""
Huggingface API Feature Extractor for Person Re-Identification

Usage:
    See tracker/configs/reid/feature_extractor_automodel.yaml for configuration
"""

from __future__ import absolute_import

from more_itertools import chunked
import supervision as sv
from typing import List
import numpy as np
import cv2
import os
import logging

import torch
from transformers import AutoImageProcessor, AutoModel, DINOv3ViTConfig

from tracker.utils.pipeline_base import MessageType, PipelineMessage

log = logging.getLogger(__name__)


class FeatureExtractorAutoModel(object):
    """AutoModel-based feature extractor for person ReID."""

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
        self.model_config = DINOv3ViTConfig.from_pretrained(cfg.model_config)
        self.model_preprocessor_config = cfg.model_preprocessor_config
        self.model_path = cfg.model_path
        self.image_size = cfg.image_size
        self.output_features = cfg.get("output_features", "cls_token")
        self.device = device
        self.batch_size = batch_size
                          
        # Try loading ONNX model first
        if self.model_path and os.path.isfile(self.model_path):
            log.info(
                f"Loading model {self.model_path}."
            )            
            self.preprocessor = AutoImageProcessor.from_pretrained(
                self.model_preprocessor_config,
                use_fast=True
            )
            self.model = AutoModel.from_config(self.model_config)
            self.model.load_state_dict(torch.load(self.model_path))
            self.model.compile(mode="max-autotune", fullgraph=True)            
            self.model.eval().to(self.device)
        else:
            log.info(
                f"Model not found at {self.model_path}. "
                f"Downloading from HuggingFace: {self.model_download_id}"
            )
            self.preprocessor = AutoImageProcessor.from_pretrained(
                self.model_download_id,
                use_fast=True
            )
            self.model = AutoModel.from_pretrained(
                self.model_download_id,
                device_map="auto", 
            )
            self.model.compile(mode="max-autotune", fullgraph=True)  
            self.model.eval().to(self.device)

    def _detections_crops(
        self,
        image: np.ndarray, 
        detections: np.ndarray,
    ) -> List:
        """
        Preprocess crops for DINOv3 inference.
                
        Args:
            image: Input BGR image (H, W, C) uint8
            detections: List of detections with bbox coordinates
            
        Returns:
            list of cropped images (PIL format)
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h_img, w_img = image_rgb.shape[:2]
        crops = []
        
        for i, detection in enumerate(detections):
            # Extract bounding box coordinates
            (l, t, r, b) = detection[:4].astype(int).tolist()            
            # Ensure coordinates are within image bounds           
            l = max(0, min(l, w_img - 1))
            t = max(0, min(t, h_img - 1))
            r = max(l + 1, min(r, w_img))
            b = max(t + 1, min(b, h_img))            
            # Crop from original image
            crop = image_rgb[t:b, l:r]         
            crops.append(sv.cv2_to_pillow(crop))
        return crops

    def process(self, input: PipelineMessage) -> PipelineMessage:
        """
        Process pipeline message with person detections.
        
        Args:
            input: PipelineMessage containing frame and detections
            
        Returns:
            PipelineMessage with extracted and normalized features
        """
        # Preprocess crops
        crops = self._detections_crops(
            input.data['frame'],    
            input.data['detections']
        )
        batches = chunked(crops, self.batch_size)

        features = []
        with torch.inference_mode():
            for batch in batches:
                inputs = self.preprocessor(images=batch, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                if self.output_features == "cls_token":
                    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                elif self.output_features == "mean":
                    embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                features.append(embeddings)

        features = np.concatenate(features, axis=0)
        
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
