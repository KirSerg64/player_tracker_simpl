"""
SAM3 segmenter implementation with text prompt support.
Uses HuggingFace transformers for open-vocabulary segmentation.
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional, List

from tracker.segment.base_segmenter import BaseSegmenter
from tracker.utils.pipeline_base import MessageType, PipelineMessage
import logging

log = logging.getLogger(__name__)


class Sam3Segmenter(BaseSegmenter):
    """
    Segmenter for SAM3 model (HuggingFace transformers).
    Supports text prompts for open-vocabulary segmentation.
    """
    
    def __init__(self, cfg, device='cuda:0', batch_size=1) -> None:
        """
        Initialize SAM3 segmenter with text prompt support.
        
        Args:
            cfg: Config with model_path, and optional text_prompt
            device: Device to run on
            batch_size: Batch size
        """
        super().__init__(cfg, device, batch_size)
        
        print(f"Initializing Sam3Segmenter on {device}")
        
        # Import SAM3 from transformers
        try:
            from transformers import Sam3Processor, Sam3Model
        except ImportError:
            raise ImportError(
                "transformers library required for SAM3. "
                "Install with: pip install transformers"
            )
        
        # Load processor and model
        self._processor = Sam3Processor.from_pretrained(cfg.model_path)
        self._model = Sam3Model.from_pretrained(
            cfg.model_path,
            torch_dtype=self._torch_dtype
        )
        self._model.to(device=self._device)
        self._model.eval()
        
        # Text prompt configuration
        self.text_prompt = getattr(cfg, 'text_prompt', 'person')
        self.use_text_prompt = getattr(cfg, 'use_text_prompt', False)
        
        self._current_image = None
        
        log.info(f"SAM3 text prompt: '{self.text_prompt}' (enabled: {self.use_text_prompt})")
    
    def process(self, input: PipelineMessage) -> PipelineMessage:
        """
        Process with either bounding boxes or text prompts.
        
        Args:
            input: Pipeline message with 'frame' and optionally 'detections'
            
        Returns:
            Pipeline message with masks
        """
        image = input.data['frame']
        detections = input.data.get('detections', None)
        
        # Choose processing mode
        if self.use_text_prompt and (detections is None or len(detections) == 0):
            # Text-based segmentation (no detector needed)
            masks = self._process_text_prompt(image)
        elif detections is not None and len(detections) > 0:
            # Box-based segmentation (with detector)
            masks = self._process_boxes(image, detections)
        else:
            # No input, return empty
            masks = []
        
        out_pipeline = PipelineMessage(
            msg_type=MessageType.DATA,
            data={
                'frame': image,
                "detections": np.stack(detections) if detections is not None and len(detections) > 0 else np.array([]),
                "masks": np.stack(masks) if len(masks) > 0 else np.array([]),
            },
            metadata=input.metadata,
            timestamp=input.timestamp,
        )
        return out_pipeline
    
    @torch.no_grad()
    def _process_boxes(self, image: np.ndarray, detections: np.ndarray) -> List[np.ndarray]:
        """
        Process image with bounding box prompts.
        
        Args:
            image: RGB image (H, W, 3)
            detections: Bounding boxes (N, 4+) with x1,y1,x2,y2
            
        Returns:
            List of binary masks
        """
        pil_image = Image.fromarray(image)
        input_boxes = detections[:, :4].tolist()
        
        # Prepare inputs
        inputs = self._processor(
            pil_image,
            input_boxes=[input_boxes],
            return_tensors="pt"
        ).to(self._device)
        
        # Run inference
        outputs = self._model(**inputs)
        pred_masks = outputs.pred_masks.squeeze(0)  # (num_boxes, num_masks, H, W)
        
        masks = []
        for i, mask in enumerate(pred_masks):
            # Take best mask from multi-mask output
            if mask.dim() == 3:  # (num_masks, H, W)
                mask = mask[0]
            
            # Post-process
            processed_mask = self._postprocess_mask(mask)
            masks.append(processed_mask)
        
        return masks
    
    @torch.no_grad()
    def _process_text_prompt(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Process image with text prompt for open-vocabulary segmentation.
        Returns separate masks for each detected instance.
        
        Args:
            image: RGB image (H, W, 3)
            
        Returns:
            List of binary masks (one per detected instance)
        """
        pil_image = Image.fromarray(image)
        
        # Prepare inputs with text prompt
        inputs = self._processor(
            pil_image,
            text=self.text_prompt,
            return_tensors="pt"
        ).to(self._device)
        
        # Run inference
        outputs = self._model(**inputs)
        
        # Extract masks - SAM3 can detect multiple instances
        pred_masks = outputs.pred_masks  # (batch, num_instances, num_masks, H, W)
        
        if pred_masks.dim() == 5:  # (1, num_instances, num_masks, H, W)
            pred_masks = pred_masks.squeeze(0)  # (num_instances, num_masks, H, W)
        
        masks = []
        if pred_masks.dim() == 4:  # Multiple instances detected
            for instance_masks in pred_masks:
                # Take best mask for each instance
                if instance_masks.dim() == 3:
                    mask = instance_masks[0]  # Take first/best mask
                else:
                    mask = instance_masks
                
                processed_mask = self._postprocess_mask(mask)
                # Only add if mask has sufficient area
                if processed_mask.sum() > 100:  # Threshold for minimum mask size
                    masks.append(processed_mask)
        elif pred_masks.dim() == 3:  # Single instance
            mask = pred_masks[0] if pred_masks.shape[0] > 1 else pred_masks
            processed_mask = self._postprocess_mask(mask)
            if processed_mask.sum() > 100:
                masks.append(processed_mask)
        
        return masks
    
    def set_image(self, image: np.ndarray) -> None:
        """Store image (SAM3 doesn't pre-embed)."""
        self._current_image = image
        self._embedded = True
    
    def reset_image(self) -> None:
        """Reset stored image."""
        self._current_image = None
        self._embedded = False
    
    @torch.no_grad()
    def predict(self, prompts: Dict, mode: str, multimask: bool = True) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Predict masks with various prompt types.
        
        Args:
            prompts: Dictionary with:
                - 'point_coords', 'point_labels' for point mode
                - 'boxes' for box mode
                - 'text' for text mode
            mode: 'point', 'box', 'text', 'both'
            multimask: Return multiple masks
            
        Returns:
            (masks, scores, logits) - logits may be None for SAM3
        """
        assert self._embedded, 'Must call set_image() before predict()'
        
        pil_image = Image.fromarray(self._current_image)
        
        if mode == 'point':
            inputs = self._processor(
                pil_image,
                input_points=[prompts['point_coords'].tolist()],
                input_labels=[prompts['point_labels'].tolist()],
                return_tensors="pt"
            ).to(self._device)
        elif mode == 'box':
            inputs = self._processor(
                pil_image,
                input_boxes=[prompts['boxes'].tolist()],
                return_tensors="pt"
            ).to(self._device)
        elif mode == 'text':
            inputs = self._processor(
                pil_image,
                text=prompts.get('text', self.text_prompt),
                return_tensors="pt"
            ).to(self._device)
        elif mode == 'both':
            # Combine point and box prompts
            inputs = self._processor(
                pil_image,
                input_points=[prompts['point_coords'].tolist()],
                input_labels=[prompts['point_labels'].tolist()],
                return_tensors="pt"
            ).to(self._device)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        
        # Run inference
        outputs = self._model(**inputs)
        
        # Extract predictions
        pred_masks = outputs.pred_masks.squeeze(0).cpu().numpy()
        pred_scores = outputs.iou_scores.squeeze(0).cpu().numpy() if hasattr(outputs, 'iou_scores') else np.ones(len(pred_masks))
        
        # Convert to binary
        masks = (pred_masks > 0.5).astype(np.uint8)
        
        # Select best mask if not multimask
        if not multimask and len(masks) > 0:
            best_idx = np.argmax(pred_scores)
            masks = masks[best_idx:best_idx+1]
            pred_scores = pred_scores[best_idx:best_idx+1]
        
        return masks, pred_scores, None  # No logits for SAM3
