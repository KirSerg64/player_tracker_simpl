"""
SAM and Mobile-SAM segmenter implementation.
Uses the segment-anything or mobile-sam predictor API.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional

from tracker.segment.base_segmenter import BaseSegmenter
from tracker.utils.pipeline_base import MessageType, PipelineMessage


class SamSegmenter(BaseSegmenter):
    """
    Segmenter for SAM and Mobile-SAM models.
    Uses the segment-anything or mobile-sam predictor API.
    """
    
    def __init__(self, cfg, device='cuda:0', batch_size=1) -> None:
        """
        Initialize SAM segmenter.
        
        Args:
            cfg: Config with model_type ('sam', 'mobile_sam'), backbone, model_path
            device: Device to run on
            batch_size: Batch size for processing
        """
        super().__init__(cfg, device, batch_size)
        
        print(f"Initializing SamSegmenter ({cfg.model_type}) on {device}")
        
        assert cfg.model_type in ['sam', 'mobile_sam'], \
            f"SamSegmenter only supports 'sam' or 'mobile_sam', got {cfg.model_type}"
        assert cfg.backbone in ['vit_b', 'vit_l', 'vit_h', 'vit_t'], \
            f"Backbone must be vit_b, vit_l, vit_h, or vit_t, got {cfg.backbone}"
        
        # Import appropriate SAM library
        if cfg.model_type == 'mobile_sam':
            from mobile_sam import sam_model_registry, SamPredictor
        else:  # 'sam'
            from segment_anything import sam_model_registry, SamPredictor
        
        # Load model
        self._model = sam_model_registry[cfg.backbone](checkpoint=cfg.model_path)
        self._model.to(device=self._device)
        self._model.eval()
        
        # Compile model for optimization
        try:
            self._model = torch.compile(self._model)
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}. Continuing without compilation.")
        
        self._predictor = SamPredictor(self._model)
        self.original_image = None
        
    def process(self, input: PipelineMessage) -> PipelineMessage:
        """
        Process pipeline message with detections to produce masks.
        
        Args:
            input: Pipeline message with 'frame' and 'detections' keys
            
        Returns:
            Pipeline message with added 'masks' key
        """
        image = input.data['frame']
        detections = input.data.get('detections', None)
        
        if detections is None or len(detections) == 0:
            # No detections, return empty masks
            return PipelineMessage(
                msg_type=MessageType.DATA,
                data={
                    'frame': image,
                    "detections": np.array([]),
                    "masks": np.array([]),
                },
                metadata=input.metadata,
                timestamp=input.timestamp,
            )
        
        # Set image for embedding
        self._predictor.set_image(image)
        
        # Process boxes
        input_boxes = torch.tensor(detections, device=self._device, dtype=torch.float32)
        transformed_boxes = self._predictor.transform.apply_boxes_torch(
            input_boxes[:, :4], 
            image.shape[:2]
        )
        
        masks = []
        for box in transformed_boxes:
            mask, _, _ = self._predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=box.unsqueeze(0),
                multimask_output=False,
            )
            # Post-process mask
            processed_mask = self._postprocess_mask(mask.squeeze(0))
            masks.append(processed_mask)
        
        self._predictor.reset_image()
        
        out_pipeline = PipelineMessage(
            msg_type=MessageType.DATA,
            data={
                'frame': image,
                "detections": np.stack(detections) if len(detections) > 0 else np.array([]),
                "masks": np.stack(masks) if len(masks) > 0 else np.array([]),
            },
            metadata=input.metadata,
            timestamp=input.timestamp,
        )
        return out_pipeline
    
    @torch.no_grad()
    def set_image(self, image: np.ndarray) -> None:
        """
        Set image and compute embeddings.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
        """
        self.original_image = image
        if self._embedded:
            print('Warning: Image already embedded. Call reset_image() first.')
            return
        
        self._predictor.set_image(image)
        self._embedded = True
    
    @torch.no_grad()
    def reset_image(self) -> None:
        """Reset image embeddings."""
        self._predictor.reset_image()
        self._embedded = False
        self.original_image = None
    
    @torch.no_grad()
    def predict(self, prompts: Dict, mode: str, multimask: bool = True) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Predict masks based on prompts.
        
        Args:
            prompts: Dictionary with keys:
                - 'point_coords': np.ndarray [N, 2]
                - 'point_labels': np.ndarray [N]
                - 'mask_input': np.ndarray [1, 256, 256]
                - 'boxes': np.ndarray [N, 4] (x1, y1, x2, y2)
            mode: 'point', 'mask', 'both', or 'box'
            multimask: Return multiple mask predictions
            
        Returns:
            (masks, scores, logits) tuple
        """
        assert self._embedded, 'Must call set_image() before predict()'
        assert mode in ['point', 'mask', 'both', 'box'], \
            f"Mode must be 'point', 'mask', 'both', or 'box', got {mode}"
        
        if mode == 'point':
            masks, scores, logits = self._predictor.predict(
                point_coords=prompts['point_coords'],
                point_labels=prompts['point_labels'],
                multimask_output=multimask
            )
        elif mode == 'mask':
            masks, scores, logits = self._predictor.predict(
                mask_input=prompts['mask_input'],
                multimask_output=multimask
            )
        elif mode == 'both':
            masks, scores, logits = self._predictor.predict(
                point_coords=prompts['point_coords'],
                point_labels=prompts['point_labels'],
                mask_input=prompts['mask_input'],
                multimask_output=multimask
            )
        elif mode == 'box':
            masks, scores, logits = self._predictor.predict(
                box=prompts['boxes'][None, :],
                multimask_output=multimask
            )
        
        return masks, scores, logits
