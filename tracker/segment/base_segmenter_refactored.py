"""
Base segmenter class and implementations for different SAM models.
Provides a universal API for segmentation with support for SAM, Mobile-SAM, SAM2, and SAM3.
"""

import time
import torch
import cv2
from abc import ABC, abstractmethod
from PIL import Image
import numpy as np
from typing import List, Optional, Dict, Tuple, Union
from tracker.visualization.mask_painter import mask_painter
from tracker.utils.pipeline_base import MessageType, PipelineMessage


class BaseSegmenter(ABC):
    """
    Abstract base class for all segmenter models.
    Provides a universal API for segmentation tasks.
    """
    
    def __init__(self, cfg, device='cuda:0', batch_size=1) -> None:
        """
        Initialize base segmenter.
        
        Args:
            cfg: Configuration object containing model settings
            device: Device to run model on ('cuda:0', 'cpu', etc.)
            batch_size: Batch size for processing
        """
        self._device = device
        self._torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.cfg = cfg
        self.batch_size = batch_size
        self._embedded = False
        
    @abstractmethod
    def process(self, input: PipelineMessage) -> PipelineMessage:
        """
        Main processing method for pipeline integration.
        
        Args:
            input: Pipeline message containing frame and optionally detections
            
        Returns:
            Pipeline message with segmentation masks
        """
        pass
    
    @abstractmethod
    def set_image(self, image: np.ndarray) -> None:
        """
        Set image for segmentation (pre-compute embeddings if applicable).
        
        Args:
            image: RGB image as numpy array (H, W, 3)
        """
        pass
    
    @abstractmethod
    def reset_image(self) -> None:
        """Reset/clear the current image and embeddings."""
        pass
    
    @abstractmethod
    def predict(self, prompts: Dict, mode: str, multimask: bool = True) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Predict masks based on prompts.
        
        Args:
            prompts: Dictionary with prompt data (points, boxes, masks, text)
            mode: Prompting mode ('point', 'mask', 'both', 'text', 'box')
            multimask: Whether to return multiple mask predictions
            
        Returns:
            Tuple of (masks, scores, logits)
            - masks: np.ndarray (n, h, w)
            - scores: np.ndarray (n,)
            - logits: Optional np.ndarray (n, 256, 256) or None
        """
        pass
    
    @staticmethod
    def _postprocess_mask(mask: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Post-process mask to ensure clean binary output.
        
        Args:
            mask: Input mask (torch.Tensor or np.ndarray)
            
        Returns:
            Processed binary mask as np.ndarray
        """
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
            
        # Convert to binary if needed
        if mask.dtype != np.uint8:
            mask = (mask > 0.5).astype(np.uint8)
        
        # Filter unique values and keep most common non-zero
        values, counts = np.unique(mask, return_counts=True)
        value_count = [(v, c) for v, c in zip(values, counts)]
        value_count = sorted(value_count, key=lambda x: x[1], reverse=True)
        
        if len(value_count) > 1:
            # Set all non-zero pixels to the most common non-zero value
            most_common = value_count[0][0] if value_count[0][0] != 0 else (value_count[1][0] if len(value_count) > 1 else 1)
            mask[mask != 0] = most_common
            
        return mask.astype(np.uint8)


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
        
        print(f"SAM3 text prompt: '{self.text_prompt}' (enabled: {self.use_text_prompt})")
    
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


class Sam2Segmenter(BaseSegmenter):
    """
    Placeholder for SAM2 segmenter.
    SAM2 will support video segmentation with temporal consistency.
    """
    
    def __init__(self, cfg, device='cuda:0', batch_size=1) -> None:
        super().__init__(cfg, device, batch_size)
        print(f"Sam2Segmenter initialized on {device} (placeholder)")
        raise NotImplementedError(
            "SAM2 is not yet implemented. "
            "This is a placeholder for future video segmentation support."
        )
    
    def process(self, input: PipelineMessage) -> PipelineMessage:
        raise NotImplementedError("SAM2 process method not implemented")
    
    def set_image(self, image: np.ndarray) -> None:
        raise NotImplementedError("SAM2 set_image method not implemented")
    
    def reset_image(self) -> None:
        raise NotImplementedError("SAM2 reset_image method not implemented")
    
    def predict(self, prompts: Dict, mode: str, multimask: bool = True) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        raise NotImplementedError("SAM2 predict method not implemented")


def create_segmenter(cfg, device='cuda:0', batch_size=1) -> BaseSegmenter:
    """
    Factory function to create the appropriate segmenter based on config.
    
    Args:
        cfg: Configuration object with model_type field
        device: Device to run on
        batch_size: Batch size for processing
        
    Returns:
        Instantiated segmenter object
        
    Raises:
        ValueError: If model_type is not supported
        
    Example:
        >>> cfg = Config(model_type='sam', backbone='vit_h', model_path='sam_vit_h.pth')
        >>> segmenter = create_segmenter(cfg, device='cuda:0')
    """
    model_type = cfg.model_type.lower()
    
    if model_type in ['sam', 'mobile_sam']:
        return SamSegmenter(cfg, device, batch_size)
    elif model_type == 'sam3':
        return Sam3Segmenter(cfg, device, batch_size)
    elif model_type == 'sam2':
        return Sam2Segmenter(cfg, device, batch_size)
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Supported types: 'sam', 'mobile_sam', 'sam3', 'sam2'"
        )


# Backward compatibility alias
BaseSegmenterLegacy = BaseSegmenter
