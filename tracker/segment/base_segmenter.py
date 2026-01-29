"""
Abstract base class for all segmenter implementations.
Defines the universal API that all segmenters must implement.
"""

from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Dict, Tuple, Optional, Union
from tracker.utils.pipeline_base import PipelineMessage


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
