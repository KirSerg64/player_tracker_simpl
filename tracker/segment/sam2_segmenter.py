"""
SAM2 segmenter placeholder for future video segmentation support.
"""

import numpy as np
from typing import Dict, Tuple, Optional

from tracker.segment.base_segmenter import BaseSegmenter
from tracker.utils.pipeline_base import PipelineMessage


class Sam2Segmenter(BaseSegmenter):
    """
    Placeholder for SAM2 segmenter.
    SAM2 will support video segmentation with temporal consistency.
    """
    
    def __init__(self, cfg, device='cuda:0', batch_size=1) -> None:
        """
        Initialize SAM2 segmenter (placeholder).
        
        Args:
            cfg: Configuration object
            device: Device to run on
            batch_size: Batch size
            
        Raises:
            NotImplementedError: SAM2 is not yet implemented
        """
        super().__init__(cfg, device, batch_size)
        print(f"Sam2Segmenter initialized on {device} (placeholder)")
        raise NotImplementedError(
            "SAM2 is not yet implemented. "
            "This is a placeholder for future video segmentation support."
        )
    
    def process(self, input: PipelineMessage) -> PipelineMessage:
        """
        Process pipeline message (not implemented).
        
        Raises:
            NotImplementedError: SAM2 is not yet implemented
        """
        raise NotImplementedError("SAM2 process method not implemented")
    
    def set_image(self, image: np.ndarray) -> None:
        """
        Set image for segmentation (not implemented).
        
        Raises:
            NotImplementedError: SAM2 is not yet implemented
        """
        raise NotImplementedError("SAM2 set_image method not implemented")
    
    def reset_image(self) -> None:
        """
        Reset image (not implemented).
        
        Raises:
            NotImplementedError: SAM2 is not yet implemented
        """
        raise NotImplementedError("SAM2 reset_image method not implemented")
    
    def predict(self, prompts: Dict, mode: str, multimask: bool = True) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Predict masks (not implemented).
        
        Raises:
            NotImplementedError: SAM2 is not yet implemented
        """
        raise NotImplementedError("SAM2 predict method not implemented")
