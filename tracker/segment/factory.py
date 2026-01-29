"""
Factory function for creating segmenter instances.
Automatically selects the appropriate segmenter class based on configuration.
"""

from tracker.segment.base_segmenter import BaseSegmenter
from tracker.segment.sam_segmenter import SamSegmenter
from tracker.segment.sam3_segmenter import Sam3Segmenter
from tracker.segment.sam2_segmenter import Sam2Segmenter


def create_segmenter(cfg, device='cuda:0', batch_size=1) -> BaseSegmenter:
    """
    Factory function to create the appropriate segmenter based on config.
    
    Args:
        cfg: Configuration object with model_type field
        device: Device to run on ('cuda:0', 'cpu', etc.)
        batch_size: Batch size for processing
        
    Returns:
        Instantiated segmenter object
        
    Raises:
        ValueError: If model_type is not supported
        
    Example:
        >>> from types import SimpleNamespace
        >>> cfg = SimpleNamespace(
        ...     model_type='sam',
        ...     backbone='vit_h',
        ...     model_path='sam_vit_h.pth'
        ... )
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
