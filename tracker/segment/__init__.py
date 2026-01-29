"""
Segmentation module for player tracking.
Provides various segmenter implementations with a universal API.
"""

from tracker.segment.base_segmenter import BaseSegmenter
from tracker.segment.sam_segmenter import SamSegmenter
from tracker.segment.sam3_segmenter import Sam3Segmenter
from tracker.segment.sam2_segmenter import Sam2Segmenter
from tracker.segment.factory import create_segmenter

__all__ = [
    'BaseSegmenter',
    'SamSegmenter',
    'Sam3Segmenter',
    'Sam2Segmenter',
    'create_segmenter'
]
