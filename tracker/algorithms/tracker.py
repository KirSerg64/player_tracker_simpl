import logging
from pathlib import Path
from typing import Any, Dict
from xml.parsers.expat import model

import torch
import pandas as pd
import supervision as sv
import numpy as np
# from tracklab.pipeline.imagelevel_module import ImageLevelModule
from tracklab.utils.coordinates import sanitize_bbox_ltrb
from tracker.utils.pipeline_base import MessageType, ProcessConfig, PipelineMessage
from boxmot import BotSort

log = logging.getLogger(__name__)


class Tracker():
    def __init__(self, cfg, device, batch_size, **kwargs):
        # super().__init__()
        self.cfg = cfg
        self.device = device
        self.batch_size = batch_size
        self.reid_weights = cfg.get("reid_weights", None)
        self.tracker_model = BotSort(
            reid_weights=Path(self.reid_weights),
            device=device,
            half=True,
            with_reid=True,
        )

    def process(self, input: PipelineMessage) -> PipelineMessage:
        images = [input.data['frame']]
        detections = input.data['detections']
        features = input.data['features']
        tracklets = self.tracker_model.update(detections, images[0], features)
        out_pipeline = PipelineMessage(
            msg_type=MessageType.DATA,
            data={
                'frame': input.data['frame'],
                "detections": detections,
                "features": features,
                "tracklets": tracklets,
            },
            metadata=input.metadata,
            timestamp=input.timestamp,
        )
        return out_pipeline