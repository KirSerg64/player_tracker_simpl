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
from tracker.algorithms.utils.botsort import BotSortCust
from tracker.algorithms.utils.strongsort import StrongSortCust

log = logging.getLogger(__name__)


class Tracker():
    def __init__(self, cfg, device, batch_size, **kwargs):
        # super().__init__()
        self.cfg = cfg
        self.device = device
        self.batch_size = batch_size
        self.reid_weights = cfg.get("reid_weights", None)
        self.tracker_type = cfg.get("tracker_type", "strong_sort")
        if self.tracker_type == "strong_sort":
            self.tracker_model = StrongSortCust(
                reid_weights=Path(self.reid_weights),
                device=device,
                half=self.cfg.get("half", True),
                per_class = self.cfg.get("per_class", False),
                min_conf = self.cfg.get("min_conf", 0.1),
                max_cos_dist=self.cfg.get("max_cos_dist", 0.2),
                max_iou_dist=self.cfg.get("max_iou_dist", 0.7),
                max_age=self.cfg.get("max_age", 30),
                n_init=self.cfg.get("n_init", 3),
                nn_budget=self.cfg.get("nn_budget", 100),
                mc_lambda=self.cfg.get("mc_lambda", 0.98),
                ema_alpha=self.cfg.get("ema_alpha", 0.9),               
                with_reid=True,
            )
        elif self.tracker_type == "bot_sort":
            self.tracker_model = BotSortCust(
                reid_weights=Path(self.reid_weights),
                device=device,
                half=self.cfg.get("half", True),
                per_class = self.cfg.get("per_class", False),
                track_high_thresh = self.cfg.get("track_high_thresh", 0.5),
                track_low_thresh = self.cfg.get("track_low_thresh", 0.1),
                new_track_thresh = self.cfg.get("new_track_thresh", 0.6),
                track_buffer = self.cfg.get("track_buffer", 100),
                match_thresh = self.cfg.get("match_thresh", 0.8),
                proximity_thresh = self.cfg.get("proximity_thresh", 0.5),
                appearance_thresh = self.cfg.get("appearance_thresh", 0.25),
                cmc_method = self.cfg.get("cmc_method", "ecc"),
                frame_rate = 30,
                fuse_first_associate = self.cfg.get("fuse_first_associate", False),               
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