import logging
from typing import Any, Dict
from xml.parsers.expat import model

import torch
import pandas as pd
import supervision as sv
import numpy as np

from ultralytics import YOLO

from tracklab.pipeline.imagelevel_module import ImageLevelModule
from tracklab.utils.coordinates import sanitize_bbox_ltrb
from tracker.utils.pipeline_base import MessageType, ProcessConfig, PipelineMessage
from boxmot import BotSort



log = logging.getLogger(__name__)


def collate_fn(batch):
    idxs = [b[0] for b in batch]
    images = [b["image"] for _, b in batch]
    shapes = [b["shape"] for _, b in batch]
    return idxs, (images, shapes)


class YOLOOnnx(ImageLevelModule):
    collate_fn = collate_fn
    input_columns = []
    output_columns = [
        "image_id",
        "video_id",
        "category_id",
        "bbox_ltwh",
        "bbox_conf",
    ]

    def __init__(self, cfg, device, batch_size, **kwargs):
        super().__init__(batch_size)
        self.cfg = cfg
        self.device = device
        self.model = YOLO(cfg.path_to_checkpoint, task="detect", verbose=False)
        # self.model.to(device)
        self.id = 0
        self.class_map = {cls: id for id, cls in enumerate(cfg.classes)}
        self.classes_to_detect = tuple([
            self.class_map['player'], 
            self.class_map['goalkeeper'], 
            self.class_map['referee'],
        ])
        self.use_slicer = cfg.get("use_slicer", False)
        self.slicer = sv.InferenceSlicer(
            callback=self.callback, thread_workers=4,
        )

    # @torch.no_grad()
    def callback(self, image_slice) -> sv.Detections:
        results = self.model.predict(image_slice, device=self.device)[0]
        return sv.Detections.from_ultralytics(results)

    # @torch.no_grad()
    def preprocess(self, image, detections, metadata: pd.Series):
        return {
            "image": image,
            "shape": (image.shape[1], image.shape[0]),
        }

    # @torch.no_grad()
    def process(self, input: PipelineMessage) -> PipelineMessage:
        images = [input.data['frame']]
        shapes = [img.shape for img in images]
        if self.use_slicer:
            results_by_image = []
            for img in images:
                sliced_results = self.slicer(img)
                results_by_image.append(sliced_results)
        else:
            results_by_image = self.model.predict(images, device=self.device)
            results_by_image = [sv.Detections.from_ultralytics(res) for res in results_by_image]
        detections = []
        for results, shape in zip(results_by_image, shapes):
            for xyxy, _, conf, class_id, _, _ in results:
                # check for `player` and 'goalkeeper' class
                if conf >= self.cfg.min_confidence and class_id in self.classes_to_detect:
                    detections.append(
                        np.concatenate([sanitize_bbox_ltrb(xyxy, (shape[1], shape[0])), [class_id, conf]], axis=0)
                    )
                    self.id += 1

        out_pipeline = PipelineMessage(
            msg_type=MessageType.DATA,
            data={
                'frame': input.data['frame'],
                "detections": detections
            },
            metadata=input.metadata,
            timestamp=input.timestamp,
        )
        return out_pipeline