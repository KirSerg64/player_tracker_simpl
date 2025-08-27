import logging
import os
import pandas as pd
import supervision as sv
import numpy as np

from ultralytics import YOLO

from tracklab.pipeline.imagelevel_module import ImageLevelModule
from tracklab.utils.coordinates import sanitize_bbox_ltrb
from tracker.utils.pipeline_base import MessageType, ProcessConfig, PipelineMessage


log = logging.getLogger(__name__)

class YOLOInference(ImageLevelModule):
 
    input_columns = []
    output_columns = []

    def __init__(self, cfg, device, batch_size, **kwargs):
        super().__init__(batch_size)
        self.cfg = cfg
        self.device = device
        
        self.model = YOLO(cfg.path_to_checkpoint, task="detect")
        
        # self.model.to(device)
        self.id = 0
        self.class_map = {cls: id for id, cls in enumerate(cfg.classes)}
        self.classes_to_detect = tuple([
            self.class_map['player'], 
            self.class_map['goalkeeper'], 
            self.class_map['referee'],
        ])
        self.use_slicer = cfg.get("use_slicer", False)
        if self.use_slicer:
            self.slicer = sv.InferenceSlicer(
                callback=self.callback, thread_workers=4,  # Reduced from 4 to 1
            )

    # @torch.no_grad()
    def callback(self, image_slice) -> sv.Detections:
        results = self.model.predict(
            image_slice, 
            # device=self.device,
            verbose=False,
            show=False,
            save=False
        )[0]
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
            results_by_image = self.model.predict(
                images, 
                device=self.device,
                show=False,
                save=False,
                save_txt=False,
                save_conf=False,
                save_crop=False,
                stream=False
            )
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

        if detections:
            out_pipeline = PipelineMessage(
                msg_type=MessageType.DATA,
                data={
                    'frame': input.data['frame'],
                    "detections": detections
                },
                metadata=input.metadata,
                timestamp=input.timestamp,
            )
        else:
            out_pipeline = PipelineMessage(
                msg_type=MessageType.NO_DATA,
                data={
                    'frame': input.data['frame'],
                    "detections": []
                },
                metadata=input.metadata,
                timestamp=input.timestamp,
            )
        return out_pipeline