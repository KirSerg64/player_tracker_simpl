import argparse
import os
from pathlib import Path

import cv2
import rich.logging
import torch
import hydra
import warnings
import logging

from tracklab.datastruct import TrackerState
from tracklab.pipeline import Pipeline
from tracklab.utils import monkeypatch_hydra, progress, wandb
from tracklab.engine.video import VideoOnlineTrackingEngine

from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch.multiprocessing as mp
from tracker.utils.video_reader import VideoReaderProcess
from tracker.utils.pipeline_base import MessageType, PipelineMessage, ProcessConfig, PipelineProcess
from tracker.algorithms.tracker import Tracker
from tracker.visualization.players_drawer import EllipseDetection

os.environ["HYDRA_FULL_ERROR"] = "1"
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


@hydra.main(version_base=None, config_path="pkg://tracker.configs", config_name="main")
def main(cfg):
    if torch.cuda.is_available():
        mp.set_start_method("spawn", force=True)

    device = init_environment(cfg)

    log.info("Start processing")
    # Create video reader process
    # video_config = ProcessConfig(
    #     name="video_reader",
    #     input_queue_size=1,  # No input needed
    #     output_queue_size=cfg.video_buffer_size
    # )
    # video_reader = VideoReaderProcess(video_config, cfg.video_path, cfg.target_fps)
    video_reader = cv2.VideoCapture(cfg.video_path)
    if not video_reader.isOpened():
        log.error(f"Failed to open video file: {cfg.video_path}")
        return 1

    #create object detector
    detector = instantiate(cfg.detector, device=device, batch_size=cfg.modules.detector.batch_size)
    #create feature extractor
    feature_extractor = instantiate(cfg.reid, device=device, batch_size=cfg.modules.feature_extractor.batch_size) 
    #create tracker
    tracker = instantiate(cfg.tracker, device=device, batch_size=cfg.modules.tracker.batch_size)
    #create visualizer
    visualizer = EllipseDetection()

    frames_processed = 0

    while True:
        ret, frame = video_reader.read()
        if not ret:
            break
        video_result = PipelineMessage(
            msg_type=MessageType.DATA,
            data={
                'frame': frame,
            },
            metadata={
                'frame_id': frames_processed
            }
        )   
        detections = detector.process(video_result)
        features = feature_extractor.process(detections)
        tracklets = tracker.process(features)
        visualizer.draw_detection(tracklets.data['frame'], tracklets.data['tracklets'])
        cv2.imshow("Video Reader", tracklets.data['frame'])
            # tracker.update()
        key = cv2.waitKey(1)
        if key == 27:  # ESC to exit
            break
        frames_processed += 1

    video_reader.release()
    cv2.destroyAllWindows()
    return 0


# def set_sharing_strategy():
#     torch.multiprocessing.set_sharing_strategy(
#         "file_system"
#     )


def init_environment(cfg):
    # set_sharing_strategy()  
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    return device


def close_environment():
    # wandb.finish()
    return


if __name__ == "__main__":
    main()

