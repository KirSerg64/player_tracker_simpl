import argparse
import os
from pathlib import Path

import cv2
import rich.logging
import torch
import hydra
import warnings
import logging
from tqdm import tqdm

from tracklab.datastruct import TrackerState
from tracklab.pipeline import Pipeline
from tracklab.utils import monkeypatch_hydra, progress, wandb
from tracklab.engine.video import VideoOnlineTrackingEngine

from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import torch.multiprocessing as mp
from tracker.utils.video_reader import VideoReaderProcess
from tracker.utils.pipeline_base import MessageType, PipelineMessage, ProcessConfig, PipelineProcess
from tracker.algorithms.tracker import Tracker
from tracker.visualization.players_drawer import EllipseDetection

os.environ["HYDRA_FULL_ERROR"] = "1"
# Suppress YOLO outputs globally
# os.environ["YOLO_VERBOSE"] = "False"
# Optimal threading for performance (not too restrictive)
import os
cpu_count = os.cpu_count() or 1
os.environ["OMP_NUM_THREADS"] = str(min(4, cpu_count))     # OpenMP threads
os.environ["MKL_NUM_THREADS"] = str(min(4, cpu_count))     # Intel MKL threads  
os.environ["NUMEXPR_NUM_THREADS"] = str(min(2, cpu_count)) # NumExpr threads
os.environ["OPENBLAS_NUM_THREADS"] = str(min(4, cpu_count)) # OpenBLAS threads

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# # Additional ultralytics silence
# import logging
# logging.getLogger('ultralytics').setLevel(logging.ERROR)


def get_hydra_output_dir():
    """Get the current Hydra output directory"""
    try:
        hydra_cfg = HydraConfig.get()
        return hydra_cfg.runtime.output_dir
    except:
        # Fallback to current working directory if Hydra config is not available
        return os.getcwd()


@hydra.main(version_base=None, config_path="pkg://tracker.configs", config_name="main")
def main(cfg):
    if torch.cuda.is_available():
        mp.set_start_method("spawn", force=True)

    device = init_environment(cfg)

    # Validate video_path is provided
    if not cfg.video_path or cfg.video_path == "":
        log.error("video_path must be provided. Use: python main.py video_path='path/to/your/video.mp4'")
        return 1
    
    if not os.path.exists(cfg.video_path):
        log.error(f"Video file not found: {cfg.video_path}")
        return 1

    log.info("Start processing")
    log.info(f"Video path: {cfg.video_path}")
    
    # Get Hydra output directory
    output_dir = get_hydra_output_dir()
    log.info(f"Output directory: {output_dir}")
    
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
    
    # Get total frame count for progress bar
    total_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get input video dimensions
    input_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = video_reader.get(cv2.CAP_PROP_FPS)
    
    log.info(f"Total frames to process: {total_frames}")
    log.info(f"Input video: {input_width}x{input_height} @ {input_fps} FPS")
    
    # result video writer
    video_writer = None
    if cfg.save_results:
        filepath = Path(output_dir) / "videos_res" / f"result.mp4"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        video_path = str(filepath)
        video_writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(input_fps),
            (input_width, input_height),
        )
        log.info(f"Output video will be saved to: {video_path}")
        log.info(f"Output video: {input_width}x{input_height} @ {input_fps} FPS")

    #create object detector
    detector = instantiate(cfg.detector, device=device, batch_size=cfg.modules.detector.batch_size)
    #create feature extractor
    feature_extractor = instantiate(cfg.reid, device=device, batch_size=cfg.modules.feature_extractor.batch_size) 
    #create tracker
    tracker = instantiate(cfg.tracker, device=device, batch_size=cfg.modules.tracker.batch_size)
    #create visualizer
    visualizer = EllipseDetection()

    frames_processed = 0

    # Initialize progress bar
    progress_bar = tqdm(
        total=total_frames,
        desc="Processing frames",
        unit="frames",
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )

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

        if cfg.save_results and video_writer is not None:
            # Draw frame using visualizers
            visualizer.draw_detection(tracklets.data['frame'], tracklets.data['tracklets'])
            # Write to video if required       
            video_writer.write(tracklets.data['frame'])

        # cv2.imshow("Video Reader", tracklets.data['frame'])
        #     # tracker.update()
        # key = cv2.waitKey(1)
        # if key == 27:  # ESC to exit
        #     break
        frames_processed += 1
        
        # Update progress bar
        progress_bar.update(1)
        progress_bar.set_postfix({
            'FPS': f"{frames_processed / (progress_bar.format_dict['elapsed'] or 1):.1f}",
            'Frame': frames_processed
        })

    # Close progress bar
    progress_bar.close()
    
    video_reader.release()
    if video_writer is not None:
        video_writer.release()
    # cv2.destroyAllWindows()
    
    log.info(f"Processing completed! Processed {frames_processed} frames")
    return 0


# def set_sharing_strategy():
#     torch.multiprocessing.set_sharing_strategy(
#         "file_system"
#     )


def init_environment(cfg):
    # Optimal PyTorch threading for performance
    import os
    cpu_count = os.cpu_count() or 1
    torch.set_num_threads(min(4, cpu_count))  # Use up to 4 threads
    
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

