import argparse
from datetime import datetime
import copy
import os
from pathlib import Path

import cv2
import rich.logging
import torch
import hydra
import warnings
import logging
from tqdm import tqdm
import numpy as np
import matplotlib
from PIL import Image

from tracklab.datastruct import TrackerState
from tracklab.pipeline import Pipeline
from tracklab.utils import monkeypatch_hydra, progress, wandb
from tracklab.engine.video import VideoOnlineTrackingEngine

from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import torch.multiprocessing as mp
from tracker.gta_link.tracklet_read_write import TrackletReadWrite
# from tracker.utils.aws_utils import upload_file_to_bucket
from tracker.utils.video_reader import VideoReaderProcess
from tracker.utils.pipeline_base import MessageType, PipelineMessage, ProcessConfig, PipelineProcess
from tracker.algorithms.tracker import Tracker
from tracker.gta_link.tracklet_refiner import TrackletsRefiner
from tracker.visualization.players_drawer import EllipseDetection
from tracker.visualization.video_drawer import create_final_tracklet_video, create_overlay_video
from tracker.visualization.mask_painter import mask_painter
from tracker.utils.statistics import save_all_statistics
import gc

os.environ["HYDRA_FULL_ERROR"] = "1"
# Enable YOLO outputs for debugging
# os.environ["YOLO_VERBOSE"] = "True"
# Optimal threading for performance (not too restrictive)
# import os
# cpu_count = os.cpu_count() or 1
# os.environ["OMP_NUM_THREADS"] = str(min(4, cpu_count))     # OpenMP threads
# os.environ["MKL_NUM_THREADS"] = str(min(4, cpu_count))     # Intel MKL threads  
# os.environ["NUMEXPR_NUM_THREADS"] = str(min(2, cpu_count)) # NumExpr threads
# os.environ["OPENBLAS_NUM_THREADS"] = str(min(4, cpu_count)) # OpenBLAS threads

log = logging.getLogger(__name__)
# warnings.filterwarnings("ignore")

# Enable ultralytics logging for debugging
# import logging
# logging.basicConfig(level=logging.INFO)  # Ensure root logger is configured
# logging.getLogger('ultralytics').setLevel(logging.DEBUG)
# logging.getLogger('ultralytics.engine').setLevel(logging.DEBUG)
# logging.getLogger('ultralytics.models').setLevel(logging.DEBUG)

# # Also enable YOLO specific loggers
# for logger_name in ['yolo', 'YOLO', 'ultralytics']:
#     logger = logging.getLogger(logger_name)
#     logger.setLevel(logging.DEBUG)
#     logger.propagate = True


def get_hydra_output_dir():
    """Get the current Hydra output directory"""
    try:
        hydra_cfg = HydraConfig.get()
        return hydra_cfg.runtime.output_dir
    except:
        # Fallback to current working directory if Hydra config is not available
        return os.getcwd()


def overlay_masks(image, masks, target_size):
    image = Image.fromarray(image).convert("RGBA")
    masks = 255 * masks.astype(np.uint8)

    n_masks = masks.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in range(n_masks)
    ]

    for mask, color in zip(masks, colors):
        mask = Image.fromarray(mask)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
    return image.convert("RGB")


@hydra.main(version_base=None, config_path="pkg://tracker.configs", config_name="segment")
def main(cfg):
    if torch.cuda.is_available():
        mp.set_start_method("spawn", force=True)

    device = init_environment(cfg)

    # Validate video_path is provided
    if not cfg.video_path or cfg.video_path == "":
        log.error("video_path must be provided. Use: python video_processor.py video_path='path/to/your/video.mp4'")
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
    if cfg.save_detection_results:
        filepath = Path(output_dir) / "videos_res" / f"result.mp4"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        video_path = str(filepath)
        save_width = 1280
        save_height = int(save_width * input_height // input_width)
        video_writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(input_fps),
            (save_width, save_height),
        )
        log.info(f"Output video will be saved to: {video_path}")
        log.info(f"Output video: {save_width}x{save_height} @ {input_fps} FPS")
        # image save path
    if cfg.save_frame_results:
        img_save_path = Path(output_dir) / "seq_0" / "img1"
        img_save_path.mkdir(parents=True, exist_ok=True)
        img_save_path = str(img_save_path)
        log.info(f"Output images will be saved to: {img_save_path}")

    #create object detector (optional)
    detector = None
    if "detector" in cfg and cfg.detector is not None:
        detector = instantiate(cfg.detector, device=device, batch_size=cfg.modules.detector.batch_size)
        log.info("Using detector for predictions")
    else:
        log.info("No detector found - using text prompts for segmentation")
    #create segmenter
    segmenter = instantiate(cfg.segment, device=device, batch_size=cfg.modules.segment.batch_size)
    # create tracklet writer
    tracklet_writer = TrackletReadWrite(file_path=os.path.join(output_dir, "original_tracklets.pkl"))
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
                'frame': frame.copy(),
            },
            metadata={
                'frame_id': frames_processed
            }
        )   
        
        # Use detector if available, otherwise pass frame directly to segmenter
        if detector is not None:
            detections = detector.process(video_result)
            masks = segmenter.process(detections)
        else:
            # Use text prompt-based segmentation
            masks = segmenter.process(video_result)
        
        painted_frame = frame.copy()  # Start with original frame
        
        if masks.msg_type == MessageType.DATA:
            
            # Paint masks on the frame
            if 'masks' in masks.data:
                mask_array = masks.data['masks']  # Shape: (N, 1, H, W) or (N, H, W)
                # log.warning(f"!!!!!!!Masks: {mask_array.shape}")
                frame_height, frame_width = painted_frame.shape[:2]
                
                painted_frame = overlay_masks(
                    painted_frame,
                    mask_array,
                    (save_width, save_height)
                )
                painted_frame = np.array(painted_frame)
                painted_frame = cv2.resize(painted_frame, (save_width, save_height))
                log.warning(f"Frame shape: {painted_frame.shape}")
                # # Paint each mask with a different color
                # for i, mask in enumerate(mask_array):
                #     # Squeeze mask to 2D if needed
                #     if mask.ndim == 3:
                #         mask = mask.squeeze(0)
                    
                #     # Convert mask to uint8 format
                #     mask_uint8 = (mask > 0).astype('uint8')
                    
                #     # Resize mask to match frame dimensions if needed
                #     if mask_uint8.shape != (save_height, save_width):
                #         mask_uint8 = cv2.resize(
                #             mask_uint8, 
                #             (save_width, save_height), 
                #             interpolation=cv2.INTER_NEAREST
                #         )
                #         painted_frame = cv2.resize(
                #             painted_frame,
                #             (save_width, save_height), 
                #             interpolation=cv2.INTER_NEAREST                            
                #         )
        else:
            log.warning(f"No detections for frame {frames_processed}")

        # Save painted frame to video if enabled
        if cfg.save_detection_results and video_writer is not None:
            video_writer.write(painted_frame)

        if cfg.save_frame_results:
            image_path = f"{img_save_path}/{frames_processed:06d}.jpg"
            assert cv2.imwrite(image_path, painted_frame), f"Error saving image {image_path}"

        # cv2.imshow("Video Reader", tracklets.data['frame'])
        #     # tracker.update()
        # key = cv2.waitKey(1)
        # if key == 27:  # ESC to exit
        #     break
        frames_processed += 1

        if frames_processed >= 100:
            break
        
        # Update progress bar
        progress_bar.update(1)
        progress_bar.set_postfix({
            'FPS': f"{frames_processed / (progress_bar.format_dict['elapsed'] or 1):.1f}",
            'Frame': frames_processed
        })

    video_reader.release()
    if video_writer is not None:
        video_writer.release()
    # Close progress bar
    progress_bar.close()

    if cfg.save_original_tracklets:
        tracklet_writer.save_tracklets(tracklet_writer.get_tracklets(), tracklet_writer.file_path)
    log.info("=======Start tracklet refiner.=======")

    # cv2.destroyAllWindows()
    clear_environment()
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


def clear_environment():
    # wandb.finish()
    gc.collect()
    torch.cuda.empty_cache()
    return


if __name__ == "__main__":
    main()

