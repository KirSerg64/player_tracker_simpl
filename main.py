import argparse
from datetime import datetime
from multiprocessing.sharedctypes import copy
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
from tracker.gta_link.tracklet_read_write import TrackletReadWrite
from tracker.utils.aws_utils import load_file_to_bucket
from tracker.utils.video_reader import VideoReaderProcess
from tracker.utils.pipeline_base import MessageType, PipelineMessage, ProcessConfig, PipelineProcess
from tracker.algorithms.tracker import Tracker
from tracker.gta_link.tracklet_refiner import TrackletsRefiner
from tracker.visualization.players_drawer import EllipseDetection
from tracker.visualization.video_creator import create_final_tracklet_video, create_overlay_video
from tracker.utils.statistics import save_all_statistics

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
        # image save path
        img_save_path = Path(output_dir) / "seq_0" / "img1"
        img_save_path.mkdir(parents=True, exist_ok=True)
        img_save_path = str(img_save_path)
        log.info(f"Output images will be saved to: {img_save_path}")

    #create object detector
    detector = instantiate(cfg.detector, device=device, batch_size=cfg.modules.detector.batch_size)
    #create feature extractor
    feature_extractor = instantiate(cfg.reid, device=device, batch_size=cfg.modules.feature_extractor.batch_size) 
    #create tracker
    tracker = instantiate(cfg.tracker, device=device, batch_size=cfg.modules.tracker.batch_size)
    #create tracklet refiner
    tracklet_refiner = instantiate(cfg.gta_link, device=device, batch_size=cfg.modules.refiner.batch_size)
    # create tracklet writer
    tracklet_writer = TrackletReadWrite(file_path=os.path.join(output_dir, "tracklets.pkl"))
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
        detections = detector.process(video_result)
        if detections.msg_type == MessageType.DATA:
            features = feature_extractor.process(detections)
            tracklets = tracker.process(features)
            tracklet_writer.add_tracklet(tracklets)

            if cfg.save_results and video_writer is not None:
                # Draw frame using visualizers
                visualizer.draw_detection(tracklets.data['frame'], tracklets.data['tracklets'])
                # Write to video if required       
                video_writer.write(tracklets.data['frame'])
        else:
            log.warning(f"No detections for frame {frames_processed}")

        if cfg.save_results:
            image_path = f"{img_save_path}/{frames_processed:06d}.jpg"
            assert cv2.imwrite(image_path, frame), f"Error saving image {image_path}"

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

    video_reader.release()
    if video_writer is not None:
        video_writer.release()
    # Close progress bar
    progress_bar.close()

    # tracklet_writer.save_tracklets(tracklet_writer.get_tracklets(), tracklet_writer.file_path)
    log.info("Start tracklet refiner.")

    real_time_tracklets = copy.deepcopy(tracklet_writer.get_tracklets())
    final_tracklets = tracklet_refiner._refine_tracklets(tracklet_writer.get_tracklets())

    # Finalize tracklet refinement and get final results
    # log.info("Finalizing tracklet refinement...")
    # final_tracklets = tracklet_refiner.finalize_and_get_results()
    # log.info(f"Final tracklet refinement completed: {len(final_tracklets)} final tracklets")
    
    # Create final video with refined tracklets
    if cfg.save_results and final_tracklets:
        _, video_name = os.path.split(cfg.video_path)
        video_name = os.path.splitext(video_name)[0]
        final_tracklets_save_path = os.path.join(output_dir, video_name + ".pkl")
        log.info(f"Final tracklets {len(final_tracklets)} saved to: {final_tracklets_save_path}")
        tracklet_writer.save_tracklets(final_tracklets, final_tracklets_save_path)

        log.info("Creating final video with refined tracklets...")      

        # Get video properties for statistics
        cap = cv2.VideoCapture(cfg.video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 25.0
        cap.release()        
        # Create final video with refined tracklets
        refined_video_path = os.path.join(output_dir, f"{video_name}_refined.mp4")
        if cfg.overlay_video:
            create_overlay_video(
                video_path=cfg.video_path,
                final_tracklets=final_tracklets,
                real_time_tracklets=real_time_tracklets,
                output_path=refined_video_path,
            )
        else:
            create_final_tracklet_video(
                video_path=cfg.video_path,
                final_tracklets=final_tracklets,
                output_path=refined_video_path,
                show_trajectories=False
            )
        # Save comprehensive statistics in all formats
        save_all_statistics(
            final_tracklets=final_tracklets,
            output_dir=output_dir,
            video_fps=video_fps
        )
        #save to AWS
        # if cfg.save_to_aws:
        #     date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        #     save_name = f"full_result_video_{date_time}.mp4"
        #     load_file_to_bucket(refined_video_path, save_name)

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

