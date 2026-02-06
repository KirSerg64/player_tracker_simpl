"""
SAM3 segmenter implementation with text prompt support.
Integrates SAM3 model for open-vocabulary segmentation using streaming API.
"""

import gc
import time
import uuid
import torch
import numpy as np
from typing import Dict, Tuple, Optional

import cv2 as cv
from sam3.build.lib.sam3.model.geometry_encoders import Prompt
from sam3.sam3.model.data_misc import FindStage, convert_my_tensors
from sam3.sam3.model.utils.misc import copy_data_to_device
from sam3.sam3.model.data_misc import BatchedDatapoint

from tracker.segment.base_segmenter import BaseSegmenter
from tracker.utils.pipeline_base import MessageType, PipelineMessage
import logging

log = logging.getLogger(__name__)


class Sam3Segmenter(BaseSegmenter):
    """
    Segmenter for SAM3 model (HuggingFace transformers).
    Supports text prompts for open-vocabulary segmentation.
    """

    _ALL_INFERENCE_STATES = {}
    
    def __init__(self, cfg, device='cuda:0', batch_size=1) -> None:
        """
        Initialize SAM3 segmenter with text prompt support.
        
        Args:
            cfg: Config with model_path, and optional text_prompt
            device: Device to run on
            batch_size: Batch size
        """
        super().__init__(cfg, device, batch_size)
        
        log.info(f"Initializing Sam3Segmenter on {device}")
        
        # Import SAM3 from transformers
        try:
            from tracker.segment.sam3.model_builder_cust import build_sam3_video_model
        except ImportError:
            raise ImportError(
                "transformers library required for SAM3Segmenter. "
            )

        self.model = (
            build_sam3_video_model(
                checkpoint_path=None, #cfg.model_path,
                load_from_HF=False,
                bpe_path=cfg.get('bpe_path', None),
                has_presence_token=cfg.get('has_presence_token', True),
                geo_encoder_use_img_cross_attn=cfg.get('geo_encoder_use_img_cross_attn', True),
                strict_state_dict_loading=cfg.get('strict_state_dict_loading', True),
                apply_temporal_disambiguation=cfg.get('apply_temporal_disambiguation', False),
                device=device, 
                compile=cfg.get('compile_model', True),              
            )
            .eval()
        )
        # Text prompt configuration
        self.text_prompt = getattr(cfg, 'text_prompt', 'person')
        self.use_text_prompt = getattr(cfg, 'use_text_prompt', False)
        
        self.streaming_session = self.start_streaming_session(
            image_height=cfg.get('image_height', 720),
            image_width=cfg.get('image_width', 1280),
            text_prompt=self.text_prompt,
            session_id=str(uuid.uuid4()),
            max_expected_frames=cfg.get('max_expected_frames', 10000),
            compile_model=cfg.get('compile_model', True),
        )
        self._current_image = None        
        log.info(f"SAM3 text prompt: '{self.text_prompt}' (enabled: {self.use_text_prompt})")
    
    def start_streaming_session(
        self,
        image_height,
        image_width,
        text_prompt,
        session_id=None,
        max_expected_frames=None,
        compile_model=False,
    )->Dict:
        """
        Initialize a streaming session for online frame-by-frame processing.
        
        Args:
            image_height: Height of incoming frames
            image_width: Width of incoming frames
            text_prompt: Text description for detection (e.g., "player")
            session_id: Optional session identifier
            max_expected_frames: Estimated total frames (for memory pre-allocation)
            compile_model: Whether to compile model for faster inference
            
        Returns:
            dict with keys:
                - "session_id": str
                - "is_streaming": bool (True)
                - "frame_count": int (starts at 0)
                
        Implementation:
            1. Create a minimal inference_state without video loading
            2. Initialize empty frame buffer
            3. Add text prompt to state
            4. Prepare for incremental frame addition
        """
        
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Create minimal inference state for streaming
        device = self.model.device
        
        # Initialize empty state structure (similar to init_state but without video loading)
        inference_state = {
            "image_size": self.model.image_size,
            "num_frames": 0,  # Will increment as frames arrive
            "max_expected_frames": max_expected_frames or 10000,
            "orig_height": image_height,
            "orig_width": image_width,
            "is_streaming": True,  # Flag for streaming mode
            "constants": {},
            "tracker_inference_states": [],
            "tracker_metadata": {},
            "feature_cache": {},
            "cached_frame_outputs": {},
            "action_history": [],
            "is_image_only": False,
            "frames_buffer": [],  # Store processed frames
        }
        
        # Initialize input batch structure (will grow dynamically)
        find_text_batch = [text_prompt if text_prompt else "<text placeholder>", "visual"]
        inference_state["input_batch"] = BatchedDatapoint(
            img_batch=[],  # Empty, will add frames dynamically
            find_text_batch=find_text_batch,
            find_inputs=[],  # Empty, will add as frames come
            find_targets=[],
            find_metadatas=[],
        )
        
        # Initialize empty geometric prompt
        bs = 1  # batch size      
        inference_state["constants"]["empty_geometric_prompt"] = Prompt(
            box_embeddings=torch.zeros(0, bs, 4, device=device),
            box_mask=torch.zeros(bs, 0, device=device, dtype=torch.bool),
            box_labels=torch.zeros(0, bs, device=device, dtype=torch.long),
            point_embeddings=torch.zeros(0, bs, 2, device=device),
            point_mask=torch.zeros(bs, 0, device=device, dtype=torch.bool),
            point_labels=torch.zeros(0, bs, device=device, dtype=torch.long),
        )
        
        # Initialize per-frame tracking structures
        inference_state["previous_stages_out"] = []
        inference_state["text_prompt"] = text_prompt
        inference_state["per_frame_raw_point_input"] = []
        inference_state["per_frame_raw_box_input"] = []
        inference_state["per_frame_visual_prompt"] = []
        inference_state["per_frame_geometric_prompt"] = []
        inference_state["per_frame_cur_step"] = []
        
        # Placeholders for visual prompts
        inference_state["visual_prompt_embed"] = None
        inference_state["visual_prompt_mask"] = None
        
        # Store session
        self._ALL_INFERENCE_STATES[session_id] = {
            "state": inference_state,
            "session_id": session_id,
            "start_time": time.time(),
            "text_prompt": text_prompt,
            "is_streaming": True,
            "frame_count": 0,
        }
        
        # Optional model compilation
        if compile_model:
            self.model._compile_model()
        
        log.info(
            f"Started streaming session {session_id} with text prompt '{text_prompt}' "
            f"for {image_height}x{image_width} frames"
        )        
        return {
            "session_id": session_id,
            "is_streaming": True,
            "frame_count": 0,
            "video_height": image_height,
            "video_width": image_width,
        }

    def close_session(self, session_id):
        """
        Close a session. This method is idempotent and can be called multiple
        times on the same "session_id".
        """
        session = self._ALL_INFERENCE_STATES.pop(session_id, None)
        if session is None:
            pass
            # log.warning(
            #     f"cannot close session {session_id} as it does not exist (it might have expired); "
            #     f"{self._get_session_stats()}"
            # )
        else:
            del session
            gc.collect()
            # log.info(f"removed session {session_id}; {self._get_session_stats()}")
        return {"is_success": True}

    @torch.inference_mode()
    def _process_frame(
        self,
        session_id,
        frame,
        frame_idx=None,
        return_masks=True,
        return_boxes=True,
    ):
        """
        Process a single incoming frame in streaming mode.
        
        Args:
            session_id: Streaming session identifier
            frame: numpy array (H, W, 3) in BGR or RGB format
            frame_idx: Optional frame index (auto-incremented if None)
            return_masks: Whether to return binary masks
            return_boxes: Whether to return bounding boxes
            
        Returns:
            dict with keys:
                - "frame_index": int
                - "outputs": {
                    "out_obj_ids": np.ndarray,
                    "out_probs": np.ndarray,
                    "out_boxes_xywh": np.ndarray (if return_boxes=True),
                    "out_binary_masks": np.ndarray (if return_masks=True),
                }
                - "frame_count": int (total frames processed so far)
                
        Implementation:
            1. Preprocess and add frame to inference_state
            2. Run detection + tracking on this frame
            3. Update tracking memory
            4. Return results immediately
        """
        session = self._get_session(session_id)
        inference_state = session["state"]
        
        if not inference_state.get("is_streaming", False):
            raise RuntimeError(
                f"Session {session_id} is not a streaming session. "
                "Use start_streaming_session() instead of start_session()"
            )
        
        # Auto-increment frame index if not provided
        if frame_idx is None:
            frame_idx = session["frame_count"]
        
        # # Validate frame
        # if frame.shape[0] != inference_state["orig_height"] or \
        # frame.shape[1] != inference_state["orig_width"]:
        #     raise ValueError(
        #         f"Frame size mismatch: expected {inference_state['orig_height']}x"
        #         f"{inference_state['orig_width']}, got {frame.shape[0]}x{frame.shape[1]}"
        #     )
        
        # Add frame to inference state
        self._add_frame_to_state(inference_state, frame, frame_idx)
        
        # Run detection + tracking on this frame
        out = self.model._run_single_frame_inference(
            inference_state=inference_state,
            frame_idx=frame_idx,
            reverse=False,
        )
        
        # Post-process outputs
        if self.model.rank == 0:
            suppressed_obj_ids = out.get("suppressed_obj_ids", set())
            postprocessed_out = self.model._postprocess_output(
                inference_state=inference_state,
                out=out,
                suppressed_obj_ids=suppressed_obj_ids,
            )
            
            # Filter outputs based on return flags
            if not return_masks:
                postprocessed_out.pop("out_binary_masks", None)
            if not return_boxes:
                postprocessed_out.pop("out_boxes_xywh", None)
        else:
            postprocessed_out = None
        
        # Increment frame counter
        session["frame_count"] = frame_idx + 1
        inference_state["num_frames"] = session["frame_count"]
        
        log.debug(
            f"Processed frame {frame_idx} in streaming session {session_id} "
            f"(total: {session['frame_count']} frames)"
        )
        
        return {
            "frame_index": frame_idx,
            "outputs": postprocessed_out,
            "frame_count": session["frame_count"],
        }


    def _add_frame_to_state(self, inference_state, frame, frame_idx):
        """
        Add a new frame to the inference state for streaming mode.
        
        Args:
            inference_state: The inference state dict
            frame: numpy array (H, W, 3)
            frame_idx: Frame index
            
        Implementation:
            1. Preprocess frame (resize, normalize)
            2. Add to img_batch
            3. Create FindStage for this frame
            4. Update per-frame tracking lists
        """
        
        device = self.model.device
        image_size = inference_state["image_size"]
        
        # Preprocess frame (resize and normalize)
        # Convert BGR to RGB if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
        
        # Resize to model input size
        resized = cv.resize(frame_rgb, (image_size, image_size))
        
        # Normalize using model's mean and std
        img_mean = np.array(self.model.image_mean).reshape(1, 1, 3)
        img_std = np.array(self.model.image_std).reshape(1, 1, 3)
        normalized = (resized.astype(np.float32) / 255.0 - img_mean) / img_std
        
        # Convert to tensor (C, H, W)
        img_tensor = torch.from_numpy(normalized).permute(2, 0, 1).to(device)
        
        # Add to img_batch
        inference_state["input_batch"].img_batch.append(img_tensor)
        
        # Create FindStage for this frame
        input_box_embedding_dim = 258
        input_points_embedding_dim = 257
        text_id = 0 if inference_state["text_prompt"] else 1  # 0 for text, 1 for visual
        
        stage = FindStage(
            img_ids=[frame_idx],
            text_ids=[text_id],
            input_boxes=[torch.zeros(input_box_embedding_dim)],
            input_boxes_mask=[torch.empty(0, dtype=torch.bool)],
            input_boxes_label=[torch.empty(0, dtype=torch.long)],
            input_points=[torch.empty(0, input_points_embedding_dim)],
            input_points_mask=[torch.empty(0)],
            object_ids=[],
        )
        stage = convert_my_tensors(stage)
        
        # Move to device
        stage = copy_data_to_device(stage, device, non_blocking=True)
        
        # Add to find_inputs
        inference_state["input_batch"].find_inputs.append(stage)
        inference_state["input_batch"].find_targets.append(None)
        inference_state["input_batch"].find_metadatas.append(None)
        
        # Extend per-frame tracking lists
        inference_state["previous_stages_out"].append(None)
        inference_state["per_frame_raw_point_input"].append(None)
        inference_state["per_frame_raw_box_input"].append(None)
        inference_state["per_frame_visual_prompt"].append(None)
        inference_state["per_frame_geometric_prompt"].append(None)
        inference_state["per_frame_cur_step"].append(0)        
        # Store original frame for visualization (optional)
        inference_state["frames_buffer"].append(frame)


    def _get_session(self, session_id):
        session = self._ALL_INFERENCE_STATES.get(session_id, None)
        if session is None:
            raise RuntimeError(
                f"Cannot find session {session_id}; it might have expired"
            )
        return session

    @torch.inference_mode()
    def process(self, input: PipelineMessage) -> PipelineMessage:
        """
        Process frame using streaming session API.
        
        Args:
            input: Pipeline message with 'frame' and optionally 'detections'
            
        Returns:
            Pipeline message with masks, boxes, scores from SAM3
        """
        image = input.data['frame']
        session_id = self.streaming_session['session_id']
        
        # Process frame through streaming session
        result = self._process_frame(
            session_id=session_id,
            frame=image,
            frame_idx=input.metadata.get('frame_id', None),
            return_masks=True,
            return_boxes=True,
        )
        
        outputs = result.get('outputs')
        if outputs is None:
            # No detections found
            masks = np.array([])
            boxes = np.array([])
            scores = np.array([])
            obj_ids = np.array([])
        else:
            # Extract outputs from SAM3
            masks = outputs.get('out_binary_masks', np.array([]))  # (N, H, W) bool
            boxes = outputs.get('out_boxes_xywh', np.array([]))    # (N, 4) normalized
            scores = outputs.get('out_probs', np.array([]))        # (N,)
            obj_ids = outputs.get('out_obj_ids', np.array([]))     # (N,)
        
        out_pipeline = PipelineMessage(
            msg_type=MessageType.DATA,
            data={
                'frame': image,
                'masks': masks,
                'boxes': boxes,
                'scores': scores,
                'obj_ids': obj_ids,
            },
            metadata=input.metadata,
            timestamp=input.timestamp,
        )
        return out_pipeline
    

    def denormalize_boxes(self, boxes_xywh: np.ndarray, image_height: int, image_width: int) -> np.ndarray:
        """
        Convert normalized boxes (0-1) to pixel coordinates.
        
        Args:
            boxes_xywh: Normalized boxes (N, 4) in [x, y, w, h] format
            image_height: Image height in pixels
            image_width: Image width in pixels
            
        Returns:
            Boxes in pixel coordinates (N, 4) [x1, y1, x2, y2]
        """
        if len(boxes_xywh) == 0:
            return np.array([])
        
        boxes_pixel = boxes_xywh.copy()
        boxes_pixel[:, 0] *= image_width   # x
        boxes_pixel[:, 1] *= image_height  # y
        boxes_pixel[:, 2] *= image_width   # w
        boxes_pixel[:, 3] *= image_height  # h
        
        # Convert from xywh to x1y1x2y2
        x1 = boxes_pixel[:, 0]
        y1 = boxes_pixel[:, 1]
        x2 = x1 + boxes_pixel[:, 2]
        y2 = y1 + boxes_pixel[:, 3]
        
        return np.stack([x1, y1, x2, y2], axis=1)
    
    def set_image(self, image: np.ndarray) -> None:
        """
        Store image for processing.
        Note: SAM3 streaming mode processes frames on-demand.
        """
        self._current_image = image
        self._embedded = True
    
    def reset_image(self) -> None:
        """Reset stored image."""
        self._current_image = None
        self._embedded = False
    
    @torch.no_grad()
    def predict(self, prompts: Dict, mode: str, multimask: bool = True) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Predict masks with text prompts using the streaming session.
        
        Args:
            prompts: Dictionary with:
                - 'text' for text mode (e.g., {'text': 'player'})
            mode: Only 'text' mode is supported for SAM3 streaming
            multimask: Return multiple masks (always True for SAM3)
            
        Returns:
            (masks, scores, logits) - logits is None for SAM3
            
        Note: Point and box modes are not implemented in streaming mode.
              Use process() method with detections for box-based segmentation.
        """
        assert self._embedded, 'Must call set_image() before predict()'
        
        if mode != 'text':
            raise NotImplementedError(
                f"SAM3 streaming mode only supports 'text' prompts. "
                f"For box prompts, pass detections to process() method."
            )
        
        # Process current image with text prompt
        session_id = self.streaming_session['session_id']
        result = self._process_frame(
            session_id=session_id,
            frame=self._current_image,
            return_masks=True,
            return_boxes=False,
        )
        
        outputs = result.get('outputs')
        if outputs is None or len(outputs.get('out_binary_masks', [])) == 0:
            return np.array([]), np.array([]), None
        
        masks = outputs['out_binary_masks']  # (N, H, W) bool
        scores = outputs['out_probs']         # (N,)
        
        # Convert bool masks to uint8
        masks = masks.astype(np.uint8)
        
        return masks, scores, None  # No logits for SAM3
