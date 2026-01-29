import time
import torch
import cv2
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from typing import Union
import matplotlib.pyplot as plt
import PIL
from tracker.visualization.mask_painter import mask_painter
from tracker.utils.pipeline_base import MessageType, PipelineMessage


class BaseSegmenter:
    def __init__(self, cfg, device='cuda:0', batch_size=1) -> None:
        """
        device: model device
        SAM_checkpoint: path of SAM checkpoint
        model_type: vit_b, vit_l, vit_h
        """
        print(f"Initializing BaseSegmenter to {device}")
        assert cfg.model_type in ['sam', 'mobile_sam', 'sam3'], 'model_type must be sam, mobile_sam, sam3'
        assert cfg.backbone  in ['vit_b', 'vit_l', 'vit_h', 'vit_t'], 'backbone must be vit_b, vit_l, vit_h, vit_t'

        self._device = device
        self._torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self._model_type = cfg.model_type
        self._embedded = False

        if cfg.model_type == 'mobile_sam':
            from mobile_sam import sam_model_registry, SamPredictor
            self._model = sam_model_registry[cfg.backbone](checkpoint=cfg.model_path)
            self._model.to(device=self._device)
            self._model.eval()
            self._model = torch.compile(self._model)
            self._predictor = SamPredictor(self._model)
        elif cfg.model_type == 'sam':
            from segment_anything import sam_model_registry, SamPredictor
            self._model = sam_model_registry[cfg.backbone](checkpoint=cfg.model_path)
            self._model.to(device=self._device)
            self._model.eval()
            self._model = torch.compile(self._model)
            self._predictor = SamPredictor(self._model)
        elif cfg.model_type == 'sam3':
            from transformers import Sam3Processor, Sam3Model
            # SAM3 uses HuggingFace transformers API
            self._processor = Sam3Processor.from_pretrained(cfg.model_path)
            self._model = Sam3Model.from_pretrained(
                cfg.model_path,
                torch_dtype=self._torch_dtype
            )
            self._model.to(device=self._device)
            self._model.eval()
            self._predictor = None  # SAM3 doesn't use SamPredictor
            self._current_image = None

    def process(self, input: PipelineMessage) -> PipelineMessage:
        image = input.data['frame']
        detections = input.data['detections']

        if self._model_type == 'sam3':
            masks = self._process_sam3(image, detections)
        else:
            masks = self._process_sam_legacy(image, detections)

        out_pipeline = PipelineMessage(
                msg_type=MessageType.DATA,
                data={
                    'frame': input.data['frame'],
                    "detections": np.stack(detections),
                    "masks": np.stack(masks),
                },
                metadata=input.metadata,
                timestamp=input.timestamp,
            )
        return out_pipeline

    def _process_sam_legacy(self, image: np.ndarray, detections: np.ndarray) -> list:
        """Process with SAM/Mobile-SAM models"""
        self._predictor.set_image(image)

        input_boxes = torch.tensor(detections, device=self._device)

        transformed_boxes = self._predictor.transform.apply_boxes_torch(
            input_boxes[:, :4], 
            image.shape[:2]
        )

        masks = []
        for box in transformed_boxes:
            mask, _, _ = self._predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=box.unsqueeze(0),
                multimask_output=False,
            )
            values, counts = torch.unique(mask, return_counts=True)
            value_count = [(v.item(), c.item())
                           for v, c in zip(values, counts)]
            value_count = sorted(value_count, key=lambda x: x[1], reverse=True)
            if len(value_count) > 1:
                mask[mask != 0] = value_count[0][0] if value_count[0][0] != 0 else value_count[1][0]
                masks.append(mask.squeeze(0).cpu().numpy())

        self._predictor.reset_image()
        return masks

    @torch.no_grad()
    def _process_sam3(self, image: np.ndarray, detections: np.ndarray) -> list:
        """Process with SAM3 model using HuggingFace transformers"""
        # Convert numpy image (H, W, C) to PIL Image
        pil_image = Image.fromarray(image)
        
        # Extract bounding boxes (x1, y1, x2, y2 format expected)
        input_boxes = detections[:, :4].tolist()
        
        # Prepare inputs using processor
        inputs = self._processor(
            pil_image,
            input_boxes=[input_boxes],  # List of list of boxes
            return_tensors="pt"
        ).to(self._device)
        
        # Run inference
        outputs = self._model(**inputs)
        
        # Extract masks from outputs
        # SAM3 returns masks with shape (batch, num_boxes, num_masks, H, W)
        pred_masks = outputs.pred_masks.squeeze(0)  # Remove batch dimension
        
        masks = []
        for i, mask in enumerate(pred_masks):
            # Take the first mask prediction (index 0) for single mask output
            # or select best mask based on IoU scores if available
            if mask.dim() == 3:  # (num_masks, H, W)
                mask = mask[0]  # Take first mask
            
            # Convert to binary mask and apply post-processing
            binary_mask = (mask > 0.5).cpu().numpy().astype(np.uint8)
            
            # Apply same post-processing as legacy SAM
            values, counts = np.unique(binary_mask, return_counts=True)
            value_count = [(v, c) for v, c in zip(values, counts)]
            value_count = sorted(value_count, key=lambda x: x[1], reverse=True)
            
            if len(value_count) > 1:
                binary_mask[binary_mask != 0] = value_count[0][0] if value_count[0][0] != 0 else value_count[1][0]
            
            masks.append(binary_mask)
        
        return masks

    @torch.no_grad()
    def set_image(self, image: np.ndarray):
        # PIL.open(image_path) 3channel: RGB
        # image embedding: avoid encode the same image multiple times
        self.original_image = image
        if self.embedded:
            print('repeat embedding, please reset_image.')
            return
        
        if self._model_type == 'sam3':
            # SAM3 doesn't pre-embed; just store the image
            self._current_image = image
        else:
            self.predictor.set_image(image)
        
        self.embedded = True
        return
    
    @torch.no_grad()
    def reset_image(self):
        # reset image embeding
        if self._model_type == 'sam3':
            self._current_image = None
        else:
            self.predictor.reset_image()
        self.embedded = False

    def predict(self, prompts, mode, multimask=True):
        """
        image: numpy array, h, w, 3
        prompts: dictionary, 3 keys: 'point_coords', 'point_labels', 'mask_input'
        prompts['point_coords']: numpy array [N,2]
        prompts['point_labels']: numpy array [1,N]
        prompts['mask_input']: numpy array [1,256,256]
        mode: 'point' (points only), 'mask' (mask only), 'both' (consider both)
        mask_outputs: True (return 3 masks), False (return 1 mask only)
        whem mask_outputs=True, mask_input=logits[np.argmax(scores), :, :][None, :, :]
        """
        assert self.embedded, 'prediction is called before set_image (feature embedding).'
        assert mode in ['point', 'mask', 'both'], 'mode must be point, mask, or both'
        
        if self._model_type == 'sam3':
            return self._predict_sam3(prompts, mode, multimask)
        else:
            return self._predict_sam_legacy(prompts, mode, multimask)
    
    def _predict_sam_legacy(self, prompts, mode, multimask=True):
        """Predict with legacy SAM/Mobile-SAM models"""
        if mode == 'point':
            masks, scores, logits = self.predictor.predict(point_coords=prompts['point_coords'], 
                                point_labels=prompts['point_labels'], 
                                multimask_output=multimask)
        elif mode == 'mask':
            masks, scores, logits = self.predictor.predict(mask_input=prompts['mask_input'], 
                                multimask_output=multimask)
        elif mode == 'both':   # both
            masks, scores, logits = self.predictor.predict(point_coords=prompts['point_coords'], 
                                point_labels=prompts['point_labels'], 
                                mask_input=prompts['mask_input'], 
                                multimask_output=multimask)
        else:
            raise("Not implement now!")
        # masks (n, h, w), scores (n,), logits (n, 256, 256)
        return masks, scores, logits
    
    @torch.no_grad()
    def _predict_sam3(self, prompts, mode, multimask=True):
        """Predict with SAM3 model using HuggingFace transformers"""
        assert self._current_image is not None, 'No image set for SAM3 prediction'
        
        # Convert numpy image to PIL
        pil_image = Image.fromarray(self._current_image)
        
        # Prepare inputs based on mode
        if mode == 'point':
            # SAM3 expects point prompts
            inputs = self._processor(
                pil_image,
                input_points=[prompts['point_coords'].tolist()],
                input_labels=[prompts['point_labels'].tolist()],
                return_tensors="pt"
            ).to(self._device)
        elif mode == 'mask':
            # SAM3 can accept mask inputs (converted to appropriate format)
            print("Warning: SAM3 mask-only mode not fully implemented, using point mode fallback")
            # Fallback or implement mask input handling
            return self._predict_sam3_fallback(prompts, mode, multimask)
        elif mode == 'both':
            # Combine point and mask inputs
            inputs = self._processor(
                pil_image,
                input_points=[prompts['point_coords'].tolist()],
                input_labels=[prompts['point_labels'].tolist()],
                return_tensors="pt"
            ).to(self._device)
        else:
            raise ValueError("Invalid mode")
        
        # Run inference
        outputs = self._model(**inputs)
        
        # Extract predictions
        pred_masks = outputs.pred_masks.squeeze(0).cpu().numpy()  # (num_masks, H, W)
        pred_scores = outputs.iou_scores.squeeze(0).cpu().numpy() if hasattr(outputs, 'iou_scores') else np.ones(len(pred_masks))
        
        # Convert to binary masks
        masks = (pred_masks > 0.5).astype(np.uint8)
        
        # Return format compatible with legacy SAM
        # For single mask output, select best mask
        if not multimask:
            best_idx = np.argmax(pred_scores)
            masks = masks[best_idx:best_idx+1]
            pred_scores = pred_scores[best_idx:best_idx+1]
        
        # SAM3 doesn't provide logits in the same format, return None or approximate
        logits = None
        
        return masks, pred_scores, logits
    
    def _predict_sam3_fallback(self, prompts, mode, multimask):
        """Fallback for unsupported SAM3 modes"""
        # Return empty predictions
        return np.array([]), np.array([]), None


if __name__ == "__main__":
    # load and show an image
    image = cv2.imread('/hhd3/gaoshang/truck.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # numpy array (h, w, 3)

    # initialise BaseSegmenter
    SAM_checkpoint= '/ssd1/gaomingqi/checkpoints/sam_vit_h_4b8939.pth'
    model_type = 'vit_h'
    device = "cuda:4"
    base_segmenter = BaseSegmenter(SAM_checkpoint=SAM_checkpoint, model_type=model_type, device=device)
    
    # image embedding (once embedded, multiple prompts can be applied)
    base_segmenter.set_image(image)
    
    # examples
    # point only ------------------------
    mode = 'point'
    prompts = {
        'point_coords': np.array([[500, 375], [1125, 625]]),
        'point_labels': np.array([1, 1]), 
    }
    masks, scores, logits = base_segmenter.predict(prompts, mode, multimask=False)  # masks (n, h, w), scores (n,), logits (n, 256, 256)
    painted_image = mask_painter(image, masks[np.argmax(scores)].astype('uint8'), background_alpha=0.8)
    painted_image = cv2.cvtColor(painted_image, cv2.COLOR_RGB2BGR)  # numpy array (h, w, 3)
    cv2.imwrite('/hhd3/gaoshang/truck_point.jpg', painted_image)

    # both ------------------------
    mode = 'both'
    mask_input  = logits[np.argmax(scores), :, :]
    prompts = {'mask_input': mask_input [None, :, :]}
    prompts = {
        'point_coords': np.array([[500, 375], [1125, 625]]),
        'point_labels': np.array([1, 0]), 
        'mask_input': mask_input[None, :, :]
    }
    masks, scores, logits = base_segmenter.predict(prompts, mode, multimask=True)  # masks (n, h, w), scores (n,), logits (n, 256, 256)
    painted_image = mask_painter(image, masks[np.argmax(scores)].astype('uint8'), background_alpha=0.8)
    painted_image = cv2.cvtColor(painted_image, cv2.COLOR_RGB2BGR)  # numpy array (h, w, 3)
    cv2.imwrite('/hhd3/gaoshang/truck_both.jpg', painted_image)

    # mask only ------------------------
    mode = 'mask'
    mask_input  = logits[np.argmax(scores), :, :]
    
    prompts = {'mask_input': mask_input[None, :, :]}
    
    masks, scores, logits = base_segmenter.predict(prompts, mode, multimask=True)  # masks (n, h, w), scores (n,), logits (n, 256, 256)
    painted_image = mask_painter(image, masks[np.argmax(scores)].astype('uint8'), background_alpha=0.8)
    painted_image = cv2.cvtColor(painted_image, cv2.COLOR_RGB2BGR)  # numpy array (h, w, 3)
    cv2.imwrite('/hhd3/gaoshang/truck_mask.jpg', painted_image)
