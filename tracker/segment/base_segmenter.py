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
        assert cfg.model_type in ['sam', 'mobile_sam'], 'model_type must be sam, mobile_sam'
        assert cfg.backbone  in ['vit_b', 'vit_l', 'vit_h', 'vit_t'], 'backbone must be vit_b, vit_l, vit_h, vit_t'

        if cfg.model_type == 'mobile_sam':
            from mobile_sam import sam_model_registry, SamPredictor
        else:
            from segment_anything import sam_model_registry, SamPredictor       

        self._device = device
        self._torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self._model = sam_model_registry[cfg.backbone](checkpoint=cfg.model_path)
        self._model.to(device=self._device)
        self._model.eval()
        self._model = torch.compile(self._model)
        self._predictor = SamPredictor(self._model)
        self._embedded = False

    def process(self, input: PipelineMessage) -> PipelineMessage:
        image = input.data['frame']
        detections = input.data['detections']

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

    @torch.no_grad()
    def set_image(self, image: np.ndarray):
        # PIL.open(image_path) 3channel: RGB
        # image embedding: avoid encode the same image multiple times
        self.original_image = image
        if self.embedded:
            print('repeat embedding, please reset_image.')
            return
        self.predictor.set_image(image)
        self.embedded = True
        return
    
    @torch.no_grad()
    def reset_image(self):
        # reset image embeding
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
