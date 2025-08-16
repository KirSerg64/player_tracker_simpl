import cv2
import numpy as np

from tracklab.visualization import DetectionVisualizer, get_fixed_colors
from tracklab.utils.cv2 import draw_text

  
class EllipseDetection(DetectionVisualizer):
    def __init__(self, print_id=True):
        self.print_id = print_id
        super().__init__()
        self.color_ellipse = (0, 255, 0)
        self.color_text = (0, 255, 0)

    def draw_detection(self, image, detection_pred):
        for detection in detection_pred:
            x1, y1, x2, y2 = detection[:4].tolist()
            track_id = int(detection[4])
            center = (int((x1 + x2) / 2), int(y2))
            width = x2 - x1
            cv2.ellipse(
                image,
                center=center,
                axes=(int(width), int(0.35 * width)),
                angle=0.0,
                startAngle=-45.0,
                endAngle=235.0,
                color=self.color_ellipse,
                thickness=2,
                lineType=cv2.LINE_AA,
            )                
            draw_text(
                image,
                f"{track_id}",
                center,
                fontFace=1,
                fontScale=1,
                thickness=1,
                alignH="c",
                alignV="c",
                color_bg=None,
                color_txt=self.color_text,
                alpha_bg=1,
            )
