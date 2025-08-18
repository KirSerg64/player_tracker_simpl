import cv2
import numpy as np
from typing import Dict, List

from tracklab.visualization import DetectionVisualizer, get_fixed_colors
from tracklab.utils.cv2 import draw_text

  
class EllipseDetection(DetectionVisualizer):
    def __init__(self, print_id=True):
        self.print_id = print_id
        super().__init__()
        self.color_ellipse = (0, 255, 0)
        self.color_text = (0, 0, 0)

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
                color_bg=self.color_ellipse,
                color_txt=self.color_text,
                alpha_bg=1,
            )

    def draw_final_tracklets(self, image, final_tracklets: Dict, frame_id: int):
        """
        Draw final refined tracklets for a specific frame.
        
        Args:
            image: Frame image to draw on
            final_tracklets: Dictionary of final refined tracklets {track_id: Tracklet}
            frame_id: Current frame number
        """
        for track_id, tracklet in final_tracklets.items():
            # Check if this tracklet has data for the current frame
            if frame_id in tracklet.times:
                frame_idx = tracklet.times.index(frame_id)
                bbox = tracklet.bboxes[frame_idx]
                score = tracklet.scores[frame_idx]
                
                # Convert bbox from [l, t, w, h] to [x1, y1, x2, y2]
                l, t, w, h = bbox
                x1, y1, x2, y2 = l, t, l + w, t + h
                
                # Draw ellipse
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
                    thickness=3,  # Thicker for final tracklets
                    lineType=cv2.LINE_AA,
                )
                
                # Draw track ID with confidence
                draw_text(
                    image,
                    f"{int(track_id):.0f}",
                    center,
                    fontFace=1,
                    fontScale=1.2,  # Larger font for final tracklets
                    thickness=2,
                    alignH="c",
                    alignV="c",
                    color_bg=self.color_ellipse,
                    color_txt=self.color_text,
                    alpha_bg=1,
                )

    def draw_tracklet_trajectories(self, image, final_tracklets: Dict, frame_id: int, trajectory_length: int = 30):
        """
        Draw trajectory trails for tracklets showing their movement history.
        
        Args:
            image: Frame image to draw on
            final_tracklets: Dictionary of final refined tracklets
            frame_id: Current frame number
            trajectory_length: Number of previous frames to show in trajectory
        """
        for track_id, tracklet in final_tracklets.items():
            if frame_id in tracklet.times:
                current_frame_idx = tracklet.times.index(frame_id)
                
                # Get trajectory points
                trajectory_points = []
                for i in range(max(0, current_frame_idx - trajectory_length), current_frame_idx + 1):
                    if i < len(tracklet.times) and i < len(tracklet.bboxes):
                        bbox = tracklet.bboxes[i]
                        l, t, w, h = bbox
                        center_x = int(l + w / 2)
                        center_y = int(t + h)
                        trajectory_points.append((center_x, center_y))
                
                # Draw trajectory
                if len(trajectory_points) > 1:
                    trajectory_color = (255, 0, 0)  # Blue trajectory
                    for i in range(1, len(trajectory_points)):
                        # Fade effect - older points are more transparent
                        alpha = i / len(trajectory_points)
                        thickness = max(1, int(3 * alpha))
                        cv2.line(image, trajectory_points[i-1], trajectory_points[i], 
                                trajectory_color, thickness, cv2.LINE_AA)
