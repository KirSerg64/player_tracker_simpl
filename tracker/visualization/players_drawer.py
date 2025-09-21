"""
Player visualization module with automatic font scaling for optimal text readability.

This module provides several algorithms for calculating optimal font scales:

ALGORITHMS:
1. "proportional" - Scales proportionally to bbox size relative to image, good for consistent appearance
2. "area_based" - Uses logarithmic scaling based on bbox area percentage, handles various sizes well  
3. "adaptive" - Advanced scaling considering aspect ratio and multiple constraints, most balanced
4. "hybrid" - Combines multiple factors including absolute and relative sizing, robust for all scenarios
5. "smart" - Precise calculation ensuring text fits within bbox bounds, prevents overflow

USAGE EXAMPLES:
    # Use adaptive algorithm (recommended default)
    detector = EllipseDetection(font_algorithm="adaptive")
    
    # Use smart algorithm for precise fitting
    detector = EllipseDetection(font_algorithm="smart")
    
    # Override algorithm for specific drawing calls
    detector.draw_tracklets_with_position(image, tracklets, frame_id, 
                                        text_position='top', 
                                        font_algorithm="hybrid")

ALGORITHM CHARACTERISTICS:
- proportional: Conservative, consistent across different image sizes
- area_based: Good for varying bbox sizes, logarithmic scaling
- adaptive: Best balance of readability and aesthetics (RECOMMENDED)
- hybrid: Robust for extreme cases, handles very small/large bboxes well
- smart: Precise fitting, prevents text overflow (good for crowded scenes)
"""

import cv2
import numpy as np
from typing import Dict, List

from tracklab.visualization import DetectionVisualizer, get_fixed_colors
from tracklab.utils.cv2 import draw_text


def calculate_optimal_font_scale(bbox_width: float, bbox_height: float, img_width: int, img_height: int, 
                                text_length: int = 2, algorithm: str = "adaptive") -> float:
    """
    Calculate optimal font scale for text overlay on bounding boxes.
    
    Args:
        bbox_width: Width of bounding box
        bbox_height: Height of bounding box  
        img_width: Image width
        img_height: Image height
        text_length: Expected length of text (number of characters)
        algorithm: Algorithm to use ('adaptive', 'proportional', 'area_based', 'hybrid')
    
    Returns:
        Optimal font scale value
    """
    
    if algorithm == "proportional":
        # Algorithm 1: Proportional to bbox size with image normalization
        # Scale based on smallest bbox dimension relative to image size
        min_bbox_dim = min(bbox_width, bbox_height)
        min_img_dim = min(img_width, img_height)
        base_scale = (min_bbox_dim / min_img_dim) * 2.0
        
        # Adjust for text length
        text_factor = max(0.5, 1.0 / np.sqrt(text_length))
        font_scale = base_scale * text_factor
        
        # Clamp to reasonable bounds
        return np.clip(font_scale, 0.3, 2.0)
    
    elif algorithm == "area_based":
        # Algorithm 2: Based on bbox area percentage of image
        bbox_area = bbox_width * bbox_height
        img_area = img_width * img_height
        area_ratio = bbox_area / img_area
        
        # Logarithmic scaling for better distribution
        font_scale = 0.8 * np.log10(area_ratio * 1000 + 1)
        
        # Adjust for text length
        text_factor = max(0.6, 1.2 / text_length)
        font_scale *= text_factor
        
        return np.clip(font_scale, 0.2, 1.8)
    
    elif algorithm == "adaptive":
        # Algorithm 3: Adaptive scaling based on bbox aspect ratio and size
        aspect_ratio = bbox_width / max(bbox_height, 1)
        
        # Base scale from width (assuming text width is primary concern)
        base_scale = bbox_width / (img_width * 0.15)  # 15% of image width as reference
        
        # Adjust for aspect ratio (wider boxes can handle larger text)
        aspect_factor = np.clip(aspect_ratio / 2.0, 0.7, 1.5)
        
        # Height constraint (text shouldn't be taller than 30% of bbox height)
        height_constraint = (bbox_height * 0.3) / (img_height * 0.03)  # 3% of img height as reference
        
        # Take minimum to respect both width and height constraints
        font_scale = min(base_scale * aspect_factor, height_constraint)
        
        # Text length adjustment
        text_factor = max(0.5, 1.0 / np.sqrt(text_length * 0.8))
        font_scale *= text_factor
        
        return np.clip(font_scale, 0.25, 1.5)
    
    elif algorithm == "hybrid":
        # Algorithm 4: Hybrid approach combining multiple factors
        # Factor 1: Relative bbox size
        relative_width = bbox_width / img_width
        relative_height = bbox_height / img_height
        size_factor = np.sqrt(relative_width * relative_height) * 3.0
        
        # Factor 2: Absolute size consideration
        min_dimension = min(bbox_width, bbox_height)
        abs_factor = np.clip(min_dimension / 50.0, 0.5, 2.0)  # 50px as reference
        
        # Factor 3: Image scale consideration
        img_scale_factor = np.sqrt(img_width * img_height) / 1000.0  # 1000px as reference
        
        # Combine factors
        font_scale = (size_factor + abs_factor) * img_scale_factor * 0.4
        
        # Text length adjustment with diminishing returns
        text_factor = max(0.6, 1.1 / (text_length ** 0.6))
        font_scale *= text_factor
        
        return np.clip(font_scale, 0.2, 2.2)
    
    else:
        # Default fallback
        return np.clip(bbox_width / (img_width * 0.2), 0.3, 1.0)


def calculate_smart_font_scale(bbox_width: float, bbox_height: float, img_width: int, img_height: int,
                              text: str, min_readable_size: int = 12) -> float:
    """
    Smart font scale calculation that ensures text readability while preventing bbox overflow.
    
    Args:
        bbox_width: Width of bounding box
        bbox_height: Height of bounding box
        img_width: Image width
        img_height: Image height
        text: Actual text to be displayed
        min_readable_size: Minimum font size in pixels for readability
    
    Returns:
        Optimal font scale
    """
    text_length = len(text)
    
    # Estimate text dimensions (rough approximation for OpenCV)
    # OpenCV font typically: char_width ≈ fontScale * 10, char_height ≈ fontScale * 15
    avg_char_width = 11  # Slightly more conservative
    avg_char_height = 16
    
    # Calculate max font scale that fits within bbox width (with 15% padding)
    max_width_scale = (bbox_width * 0.85) / (text_length * avg_char_width)
    
    # Calculate max font scale that fits within bbox height (with 25% padding)
    max_height_scale = (bbox_height * 0.75) / avg_char_height
    
    # Take the smaller constraint
    max_font_scale = min(max_width_scale, max_height_scale)
    
    # Ensure minimum readability based on image resolution
    resolution_factor = np.sqrt(img_width * img_height) / 800.0  # More conservative base
    min_scale = (min_readable_size / avg_char_height) / resolution_factor
    
    # Final scale with conservative approach
    optimal_scale = max(min_scale, max_font_scale * 0.9)  # 10% safety margin
    
    return np.clip(optimal_scale, 0.15, 2.0)  # More reasonable bounds

  
class EllipseDetection(DetectionVisualizer):
    def __init__(self, print_id=True, font_algorithm="adaptive"):
        self.print_id = print_id
        self.font_algorithm = font_algorithm  # Choose font scaling algorithm
        super().__init__()
        self.color_ellipse = (0, 255, 0)
        self.color_text = (0, 0, 0)

    def draw_detection(self, image, detection_pred):
        img_height, img_width = image.shape[:2]
        for detection in detection_pred:
            x1, y1, x2, y2 = detection[:4].tolist()
            track_id = int(detection[4])
            center = (int((x1 + x2) / 2), int(y2))
            width = x2 - x1
            height = y2 - y1
            
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
            
            # Calculate optimal font scale using selected algorithm
            text = str(track_id)
            if self.font_algorithm == "smart":
                font_scale = calculate_smart_font_scale(width, height, img_width, img_height, text)
            else:
                font_scale = calculate_optimal_font_scale(
                    width, height, img_width, img_height, 
                    len(text), self.font_algorithm
                )
            
            draw_text(
                image,
                text,
                center,
                fontFace=1,
                fontScale=font_scale,
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
        img_height, img_width = image.shape[:2]
        
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
                height = y2 - y1
                                
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
                
                # Calculate optimal font scale
                text = f"{int(track_id):.0f}"
                if self.font_algorithm == "smart":
                    font_scale = calculate_smart_font_scale(width, height, img_width, img_height, text)
                else:
                    font_scale = calculate_optimal_font_scale(
                        width, height, img_width, img_height, 
                        len(text), self.font_algorithm
                    )
                
                # Draw track ID with confidence
                draw_text(
                    image,
                    text,
                    center,
                    fontFace=1,
                    fontScale=font_scale,
                    thickness=1,
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

    def draw_tracklets_with_position(self, image, final_tracklets: Dict, frame_id: int, text_position='center', color=(0, 255, 255), font_algorithm=None):
        """
        Draw final refined tracklets with configurable text position and color.
        
        Args:
            image: Frame image to draw on
            final_tracklets: Dictionary of final refined tracklets {track_id: Tracklet}
            frame_id: Current frame number
            text_position: Position of text ('top', 'bottom', 'center')
            color: Color tuple for ellipse and background
            font_algorithm: Override font algorithm for this specific call
        """
        img_height, img_width = image.shape[:2]
        selected_algorithm = font_algorithm or self.font_algorithm
        
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
                height = y2 - y1
                                
                cv2.ellipse(
                    image,
                    center=center,
                    axes=(int(width), int(0.35 * width)),
                    angle=0.0,
                    startAngle=-45.0,
                    endAngle=235.0,
                    color=color,
                    thickness=2,  # Thicker for final tracklets
                    lineType=cv2.LINE_AA,
                )
                
                # Determine text position
                if text_position == 'top':
                    text_center = (int((x1 + x2) / 2), int(y1 - 10))
                    alignV = "b"
                elif text_position == 'bottom':
                    text_center = (int((x1 + x2) / 2), int(y2 + 15))
                    alignV = "t"
                else:  # center
                    text_center = center
                    alignV = "c"
                
                # Calculate optimal font scale
                text = f"{int(track_id):.0f}"
                if selected_algorithm == "smart":
                    font_scale = calculate_smart_font_scale(width, height, img_width, img_height, text)
                else:
                    font_scale = calculate_optimal_font_scale(
                        width, height, img_width, img_height, 
                        len(text), selected_algorithm
                    )
                
                # Draw track ID
                draw_text(
                    image,
                    text,
                    text_center,
                    fontFace=1,
                    fontScale=font_scale,
                    thickness=1,
                    alignH="c",
                    alignV=alignV,
                    color_bg=color,
                    color_txt=self.color_text,
                    alpha_bg=1,
                )


def test_font_algorithms(bbox_width=100, bbox_height=150, img_width=1920, img_height=1080, text="123"):
    """
    Test function to compare different font scaling algorithms.
    
    Args:
        bbox_width: Width of test bounding box
        bbox_height: Height of test bounding box
        img_width: Image width
        img_height: Image height
        text: Test text
    
    Returns:
        Dictionary with algorithm results
    """
    algorithms = ["proportional", "area_based", "adaptive", "hybrid"]
    results = {}
    
    print(f"\nFont Scale Comparison for bbox({bbox_width}x{bbox_height}) in image({img_width}x{img_height})")
    print(f"Text: '{text}' (length: {len(text)})")
    print("-" * 60)
    
    for algo in algorithms:
        scale = calculate_optimal_font_scale(
            bbox_width, bbox_height, img_width, img_height, len(text), algo
        )
        results[algo] = scale
        print(f"{algo:12s}: {scale:.3f}")
    
    # Smart algorithm
    smart_scale = calculate_smart_font_scale(bbox_width, bbox_height, img_width, img_height, text)
    results["smart"] = smart_scale
    print(f"{'smart':12s}: {smart_scale:.3f}")
    
    return results


if __name__ == "__main__":
    # Example usage and testing
    print("Testing font scaling algorithms...")
    
    # Test with different bbox sizes
    test_cases = [
        (50, 80, 1920, 1080, "1"),      # Small bbox, short text
        (100, 150, 1920, 1080, "12"),   # Medium bbox, medium text
        (200, 300, 1920, 1080, "123"),  # Large bbox, longer text
        (80, 120, 640, 480, "45"),      # Medium bbox, lower resolution
    ]
    
    for bbox_w, bbox_h, img_w, img_h, text in test_cases:
        test_font_algorithms(bbox_w, bbox_h, img_w, img_h, text)
        print()
