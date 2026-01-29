"""
Example usage of the refactored segmentation module.
Demonstrates different segmenter types and use cases.
"""

import cv2
import numpy as np
from pathlib import Path
import torch

from tracker.segment import (
    create_segmenter,
    SamSegmenter,
    Sam3Segmenter
)
from tracker.utils.pipeline_base import MessageType, PipelineMessage


def example_sam_with_boxes():
    """Example: Using SAM with bounding box detections."""
    print("\n=== Example 1: SAM with Bounding Boxes ===")
    
    # Load image
    image_path = "path/to/your/image.jpg"
    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        # Create dummy image for demo
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    else:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create config
    class Config:
        model_type = 'mobile_sam'
        backbone = 'vit_t'
        model_path = 'pretrained_models/segmenter/mobile_sam.pt'
    
    cfg = Config()
    
    # Create segmenter using factory
    segmenter = create_segmenter(cfg, device='cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Simulate detector outputs (x1, y1, x2, y2, confidence, class)
    detections = np.array([
        [100, 100, 300, 400, 0.9, 0],  # Person 1
        [350, 150, 550, 450, 0.85, 0], # Person 2
    ], dtype=np.float32)
    
    # Create pipeline message
    pipeline_msg = PipelineMessage(
        msg_type=MessageType.DATA,
        data={
            'frame': image,
            'detections': detections
        },
        metadata={},
        timestamp=0.0
    )
    
    # Process
    result = segmenter.process(pipeline_msg)
    
    print(f"Input detections: {result.data['detections'].shape}")
    print(f"Output masks: {result.data['masks'].shape}")
    print("✓ Segmentation complete")


def example_sam3_with_text():
    """Example: Using SAM3 with text prompts (no detector needed)."""
    print("\n=== Example 2: SAM3 with Text Prompts ===")
    
    # Load image
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Create config with text prompt enabled
    class Config:
        model_type = 'sam3'
        model_path = 'facebook/sam3-vit-base'
        text_prompt = 'person'
        use_text_prompt = True  # Enable text-based segmentation
    
    cfg = Config()
    
    # Create SAM3 segmenter
    try:
        segmenter = Sam3Segmenter(cfg, device='cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Create pipeline message WITHOUT detections
        pipeline_msg = PipelineMessage(
            msg_type=MessageType.DATA,
            data={
                'frame': image,
                'detections': None  # No detector output needed!
            },
            metadata={},
            timestamp=0.0
        )
        
        # Process - SAM3 will detect persons using text prompt
        result = segmenter.process(pipeline_msg)
        
        print(f"Text prompt: '{cfg.text_prompt}'")
        print(f"Detected instances: {result.data['masks'].shape[0]}")
        print(f"Mask resolution: {result.data['masks'].shape[1:] if len(result.data['masks']) > 0 else 'N/A'}")
        print("✓ Text-based segmentation complete")
        print("Note: Each mask corresponds to one detected person instance")
        
    except ImportError:
        print("⚠ SAM3 requires transformers library: pip install transformers")
    except Exception as e:
        print(f"⚠ SAM3 not available: {e}")


def example_manual_prediction():
    """Example: Using predict() method with custom prompts."""
    print("\n=== Example 3: Manual Prediction with Prompts ===")
    
    # Create dummy image
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Create config
    class Config:
        model_type = 'mobile_sam'
        backbone = 'vit_t'
        model_path = 'pretrained_models/segmenter/mobile_sam.pt'
    
    cfg = Config()
    
    # Create segmenter
    segmenter = SamSegmenter(cfg, device='cpu')
    
    # Set image (pre-compute embeddings)
    segmenter.set_image(image)
    
    # Point prompts
    prompts = {
        'point_coords': np.array([[320, 240], [400, 300]]),  # Two points
        'point_labels': np.array([1, 1])  # Both are foreground
    }
    
    # Predict with point prompts
    masks, scores, logits = segmenter.predict(prompts, mode='point', multimask=False)
    
    print(f"Point prompts: {prompts['point_coords'].shape[0]} points")
    print(f"Output masks: {masks.shape}")
    print(f"Scores: {scores}")
    print("✓ Manual prediction complete")
    
    # Reset when done
    segmenter.reset_image()


def example_pipeline_comparison():
    """Example: Comparing detector-based vs text-based pipelines."""
    print("\n=== Example 4: Pipeline Comparison ===")
    
    print("\nStandard Pipeline (with detector):")
    print("Video → Detector → Segmenter → Tracker")
    print("         ↓           ↓")
    print("      Boxes      Boxes→Masks")
    print("Advantages: Fast, accurate bounding boxes")
    print("Use case: When you have a good object detector")
    
    print("\nText-Prompt Pipeline (no detector):")
    print("Video → Segmenter (SAM3) → Tracker")
    print("         ↓")
    print("      Text→Masks (per instance)")
    print("Advantages: No detector needed, open-vocabulary")
    print("Use case: Detector unavailable or for generic objects")
    print("\nKey insight: Text-based approach returns SEPARATE masks")
    print("for each instance, which is optimal for tracking!")


def example_factory_usage():
    """Example: Using factory function for flexible model selection."""
    print("\n=== Example 5: Factory Function Usage ===")
    
    # Different configs for different scenarios
    configs = {
        'mobile_sam': {
            'model_type': 'mobile_sam',
            'backbone': 'vit_t',
            'model_path': 'pretrained_models/segmenter/mobile_sam.pt'
        },
        'sam': {
            'model_type': 'sam',
            'backbone': 'vit_h',
            'model_path': 'pretrained_models/segmenter/sam_vit_h.pth'
        },
        'sam3': {
            'model_type': 'sam3',
            'model_path': 'facebook/sam3-vit-base',
            'text_prompt': 'person',
            'use_text_prompt': False
        }
    }
    
    for name, config_dict in configs.items():
        # Convert dict to config object
        class Config:
            pass
        
        cfg = Config()
        for key, value in config_dict.items():
            setattr(cfg, key, value)
        
        # Factory automatically selects the right class
        try:
            segmenter = create_segmenter(cfg, device='cpu')
            print(f"✓ {name}: {type(segmenter).__name__}")
        except Exception as e:
            print(f"⚠ {name}: {e}")
    
    print("\nFactory function makes it easy to switch models!")


def main():
    """Run all examples."""
    print("=" * 60)
    print("Segmentation Module Examples")
    print("=" * 60)
    
    try:
        example_factory_usage()
        example_pipeline_comparison()
        
        # These examples require actual models/data
        # Uncomment when you have the necessary files
        # example_sam_with_boxes()
        # example_sam3_with_text()
        # example_manual_prediction()
        
    except Exception as e:
        print(f"\n⚠ Error: {e}")
        print("Some examples require model files to be present.")
    
    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
