"""
Export DINOv3 model from HuggingFace to ONNX format.

This script exports Meta's DINOv3 vision transformer to ONNX for optimized inference.
The exported model is compatible with the FeatureExtractorDINOv3 class.

Features:
- Dynamic batch size support
- Configurable image resolution (224 or 384)
- Model simplification and optimization
- Output validation against PyTorch reference

Usage:
    # Export DINOv3-base with 224x224 input (recommended)
    python scripts/export_dinov3_to_onnx.py --model facebook/dinov2-base --size 224
    
    # Export DINOv3-small (faster, smaller features)
    python scripts/export_dinov3_to_onnx.py --model facebook/dinov2-small --size 224
    
    # Export with higher resolution (better quality, slower)
    python scripts/export_dinov3_to_onnx.py --model facebook/dinov2-base --size 384

Requirements:
    pip install torch transformers onnx onnxruntime
    pip install onnxsim  # Optional, for model simplification
"""

import argparse
import logging
import os
from pathlib import Path
import sys

import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


def export_dinov3_to_onnx(
    model_name: str = "facebook/dinov2-base",
    output_path: str = None,
    image_size: int = 224,
    opset_version: int = 17,
    simplify: bool = True,
    validate: bool = True,
):
    """
    Export DINOv3 model to ONNX format.
    
    Args:
        model_name: HuggingFace model identifier
        output_path: Path to save ONNX model (auto-generated if None)
        image_size: Input image size (224 or 384)
        opset_version: ONNX opset version (17+ recommended for ViT)
        simplify: Apply ONNX simplification (requires onnxsim)
        validate: Validate output against PyTorch reference
    """
    try:
        from transformers import AutoModel, AutoImageProcessor
    except ImportError:
        log.error("transformers library not found. Install with: pip install transformers")
        sys.exit(1)
    
    # Generate output path if not provided
    if output_path is None:
        model_variant = model_name.split('/')[-1]  # e.g., 'dinov2-base'
        output_dir = Path("pretrained_models/reid")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{model_variant}_{image_size}.onnx"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    log.info(f"Exporting {model_name} to ONNX...")
    log.info(f"Image size: {image_size}x{image_size}")
    log.info(f"Output path: {output_path}")
    
    # Load model and processor
    log.info("Loading model from HuggingFace...")
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, image_size, image_size)
    
    # Test forward pass
    log.info("Testing forward pass...")
    with torch.no_grad():
        outputs = model(pixel_values=dummy_input)
    
    # Determine output type
    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
        output_key = 'pooler_output'
        log.info(f"Using pooler_output (CLS token): shape {outputs.pooler_output.shape}")
    else:
        output_key = 'last_hidden_state'
        log.info(f"Using last_hidden_state[:, 0, :] (CLS token): shape {outputs.last_hidden_state[:, 0, :].shape}")
    
    # Export to ONNX
    log.info("Exporting to ONNX...")
    
    # Prepare input/output names
    input_names = ["pixel_values"]
    output_names = ["features"]
    
    # Dynamic axes for batch size
    dynamic_axes = {
        "pixel_values": {0: "batch_size"},
        "features": {0: "batch_size"}
    }
    
    # Custom forward wrapper to output only CLS token features
    class DINOv3Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, pixel_values):
            outputs = self.model(pixel_values=pixel_values)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                return outputs.pooler_output
            else:
                # Extract CLS token from last_hidden_state
                return outputs.last_hidden_state[:, 0, :]
    
    wrapped_model = DINOv3Wrapper(model)
    
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
        verbose=False,
    )
    
    log.info(f"ONNX model exported to: {output_path}")
    
    # Simplify ONNX model (optional but recommended)
    if simplify:
        try:
            import onnxsim
            log.info("Simplifying ONNX model...")
            
            import onnx
            onnx_model = onnx.load(str(output_path))
            onnx_model_simplified, check = onnxsim.simplify(onnx_model)
            
            if check:
                simplified_path = output_path.with_suffix('.simplified.onnx')
                onnx.save(onnx_model_simplified, str(simplified_path))
                log.info(f"Simplified model saved to: {simplified_path}")
                
                # Replace original with simplified
                simplified_path.replace(output_path)
                log.info("Replaced original with simplified model")
            else:
                log.warning("Simplification check failed, keeping original model")
        
        except ImportError:
            log.warning("onnxsim not installed. Skipping simplification.")
            log.info("Install with: pip install onnxsim")
        except Exception as e:
            log.warning(f"Simplification failed: {e}. Keeping original model.")
    
    # Validate ONNX model
    if validate:
        log.info("Validating ONNX model...")
        try:
            import onnxruntime as ort
            
            # Load ONNX model
            session = ort.InferenceSession(
                str(output_path),
                providers=['CPUExecutionProvider']
            )
            
            # Test with different batch sizes
            for batch_size in [1, 2, 4]:
                test_input = torch.randn(batch_size, 3, image_size, image_size)
                
                # PyTorch inference
                with torch.no_grad():
                    torch_output = wrapped_model(test_input).numpy()
                
                # ONNX inference
                onnx_output = session.run(
                    None,
                    {"pixel_values": test_input.numpy()}
                )[0]
                
                # Compare outputs
                diff = np.abs(torch_output - onnx_output).max()
                log.info(f"Batch size {batch_size}: max difference = {diff:.6f}")
                
                if diff > 1e-4:
                    log.warning(f"Large difference detected: {diff}")
                else:
                    log.info(f"✓ Batch size {batch_size} validated successfully")
            
            # Print model info
            log.info("\n=== Model Information ===")
            log.info(f"Input name: {session.get_inputs()[0].name}")
            log.info(f"Input shape: {session.get_inputs()[0].shape}")
            log.info(f"Output name: {session.get_outputs()[0].name}")
            log.info(f"Output shape: {session.get_outputs()[0].shape}")
            log.info(f"Feature dimension: {session.get_outputs()[0].shape[1]}")
            
            # Calculate model size
            model_size_mb = output_path.stat().st_size / (1024 * 1024)
            log.info(f"Model size: {model_size_mb:.2f} MB")
            
        except ImportError:
            log.warning("onnxruntime not installed. Skipping validation.")
            log.info("Install with: pip install onnxruntime")
        except Exception as e:
            log.error(f"Validation failed: {e}")
    
    log.info("\n✓ Export completed successfully!")
    log.info(f"\nTo use this model, update your config:")
    log.info(f"  model_path: '{output_path}'")
    log.info(f"  image_size: [{image_size}, {image_size}]")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Export DINOv3 model to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export DINOv3-base (recommended for ReID)
  python scripts/export_dinov3_to_onnx.py --model facebook/dinov2-base --size 224
  
  # Export DINOv3-small (faster, smaller features)
  python scripts/export_dinov3_to_onnx.py --model facebook/dinov2-small --size 224
  
  # Export with high resolution
  python scripts/export_dinov3_to_onnx.py --model facebook/dinov2-base --size 384
  
  # Export to custom path
  python scripts/export_dinov3_to_onnx.py --model facebook/dinov2-base --output my_model.onnx

Available models:
  - facebook/dinov2-small  (384-dim features)
  - facebook/dinov2-base   (768-dim features, recommended)
  - facebook/dinov2-large  (1024-dim features)
  - facebook/dinov2-giant  (1536-dim features)
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='facebook/dinov2-base',
        help='HuggingFace model identifier (default: facebook/dinov2-base)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output ONNX file path (auto-generated if not specified)'
    )
    
    parser.add_argument(
        '--size',
        type=int,
        default=224,
        choices=[224, 384, 518],
        help='Input image size (default: 224)'
    )
    
    parser.add_argument(
        '--opset',
        type=int,
        default=17,
        help='ONNX opset version (default: 17, minimum 14 for ViT)'
    )
    
    parser.add_argument(
        '--no-simplify',
        action='store_true',
        help='Skip ONNX model simplification'
    )
    
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip validation against PyTorch reference'
    )
    
    args = parser.parse_args()
    
    export_dinov3_to_onnx(
        model_name=args.model,
        output_path=args.output,
        image_size=args.size,
        opset_version=args.opset,
        simplify=not args.no_simplify,
        validate=not args.no_validate,
    )


if __name__ == '__main__':
    main()
