#!/usr/bin/env python3
"""
ONNX to TensorRT Conversion Utility

This script converts ONNX models to optimized TensorRT engines for faster inference.
"""

import argparse
import os
import logging
from pathlib import Path

try:
    import tensorrt as trt
    import onnx
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def build_tensorrt_engine(
    onnx_path: str,
    engine_path: str,
    max_batch_size: int = 8,
    max_workspace_size: int = 1 << 30,  # 1GB
    fp16_mode: bool = True,
    int8_mode: bool = False,
    verbose: bool = False
):
    """
    Build TensorRT engine from ONNX model.
    
    Args:
        onnx_path: Path to ONNX model
        engine_path: Path to save TensorRT engine
        max_batch_size: Maximum batch size for the engine
        max_workspace_size: Maximum workspace size in bytes
        fp16_mode: Enable FP16 precision
        int8_mode: Enable INT8 quantization
        verbose: Enable verbose logging
    """
    if not TENSORRT_AVAILABLE:
        raise ImportError("TensorRT not available. Please install tensorrt package.")
    
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    
    # Set logging level
    trt_logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    
    # Create builder and network
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)
    
    # Parse ONNX model
    log.info(f"Parsing ONNX model: {onnx_path}")
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            log.error("Failed to parse ONNX model")
            for error in range(parser.num_errors):
                log.error(parser.get_error(error))
            return False
    
    # Configure builder
    config = builder.create_builder_config()
    
    # Set workspace/memory pool size (handle both old and new TensorRT versions)
    try:
        # TensorRT 8.5+ uses memory pool limits
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
        log.info(f"Set memory pool limit: {max_workspace_size} bytes")
    except AttributeError:
        try:
            # TensorRT 8.4 and earlier use max_workspace_size
            config.max_workspace_size = max_workspace_size
            log.info(f"Set max workspace size: {max_workspace_size} bytes")
        except AttributeError:
            log.warning("Could not set workspace size - using default")
    
    # Enable optimizations
    if fp16_mode and builder.platform_has_fast_fp16:
        log.info("Enabling FP16 mode")
        config.set_flag(trt.BuilderFlag.FP16)
    
    if int8_mode and builder.platform_has_fast_int8:
        log.info("Enabling INT8 mode")
        config.set_flag(trt.BuilderFlag.INT8)
        # Note: INT8 calibration would be needed here for production use
        log.warning("INT8 mode enabled but no calibration provided. This may result in accuracy loss.")
    
    # Set up dynamic shapes for better flexibility
    profile = builder.create_optimization_profile()
    
    # Get input tensor info
    input_tensor = network.get_input(0)
    input_shape = input_tensor.shape
    
    log.info(f"Input tensor shape: {input_shape}")
    
    # Configure dynamic batch size if the first dimension is -1 or we want flexibility
    if input_shape[0] == -1 or max_batch_size > 1:
        # Dynamic batch size: min=1, opt=max_batch_size//2, max=max_batch_size
        min_shape = (1,) + input_shape[1:]
        opt_shape = (max(1, max_batch_size // 2),) + input_shape[1:]
        max_shape = (max_batch_size,) + input_shape[1:]
        
        profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        
        log.info(f"Dynamic shapes - Min: {min_shape}, Opt: {opt_shape}, Max: {max_shape}")
    
    # Build engine
    log.info("Building TensorRT engine... This may take several minutes.")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        log.error("Failed to build TensorRT engine")
        return False
    
    # Save engine
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    log.info(f"TensorRT engine saved to: {engine_path}")
    
    # Print engine info
    runtime = trt.Runtime(trt_logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    
    # log.info(f"Engine max batch size: {engine.max_batch_size}")
    log.info(f"Engine num bindings: {engine.num_io_tensors}")
    
    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        tensor_shape = engine.get_tensor_shape(tensor_name)
        tensor_dtype = engine.get_tensor_dtype(tensor_name)
        is_input = engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT
        log.info(f"Binding {i}: {tensor_name} - {'Input' if is_input else 'Output'} - Shape: {tensor_shape} - Dtype: {tensor_dtype}")

    return True


def validate_onnx_model(onnx_path: str) -> bool:
    """Validate ONNX model."""
    try:
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        log.info(f"ONNX model validation passed: {onnx_path}")
        return True
    except Exception as e:
        log.error(f"ONNX model validation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX model to TensorRT engine")
    parser.add_argument('--onnx_path', type=str, required=True,
                        help='Path to input ONNX model')
    parser.add_argument('--engine_path', type=str, required=True,
                        help='Path to output TensorRT engine')
    parser.add_argument('--max_batch_size', type=int, default=8,
                        help='Maximum batch size for the engine')
    parser.add_argument('--max_workspace_size', type=int, default=1073741824,
                        help='Maximum workspace size in bytes (default: 1GB)')
    parser.add_argument('--fp16', action='store_true',
                        help='Enable FP16 precision')
    parser.add_argument('--int8', action='store_true',
                        help='Enable INT8 quantization')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--validate', action='store_true',
                        help='Validate ONNX model before conversion')
    
    args = parser.parse_args()
    
    # Validate ONNX model if requested
    if args.validate:
        if not validate_onnx_model(args.onnx_path):
            return 1
    
    # Convert to TensorRT
    success = build_tensorrt_engine(
        onnx_path=args.onnx_path,
        engine_path=args.engine_path,
        max_batch_size=args.max_batch_size,
        max_workspace_size=args.max_workspace_size,
        fp16_mode=args.fp16,
        int8_mode=args.int8,
        verbose=args.verbose
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
