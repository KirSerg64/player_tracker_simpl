#!/usr/bin/env python3
"""
Feature Extractor Benchmark

Compare performance between ONNX and TensorRT feature extractors.
"""

import argparse
import time
import numpy as np
import cv2
from pathlib import Path
import logging
from typing import Dict, List

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def create_dummy_detections(batch_size: int, image_shape: tuple) -> List[Dict]:
    """Create dummy detection data for benchmarking."""
    h, w = image_shape[:2]
    detections = []
    
    for i in range(batch_size):
        # Random bounding box
        x1 = np.random.randint(0, w // 2)
        y1 = np.random.randint(0, h // 2)
        x2 = np.random.randint(x1 + 50, min(x1 + 200, w))
        y2 = np.random.randint(y1 + 50, min(y1 + 200, h))
        
        detection = np.array([x1, y1, x2, y2, 0.9])  # bbox + confidence
        detections.append(detection)
    
    return detections

def benchmark_feature_extractor(extractor, image: np.ndarray, detections: List[Dict], 
                               num_iterations: int = 100) -> Dict:
    """Benchmark a feature extractor."""
    from tracker.utils.pipeline_base import PipelineMessage, MessageType
    
    # Warmup
    log.info("Warming up...")
    for _ in range(5):
        input_msg = PipelineMessage(
            msg_type=MessageType.DATA,
            data={'frame': image, 'detections': detections},
            metadata={'frame_id': 0}
        )
        _ = extractor.process(input_msg)
    
    # Benchmark
    log.info(f"Running {num_iterations} iterations...")
    times = []
    
    for i in range(num_iterations):
        input_msg = PipelineMessage(
            msg_type=MessageType.DATA,
            data={'frame': image, 'detections': detections},
            metadata={'frame_id': i}
        )
        
        start_time = time.time()
        result = extractor.process(input_msg)
        end_time = time.time()
        
        times.append(end_time - start_time)
        
        if (i + 1) % 20 == 0:
            avg_time = np.mean(times[-20:])
            log.info(f"Progress: {i+1}/{num_iterations}, Recent avg: {avg_time*1000:.2f}ms")
    
    # Calculate statistics
    times = np.array(times)
    
    return {
        'avg_time_ms': np.mean(times) * 1000,
        'min_time_ms': np.min(times) * 1000,
        'max_time_ms': np.max(times) * 1000,
        'std_time_ms': np.std(times) * 1000,
        'throughput_fps': len(detections) / np.mean(times),
        'total_iterations': num_iterations
    }

def run_benchmark(
    onnx_model_path: str = None,
    tensorrt_model_path: str = None,
    batch_sizes: List[int] = [1, 4, 8, 16],
    num_iterations: int = 100,
    image_size: tuple = (640, 480),
    device: str = "cuda"
) -> Dict:
    """Run comprehensive benchmark."""
    
    # Create dummy image
    image = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
    
    results = {}
    
    # Benchmark ONNX if available
    if onnx_model_path and Path(onnx_model_path).exists():
        log.info("=" * 50)
        log.info("Benchmarking ONNX Feature Extractor")
        log.info("=" * 50)
        
        try:
            from tracker.reid.feature_extractor import FeatureExtractor
            
            # Create ONNX config
            class Config:
                def __init__(self):
                    self.model_name = 'osnet_x1_0'
                    self.model_path = onnx_model_path
                    self.image_size = [256, 128]
                    self.pixel_mean = [0.485, 0.456, 0.406]
                    self.pixel_std = [0.229, 0.224, 0.225]
                    self.pixel_norm = True
                    self.device = device
                    self.verbose = False
                    self.enable_batch_padding = True
            
            results['onnx'] = {}
            
            for batch_size in batch_sizes:
                log.info(f"\nTesting ONNX with batch size {batch_size}")
                
                extractor = FeatureExtractor(Config(), device, batch_size)
                detections = create_dummy_detections(batch_size, image_size)
                
                batch_results = benchmark_feature_extractor(
                    extractor, image, detections, num_iterations
                )
                
                results['onnx'][batch_size] = batch_results
                
                log.info(f"ONNX Batch {batch_size} Results:")
                log.info(f"  Avg time: {batch_results['avg_time_ms']:.2f}ms")
                log.info(f"  Throughput: {batch_results['throughput_fps']:.1f} FPS")
                
        except Exception as e:
            log.error(f"ONNX benchmark failed: {e}")
    
    # Benchmark TensorRT if available
    if tensorrt_model_path and Path(tensorrt_model_path).exists():
        log.info("=" * 50)
        log.info("Benchmarking TensorRT Feature Extractor")
        log.info("=" * 50)
        
        try:
            from tracker.reid.feature_extractor_tensorrt import TensorRTFeatureExtractor
            
            # Create TensorRT config
            class Config:
                def __init__(self):
                    self.model_name = 'osnet_x1_0_tensorrt'
                    self.model_path = tensorrt_model_path
                    self.image_size = [256, 128]
                    self.pixel_mean = [0.485, 0.456, 0.406]
                    self.pixel_std = [0.229, 0.224, 0.225]
                    self.pixel_norm = True
                    self.device = device
                    self.verbose = False
                    self.enable_batch_padding = True
            
            results['tensorrt'] = {}
            
            for batch_size in batch_sizes:
                log.info(f"\nTesting TensorRT with batch size {batch_size}")
                
                extractor = TensorRTFeatureExtractor(Config(), device, batch_size)
                detections = create_dummy_detections(batch_size, image_size)
                
                batch_results = benchmark_feature_extractor(
                    extractor, image, detections, num_iterations
                )
                
                # Add TensorRT-specific stats if available
                if hasattr(extractor, 'get_performance_stats'):
                    trt_stats = extractor.get_performance_stats()
                    batch_results.update(trt_stats)
                
                results['tensorrt'][batch_size] = batch_results
                
                log.info(f"TensorRT Batch {batch_size} Results:")
                log.info(f"  Avg time: {batch_results['avg_time_ms']:.2f}ms")
                log.info(f"  Throughput: {batch_results['throughput_fps']:.1f} FPS")
                
        except Exception as e:
            log.error(f"TensorRT benchmark failed: {e}")
    
    return results

def print_comparison(results: Dict):
    """Print detailed comparison between backends."""
    if 'onnx' not in results or 'tensorrt' not in results:
        log.warning("Cannot compare - missing results for one or both backends")
        return
    
    log.info("\n" + "=" * 70)
    log.info("PERFORMANCE COMPARISON")
    log.info("=" * 70)
    
    print(f"{'Batch Size':<12} {'ONNX (ms)':<12} {'TensorRT (ms)':<15} {'Speedup':<10} {'ONNX FPS':<12} {'TensorRT FPS':<15}")
    print("-" * 70)
    
    for batch_size in sorted(results['onnx'].keys()):
        if batch_size in results['tensorrt']:
            onnx_time = results['onnx'][batch_size]['avg_time_ms']
            trt_time = results['tensorrt'][batch_size]['avg_time_ms']
            speedup = onnx_time / trt_time
            
            onnx_fps = results['onnx'][batch_size]['throughput_fps']
            trt_fps = results['tensorrt'][batch_size]['throughput_fps']
            
            print(f"{batch_size:<12} {onnx_time:<12.2f} {trt_time:<15.2f} {speedup:<10.2f}x {onnx_fps:<12.1f} {trt_fps:<15.1f}")

def main():
    parser = argparse.ArgumentParser(description="Benchmark feature extractors")
    parser.add_argument('--onnx_model', type=str,
                        default='pretrained_models/reid/feature_extractor_osnet_x1_0.onnx',
                        help='Path to ONNX model')
    parser.add_argument('--tensorrt_model', type=str,
                        default='pretrained_models/reid/feature_extractor_osnet_x1_0.engine',
                        help='Path to TensorRT engine')
    parser.add_argument('--batch_sizes', nargs='+', type=int,
                        default=[1, 4, 8, 16],
                        help='Batch sizes to test')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of iterations per test')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--image_width', type=int, default=640,
                        help='Input image width')
    parser.add_argument('--image_height', type=int, default=480,
                        help='Input image height')
    
    args = parser.parse_args()
    
    image_size = (args.image_height, args.image_width)
    
    log.info("Starting feature extractor benchmark")
    log.info(f"ONNX model: {args.onnx_model}")
    log.info(f"TensorRT model: {args.tensorrt_model}")
    log.info(f"Batch sizes: {args.batch_sizes}")
    log.info(f"Iterations: {args.iterations}")
    log.info(f"Device: {args.device}")
    log.info(f"Image size: {image_size}")
    
    results = run_benchmark(
        onnx_model_path=args.onnx_model,
        tensorrt_model_path=args.tensorrt_model,
        batch_sizes=args.batch_sizes,
        num_iterations=args.iterations,
        image_size=image_size,
        device=args.device
    )
    
    # Print comparison
    print_comparison(results)
    
    # Save results
    import json
    output_file = f"benchmark_results_{int(time.time())}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
