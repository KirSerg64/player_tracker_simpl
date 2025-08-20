#!/usr/bin/env python3
"""
Test script to verify GPU/CPU selection logic in tracklet_refiner.py
"""

import os
import sys
from types import SimpleNamespace

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tracker.gta_link.tracklet_refiner import TrackletsRefiner

def test_gpu_cpu_selection():
    """Test GPU/CPU function selection based on configuration."""
    
    print("="*60)
    print("Testing GPU/CPU Selection Logic")
    print("="*60)
    
    # Test configurations
    configs = [
        ("GPU Enabled", True),
        ("GPU Disabled", False),
    ]
    
    for config_name, use_gpu in configs:
        print(f"\nTesting: {config_name}")
        print("-" * 30)
        
        # Create mock configuration
        cfg = SimpleNamespace()
        cfg.use_split = True
        cfg.eps = 0.5
        cfg.min_samples = 3
        cfg.max_k = 10
        cfg.min_len = 5
        cfg.merge_dist_thres = 50.0
        cfg.spatial_factor = 2.0
        cfg.use_batched_merge = True
        cfg.mapping_strategy = "sequential"
        cfg.return_refined_detections = True
        cfg.max_wait_time = 30
        cfg.wait_interval = 0.5
        cfg.save_tracklets = True
        cfg.use_gpu_acceleration = use_gpu  # This is the key flag
        cfg.gpu_batch_size = None
        cfg.enable_benchmarking = False
        
        try:
            # Create refiner instance
            refiner = TrackletsRefiner(
                cfg=cfg,
                device="cuda:0",
                batch_size=50
            )
            
            # Print status
            print(f"   use_gpu_acceleration: {refiner.use_gpu_acceleration}")
            print(f"   GPU_AVAILABLE: {refiner.GPU_AVAILABLE}")
            print(f"   Functions loaded: {'GPU' if refiner.GPU_AVAILABLE else 'CPU'}")
            
            # Get detailed GPU info
            gpu_info = refiner.get_gpu_info()
            print(f"   CuPy available: {gpu_info['cupy_available']}")
            print(f"   Functions type: {gpu_info['functions_loaded']}")
            
            if 'gpu_error' in gpu_info:
                print(f"   GPU Error: {gpu_info['gpu_error']}")
            elif gpu_info['cupy_available']:
                print(f"   CUDA devices: {gpu_info.get('cuda_devices', 'Unknown')}")
                if 'gpu_memory' in gpu_info:
                    mem = gpu_info['gpu_memory']
                    print(f"   GPU Memory: {mem['free_gb']:.1f}GB free / {mem['total_gb']:.1f}GB total")
            
            print("   Configuration successful")
            
        except Exception as e:
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("Test completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    test_gpu_cpu_selection()
