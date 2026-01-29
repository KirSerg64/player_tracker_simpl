"""
Simple validation script to check refactored module imports and structure.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_imports():
    """Test that all classes can be imported."""
    print("Testing imports...")
    
    try:
        from tracker.segment import (
            BaseSegmenter,
            SamSegmenter,
            Sam3Segmenter,
            Sam2Segmenter,
            create_segmenter
        )
        print("✓ All classes imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_class_structure():
    """Test class hierarchy and methods."""
    print("\nTesting class structure...")
    
    from tracker.segment import (
        BaseSegmenter,
        SamSegmenter,
        Sam3Segmenter,
        Sam2Segmenter
    )
    import inspect
    
    # Check BaseSegmenter is abstract
    assert inspect.isabstract(BaseSegmenter), "BaseSegmenter should be abstract"
    print("✓ BaseSegmenter is abstract")
    
    # Check all required methods exist
    required_methods = ['process', 'set_image', 'reset_image', 'predict']
    for method in required_methods:
        assert hasattr(BaseSegmenter, method), f"Missing method: {method}"
    print(f"✓ All required methods present: {required_methods}")
    
    # Check concrete classes
    concrete_classes = [SamSegmenter, Sam3Segmenter, Sam2Segmenter]
    for cls in concrete_classes:
        assert issubclass(cls, BaseSegmenter), f"{cls.__name__} should inherit from BaseSegmenter"
    print(f"✓ All concrete classes inherit from BaseSegmenter")
    
    return True


def test_factory():
    """Test factory function."""
    print("\nTesting factory function...")
    
    from tracker.segment import create_segmenter
    from unittest.mock import Mock
    
    # Test unknown model type
    cfg = Mock()
    cfg.model_type = 'unknown'
    
    try:
        create_segmenter(cfg)
        print("✗ Factory should raise ValueError for unknown type")
        return False
    except ValueError as e:
        print(f"✓ Factory correctly raises ValueError: {e}")
        return True


def test_postprocessing():
    """Test mask postprocessing function."""
    print("\nTesting mask postprocessing...")
    
    from tracker.segment import BaseSegmenter
    import numpy as np
    
    # Test with numpy
    mask_np = np.array([[0.1, 0.8], [0.9, 0.2]])
    result = BaseSegmenter._postprocess_mask(mask_np)
    assert result.dtype == np.uint8, "Output should be uint8"
    assert result.shape == mask_np.shape, "Shape should be preserved"
    print("✓ Numpy postprocessing works")
    
    # Test with torch
    try:
        import torch
        mask_torch = torch.tensor([[0.1, 0.8], [0.9, 0.2]])
        result = BaseSegmenter._postprocess_mask(mask_torch)
        assert isinstance(result, np.ndarray), "Output should be numpy"
        assert result.dtype == np.uint8, "Output should be uint8"
        print("✓ Torch postprocessing works")
    except ImportError:
        print("⚠ Torch not available, skipping torch test")
    
    return True


def test_configs():
    """Test that config files exist."""
    print("\nTesting configuration files...")
    
    config_dir = Path(__file__).parent.parent.parent / 'tracker' / 'configs' / 'segment'
    
    expected_configs = [
        'base_segmenter.yaml',
        'sam.yaml',
        'mobile_sam.yaml',
        'sam3.yaml',
        'sam3_text.yaml'
    ]
    
    missing = []
    for config in expected_configs:
        config_path = config_dir / config
        if config_path.exists():
            print(f"✓ {config}")
        else:
            print(f"✗ {config} missing")
            missing.append(config)
    
    if missing:
        print(f"⚠ Missing configs: {missing}")
        return False
    
    return True


def test_documentation():
    """Test that documentation files exist."""
    print("\nTesting documentation...")
    
    doc_dir = Path(__file__).parent
    
    expected_docs = [
        'SEGMENTATION_REFACTOR_README.md',
        'QUICK_REFERENCE.md',
        'examples_usage.py'
    ]
    
    for doc in expected_docs:
        doc_path = doc_dir / doc
        if doc_path.exists():
            print(f"✓ {doc}")
        else:
            print(f"✗ {doc} missing")
    
    return True


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("Segmentation Module Validation")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Class Structure", test_class_structure),
        ("Factory Function", test_factory),
        ("Postprocessing", test_postprocessing),
        ("Configuration Files", test_configs),
        ("Documentation", test_documentation)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, r in results if r)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All validations passed!")
        return 0
    else:
        print(f"\n⚠ {total - passed} validation(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
