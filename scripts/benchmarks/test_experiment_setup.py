#!/usr/bin/env python3
"""Test that all experiment dependencies are available."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"✗ PyTorch: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy: {e}")
        return False
    
    try:
        from src.models.resnet_bk import ResNetBK
        print("✓ ResNetBK model")
    except ImportError as e:
        print(f"✗ ResNetBK: {e}")
        return False
    
    try:
        from src.models.mamba_baseline import MambaBaseline
        print("✓ Mamba baseline")
    except ImportError as e:
        print(f"✗ Mamba baseline: {e}")
        return False
    
    try:
        from src.benchmarks.flops_counter import FLOPsCounter
        print("✓ FLOPs counter")
    except ImportError as e:
        print(f"✗ FLOPs counter: {e}")
        return False
    
    try:
        from src.benchmarks.wikitext2_benchmark import WikiText2Benchmark
        print("✓ WikiText2 benchmark")
    except ImportError as e:
        print(f"✗ WikiText2 benchmark: {e}")
        return False
    
    return True


def test_model_creation():
    """Test that models can be created."""
    print("\nTesting model creation...")
    
    try:
        import torch
        from src.models.resnet_bk import ResNetBK
        
        model = ResNetBK(d_model=64, num_layers=2)
        print(f"✓ ResNetBK created: {sum(p.numel() for p in model.parameters())} params")
    except Exception as e:
        print(f"✗ ResNetBK creation failed: {e}")
        return False
    
    try:
        from src.models.mamba_baseline import MambaBaseline
        
        model = MambaBaseline(d_model=64, n_layer=2)
        print(f"✓ Mamba created: {sum(p.numel() for p in model.parameters())} params")
    except Exception as e:
        print(f"✗ Mamba creation failed: {e}")
        return False
    
    return True


def test_forward_pass():
    """Test that models can do forward passes."""
    print("\nTesting forward passes...")
    
    try:
        import torch
        from src.models.resnet_bk import ResNetBK
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = ResNetBK(d_model=64, num_layers=2).to(device)
        x = torch.randint(0, 1000, (2, 128), device=device)
        
        with torch.no_grad():
            output = model(x)
        
        print(f"✓ ResNetBK forward pass: {output.shape}")
    except Exception as e:
        print(f"✗ ResNetBK forward failed: {e}")
        return False
    
    try:
        from src.models.mamba_baseline import MambaBaseline
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = MambaBaseline(d_model=64, n_layer=2).to(device)
        x = torch.randint(0, 1000, (2, 128), device=device)
        
        with torch.no_grad():
            output = model(x)
        
        print(f"✓ Mamba forward pass: {output.shape}")
    except Exception as e:
        print(f"✗ Mamba forward failed: {e}")
        return False
    
    return True


def main():
    print("=" * 60)
    print("Experiment Setup Test")
    print("=" * 60)
    
    all_passed = True
    
    if not test_imports():
        all_passed = False
    
    if not test_model_creation():
        all_passed = False
    
    if not test_forward_pass():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! Ready to run experiments.")
        return 0
    else:
        print("✗ Some tests failed. Please fix issues before running experiments.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
