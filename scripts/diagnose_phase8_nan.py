#!/usr/bin/env python3
"""
Phase 8 NaN診断スクリプト
どこでNaNが発生しているか特定する
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn

# Phase 8 imports
from src.models.phase8.linear_attention import TangentSpaceLinearAttention, LinearAttentionConfig

def check_nan(tensor, name):
    """テンソルにNaNが含まれているかチェック"""
    if torch.isnan(tensor).any():
        print(f"❌ NaN detected in {name}!")
        print(f"   Shape: {tensor.shape}")
        print(f"   Min: {tensor.min().item()}, Max: {tensor.max().item()}")
        return True
    else:
        print(f"✓ {name} is OK (min: {tensor.min().item():.4f}, max: {tensor.max().item():.4f})")
        return False

def test_linear_attention():
    """Linear Attentionのテスト"""
    print("\n" + "="*60)
    print("Testing TangentSpaceLinearAttention")
    print("="*60)
    
    # 設定
    config = LinearAttentionConfig(
        d_model=512,
        num_heads=8,
        curvature=0.01,
        low_curvature_threshold=0.1,
        high_curvature_threshold=1.0,
        num_features=64,
        kernel_type="elu"
    )
    
    # モデル作成
    model = TangentSpaceLinearAttention(config)
    model.eval()
    
    # 入力データ（小さい値で初期化）
    B, N, D = 2, 128, 512
    x = torch.randn(B, N, D) * 0.01  # 小さい値
    
    print(f"\nInput shape: {x.shape}")
    check_nan(x, "Input")
    
    # Forward pass with hooks
    has_nan = False
    
    def hook_fn(name):
        def hook(module, input, output):
            nonlocal has_nan
            if isinstance(output, tuple):
                output = output[0]
            if check_nan(output, f"{name} output"):
                has_nan = True
        return hook
    
    # フックを登録
    model.q_proj.register_forward_hook(hook_fn("Q projection"))
    model.k_proj.register_forward_hook(hook_fn("K projection"))
    model.v_proj.register_forward_hook(hook_fn("V projection"))
    model.out_proj.register_forward_hook(hook_fn("Output projection"))
    
    try:
        with torch.no_grad():
            output = model(x)
            if isinstance(output, tuple):
                output = output[0]
            check_nan(output, "Final output")
    except Exception as e:
        print(f"\n❌ Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    if has_nan:
        print("\n❌ NaN detected during forward pass!")
        return False
    else:
        print("\n✓ No NaN detected!")
        return True

def test_simple_model():
    """簡易モデルのテスト"""
    print("\n" + "="*60)
    print("Testing Simple Phase8Layer")
    print("="*60)
    
    from scripts.train_phase8 import Phase8Config, Phase8Layer
    
    config = Phase8Config(
        d_model=256,
        n_layers=2,
        num_heads=4,
        max_seq_len=128,
        curvature=0.01,
        use_hyperbolic_ssm=False
    )
    
    layer = Phase8Layer(config)
    layer.eval()
    
    B, N, D = 2, 128, 256
    x = torch.randn(B, N, D) * 0.01
    
    print(f"\nInput shape: {x.shape}")
    check_nan(x, "Input")
    
    try:
        with torch.no_grad():
            output = layer(x)
            check_nan(output, "Layer output")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✓ Layer test passed!")
    return True

def test_initialization():
    """初期化のテスト"""
    print("\n" + "="*60)
    print("Testing Model Initialization")
    print("="*60)
    
    from scripts.train_phase8 import Phase8Config, Phase8Model
    
    config = Phase8Config(
        d_model=256,
        n_layers=2,
        num_heads=4,
        max_seq_len=128,
        curvature=0.01,
        use_hyperbolic_ssm=False
    )
    
    model = Phase8Model(config)
    
    # パラメータチェック
    print("\nChecking parameters...")
    has_nan = False
    for name, param in model.named_parameters():
        if check_nan(param, f"Parameter: {name}"):
            has_nan = True
    
    if has_nan:
        print("\n❌ NaN in parameters!")
        return False
    
    print("\n✓ All parameters initialized correctly!")
    return True

if __name__ == "__main__":
    print("Phase 8 NaN Diagnosis")
    print("="*60)
    
    # テスト実行
    test1 = test_initialization()
    test2 = test_linear_attention()
    test3 = test_simple_model()
    
    print("\n" + "="*60)
    print("Summary:")
    print(f"  Initialization: {'✓ PASS' if test1 else '❌ FAIL'}")
    print(f"  Linear Attention: {'✓ PASS' if test2 else '❌ FAIL'}")
    print(f"  Simple Model: {'✓ PASS' if test3 else '❌ FAIL'}")
    print("="*60)
    
    if all([test1, test2, test3]):
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
