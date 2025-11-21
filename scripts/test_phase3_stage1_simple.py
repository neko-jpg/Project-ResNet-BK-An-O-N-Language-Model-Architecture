"""
Simple test script for Phase 3 Stage 1 model

このスクリプトは、Phase 3 Stage 1モデルの基本的な動作を確認します。
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.phase3.stage1_model import Phase3Stage1Model, Phase3Stage1Config

def test_model_creation():
    """モデルの作成をテスト"""
    print("\n[Test 1] Model Creation")
    print("=" * 60)
    
    config = Phase3Stage1Config(
        vocab_size=50257,
        d_model=512,
        n_layers=6,
        n_seq=1024,
        use_complex32=True
    )
    
    model = Phase3Stage1Model(config)
    print(f"✓ Model created successfully")
    print(f"  - vocab_size: {model.vocab_size}")
    print(f"  - d_model: {model.d_model}")
    print(f"  - n_layers: {model.n_layers}")
    print(f"  - max_seq_len: {model.max_seq_len}")
    print(f"  - use_complex32: {model.use_complex32}")
    
    # パラメータ数を計算
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - Total parameters: {total_params:,}")
    
    return model

def test_forward_pass(model, device='cpu'):
    """Forward passをテスト"""
    print("\n[Test 2] Forward Pass")
    print("=" * 60)
    
    model = model.to(device)
    model.eval()
    
    # ダミー入力
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_length), device=device)
    
    print(f"  - Input shape: {input_ids.shape}")
    
    # Forward pass
    with torch.no_grad():
        logits = model(input_ids)
    
    print(f"  - Output shape: {logits.shape}")
    print(f"  - Expected shape: ({batch_size}, {seq_length}, {model.vocab_size})")
    
    # 形状チェック
    assert logits.shape == (batch_size, seq_length, model.vocab_size), "Output shape mismatch"
    print(f"✓ Forward pass successful")
    
    # NaN/Infチェック
    has_nan = torch.isnan(logits).any().item()
    has_inf = torch.isinf(logits).any().item()
    
    print(f"  - NaN detected: {has_nan}")
    print(f"  - Inf detected: {has_inf}")
    
    if has_nan or has_inf:
        print(f"✗ Numerical instability detected!")
        return False
    else:
        print(f"✓ Numerical stability confirmed")
        return True

def test_backward_pass(model, device='cpu'):
    """Backward passをテスト"""
    print("\n[Test 3] Backward Pass")
    print("=" * 60)
    
    model = model.to(device)
    model.train()
    
    # ダミー入力
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_length), device=device)
    labels = torch.randint(0, model.vocab_size, (batch_size, seq_length), device=device)
    
    # Forward pass
    logits = model(input_ids)
    
    # Loss計算
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1)
    )
    
    print(f"  - Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    print(f"✓ Backward pass successful")
    
    # 勾配チェック
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            if grad_norm < 1e-6 or grad_norm > 1e3:
                print(f"  ⚠ {name}: grad_norm={grad_norm:.6e} (out of range)")
    
    if grad_norms:
        print(f"  - Gradient norms: min={min(grad_norms):.6e}, max={max(grad_norms):.6e}")
        
        # 勾配健全性チェック
        all_healthy = all(1e-6 <= g <= 1e3 for g in grad_norms)
        if all_healthy:
            print(f"✓ All gradients are healthy")
            return True
        else:
            print(f"✗ Some gradients are unhealthy")
            return False
    else:
        print(f"✗ No gradients found")
        return False

def test_memory_usage(model, device='cpu'):
    """メモリ使用量をテスト"""
    print("\n[Test 4] Memory Usage")
    print("=" * 60)
    
    if device == 'cpu':
        print("  - Skipping memory test on CPU")
        return True
    
    model = model.to(device)
    model.train()
    
    # VRAMをリセット
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    # ダミー入力
    batch_size = 2
    seq_length = 1024  # max_seq_lenに合わせる
    input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_length), device=device)
    labels = torch.randint(0, model.vocab_size, (batch_size, seq_length), device=device)
    
    # Forward + Backward pass
    logits = model(input_ids)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1)
    )
    loss.backward()
    
    # VRAM測定
    peak_memory = torch.cuda.max_memory_allocated(device)
    current_memory = torch.cuda.memory_allocated(device)
    
    vram_gb = current_memory / (1024 ** 3)
    peak_vram_gb = peak_memory / (1024 ** 3)
    
    print(f"  - Current VRAM: {vram_gb:.2f} GB")
    print(f"  - Peak VRAM: {peak_vram_gb:.2f} GB")
    
    # 目標: 8GB以下
    if peak_vram_gb < 8.0:
        print(f"✓ Memory usage is within target (< 8GB)")
        return True
    else:
        print(f"✗ Memory usage exceeds target (>= 8GB)")
        return False

def main():
    print("=" * 60)
    print("Phase 3 Stage 1 Model - Simple Test")
    print("=" * 60)
    
    # デバイスの選択
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Test 1: モデルの作成
    model = test_model_creation()
    
    # Test 2: Forward pass
    forward_ok = test_forward_pass(model, device)
    
    # Test 3: Backward pass
    backward_ok = test_backward_pass(model, device)
    
    # Test 4: Memory usage (CUDA only)
    if device == 'cuda':
        memory_ok = test_memory_usage(model, device)
    else:
        memory_ok = True
    
    # 総合結果
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"  - Forward pass: {'✓ PASS' if forward_ok else '✗ FAIL'}")
    print(f"  - Backward pass: {'✓ PASS' if backward_ok else '✗ FAIL'}")
    print(f"  - Memory usage: {'✓ PASS' if memory_ok else '✗ FAIL'}")
    
    all_pass = forward_ok and backward_ok and memory_ok
    print(f"\nOverall: {'✓ ALL PASS' if all_pass else '✗ SOME FAILED'}")
    
    return 0 if all_pass else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
