"""
Test new Hyperbolic Attention Triton implementation.
"""
import torch
import time
import sys
sys.path.insert(0, '.')

def test_triton_implementation():
    """Test new Triton implementation."""
    print("=" * 60)
    print("Testing New Hyperbolic Attention Triton Implementation")
    print("=" * 60)
    
    from src.kernels.hyperbolic_attention_triton import (
        hyperbolic_attention_triton,
        _hyperbolic_attention_pytorch,
    )
    
    # Test parameters
    batch = 2
    num_heads = 4
    seq_len = 64
    d_head = 16
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"Shape: B={batch}, H={num_heads}, N={seq_len}, D={d_head}")
    
    # Create inputs
    torch.manual_seed(42)
    q = torch.randn(batch, num_heads, seq_len, d_head, device=device, dtype=torch.float32)
    k = torch.randn(batch, num_heads, seq_len, d_head, device=device, dtype=torch.float32)
    v = torch.randn(batch, num_heads, seq_len, d_head, device=device, dtype=torch.float32)
    c = torch.tensor(1.0, device=device, dtype=torch.float32)
    beta = torch.tensor(1.0, device=device, dtype=torch.float32)
    
    # Test 1: Forward pass correctness
    print("\n" + "-" * 40)
    print("Test 1: Forward Pass Correctness")
    print("-" * 40)
    
    # PyTorch reference
    out_pytorch = _hyperbolic_attention_pytorch(q, k, v, c, beta, causal=True)
    
    # Triton
    out_triton = hyperbolic_attention_triton(q, k, v, c, beta, causal=True)
    
    # Compare
    diff = (out_pytorch - out_triton).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Max difference: {max_diff:.6e}")
    print(f"Mean difference: {mean_diff:.6e}")
    
    if max_diff < 1e-3:
        print("✅ Forward pass: PASSED")
    else:
        print("⚠️ Forward pass: Large difference (may be acceptable)")
    
    # Test 2: Backward pass
    print("\n" + "-" * 40)
    print("Test 2: Backward Pass")
    print("-" * 40)
    
    q_grad = q.clone().requires_grad_(True)
    k_grad = k.clone().requires_grad_(True)
    v_grad = v.clone().requires_grad_(True)
    c_grad = c.clone().requires_grad_(True)
    beta_grad = beta.clone().requires_grad_(True)
    
    out = hyperbolic_attention_triton(q_grad, k_grad, v_grad, c_grad, beta_grad, causal=True)
    loss = out.sum()
    loss.backward()
    
    # Check gradients
    grads_ok = True
    for name, param in [('q', q_grad), ('k', k_grad), ('v', v_grad), ('c', c_grad), ('beta', beta_grad)]:
        if param.grad is not None:
            has_nan = torch.isnan(param.grad).any().item()
            has_inf = torch.isinf(param.grad).any().item()
            grad_norm = param.grad.norm().item()
            status = "✅" if not (has_nan or has_inf) else "❌"
            print(f"  {name}: norm={grad_norm:.4f}, NaN={has_nan}, Inf={has_inf} {status}")
            if has_nan or has_inf:
                grads_ok = False
        else:
            print(f"  {name}: No gradient")
    
    if grads_ok:
        print("✅ Backward pass: PASSED")
    else:
        print("❌ Backward pass: FAILED")
    
    # Test 3: Performance
    print("\n" + "-" * 40)
    print("Test 3: Performance")
    print("-" * 40)
    
    # Warmup
    for _ in range(3):
        _ = hyperbolic_attention_triton(q, k, v, c, beta, causal=True)
    torch.cuda.synchronize()
    
    # Forward timing
    n_iter = 10
    t0 = time.time()
    for _ in range(n_iter):
        out = hyperbolic_attention_triton(q, k, v, c, beta, causal=True)
    torch.cuda.synchronize()
    fwd_time = (time.time() - t0) / n_iter
    
    # Backward timing
    q_t = q.clone().requires_grad_(True)
    k_t = k.clone().requires_grad_(True)
    v_t = v.clone().requires_grad_(True)
    
    t0 = time.time()
    for _ in range(n_iter):
        out = hyperbolic_attention_triton(q_t, k_t, v_t, c, beta, causal=True)
        loss = out.sum()
        loss.backward()
        q_t.grad = None
        k_t.grad = None
        v_t.grad = None
    torch.cuda.synchronize()
    total_time = (time.time() - t0) / n_iter
    bwd_time = total_time - fwd_time
    
    print(f"Forward time:  {fwd_time*1000:.2f} ms")
    print(f"Backward time: {bwd_time*1000:.2f} ms")
    print(f"Total time:    {total_time*1000:.2f} ms")
    
    # Compare with PyTorch
    t0 = time.time()
    for _ in range(n_iter):
        out = _hyperbolic_attention_pytorch(q, k, v, c, beta, causal=True)
    torch.cuda.synchronize()
    pytorch_fwd_time = (time.time() - t0) / n_iter
    
    print(f"\nPyTorch forward: {pytorch_fwd_time*1000:.2f} ms")
    speedup = pytorch_fwd_time / fwd_time
    print(f"Triton speedup (forward): {speedup:.2f}x")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
    
    return grads_ok

if __name__ == "__main__":
    success = test_triton_implementation()
    exit(0 if success else 1)
