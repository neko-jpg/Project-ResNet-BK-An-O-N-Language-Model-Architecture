"""
Test Hyperbolic Attention with PyTorch reference implementation only.
"""
import torch
import time

def test_pytorch_reference():
    """Test PyTorch reference implementation."""
    print("Testing PyTorch Reference Implementation...")
    
    # Small test case
    batch = 2
    seq_len = 64
    d_model = 64
    num_heads = 4
    d_head = d_model // num_heads
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create inputs
    q = torch.randn(batch, num_heads, seq_len, d_head, device=device, requires_grad=True)
    k = torch.randn(batch, num_heads, seq_len, d_head, device=device, requires_grad=True)
    v = torch.randn(batch, num_heads, seq_len, d_head, device=device, requires_grad=True)
    c = torch.tensor(1.0, device=device, requires_grad=True)
    beta = torch.tensor(1.0, device=device, requires_grad=True)
    attention_bias = torch.tensor(0.0, device=device, requires_grad=True)
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).view(1, 1, seq_len, seq_len)
    
    # Import reference implementation
    from src.kernels.hyperbolic_attention_kernel import _hyperbolic_attention_reference
    
    # Forward pass
    print("\nForward pass...")
    t0 = time.time()
    output = _hyperbolic_attention_reference(q, k, v, c, beta, attention_bias, mask)
    fwd_time = time.time() - t0
    print(f"Forward time: {fwd_time:.4f}s")
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.6f}")
    print(f"Output std: {output.std().item():.6f}")
    print(f"Output min: {output.min().item():.6f}")
    print(f"Output max: {output.max().item():.6f}")
    
    # Check for NaN/Inf
    if torch.isnan(output).any():
        print("❌ NaN detected in output!")
        return False
    if torch.isinf(output).any():
        print("❌ Inf detected in output!")
        return False
    print("✅ No NaN/Inf in output")
    
    # Backward pass
    print("\nBackward pass...")
    loss = output.sum()
    t1 = time.time()
    loss.backward()
    bwd_time = time.time() - t1
    print(f"Backward time: {bwd_time:.4f}s")
    
    # Check gradients
    print("\nGradient check:")
    for name, param in [('q', q), ('k', k), ('v', v), ('c', c), ('beta', beta), ('attention_bias', attention_bias)]:
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            has_nan = torch.isnan(param.grad).any().item()
            has_inf = torch.isinf(param.grad).any().item()
            print(f"  {name}: norm={grad_norm:.6f}, NaN={has_nan}, Inf={has_inf}")
            if has_nan or has_inf:
                print(f"  ❌ {name} has NaN or Inf!")
                return False
    
    print("\n✅ All gradients are finite!")
    print(f"\nTotal time: {fwd_time + bwd_time:.4f}s")
    return True

if __name__ == "__main__":
    success = test_pytorch_reference()
    if success:
        print("\n✅ PyTorch reference implementation works correctly!")
    else:
        print("\n❌ PyTorch reference implementation has issues!")
    exit(0 if success else 1)
