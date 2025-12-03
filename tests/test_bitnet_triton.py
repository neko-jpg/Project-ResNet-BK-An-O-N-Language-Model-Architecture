
import torch
import pytest
import math

try:
    import triton
    import triton.language as tl
    from src.kernels.bitnet_triton import bitnet_matmul
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bitnet_matmul_correctness():
    torch.manual_seed(0)
    
    B, N, K = 32, 128, 64
    
    # Inputs
    x = torch.randn(B, N, device='cuda', dtype=torch.float16)
    
    # Weights (int8 {-1, 0, 1})
    w_fp = torch.randn(N, K, device='cuda', dtype=torch.float16)
    scale = w_fp.abs().mean(dim=0) # Per-channel scale? Or global?
    # bitnet_matmul expects scale per output channel (K,)
    # Let's assume w_fp is already scaled somewhat.
    
    # Quantize w_fp to simulate int8 weights
    # w_q = clamp(round(w / s), -1, 1)
    # We need to generate valid int8 weights first.
    w_int8 = torch.randint(-1, 2, (N, K), device='cuda', dtype=torch.int8)
    
    # Scale
    w_scale = torch.rand(K, device='cuda', dtype=torch.float16) + 0.5
    
    # Reference computation
    w_fp_ref = w_int8.to(torch.float16) * w_scale.unsqueeze(0)
    y_ref = torch.matmul(x, w_fp_ref)
    
    # Triton computation
    y_triton = bitnet_matmul(x, w_int8, w_scale)
    
    # Compare
    # Tolerances for float16 can be loose
    assert torch.allclose(y_ref, y_triton, atol=1e-2, rtol=1e-2)

@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bitnet_matmul_shapes():
    B, N, K = 1, 32, 32
    x = torch.randn(B, N, device='cuda', dtype=torch.float32)
    w = torch.randint(-1, 2, (N, K), device='cuda', dtype=torch.int8)
    s = torch.ones(K, device='cuda', dtype=torch.float32)
    
    y = bitnet_matmul(x, w, s)
    assert y.shape == (B, K)

