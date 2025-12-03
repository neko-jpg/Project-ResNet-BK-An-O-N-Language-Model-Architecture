
import torch
import pytest
import math

try:
    import triton
    import triton.language as tl
    from src.kernels.htt_triton import htt_fused_contraction
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_htt_fused_contraction():
    torch.manual_seed(0)
    
    B, L = 2, 16
    v1, v2 = 10, 10 # V=100
    rank = 4
    d1, d2 = 8, 8 # D=64
    d_model = 64
    
    # Indices
    idx1 = torch.randint(0, v1, (B, L), device='cuda')
    idx2 = torch.randint(0, v2, (B, L), device='cuda')
    
    # Cores (int8)
    core1 = torch.randint(-127, 128, (v1, rank, d1), device='cuda', dtype=torch.int8)
    core2 = torch.randint(-127, 128, (v2, rank, d2), device='cuda', dtype=torch.int8)
    
    scale1 = torch.tensor(0.01, device='cuda')
    scale2 = torch.tensor(0.01, device='cuda')
    
    # Reference
    c1_float = core1.to(torch.float32) * scale1
    c2_float = core2.to(torch.float32) * scale2
    
    # Gather
    c1_gathered = c1_float[idx1] # (B, L, rank, d1)
    c2_gathered = c2_float[idx2] # (B, L, rank, d2)
    
    # Contract
    # einsum('blrd,blrf->bldf', c1, c2)
    out_ref = torch.einsum('blrd,blrf->bldf', c1_gathered, c2_gathered)
    out_ref = out_ref.reshape(B, L, -1)
    
    # Triton
    out_triton = htt_fused_contraction(
        idx1, idx2,
        core1, core2,
        scale1, scale2,
        d_model
    )
    
    # Compare
    assert torch.allclose(out_ref, out_triton, atol=1e-3, rtol=1e-3)
