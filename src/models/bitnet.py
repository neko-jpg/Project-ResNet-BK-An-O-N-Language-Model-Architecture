import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    from src.kernels.bitnet_triton import bitnet_matmul
    TRITON_AVAILABLE = False # Force disable for debugging
except ImportError:
    TRITON_AVAILABLE = False

def ternary_weight(w: torch.Tensor) -> torch.Tensor:
    """
    BitNet b1.58 quantization function with Straight-Through Estimator (STE).
    w_q = clamp(round(w / s), -1, 1) * s
    s = mean(|w|)
    """
    if w.numel() == 0:
        return w

    # 1. Calculate scale s = mean(|w|)
    scale = w.abs().mean()
    scale = torch.clamp(scale, min=1e-3)  # Increased from 1e-4 for better stability

    # 2. Quantize: round(w/s) clipped to {-1, 0, 1}
    w_scaled = w / scale
    w_quant = torch.clamp(torch.round(w_scaled), -1, 1)

    # 3. Rescale
    w_out = w_quant * scale

    # 4. Straight-Through Estimator (STE)
    return (w_out - w).detach() + w

class BitNetLinear(nn.Linear):
    """
    Linear layer with BitNet b1.58 quantization.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Quantize weights
        w_quant = ternary_weight(self.weight)
        
        # Use Triton kernel if available and input is CUDA
        if TRITON_AVAILABLE and input.is_cuda and self.weight.is_cuda:
            # Prepare inputs for Triton kernel
            # w_quant is float, but effectively {-s, 0, s}
            # We need to extract int8 weights and scale
            scale = self.weight.abs().mean()
            scale = torch.clamp(scale, min=1e-6)
            
            w_scaled = self.weight / scale
            w_int8 = torch.clamp(torch.round(w_scaled), -1, 1).to(torch.int8)
            
            # bitnet_matmul expects (M, K) @ (K, N) -> (M, N)
            # input: (..., in_features) -> flatten to (M, K)
            # weight: (out_features, in_features) -> (N, K)
            # We need weight.T for matmul: (K, N)
            
            input_shape = input.shape
            input_flat = input.view(-1, self.in_features)
            
            # Transpose weight for matmul: (K, N)
            w_int8_t = w_int8.t().contiguous()
            
            # Scale vector: (N,)
            scale_vec = torch.full((self.out_features,), scale.item(), device=input.device, dtype=input.dtype)
            
            output_flat = bitnet_matmul(input_flat, w_int8_t, scale_vec)
            
            output = output_flat.view(*input_shape[:-1], self.out_features)
            
            if self.bias is not None:
                output += self.bias
                
            return output
        else:
            # Fallback to PyTorch (using quantized float weights)
            return F.linear(input, w_quant, self.bias)

class LowRankLinear(nn.Module):
    """
    Low-Rank Linear Layer: W = U @ V^T
    U: (out_features, rank)
    V: (in_features, rank)
    
    Supports BitNet quantization for U and V.
    """
    def __init__(self, in_features, out_features, rank, bias=True, use_bitnet=False):
        super().__init__()
        self.rank = rank
        self.use_bitnet = use_bitnet
        
        LinearClass = BitNetLinear if use_bitnet else nn.Linear
        
        self.U = LinearClass(rank, out_features, bias=bias)
        self.V = LinearClass(in_features, rank, bias=False)
        
        self._init_weights()
        
    def _init_weights(self):
        # Robust initialization for low-rank matrices
        # Scale down aggressively to prevent explosion in deep networks
        nn.init.normal_(self.U.weight, std=0.001)  # Reduced from 0.02
        if self.U.bias is not None:
            nn.init.zeros_(self.U.bias)
            
        nn.init.normal_(self.V.weight, std=0.001)  # Reduced from 0.02

    def forward(self, x):
        # x: (..., in_features)
        # V(x): (..., rank)
        # U(V(x)): (..., out_features)
        return self.U(self.V(x))
