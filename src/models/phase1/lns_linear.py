"""
LNS Linear Layer - Phase 1.3

物理的直観 (Physical Intuition):
推論時に対数領域で計算を行うことで、乗算器(FMA)を使わず
加算器(ADD)のみで線形層を実装します。これにより消費電力を
大幅に削減し、推論速度を向上させます。

Mathematical Foundation:
Standard Linear Layer: y = xW^T + b
LNS Linear Layer (inference): 
  log(y) ≈ lns_matmul(log(|x|), log(|W|)) + log(|b|)
  y = sign(x) * sign(W) * exp(log(y))

Note: This is primarily for inference-only deployment.
Training in log domain is complex and not implemented here.

References:
- Requirements: 3.1, 3.3, 3.4, 3.5
- Design: Section "Logarithmic Number System (LNS) Kernel"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import warnings

try:
    from ...kernels.lns_kernel import lns_matmul, TRITON_AVAILABLE
except ImportError:
    TRITON_AVAILABLE = False
    lns_matmul = None


class LNSLinear(nn.Module):
    """
    Linear layer using LNS kernel for matrix multiplication.
    
    物理的直観:
    推論時に対数領域で計算を実行。学習時は通常のFP16/FP32で学習し、
    推論時のみLNSカーネルを使用することで、精度を保ちながら
    計算コストを削減します。
    
    This layer provides a drop-in replacement for nn.Linear with
    LNS-based inference for reduced power consumption and improved
    throughput on CUDA devices.
    
    Use cases:
    - Inference-only models (weights pre-converted to log domain)
    - Large matrix multiplications where FMA is bottleneck
    - Experimental log-domain training
    
    Args:
        in_features: Size of input features
        out_features: Size of output features
        bias: If True, adds learnable bias (default: True)
        use_lns: If True, use LNS kernel for inference (default: True)
        lns_block_size_m: Block size for M dimension (default: 128)
        lns_block_size_n: Block size for N dimension (default: 128)
        lns_block_size_k: Block size for K dimension (default: 32)
    
    Requirements: 3.1, 3.3, 3.4, 3.5
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        use_lns: bool = True,
        lns_block_size_m: int = 128,
        lns_block_size_n: int = 128,
        lns_block_size_k: int = 32,
        gradient_clip_value: Optional[float] = 10.0,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.use_lns = use_lns and TRITON_AVAILABLE
        self.lns_block_size_m = lns_block_size_m
        self.lns_block_size_n = lns_block_size_n
        self.lns_block_size_k = lns_block_size_k
        self.gradient_clip_value = gradient_clip_value
        
        # Standard weight and bias parameters
        # Requirement 3.3: Standard parameters for training
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Pre-computed log weights for inference
        # Requirement 3.4: Add pre-computation of log weights for inference
        # 物理的直観: 推論時に毎回log計算するのは無駄なので、事前計算して保存
        self.register_buffer('log_weight', None)
        self.register_buffer('weight_sign', None)
        if bias:
            self.register_buffer('log_bias', None)
            self.register_buffer('bias_sign', None)
        
        # Initialize parameters
        self.reset_parameters()
        
        # Warn if LNS requested but not available
        if use_lns and not TRITON_AVAILABLE:
            warnings.warn(
                "LNS kernel requested but Triton is not available. "
                "Falling back to standard torch.matmul.",
                RuntimeWarning
            )
    
    def reset_parameters(self):
        """Initialize parameters using Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def prepare_lns_weights(self):
        """
        Pre-compute log-domain weights for inference.
        
        物理的直観:
        重みを対数領域に変換し、符号を別途保存。
        推論時にこれを使うことで、毎回のlog計算を回避します。
        
        This should be called once before inference to convert weights
        to log domain. The conversion is done in-place to save memory.
        
        Requirements: 3.4
        """
        if not self.use_lns:
            return
        
        with torch.no_grad():
            # Compute log of absolute values
            # Add small epsilon to avoid log(0)
            eps = 1e-8
            self.log_weight = torch.log(torch.abs(self.weight) + eps)
            self.weight_sign = torch.sign(self.weight)
            
            if self.bias is not None:
                self.log_bias = torch.log(torch.abs(self.bias) + eps)
                self.bias_sign = torch.sign(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional LNS computation.
        
        物理的直観:
        学習時: 通常のFP16/FP32計算（勾配が必要）
        推論時: LNSカーネルで対数領域計算（高速・低消費電力）
        
        Args:
            x: Input tensor of shape (..., in_features)
        
        Returns:
            output: Output tensor of shape (..., out_features)
        
        Requirements: 3.3, 3.4, 3.5
        """
        # Requirement 3.5: Implement fallback to standard matmul for training
        if self.training or not self.use_lns:
            # Training mode: use standard linear layer
            # 物理的直観: 学習時は通常の計算（勾配が必要）
            return F.linear(x, self.weight, self.bias)
        
        # Inference mode: use LNS kernel
        # Requirement 3.4: Use pre-computed log weights
        if self.log_weight is None:
            # First inference call: prepare log weights
            self.prepare_lns_weights()
        
        # Check if we can use LNS kernel
        if not x.is_cuda or not TRITON_AVAILABLE:
            # Requirement 3.5: Fallback to standard matmul
            # CPU or Triton not available: use standard computation
            return F.linear(x, self.weight, self.bias)
        
        # Save original shape for reshaping later
        original_shape = x.shape
        
        # Flatten batch dimensions
        # x: (..., in_features) -> (batch_size, in_features)
        x_flat = x.reshape(-1, self.in_features)
        batch_size = x_flat.shape[0]
        
        # Convert input to log domain
        # 物理的直観: 入力を対数領域に変換
        eps = 1e-8
        log_x = torch.log(torch.abs(x_flat) + eps)
        x_sign = torch.sign(x_flat)
        
        # Perform LNS matrix multiplication
        # log_out = lns_matmul(log_x, log_weight^T)
        # Requirement 3.1: Use LNS kernel for matrix multiplication
        # Task 8.3: Add gradient clipping at kernel level
        try:
            log_out = lns_matmul(
                log_x,
                self.log_weight.T,
                block_size_m=self.lns_block_size_m,
                block_size_n=self.lns_block_size_n,
                block_size_k=self.lns_block_size_k,
                gradient_clip_value=self.gradient_clip_value,
            )
            
            # Compute output sign
            # sign(out) = sign(x) @ sign(weight^T)
            # This is approximate but works for most cases
            out_sign = torch.sign(torch.matmul(x_sign, self.weight_sign.T))
            
            # Convert back to linear domain
            # out = sign * exp(log_out)
            out = out_sign * torch.exp(log_out)
            
            # Add bias if present
            if self.bias is not None:
                # Bias is added in linear domain
                # For simplicity, we add it directly
                # A more accurate approach would handle sign properly
                out = out + self.bias
            
        except Exception as e:
            # Requirement 3.5: Fallback on error
            warnings.warn(
                f"LNS kernel failed: {e}. Falling back to standard matmul.",
                RuntimeWarning
            )
            out = F.linear(x_flat, self.weight, self.bias)
        
        # Reshape to original shape
        output_shape = list(original_shape[:-1]) + [self.out_features]
        out = out.reshape(output_shape)
        
        return out
    
    def extra_repr(self) -> str:
        """String representation of the layer."""
        return (
            f'in_features={self.in_features}, '
            f'out_features={self.out_features}, '
            f'bias={self.bias is not None}, '
            f'use_lns={self.use_lns}'
        )


def convert_linear_to_lns(
    module: nn.Module,
    inplace: bool = True,
) -> nn.Module:
    """
    Convert all nn.Linear layers in a module to LNSLinear layers.
    
    物理的直観:
    既存のモデルの線形層をLNS版に置き換え。
    学習済みの重みをそのまま使用できます。
    
    This function recursively replaces all nn.Linear layers with
    LNSLinear layers while preserving the trained weights.
    
    Args:
        module: PyTorch module to convert
        inplace: If True, modify module in-place (default: True)
    
    Returns:
        Converted module with LNSLinear layers
    
    Example:
        >>> model = MyModel()
        >>> model.load_state_dict(torch.load('checkpoint.pt'))
        >>> model = convert_linear_to_lns(model)
        >>> model.eval()  # Switch to inference mode
        >>> # Now all linear layers use LNS kernel
    
    Requirements: 3.3, 3.4
    """
    if not inplace:
        import copy
        module = copy.deepcopy(module)
    
    # Recursively replace Linear layers
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            # Create LNSLinear with same parameters
            lns_linear = LNSLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
                use_lns=True,
            )
            
            # Copy weights and bias
            with torch.no_grad():
                lns_linear.weight.copy_(child.weight)
                if child.bias is not None:
                    lns_linear.bias.copy_(child.bias)
            
            # Replace the layer
            setattr(module, name, lns_linear)
        else:
            # Recursively convert child modules
            convert_linear_to_lns(child, inplace=True)
    
    return module


import math

__all__ = [
    'LNSLinear',
    'convert_linear_to_lns',
]
