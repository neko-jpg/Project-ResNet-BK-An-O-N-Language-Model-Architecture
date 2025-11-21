"""
Phase 3.1: Complex Dynamics Foundation

複素ニューラルネットワークの基本コンポーネント。
実部（振幅/強度）と虚部（位相/文脈）を独立かつ相互作用させながら処理する。

Mathematical Basis:
    z = x + iy = r * e^(iθ)
    - Magnitude (r): Information importance
    - Phase (θ): Contextual relationship / Interference

Features:
    - ComplexLinear: Wirtinger calculus compliant linear layer
    - ComplexReLU: Phase-preserving activation (CReLU or ModReLU)
    - ComplexLayerNorm: Normalization in complex plane
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class ComplexLinear(nn.Module):
    """
    複素線形層: W * z + b
    W = A + iB, z = x + iy
    Wz = (Ax - By) + i(Bx + Ay)
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 実部と虚部の重みを独立して定義（実数空間で最適化）
        self.weight_real = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_imag = nn.Parameter(torch.Tensor(out_features, in_features))
        
        if bias:
            self.bias_real = nn.Parameter(torch.Tensor(out_features))
            self.bias_imag = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        # 複素数の初期化: 位相はランダム、振幅はRayleigh分布
        # ここでは簡易的にHe初期化を適用
        nn.init.kaiming_uniform_(self.weight_real, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_imag, a=math.sqrt(5))
        if self.bias_real is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_real)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_real, -bound, bound)
            nn.init.uniform_(self.bias_imag, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input: (..., in_features) complex64/128 or (..., in_features, 2)
        if input.is_complex():
            x = input.real
            y = input.imag
        elif input.dim() == 3 and input.shape[-1] == 2: # (B, N, 2) format from Phase 2
             x = input[..., 0]
             y = input[..., 1]
        else:
            # Assume real input, promote to complex
            x = input
            y = torch.zeros_like(input)

        # Complex Multiplication
        # Re = W_r * x - W_i * y
        # Im = W_i * x + W_r * y
        
        out_real = F.linear(x, self.weight_real, self.bias_real) - \
                   F.linear(y, self.weight_imag, None)
                   
        out_imag = F.linear(x, self.weight_imag, self.bias_imag) + \
                   F.linear(y, self.weight_real, None)
        
        return torch.complex(out_real, out_imag)

class ModReLU(nn.Module):
    """
    モジュライReLU (ModReLU)
    振幅に対してReLUを適用し、位相は保持する。
    z' = ReLU(|z| + b) * z / |z|
    
    物理的意味: 信号の強度はフィルタリングされるが、文脈（位相）の流れは乱さない。
    """
    def __init__(self, features):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, N, D) complex
        mag = torch.abs(z)
        
        # Avoid division by zero
        normalized = z / (mag + 1e-6)
        
        # Activate magnitude
        new_mag = F.relu(mag + self.bias)
        
        return new_mag * normalized

import math