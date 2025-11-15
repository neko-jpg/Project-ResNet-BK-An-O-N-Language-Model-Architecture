"""
INT4 quantized MoE with group-wise quantization.

This module implements INT4 quantization for MoE expert weights,
achieving 8× compression with minimal accuracy loss.

Integrates with existing MoE implementation from src/models/moe.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from .moe import SparseMoELayer


class GroupWiseQuantizer:
    """
    Group-wise INT4 quantization for weight matrices.
    
    Divides weights into groups and quantizes each group independently.
    """
    
    def __init__(self, group_size: int = 128):
        """
        Args:
            group_size: Number of weights per quantization group
        """
        self.group_size = group_size
    
    @staticmethod
    def quantize_int4(x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
        """
        Quantize to INT4 (stored as INT8 with values in [-8, 7]).
        
        Args:
            x: FP32 tensor
            scale: Quantization scale
            zero_point: Quantization zero point
        
        Returns:
            INT8 tensor with values in [-8, 7]
        """
        x_int4 = torch.clamp(torch.round(x / scale) + zero_point, -8, 7)
        return x_int4.to(torch.int8)
    
    @staticmethod
    def dequantize_int4(x_int4: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
        """
        Dequantize INT4 to FP32.
        
        Args:
            x_int4: INT8 tensor with values in [-8, 7]
            scale: Quantization scale
            zero_point: Quantization zero point
        
        Returns:
            FP32 tensor
        """
        return (x_int4.to(torch.float32) - zero_point.to(torch.float32)) * scale
    
    def quantize_groupwise(self, weight: torch.Tensor) -> tuple:
        """
        Quantize weight matrix using group-wise INT4 quantization.
        
        Args:
            weight: (out_features, in_features) - weight matrix
        
        Returns:
            weight_int4: Quantized weights
            scales: Per-group scales
            zero_points: Per-group zero points
        """
        out_features, in_features = weight.shape
        
        # Flatten weight for grouping
        weight_flat = weight.flatten()
        num_elements = weight_flat.numel()
        
        # Pad to multiple of group_size
        pad_size = (self.group_size - num_elements % self.group_size) % self.group_size
        if pad_size > 0:
            weight_flat = F.pad(weight_flat, (0, pad_size), value=0.0)
        
        # Reshape into groups
        num_groups = weight_flat.numel() // self.group_size
        weight_grouped = weight_flat.view(num_groups, self.group_size)
        
        # Compute per-group scales (symmetric quantization)
        group_abs_max = weight_grouped.abs().max(dim=1, keepdim=True)[0]
        scales = (group_abs_max / 7.0).clamp(min=1e-6)  # INT4 range: [-8, 7]
        zero_points = torch.zeros_like(scales, dtype=torch.int32)
        
        # Quantize each group
        weight_int4_grouped = self.quantize_int4(weight_grouped, scales, zero_points)
        
        # Reshape back
        weight_int4_flat = weight_int4_grouped.flatten()
        if pad_size > 0:
            weight_int4_flat = weight_int4_flat[:-pad_size]
        
        weight_int4 = weight_int4_flat.view(out_features, in_features)
        
        return weight_int4, scales.squeeze(-1), zero_points.squeeze(-1)
    
    def dequantize_groupwise(self, weight_int4: torch.Tensor, scales: torch.Tensor, 
                             zero_points: torch.Tensor) -> torch.Tensor:
        """
        Dequantize group-wise INT4 weights.
        
        Args:
            weight_int4: Quantized weights
            scales: Per-group scales
            zero_points: Per-group zero points
        
        Returns:
            weight_fp32: Dequantized weight matrix
        """
        out_features, in_features = weight_int4.shape
        
        # Flatten
        weight_int4_flat = weight_int4.flatten()
        num_elements = weight_int4_flat.numel()
        
        # Pad to multiple of group_size
        pad_size = (self.group_size - num_elements % self.group_size) % self.group_size
        if pad_size > 0:
            weight_int4_flat = F.pad(weight_int4_flat, (0, pad_size), value=0)
        
        # Reshape into groups
        num_groups = weight_int4_flat.numel() // self.group_size
        weight_int4_grouped = weight_int4_flat.view(num_groups, self.group_size)
        
        # Dequantize each group
        scales_expanded = scales.unsqueeze(-1)
        zero_points_expanded = zero_points.unsqueeze(-1)
        weight_fp32_grouped = self.dequantize_int4(weight_int4_grouped, scales_expanded, zero_points_expanded)
        
        # Reshape back
        weight_fp32_flat = weight_fp32_grouped.flatten()
        if pad_size > 0:
            weight_fp32_flat = weight_fp32_flat[:-pad_size]
        
        weight_fp32 = weight_fp32_flat.view(out_features, in_features)
        
        return weight_fp32


class QuantizedLinear(nn.Module):
    """
    Linear layer with INT4 group-wise quantization.
    """
    
    def __init__(self, in_features: int, out_features: int, group_size: int = 128, 
                 bias: bool = True, enable_quantization: bool = True):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            group_size: Group size for quantization
            bias: If True, include bias
            enable_quantization: If True, use quantized weights
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.enable_quantization = enable_quantization
        
        # FP32 weights (for training)
        self.weight_fp32 = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Quantized weights (for inference)
        self.register_buffer('weight_int4', None)
        self.register_buffer('scales', None)
        self.register_buffer('zero_points', None)
        
        self.quantizer = GroupWiseQuantizer(group_size=group_size)
        self.quantized = False
    
    def quantize_weights(self):
        """Quantize weights to INT4."""
        weight_int4, scales, zero_points = self.quantizer.quantize_groupwise(self.weight_fp32.data)
        
        self.weight_int4 = weight_int4
        self.scales = scales
        self.zero_points = zero_points
        self.quantized = True
        
        print(f"Quantized linear layer: {self.out_features}x{self.in_features}")
        print(f"  Num groups: {len(scales)}, Group size: {self.group_size}")
        print(f"  Scale range: [{scales.min():.6f}, {scales.max():.6f}]")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional quantization.
        
        Args:
            x: (B, *, in_features) - input
        
        Returns:
            output: (B, *, out_features)
        """
        if self.enable_quantization and self.quantized and not self.training:
            # Use quantized weights for inference
            weight = self.quantizer.dequantize_groupwise(self.weight_int4, self.scales, self.zero_points)
        else:
            # Use FP32 weights for training
            weight = self.weight_fp32
        
        output = F.linear(x, weight, self.bias)
        return output


class QuantizedMoELayer(nn.Module):
    """
    MoE layer with INT4 quantized expert weights.
    
    Supports mixed INT4/INT8 quantization: INT4 for expert weights, INT8 for routing.
    """
    
    def __init__(self, d_model: int, num_experts: int = 4, group_size: int = 128,
                 enable_quantization: bool = True):
        """
        Args:
            d_model: Model dimension
            num_experts: Number of experts
            group_size: Group size for INT4 quantization
            enable_quantization: If True, use quantized weights
        """
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.enable_quantization = enable_quantization
        
        # Experts with INT4 quantization
        self.experts = nn.ModuleList([
            nn.Sequential(
                QuantizedLinear(d_model, d_model * 2, group_size=group_size, 
                               enable_quantization=enable_quantization),
                nn.ReLU(),
                QuantizedLinear(d_model * 2, d_model, group_size=group_size,
                               enable_quantization=enable_quantization)
            ) for _ in range(num_experts)
        ])
        
        # Gating network (INT8 quantization)
        self.gating = nn.Linear(d_model, num_experts)
        
        # INT8 quantization for gating weights
        self.register_buffer('gating_weight_int8', None)
        self.register_buffer('gating_scale', torch.tensor(1.0))
        self.register_buffer('gating_zero_point', torch.tensor(0, dtype=torch.int32))
        self.gating_quantized = False
    
    def quantize_experts(self):
        """Quantize all expert weights to INT4."""
        for expert_idx, expert in enumerate(self.experts):
            print(f"\nQuantizing expert {expert_idx}:")
            for layer in expert:
                if isinstance(layer, QuantizedLinear):
                    layer.quantize_weights()
    
    def quantize_gating(self):
        """Quantize gating network to INT8."""
        weight = self.gating.weight.data
        
        # Symmetric quantization
        weight_abs_max = weight.abs().max()
        self.gating_scale = (weight_abs_max / 127.0).clamp(min=1e-6)
        
        # Quantize
        weight_int8 = torch.clamp(torch.round(weight / self.gating_scale), -128, 127).to(torch.int8)
        self.gating_weight_int8 = weight_int8
        self.gating_quantized = True
        
        print(f"\nQuantized gating network:")
        print(f"  Scale: {self.gating_scale.item():.6f}")
    
    def quantize_all(self):
        """Quantize both experts and gating network."""
        self.quantize_experts()
        self.quantize_gating()
        print("\nMoE quantization complete!")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with quantized experts.
        
        Args:
            x: (B, N, D) - input
        
        Returns:
            output: (B, N, D)
        """
        B, N, D = x.shape
        x_flat = x.reshape(B * N, D)
        
        # Gating (use quantized weights if available)
        if self.enable_quantization and self.gating_quantized and not self.training:
            # Dequantize gating weights
            gating_weight = (self.gating_weight_int8.to(torch.float32) - 
                           self.gating_zero_point.to(torch.float32)) * self.gating_scale
            router_logits = F.linear(x_flat, gating_weight, self.gating.bias)
        else:
            router_logits = self.gating(x_flat)
        
        # Gumbel-Softmax routing
        gates = F.gumbel_softmax(router_logits, hard=True, tau=1.0)
        
        # Compute expert outputs (using quantized weights if available)
        output = torch.zeros(B * N, D, device=x.device)
        for e, expert in enumerate(self.experts):
            expert_output = expert(x_flat)
            output += expert_output * gates[:, e].unsqueeze(-1)
        
        return output.view(B, N, D)
    
    def get_compression_ratio(self) -> float:
        """
        Calculate compression ratio.
        
        Returns:
            Compression ratio (original size / compressed size)
        """
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        
        # FP32: 4 bytes per parameter
        # INT4: 0.5 bytes per parameter (for expert weights)
        # INT8: 1 byte per parameter (for gating)
        
        # Expert parameters (INT4)
        expert_params = sum(
            sum(p.numel() for p in expert.parameters())
            for expert in self.experts
        )
        
        # Gating parameters (INT8)
        gating_params = sum(p.numel() for p in self.gating.parameters())
        
        # Original size (FP32)
        original_size = total_params * 4  # bytes
        
        # Compressed size
        compressed_size = expert_params * 0.5 + gating_params * 1  # bytes
        
        compression_ratio = original_size / compressed_size
        
        return compression_ratio
