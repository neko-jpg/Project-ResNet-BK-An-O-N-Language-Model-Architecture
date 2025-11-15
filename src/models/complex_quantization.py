"""
Complex number quantization with separate real/imaginary quantization.

This module provides per-channel quantization for complex-valued tensors,
enabling efficient INT8 representation of BK-Core outputs.

Integrates with existing BK-Core implementation from src/models/bk_core.py
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from .bk_core import vmapped_get_diag


class ComplexQuantizer:
    """
    Quantizer for complex-valued tensors with separate real/imaginary scales.
    
    Supports per-channel quantization for better accuracy.
    """
    
    def __init__(self, num_channels: int, per_channel: bool = True):
        """
        Args:
            num_channels: Number of channels (for per-channel quantization)
            per_channel: If True, use per-channel scales; else use per-tensor scale
        """
        self.num_channels = num_channels
        self.per_channel = per_channel
        self.training = True
        
        # Quantization parameters
        if per_channel:
            self.real_scale = torch.ones(num_channels)
            self.real_zero_point = torch.zeros(num_channels, dtype=torch.int32)
            self.imag_scale = torch.ones(num_channels)
            self.imag_zero_point = torch.zeros(num_channels, dtype=torch.int32)
        else:
            self.real_scale = torch.tensor(1.0)
            self.real_zero_point = torch.tensor(0, dtype=torch.int32)
            self.imag_scale = torch.tensor(1.0)
            self.imag_zero_point = torch.tensor(0, dtype=torch.int32)
        
        self.calibrated = False
    
    @staticmethod
    def quantize_tensor(x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
        """Quantize FP32 to INT8."""
        x_int8 = torch.clamp(torch.round(x / scale) + zero_point, -128, 127)
        return x_int8.to(torch.int8)
    
    @staticmethod
    def dequantize_tensor(x_int8: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
        """Dequantize INT8 to FP32."""
        return (x_int8.to(torch.float32) - zero_point.to(torch.float32)) * scale
    
    def calibrate(self, complex_samples: torch.Tensor):
        """
        Calibrate quantization parameters from sample data.
        
        Args:
            complex_samples: (num_samples, num_channels) - complex tensor samples
        """
        real_part = complex_samples.real
        imag_part = complex_samples.imag
        
        if self.per_channel:
            # Per-channel calibration
            for ch in range(self.num_channels):
                # Real part
                real_ch = real_part[:, ch]
                real_abs_max = real_ch.abs().max()
                self.real_scale[ch] = (real_abs_max / 127.0).clamp(min=1e-6)
                
                # Imaginary part
                imag_ch = imag_part[:, ch]
                imag_abs_max = imag_ch.abs().max()
                self.imag_scale[ch] = (imag_abs_max / 127.0).clamp(min=1e-6)
        else:
            # Per-tensor calibration
            real_abs_max = real_part.abs().max()
            self.real_scale = (real_abs_max / 127.0).clamp(min=1e-6)
            
            imag_abs_max = imag_part.abs().max()
            self.imag_scale = (imag_abs_max / 127.0).clamp(min=1e-6)
        
        self.calibrated = True
        print(f"Complex quantizer calibrated (per_channel={self.per_channel})")
        if self.per_channel:
            print(f"  Real scales: min={self.real_scale.min():.6f}, max={self.real_scale.max():.6f}")
            print(f"  Imag scales: min={self.imag_scale.min():.6f}, max={self.imag_scale.max():.6f}")
        else:
            print(f"  Real scale: {self.real_scale.item():.6f}")
            print(f"  Imag scale: {self.imag_scale.item():.6f}")
    
    def quantize(self, x_complex: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize complex tensor to INT8 (separate real/imag).
        
        Args:
            x_complex: (..., num_channels) - complex tensor
        
        Returns:
            real_int8: (..., num_channels) - quantized real part
            imag_int8: (..., num_channels) - quantized imaginary part
        """
        real_part = x_complex.real
        imag_part = x_complex.imag
        
        # Reshape scales for broadcasting
        if self.per_channel:
            scale_shape = [1] * (real_part.ndim - 1) + [self.num_channels]
            real_scale = self.real_scale.view(scale_shape)
            real_zero_point = self.real_zero_point.view(scale_shape)
            imag_scale = self.imag_scale.view(scale_shape)
            imag_zero_point = self.imag_zero_point.view(scale_shape)
        else:
            real_scale = self.real_scale
            real_zero_point = self.real_zero_point
            imag_scale = self.imag_scale
            imag_zero_point = self.imag_zero_point
        
        # Quantize
        real_int8 = self.quantize_tensor(real_part, real_scale, real_zero_point)
        imag_int8 = self.quantize_tensor(imag_part, imag_scale, imag_zero_point)
        
        return real_int8, imag_int8
    
    def dequantize(self, real_int8: torch.Tensor, imag_int8: torch.Tensor) -> torch.Tensor:
        """
        Dequantize INT8 to complex tensor.
        
        Args:
            real_int8: (..., num_channels) - quantized real part
            imag_int8: (..., num_channels) - quantized imaginary part
        
        Returns:
            x_complex: (..., num_channels) - complex tensor
        """
        # Reshape scales for broadcasting
        if self.per_channel:
            scale_shape = [1] * (real_int8.ndim - 1) + [self.num_channels]
            real_scale = self.real_scale.view(scale_shape)
            real_zero_point = self.real_zero_point.view(scale_shape)
            imag_scale = self.imag_scale.view(scale_shape)
            imag_zero_point = self.imag_zero_point.view(scale_shape)
        else:
            real_scale = self.real_scale
            real_zero_point = self.real_zero_point
            imag_scale = self.imag_scale
            imag_zero_point = self.imag_zero_point
        
        # Dequantize
        real_part = self.dequantize_tensor(real_int8, real_scale, real_zero_point)
        imag_part = self.dequantize_tensor(imag_int8, imag_scale, imag_zero_point)
        
        # Combine to complex
        x_complex = torch.complex(real_part, imag_part)
        
        return x_complex
    
    def fake_quantize(self, x_complex: torch.Tensor) -> torch.Tensor:
        """
        Fake quantization for training (quantize then dequantize).
        
        Args:
            x_complex: (..., num_channels) - complex tensor
        
        Returns:
            x_dequant: (..., num_channels) - fake-quantized complex tensor
        """
        real_int8, imag_int8 = self.quantize(x_complex)
        x_dequant = self.dequantize(real_int8, imag_int8)
        return x_dequant


class PerChannelQuantizedBKCore:
    """
    BK-Core with per-channel quantization for complex outputs.
    
    Each sequence position gets its own quantization scale for better accuracy.
    
    Note: This is a callable class (not nn.Module) to match the interface of BKCoreFunction.apply
    """
    
    def __init__(self, n_seq: int, enable_quantization: bool = True):
        """
        Args:
            n_seq: Sequence length (number of channels)
            enable_quantization: If True, apply quantization
        """
        self.n_seq = n_seq
        self.enable_quantization = enable_quantization
        self.training = True  # Track training mode
        
        # Complex quantizer for G_ii output
        self.complex_quantizer = ComplexQuantizer(num_channels=n_seq, per_channel=True)
        
        # Input quantization (per-tensor for potential v)
        self.v_scale = torch.tensor(1.0)
        self.v_zero_point = torch.tensor(0, dtype=torch.int32)
        
        self.z = torch.tensor(1.0j, dtype=torch.complex128)
        
        # Calibration
        self.calibration_mode = False
        self.calibration_samples_v = []
        self.calibration_samples_G = []
    
    def start_calibration(self):
        """Start calibration mode."""
        self.calibration_mode = True
        self.calibration_samples_v = []
        self.calibration_samples_G = []
    
    def end_calibration(self):
        """End calibration and compute quantization parameters."""
        self.calibration_mode = False
        
        if len(self.calibration_samples_v) > 0:
            # Calibrate input (v)
            v_samples = torch.cat(self.calibration_samples_v, dim=0)
            v_abs_max = v_samples.abs().max()
            self.v_scale = (v_abs_max / 127.0).clamp(min=1e-6)
            
            # Calibrate output (G_ii) - per-channel
            G_samples = torch.cat(self.calibration_samples_G, dim=0)
            self.complex_quantizer.calibrate(G_samples)
            
            # Clear samples
            self.calibration_samples_v = []
            self.calibration_samples_G = []
            
            print(f"Per-channel calibration complete with {len(v_samples)} samples")
            print(f"  v_scale: {self.v_scale.item():.6f}")
    
    def _compute_bk_core(self, he_diag: torch.Tensor, h0_super: torch.Tensor, 
                         h0_sub: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Compute BK-Core using existing implementation."""
        # Use existing vmapped BK-Core implementation
        G_ii = vmapped_get_diag(he_diag, h0_super, h0_sub, z)
        return G_ii
    
    def __call__(self, he_diag: torch.Tensor, h0_super: torch.Tensor, 
                 h0_sub: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with per-channel quantization.
        
        This matches the interface of BKCoreFunction.apply(he_diag, h0_super, h0_sub, z)
        
        Args:
            he_diag: (B, N) - effective Hamiltonian diagonal
            h0_super: (B, N-1) - super-diagonal
            h0_sub: (B, N-1) - sub-diagonal
            z: complex scalar - spectral shift
        
        Returns:
            features: (B, N, 2) - [real(G_ii), imag(G_ii)] in FP32
        """
        # Extract potential v from he_diag
        v_fp32 = he_diag + 2.0  # Approximate v (h0_diag is typically -2.0)
        B, N = he_diag.shape
        device = he_diag.device
        
        # Fake quantize input
        if self.enable_quantization and self.training:
            v_int8 = ComplexQuantizer.quantize_tensor(v_fp32, self.v_scale, self.v_zero_point)
            v = ComplexQuantizer.dequantize_tensor(v_int8, self.v_scale, self.v_zero_point)
            # Recompute he_diag with quantized v
            he_diag_quant = he_diag - v_fp32 + v
        else:
            he_diag_quant = he_diag
        
        he_diag_quant = torch.clamp(he_diag_quant, -10.0, 10.0)
        
        # Compute BK-Core using existing implementation
        G_ii = self._compute_bk_core(he_diag_quant, h0_super, h0_sub, z)
        
        # Collect calibration samples
        if self.calibration_mode:
            self.calibration_samples_v.append(v.detach().cpu())
            self.calibration_samples_G.append(G_ii.detach().cpu())
        
        # Fake quantize output (per-channel)
        if self.enable_quantization and self.training and self.complex_quantizer.calibrated:
            G_ii = self.complex_quantizer.fake_quantize(G_ii)
        
        # Convert to features
        G_real = G_ii.real.to(torch.float32)
        G_imag = G_ii.imag.to(torch.float32)
        
        G_real = torch.clamp(G_real, -100.0, 100.0)
        G_imag = torch.clamp(G_imag, -100.0, 100.0)
        
        features = torch.stack([G_real, G_imag], dim=-1)
        
        return features
    
    def train(self):
        """Set to training mode."""
        self.training = True
        self.complex_quantizer.training = True
    
    def eval(self):
        """Set to evaluation mode."""
        self.training = False
        self.complex_quantizer.training = False
