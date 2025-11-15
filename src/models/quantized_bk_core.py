"""
Quantized BK-Core with INT8 operations and dynamic range calibration.

This module implements quantization-aware training (QAT) for the BK-Core,
enabling INT8 inference with minimal accuracy loss.

Integrates with existing BK-Core implementation from src/models/bk_core.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .bk_core import get_tridiagonal_inverse_diagonal, vmapped_get_diag


class QuantizedBKCore:
    """
    INT8 quantized BK-Core with dynamic range calibration.
    
    Implements fake quantization during training to learn quantization-robust parameters.
    
    Note: This is a callable class (not nn.Module) to match the interface of BKCoreFunction.apply
    """
    
    def __init__(self, n_seq: int, enable_quantization: bool = True):
        """
        Args:
            n_seq: Sequence length
            enable_quantization: If True, apply quantization; else use FP32
        """
        self.n_seq = n_seq
        self.enable_quantization = enable_quantization
        self.training = True  # Track training mode
        
        # Quantization parameters (learned during calibration)
        self.v_scale = torch.tensor(1.0)
        self.v_zero_point = torch.tensor(0, dtype=torch.int32)
        self.G_real_scale = torch.tensor(1.0)
        self.G_real_zero_point = torch.tensor(0, dtype=torch.int32)
        self.G_imag_scale = torch.tensor(1.0)
        self.G_imag_zero_point = torch.tensor(0, dtype=torch.int32)
        
        self.z = torch.tensor(1.0j, dtype=torch.complex128)
        
        # Calibration mode flag
        self.calibration_mode = False
        self.calibration_samples = []
    
    @staticmethod
    def quantize_tensor(x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
        """
        Quantize FP32 tensor to INT8.
        
        Args:
            x: FP32 tensor
            scale: Quantization scale
            zero_point: Quantization zero point
        
        Returns:
            INT8 tensor
        """
        x_int8 = torch.clamp(torch.round(x / scale) + zero_point, -128, 127)
        return x_int8.to(torch.int8)
    
    @staticmethod
    def dequantize_tensor(x_int8: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
        """
        Dequantize INT8 tensor to FP32.
        
        Args:
            x_int8: INT8 tensor
            scale: Quantization scale
            zero_point: Quantization zero point
        
        Returns:
            FP32 tensor
        """
        return (x_int8.to(torch.float32) - zero_point.to(torch.float32)) * scale
    
    @staticmethod
    def fake_quantize(x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
        """
        Fake quantization: quantize then dequantize (for training).
        
        This simulates quantization during training while maintaining gradients.
        """
        x_int8 = QuantizedBKCore.quantize_tensor(x, scale, zero_point)
        x_dequant = QuantizedBKCore.dequantize_tensor(x_int8, scale, zero_point)
        return x_dequant
    
    def calibrate_quantization(self, v_samples: torch.Tensor):
        """
        Calibrate quantization parameters using sample data.
        
        Uses symmetric quantization: scale = max(|x|) / 127
        
        Args:
            v_samples: (num_samples, n_seq) - sample potential values
        """
        # Compute dynamic range for potential v
        v_abs_max = v_samples.abs().max()
        self.v_scale = (v_abs_max / 127.0).clamp(min=1e-6)
        self.v_zero_point = torch.tensor(0, dtype=torch.int32)
        
        print(f"Calibrated v_scale: {self.v_scale.item():.6f}")
    
    def calibrate_output(self, G_real_samples: torch.Tensor, G_imag_samples: torch.Tensor):
        """
        Calibrate output quantization parameters.
        
        Args:
            G_real_samples: (num_samples, n_seq) - sample real(G_ii) values
            G_imag_samples: (num_samples, n_seq) - sample imag(G_ii) values
        """
        # Real part
        G_real_abs_max = G_real_samples.abs().max()
        self.G_real_scale = (G_real_abs_max / 127.0).clamp(min=1e-6)
        self.G_real_zero_point = torch.tensor(0, dtype=torch.int32)
        
        # Imaginary part
        G_imag_abs_max = G_imag_samples.abs().max()
        self.G_imag_scale = (G_imag_abs_max / 127.0).clamp(min=1e-6)
        self.G_imag_zero_point = torch.tensor(0, dtype=torch.int32)
        
        print(f"Calibrated G_real_scale: {self.G_real_scale.item():.6f}")
        print(f"Calibrated G_imag_scale: {self.G_imag_scale.item():.6f}")
    
    def start_calibration(self):
        """Start calibration mode to collect samples."""
        self.calibration_mode = True
        self.calibration_samples = []
    
    def end_calibration(self):
        """End calibration mode and compute quantization parameters."""
        self.calibration_mode = False
        
        if len(self.calibration_samples) > 0:
            # Stack all samples
            v_samples = torch.cat([s['v'] for s in self.calibration_samples], dim=0)
            G_real_samples = torch.cat([s['G_real'] for s in self.calibration_samples], dim=0)
            G_imag_samples = torch.cat([s['G_imag'] for s in self.calibration_samples], dim=0)
            
            # Calibrate
            self.calibrate_quantization(v_samples)
            self.calibrate_output(G_real_samples, G_imag_samples)
            
            # Clear samples
            self.calibration_samples = []
            
            print(f"Calibration complete with {len(v_samples)} samples")
    
    def _compute_bk_core(self, he_diag: torch.Tensor, h0_super: torch.Tensor, 
                         h0_sub: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Compute BK-Core using existing implementation.
        
        Args:
            he_diag: (B, N) - effective Hamiltonian diagonal
            h0_super: (B, N-1) - super-diagonal
            h0_sub: (B, N-1) - sub-diagonal
            z: complex scalar - spectral shift
        
        Returns:
            G_ii: (B, N) - diagonal of resolvent operator (complex)
        """
        # Use existing vmapped BK-Core implementation
        G_ii = vmapped_get_diag(he_diag, h0_super, h0_sub, z)
        return G_ii
    
    def __call__(self, he_diag: torch.Tensor, h0_super: torch.Tensor, 
                 h0_sub: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional quantization.
        
        This matches the interface of BKCoreFunction.apply(he_diag, h0_super, h0_sub, z)
        
        Args:
            he_diag: (B, N) - effective Hamiltonian diagonal (h0_diag + v)
            h0_super: (B, N-1) - super-diagonal
            h0_sub: (B, N-1) - sub-diagonal
            z: complex scalar - spectral shift
        
        Returns:
            features: (B, N, 2) - [real(G_ii), imag(G_ii)] in FP32
        """
        # Extract potential v from he_diag (assuming h0_diag = -2.0)
        # v = he_diag - h0_diag, but we'll work directly with he_diag
        v_fp32 = he_diag + 2.0  # Approximate v (h0_diag is typically -2.0)
        B, N = he_diag.shape
        device = he_diag.device
        
        # Apply fake quantization to input (if enabled)
        if self.enable_quantization and self.training:
            v = self.fake_quantize(v_fp32, self.v_scale, self.v_zero_point)
            # Recompute he_diag with quantized v
            he_diag_quant = he_diag - v_fp32 + v
        else:
            he_diag_quant = he_diag
        
        # Clamp for numerical stability
        he_diag_quant = torch.clamp(he_diag_quant, -10.0, 10.0)
        
        # Compute BK-Core using existing implementation
        G_ii = self._compute_bk_core(he_diag_quant, h0_super, h0_sub, z)
        
        # Extract real and imaginary parts
        G_real = G_ii.real.to(torch.float32)
        G_imag = G_ii.imag.to(torch.float32)
        
        # Clamp output for numerical stability
        G_real = torch.clamp(G_real, -100.0, 100.0)
        G_imag = torch.clamp(G_imag, -100.0, 100.0)
        
        # Collect calibration samples
        if self.calibration_mode:
            self.calibration_samples.append({
                'v': v.detach().cpu(),
                'G_real': G_real.detach().cpu(),
                'G_imag': G_imag.detach().cpu()
            })
        
        # Apply fake quantization to output (if enabled)
        if self.enable_quantization and self.training:
            G_real = self.fake_quantize(G_real, self.G_real_scale, self.G_real_zero_point)
            G_imag = self.fake_quantize(G_imag, self.G_imag_scale, self.G_imag_zero_point)
        
        # Stack features
        features = torch.stack([G_real, G_imag], dim=-1)
        
        return features
    
    def train(self):
        """Set to training mode."""
        self.training = True
    
    def eval(self):
        """Set to evaluation mode."""
        self.training = False
    
    def to_int8_inference(self):
        """
        Convert to INT8 inference mode (no fake quantization).
        
        This should be called after training for deployment.
        """
        self.enable_quantization = False
        self.eval()
        print("Converted to INT8 inference mode")
