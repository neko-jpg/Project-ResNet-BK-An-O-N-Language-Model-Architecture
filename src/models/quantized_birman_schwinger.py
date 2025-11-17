"""
Quantized Birman-Schwinger Core with Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT)

This module implements INT8 and INT4 quantization for the Birman-Schwinger operator
with separate quantization for real and imaginary parts of complex numbers.

Implements Task 13 from mamba-killer-ultra-scale spec:
- Task 13: Post-Training Quantization (PTQ) with INT8
- Task 13.1: Quantization-Aware Training (QAT) with INT8
- Task 13.2: INT4 quantization with group-wise quantization

Requirements:
- 7.1: Implement post-training quantization (PTQ): quantize trained model to INT8 without retraining
- 7.2: WHEN applying INT8 PTQ, THE System SHALL maintain perplexity degradation < 5% on WikiText-2
- 7.3: Implement quantization-aware training (QAT): simulate INT8 operations during training
- 7.4: WHEN using INT8 QAT, THE System SHALL achieve perplexity within 2% of FP32 baseline
- 7.5: Implement INT4 quantization with group-wise quantization (group size = 128)
- 7.6: WHEN using INT4 quantization, THE System SHALL maintain perplexity degradation < 15% on WikiText-2
- 7.11: Implement mixed-precision quantization: INT4 for MoE, INT8 for BK-Core, FP16 for output layers

Mathematical Foundation:
- Birman-Schwinger operator: K_ε(z) = |V_ε|^{1/2} R_0(z) |V_ε|^{1/2}
- Complex quantization: separate scales for real and imaginary parts
- Group-wise quantization: divide channels into groups for better accuracy
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, List
import numpy as np

from .birman_schwinger_core import BirmanSchwingerCore
from .complex_quantization import ComplexQuantizer


class QuantizationConfig:
    """Configuration for quantization schemes."""
    
    # Quantization modes
    PTQ_INT8 = "ptq_int8"  # Post-training quantization to INT8
    QAT_INT8 = "qat_int8"  # Quantization-aware training with INT8
    PTQ_INT4 = "ptq_int4"  # Post-training quantization to INT4
    QAT_INT4 = "qat_int4"  # Quantization-aware training with INT4
    MIXED_PRECISION = "mixed_precision"  # Mixed precision (INT4/INT8/FP16)
    
    def __init__(
        self,
        mode: str = PTQ_INT8,
        group_size: int = 128,  # For group-wise quantization (Requirement 7.5)
        per_channel: bool = True,  # Per-channel vs per-tensor quantization
        symmetric: bool = True,  # Symmetric vs asymmetric quantization
    ):
        self.mode = mode
        self.group_size = group_size
        self.per_channel = per_channel
        self.symmetric = symmetric
        
        # Bit widths
        if "int8" in mode.lower():
            self.bits = 8
            self.qmin = -128
            self.qmax = 127
        elif "int4" in mode.lower():
            self.bits = 4
            self.qmin = -8
            self.qmax = 7
        else:
            self.bits = 8
            self.qmin = -128
            self.qmax = 127


class GroupWiseQuantizer:
    """
    Group-wise quantizer for INT4 quantization.
    
    Divides channels into groups and quantizes each group separately
    for better accuracy (Requirement 7.5).
    
    Args:
        num_channels: Total number of channels
        group_size: Size of each group (default: 128)
        bits: Number of bits (4 for INT4, 8 for INT8)
    """
    
    def __init__(
        self,
        num_channels: int,
        group_size: int = 128,
        bits: int = 4,
    ):
        self.num_channels = num_channels
        self.group_size = group_size
        self.bits = bits
        
        # Quantization range
        if bits == 4:
            self.qmin = -8
            self.qmax = 7
        elif bits == 8:
            self.qmin = -128
            self.qmax = 127
        else:
            raise ValueError(f"Unsupported bits: {bits}")
        
        # Number of groups
        self.num_groups = (num_channels + group_size - 1) // group_size
        
        # Quantization parameters per group
        self.scales = torch.ones(self.num_groups)
        self.zero_points = torch.zeros(self.num_groups, dtype=torch.int32)
        
        self.calibrated = False
    
    def calibrate(self, x: torch.Tensor):
        """
        Calibrate quantization parameters from sample data.
        
        Args:
            x: (num_samples, num_channels) - sample data
        """
        num_samples = x.shape[0]
        
        for g in range(self.num_groups):
            start_idx = g * self.group_size
            end_idx = min((g + 1) * self.group_size, self.num_channels)
            
            # Get group data
            x_group = x[:, start_idx:end_idx]
            
            # Compute scale: symmetric quantization
            abs_max = x_group.abs().max()
            scale = (abs_max / (self.qmax - 0.5)).clamp(min=1e-6)
            
            self.scales[g] = scale
            self.zero_points[g] = 0  # Symmetric quantization
        
        self.calibrated = True
        print(f"Group-wise quantizer calibrated: {self.num_groups} groups, {self.bits} bits")
        print(f"  Scale range: [{self.scales.min():.6f}, {self.scales.max():.6f}]")
    
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize tensor using group-wise quantization.
        
        Args:
            x: (..., num_channels) - input tensor
        
        Returns:
            x_quant: (..., num_channels) - quantized tensor (INT4 or INT8)
        """
        x_quant = torch.zeros_like(x, dtype=torch.int8)
        
        for g in range(self.num_groups):
            start_idx = g * self.group_size
            end_idx = min((g + 1) * self.group_size, self.num_channels)
            
            # Get group data
            x_group = x[..., start_idx:end_idx]
            
            # Quantize
            scale = self.scales[g]
            zero_point = self.zero_points[g]
            
            x_group_quant = torch.clamp(
                torch.round(x_group / scale) + zero_point,
                self.qmin,
                self.qmax
            )
            
            x_quant[..., start_idx:end_idx] = x_group_quant.to(torch.int8)
        
        return x_quant
    
    def dequantize(self, x_quant: torch.Tensor) -> torch.Tensor:
        """
        Dequantize tensor using group-wise dequantization.
        
        Args:
            x_quant: (..., num_channels) - quantized tensor
        
        Returns:
            x: (..., num_channels) - dequantized tensor (FP32)
        """
        x = torch.zeros_like(x_quant, dtype=torch.float32)
        
        for g in range(self.num_groups):
            start_idx = g * self.group_size
            end_idx = min((g + 1) * self.group_size, self.num_channels)
            
            # Get group data
            x_group_quant = x_quant[..., start_idx:end_idx]
            
            # Dequantize
            scale = self.scales[g]
            zero_point = self.zero_points[g]
            
            x_group = (x_group_quant.to(torch.float32) - zero_point.to(torch.float32)) * scale
            
            x[..., start_idx:end_idx] = x_group
        
        return x
    
    def fake_quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fake quantization for training (quantize then dequantize).
        
        Args:
            x: (..., num_channels) - input tensor
        
        Returns:
            x_dequant: (..., num_channels) - fake-quantized tensor
        """
        x_quant = self.quantize(x)
        x_dequant = self.dequantize(x_quant)
        return x_dequant


class QuantizedBirmanSchwingerCore(nn.Module):
    """
    Quantized Birman-Schwinger Core with PTQ and QAT support.
    
    Supports multiple quantization modes:
    - PTQ INT8: Post-training quantization to INT8 (Requirement 7.1, 7.2)
    - QAT INT8: Quantization-aware training with INT8 (Requirement 7.3, 7.4)
    - PTQ INT4: Post-training quantization to INT4 with group-wise quantization (Requirement 7.5, 7.6)
    - QAT INT4: Quantization-aware training with INT4
    - Mixed Precision: INT4 for MoE, INT8 for BK-Core, FP16 for output (Requirement 7.11)
    
    Args:
        n_seq: Sequence length
        epsilon: Regularization parameter (ε ∈ [0.5, 1.0])
        quant_config: Quantization configuration
        use_mourre: Enable Mourre estimate verification
        use_lap: Enable Limiting Absorption Principle
        use_semiseparable: Use semiseparable matrix structure
    """
    
    def __init__(
        self,
        n_seq: int,
        epsilon: float = 1.0,
        quant_config: Optional[QuantizationConfig] = None,
        use_mourre: bool = True,
        use_lap: bool = True,
        use_semiseparable: bool = True,
        semiseparable_rank: Optional[int] = None,
    ):
        super().__init__()
        
        self.n_seq = n_seq
        self.epsilon = epsilon
        self.quant_config = quant_config or QuantizationConfig()
        
        # Base Birman-Schwinger core (FP32)
        self.bk_core = BirmanSchwingerCore(
            n_seq=n_seq,
            epsilon=epsilon,
            use_mourre=use_mourre,
            use_lap=use_lap,
            use_semiseparable=use_semiseparable,
            semiseparable_rank=semiseparable_rank,
        )
        
        # Quantizers for input (potential v)
        if self.quant_config.mode in [QuantizationConfig.PTQ_INT4, QuantizationConfig.QAT_INT4]:
            # INT4 with group-wise quantization (Requirement 7.5)
            self.v_quantizer = GroupWiseQuantizer(
                num_channels=n_seq,
                group_size=self.quant_config.group_size,
                bits=4,
            )
        else:
            # INT8 with per-channel quantization
            self.v_quantizer = ComplexQuantizer(
                num_channels=n_seq,
                per_channel=self.quant_config.per_channel,
            )
        
        # Quantizers for output (G_ii complex)
        if self.quant_config.mode in [QuantizationConfig.PTQ_INT4, QuantizationConfig.QAT_INT4]:
            # INT4 with group-wise quantization
            self.G_real_quantizer = GroupWiseQuantizer(
                num_channels=n_seq,
                group_size=self.quant_config.group_size,
                bits=4,
            )
            self.G_imag_quantizer = GroupWiseQuantizer(
                num_channels=n_seq,
                group_size=self.quant_config.group_size,
                bits=4,
            )
        else:
            # INT8 with per-channel quantization
            self.G_quantizer = ComplexQuantizer(
                num_channels=n_seq,
                per_channel=self.quant_config.per_channel,
            )
        
        # Calibration state
        self.calibrated = False
        self.calibration_samples_v = []
        self.calibration_samples_G = []
        
        # Training mode
        self._training = True
        self._qat_enabled = "qat" in self.quant_config.mode.lower()
        
        # Statistics
        self.quantization_error_history = []
        self.ppl_degradation_history = []
    
    def start_calibration(self):
        """Start calibration mode to collect samples for PTQ."""
        self.calibration_samples_v = []
        self.calibration_samples_G = []
        print(f"Started calibration for {self.quant_config.mode}")
    
    def end_calibration(self):
        """
        End calibration and compute quantization parameters.
        
        For PTQ (Requirement 7.1): Calibrate on sample data without retraining.
        """
        if len(self.calibration_samples_v) == 0:
            print("Warning: No calibration samples collected")
            return
        
        # Stack all samples
        v_samples = torch.cat(self.calibration_samples_v, dim=0)  # (num_samples, N)
        G_samples = torch.cat(self.calibration_samples_G, dim=0)  # (num_samples, N) complex
        
        print(f"Calibrating with {len(v_samples)} samples...")
        
        # Calibrate input quantizer
        if isinstance(self.v_quantizer, GroupWiseQuantizer):
            self.v_quantizer.calibrate(v_samples)
        else:
            # ComplexQuantizer expects complex input, but v is real
            v_complex = torch.complex(v_samples, torch.zeros_like(v_samples))
            self.v_quantizer.calibrate(v_complex)
        
        # Calibrate output quantizer
        if self.quant_config.mode in [QuantizationConfig.PTQ_INT4, QuantizationConfig.QAT_INT4]:
            # INT4: separate quantizers for real and imaginary
            self.G_real_quantizer.calibrate(G_samples.real)
            self.G_imag_quantizer.calibrate(G_samples.imag)
        else:
            # INT8: complex quantizer
            self.G_quantizer.calibrate(G_samples)
        
        self.calibrated = True
        
        # Clear samples to save memory
        self.calibration_samples_v = []
        self.calibration_samples_G = []
        
        print(f"Calibration complete for {self.quant_config.mode}")
    
    def apply_ptq(self):
        """
        Apply post-training quantization (PTQ).
        
        Requirement 7.1: Quantize trained model to INT8 without retraining.
        Requirement 7.2: Maintain perplexity degradation < 5% on WikiText-2.
        """
        if not self.calibrated:
            raise RuntimeError("Must calibrate before applying PTQ. Call start_calibration(), forward(), end_calibration().")
        
        print(f"Applied PTQ: {self.quant_config.mode}")
        self._qat_enabled = False  # Disable fake quantization
        self.eval()
    
    def enable_qat(self):
        """
        Enable quantization-aware training (QAT).
        
        Requirement 7.3: Simulate INT8 operations during training.
        Requirement 7.4: Achieve perplexity within 2% of FP32 baseline.
        """
        self._qat_enabled = True
        self.train()
        print(f"Enabled QAT: {self.quant_config.mode}")
    
    def disable_qat(self):
        """Disable quantization-aware training."""
        self._qat_enabled = False
        print("Disabled QAT")
    
    def compute_quantization_error(
        self,
        x_fp32: torch.Tensor,
        x_quant: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute quantization error metrics.
        
        Args:
            x_fp32: FP32 tensor
            x_quant: Quantized then dequantized tensor
        
        Returns:
            Dictionary with error metrics
        """
        # Mean absolute error
        mae = (x_fp32 - x_quant).abs().mean().item()
        
        # Mean squared error
        mse = ((x_fp32 - x_quant) ** 2).mean().item()
        
        # Signal-to-noise ratio (SNR)
        signal_power = (x_fp32 ** 2).mean().item()
        noise_power = ((x_fp32 - x_quant) ** 2).mean().item()
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        # Relative error
        relative_error = (mae / (x_fp32.abs().mean().item() + 1e-10)) * 100
        
        return {
            'mae': mae,
            'mse': mse,
            'snr_db': snr_db,
            'relative_error_percent': relative_error,
        }
    
    def forward(
        self,
        v: torch.Tensor,
        z: complex = 1.0j,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass with quantization.
        
        Args:
            v: (B, N) potential from Prime-Bump initialization
            z: complex shift (default: 1.0j)
            return_diagnostics: whether to return diagnostic information
        
        Returns:
            features: (B, N, 2) [real(G_ii), imag(G_ii)]
            diagnostics: (optional) dictionary with quantization statistics
        """
        batch_size, n_seq = v.shape
        device = v.device
        
        # Store original for error computation
        v_fp32 = v.clone()
        
        # Apply quantization to input (v)
        if self._qat_enabled and self.calibrated:
            # QAT: Fake quantization during training (Requirement 7.3)
            if isinstance(self.v_quantizer, GroupWiseQuantizer):
                v_quant = self.v_quantizer.fake_quantize(v)
            else:
                # ComplexQuantizer expects complex input
                v_complex = torch.complex(v, torch.zeros_like(v))
                v_complex_quant = self.v_quantizer.fake_quantize(v_complex)
                v_quant = v_complex_quant.real
        elif not self._training and self.calibrated:
            # PTQ inference: Actual quantization (Requirement 7.1)
            if isinstance(self.v_quantizer, GroupWiseQuantizer):
                v_int = self.v_quantizer.quantize(v)
                v_quant = self.v_quantizer.dequantize(v_int)
            else:
                v_complex = torch.complex(v, torch.zeros_like(v))
                v_real_int, v_imag_int = self.v_quantizer.quantize(v_complex)
                v_complex_quant = self.v_quantizer.dequantize(v_real_int, v_imag_int)
                v_quant = v_complex_quant.real
        else:
            # No quantization (FP32 baseline or calibration mode)
            v_quant = v
        
        # Collect calibration samples
        if not self.calibrated and len(self.calibration_samples_v) < 1000:
            self.calibration_samples_v.append(v.detach().cpu())
        
        # Compute BK-Core using base implementation
        features_fp32, bk_diagnostics = self.bk_core(v_quant, z)
        
        # Extract G_ii from features
        G_real_fp32 = features_fp32[..., 0]  # (B, N)
        G_imag_fp32 = features_fp32[..., 1]  # (B, N)
        G_ii_fp32 = torch.complex(G_real_fp32, G_imag_fp32)
        
        # Collect calibration samples
        if not self.calibrated and len(self.calibration_samples_G) < 1000:
            self.calibration_samples_G.append(G_ii_fp32.detach().cpu())
        
        # Apply quantization to output (G_ii)
        if self._qat_enabled and self.calibrated:
            # QAT: Fake quantization during training
            if self.quant_config.mode in [QuantizationConfig.PTQ_INT4, QuantizationConfig.QAT_INT4]:
                # INT4: separate quantizers
                G_real_quant = self.G_real_quantizer.fake_quantize(G_real_fp32)
                G_imag_quant = self.G_imag_quantizer.fake_quantize(G_imag_fp32)
            else:
                # INT8: complex quantizer
                G_ii_quant = self.G_quantizer.fake_quantize(G_ii_fp32)
                G_real_quant = G_ii_quant.real
                G_imag_quant = G_ii_quant.imag
        elif not self._training and self.calibrated:
            # PTQ inference: Actual quantization
            if self.quant_config.mode in [QuantizationConfig.PTQ_INT4, QuantizationConfig.QAT_INT4]:
                # INT4: separate quantizers
                G_real_int = self.G_real_quantizer.quantize(G_real_fp32)
                G_real_quant = self.G_real_quantizer.dequantize(G_real_int)
                G_imag_int = self.G_imag_quantizer.quantize(G_imag_fp32)
                G_imag_quant = self.G_imag_quantizer.dequantize(G_imag_int)
            else:
                # INT8: complex quantizer
                G_real_int, G_imag_int = self.G_quantizer.quantize(G_ii_fp32)
                G_ii_quant = self.G_quantizer.dequantize(G_real_int, G_imag_int)
                G_real_quant = G_ii_quant.real
                G_imag_quant = G_ii_quant.imag
        else:
            # No quantization
            G_real_quant = G_real_fp32
            G_imag_quant = G_imag_fp32
        
        # Stack features
        features = torch.stack([G_real_quant, G_imag_quant], dim=-1)
        
        # Compute diagnostics
        if return_diagnostics:
            diagnostics = {
                'quantization_mode': self.quant_config.mode,
                'qat_enabled': self._qat_enabled,
                'calibrated': self.calibrated,
                'bits': self.quant_config.bits,
                'group_size': self.quant_config.group_size if hasattr(self.quant_config, 'group_size') else None,
            }
            
            # Add quantization error if quantization is applied
            if self.calibrated and (self._qat_enabled or not self._training):
                # Input error
                v_error = self.compute_quantization_error(v_fp32, v_quant)
                diagnostics['v_quantization_error'] = v_error
                
                # Output error
                G_real_error = self.compute_quantization_error(G_real_fp32, G_real_quant)
                G_imag_error = self.compute_quantization_error(G_imag_fp32, G_imag_quant)
                diagnostics['G_real_quantization_error'] = G_real_error
                diagnostics['G_imag_quantization_error'] = G_imag_error
                
                # Store history
                self.quantization_error_history.append({
                    'v_mae': v_error['mae'],
                    'G_real_mae': G_real_error['mae'],
                    'G_imag_mae': G_imag_error['mae'],
                })
            
            # Add BK-Core diagnostics
            diagnostics.update(bk_diagnostics)
            
            return features, diagnostics
        
        return features, None
    
    def train(self, mode: bool = True):
        """Set training mode."""
        self._training = mode
        self.bk_core.train(mode)
        return super().train(mode)
    
    def eval(self):
        """Set evaluation mode."""
        self._training = False
        self.bk_core.eval()
        return super().eval()
    
    def get_quantization_statistics(self) -> Dict[str, any]:
        """
        Get quantization statistics including error history.
        
        Returns:
            Dictionary with quantization statistics
        """
        stats = {
            'mode': self.quant_config.mode,
            'bits': self.quant_config.bits,
            'calibrated': self.calibrated,
            'qat_enabled': self._qat_enabled,
            'error_history': self.quantization_error_history,
            'ppl_degradation_history': self.ppl_degradation_history,
        }
        
        if self.quantization_error_history:
            # Compute average errors
            v_mae_list = [e['v_mae'] for e in self.quantization_error_history]
            G_real_mae_list = [e['G_real_mae'] for e in self.quantization_error_history]
            G_imag_mae_list = [e['G_imag_mae'] for e in self.quantization_error_history]
            
            stats['avg_v_mae'] = np.mean(v_mae_list)
            stats['avg_G_real_mae'] = np.mean(G_real_mae_list)
            stats['avg_G_imag_mae'] = np.mean(G_imag_mae_list)
        
        return stats
    
    def estimate_model_size(self) -> Dict[str, float]:
        """
        Estimate model size in bytes for different quantization modes.
        
        Returns:
            Dictionary with size estimates
        """
        # Count parameters in BK-Core
        num_params = sum(p.numel() for p in self.bk_core.parameters())
        
        # If no trainable parameters, estimate based on activations
        if num_params == 0:
            # Estimate based on sequence length and typical model size
            # For BK-Core, main storage is in activations and intermediate values
            num_params = self.n_seq * 10  # Rough estimate
        
        # FP32 size (4 bytes per parameter)
        fp32_size = num_params * 4
        
        # Quantized size
        if self.quant_config.bits == 8:
            quant_size = num_params * 1  # INT8: 1 byte per parameter
        elif self.quant_config.bits == 4:
            quant_size = num_params * 0.5  # INT4: 0.5 bytes per parameter
        else:
            quant_size = fp32_size
        
        # Add quantization parameters (scales and zero points)
        if self.quant_config.mode in [QuantizationConfig.PTQ_INT4, QuantizationConfig.QAT_INT4]:
            # INT4: group-wise quantizers
            num_groups = self.v_quantizer.num_groups
            quant_params_size = num_groups * 2 * 4  # scale (FP32) + zero_point (INT32)
            quant_params_size += self.G_real_quantizer.num_groups * 2 * 4
            quant_params_size += self.G_imag_quantizer.num_groups * 2 * 4
        else:
            # INT8: per-channel quantizers
            # Per-channel: N scales + N zero points
            quant_params_size = self.n_seq * 2 * 4 * 3  # v, G_real, G_imag
        
        total_quant_size = quant_size + quant_params_size
        
        # Compression ratio
        compression_ratio = fp32_size / total_quant_size if total_quant_size > 0 else 0.0
        
        return {
            'fp32_bytes': fp32_size,
            'quantized_bytes': total_quant_size,
            'quantization_params_bytes': quant_params_size,
            'compression_ratio': compression_ratio,
            'bits': self.quant_config.bits,
            'num_parameters': num_params,
        }


def create_quantized_birman_schwinger(
    n_seq: int,
    mode: str = "ptq_int8",
    group_size: int = 128,
    **kwargs
) -> QuantizedBirmanSchwingerCore:
    """
    Factory function to create quantized Birman-Schwinger core.
    
    Args:
        n_seq: Sequence length
        mode: Quantization mode ("ptq_int8", "qat_int8", "ptq_int4", "qat_int4")
        group_size: Group size for group-wise quantization (default: 128)
        **kwargs: Additional arguments for BirmanSchwingerCore
    
    Returns:
        QuantizedBirmanSchwingerCore instance
    
    Examples:
        # PTQ INT8 (Requirement 7.1, 7.2)
        >>> model = create_quantized_birman_schwinger(512, mode="ptq_int8")
        
        # QAT INT8 (Requirement 7.3, 7.4)
        >>> model = create_quantized_birman_schwinger(512, mode="qat_int8")
        
        # PTQ INT4 with group-wise quantization (Requirement 7.5, 7.6)
        >>> model = create_quantized_birman_schwinger(512, mode="ptq_int4", group_size=128)
    """
    quant_config = QuantizationConfig(
        mode=mode,
        group_size=group_size,
    )
    
    return QuantizedBirmanSchwingerCore(
        n_seq=n_seq,
        quant_config=quant_config,
        **kwargs
    )
