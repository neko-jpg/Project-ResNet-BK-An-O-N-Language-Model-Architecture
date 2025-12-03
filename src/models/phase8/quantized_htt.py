"""
Quantized Holographic Tensor Train (QHTT) Embedding

Phase 8: High-Fidelity Compression for Real-World Data.
Implements the strategy of "Preserving the Prototype while Compressing Storage".

Problem Solved:
    Standard HTT requires high rank for real data, reducing compression ratio to ~60%.
    QHTT maintains high rank (accuracy) but compresses storage using Manifold-Aware Quantization,
    restoring the ~99% compression ratio.

Mechanism:
    - Stores TT-Cores in quantized integer format (INT8/INT4).
    - Uses Logarithmic Quantization to preserve high-fidelity information (the "prototype").
    - Dequantizes on-the-fly during the forward pass.

Author: Project MUSE Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math

from src.models.phase1.htt_embedding import HolographicTTEmbedding
from src.models.phase8.quantization import LogarithmicQuantizer, QuantizationConfig

class QuantizedHolographicTTEmbedding(nn.Module):
    """
    Quantized Holographic Tensor Train Embedding.

    Wraps the logic of HTT but stores parameters in a compressed quantized format.

    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        rank: TT Rank (can be high, e.g., 64, 128)
        bits: Quantization bits (default 8)
        phase_encoding: Whether to use phase encoding
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        rank: int = 64,  # Default higher rank for real data
        bits: int = 8,
        phase_encoding: bool = True,
        quantization_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.rank = rank
        self.bits = bits
        self.phase_encoding = phase_encoding

        # Factorization (Same as HTT)
        self.v1 = int(math.ceil(math.sqrt(vocab_size)))
        self.v2 = int(math.ceil(vocab_size / self.v1))
        self.d1 = int(math.ceil(math.sqrt(d_model)))
        self.d2 = int(math.ceil(d_model / self.d1))

        # Quantizer
        # Force per_channel=False to simplify storage (1 scale per core)
        # Logarithmic Quantization handles the dynamic range well even without per-channel scaling.
        if quantization_config is None:
            self.quantizer = LogarithmicQuantizer(bits=bits, per_channel=False)
        else:
            quantization_config.per_channel = False
            self.quantizer = LogarithmicQuantizer(config_or_bits=quantization_config)
            self.bits = quantization_config.bits

        # Storage for Quantized Parameters
        # Core1: (v1, 1, rank, d1)
        self.register_buffer('core1_q', torch.zeros(self.v1, 1, rank, self.d1, dtype=torch.int8))
        self.register_buffer('core1_scale', torch.tensor(1.0))
        self.register_buffer('core1_zp', torch.tensor(0.0))

        # Core2: (v2, rank, 1, d2)
        self.register_buffer('core2_q', torch.zeros(self.v2, rank, 1, self.d2, dtype=torch.int8))
        self.register_buffer('core2_scale', torch.tensor(1.0))
        self.register_buffer('core2_zp', torch.tensor(0.0))

        # Phase Shift
        if phase_encoding:
            self.phase_shift = nn.Parameter(torch.zeros(rank))
        else:
            self.register_buffer('phase_shift', torch.zeros(rank))

        # State flag
        self.is_quantized = False

    @classmethod
    def from_htt(cls, htt_model: HolographicTTEmbedding, bits: int = 8) -> 'QuantizedHolographicTTEmbedding':
        """
        Convert a standard HolographicTTEmbedding to QuantizedHolographicTTEmbedding.

        Args:
            htt_model: Source HTT model (trained)
            bits: Target bits per parameter

        Returns:
            Quantized instance
        """
        q_model = cls(
            vocab_size=htt_model.vocab_size,
            d_model=htt_model.d_model,
            rank=htt_model.rank,
            bits=bits,
            phase_encoding=htt_model.phase_encoding
        )

        # Copy Phase Shift
        if htt_model.phase_encoding:
            q_model.phase_shift.data.copy_(htt_model.phase_shift.data)

        # Quantize Core 1
        c1_data = htt_model.core1.data
        # Reshape to flat for reliable scalar calibration
        c1_flat = c1_data.view(-1)
        q_model.quantizer.calibrate(c1_flat)
        c1_q, c1_s, c1_zp = q_model.quantizer.quantize_int(c1_flat)
        q_model.core1_q.copy_(c1_q.view(c1_data.shape))
        q_model.core1_scale.copy_(c1_s)
        q_model.core1_zp.copy_(c1_zp)

        # Quantize Core 2
        c2_data = htt_model.core2.data
        c2_flat = c2_data.view(-1)
        q_model.quantizer.calibrate(c2_flat)
        c2_q, c2_s, c2_zp = q_model.quantizer.quantize_int(c2_flat)
        q_model.core2_q.copy_(c2_q.view(c2_data.shape))
        q_model.core2_scale.copy_(c2_s)
        q_model.core2_zp.copy_(c2_zp)

        q_model.is_quantized = True

        return q_model

    def dequantize_cores(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dequantize cores for computation.

        Returns:
            (core1_float, core2_float)
        """
        c1 = self.quantizer.dequantize_int(self.core1_q, self.core1_scale, self.core1_zp)
        c2 = self.quantizer.dequantize_int(self.core2_q, self.core2_scale, self.core2_zp)
        return c1, c2

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with on-the-fly dequantization.
        """
        # Dequantize
        c1_full, c2_full = self.dequantize_cores()

        # --- HTT Logic Below (Adapted from HolographicTTEmbedding) ---

        B, L = input_ids.shape

        # Index decomposition
        idx1 = input_ids // self.v2
        idx2 = input_ids % self.v2

        idx1 = torch.clamp(idx1, 0, self.v1 - 1)
        idx2 = torch.clamp(idx2, 0, self.v2 - 1)

        # Gather
        c1 = c1_full[idx1].squeeze(2)  # (B, L, rank, d1)
        c2 = c2_full[idx2].squeeze(3)  # (B, L, rank, d2)

        # Phase Rotation
        if self.phase_encoding:
            phase_mod = torch.cos(self.phase_shift)
            c1 = c1 * phase_mod.view(1, 1, -1, 1)

        # Contraction
        out_tensor = torch.einsum('blrd,blrf->bldf', c1, c2)

        # Reshape and Crop
        out = out_tensor.reshape(B, L, -1)
        out = out[:, :, :self.d_model]

        return out

    def get_storage_memory_mb(self) -> float:
        """Calculate actual storage memory in MB."""
        param_count = self.core1_q.numel() + self.core2_q.numel()
        bits_total = param_count * self.bits
        bytes_total = bits_total / 8

        # Phase shift (FP32)
        bytes_total += self.phase_shift.numel() * 4

        # Scales/ZPs
        bytes_total += 16

        return bytes_total / (1024 * 1024)

    def get_compression_stats(self) -> Dict:
        """Get compression statistics compared to standard embedding."""
        standard_bytes = (self.vocab_size * self.d_model) * 4  # FP32

        qhtt_mb = self.get_storage_memory_mb()
        standard_mb = standard_bytes / (1024 * 1024)

        ratio = qhtt_mb / standard_mb

        return {
            "standard_mb": standard_mb,
            "qhtt_mb": qhtt_mb,
            "compression_ratio": ratio,
            "reduction_percentage": (1 - ratio) * 100
        }
