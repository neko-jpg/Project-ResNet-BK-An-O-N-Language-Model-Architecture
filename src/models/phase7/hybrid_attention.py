# src/models/phase7/hybrid_attention.py
# Implementation of Hybrid Hyperbolic Attention

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .hyperbolic_attention import HyperbolicMultiHeadAttention
from ..phase1.ar_ssm_layer import AdaptiveRankSemiseparableLayer
from ..config import ResNetBKConfig

class HybridHyperbolicAttention(nn.Module):
    """
    Implements a hybrid attention mechanism that combines local, high-resolution
    hyperbolic attention with a global, linear-complexity SSM.

    - Local Context: Processed by HyperbolicMultiHeadAttention in non-overlapping windows.
                     Captures fine-grained hierarchical relationships in a local neighborhood.
    - Global Context: Processed by AdaptiveRankSemiseparableLayer (SSM).
                      Efficiently propagates information across the entire sequence.
    """
    def __init__(self, config: ResNetBKConfig):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.local_window_size = config.hyperbolic_window_size

        # 1. Local hyperbolic attention module
        kernel_version = getattr(config, 'triton_kernel_version', 'fast')
        self.local_attn = HyperbolicMultiHeadAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            use_triton_kernel=config.use_triton_kernel,
            kernel_version=kernel_version,
            use_bitnet=config.use_bitnet,
            low_rank_attention=config.low_rank_attention,
            low_rank_rank=config.low_rank_rank,
        )

        # 2. Global SSM module, configured from the main config object
        self.global_attn = AdaptiveRankSemiseparableLayer(
            d_model=config.d_model,
            max_rank=config.ar_ssm_max_rank,
            min_rank=config.ar_ssm_min_rank
        )

        # 3. Dynamic gating based on scattering phase
        self.gate_sensitivity = nn.Parameter(torch.tensor(1.0))

        # 4. Final layer normalization for stability
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, x, g_ii: torch.Tensor = None, return_diagnostics: bool = True):
        """
        Args:
            x (torch.Tensor): Input tensor. Shape: (batch, seq_len, d_model)
            g_ii (torch.Tensor, optional): Green's function diagonal from BK-Core. 
                                           Shape: (batch, seq_len, 1). If None, uses learned gating.
            return_diagnostics (bool): If True, returns a dictionary of monitoring metrics.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, dict]:
            - Output tensor. Shape: (batch, seq_len, d_model)
            - (Optional) Dictionary with diagnostic metrics.
        """
        batch_size, seq_len, _ = x.shape
        diagnostics = {}

        # --- 1. Global Attention Path (SSM) ---
        # NOTE: SSMs are causal by nature, no explicit mask needed here.
        global_out, ssm_diagnostics = self.global_attn(x)

        if return_diagnostics:
            diagnostics['ssm_effective_rank'] = ssm_diagnostics.get('effective_rank', torch.tensor(0.0))

        # --- 2. Local Attention Path (Windowed Hyperbolic) ---
        pad_len = (self.local_window_size - seq_len % self.local_window_size) % self.local_window_size
        if pad_len > 0:
            x_padded = F.pad(x, (0, 0, 0, pad_len))
        else:
            x_padded = x

        num_windows = x_padded.shape[1] // self.local_window_size
        x_windowed = x_padded.reshape(batch_size * num_windows, self.local_window_size, self.d_model)

        # Create a causal mask for the window size
        local_mask = torch.tril(torch.ones(self.local_window_size, self.local_window_size, device=x.device)).view(1, 1, self.local_window_size, self.local_window_size)

        # Apply local attention within each window
        local_out_windowed, local_diagnostics = self.local_attn(x_windowed, mask=local_mask, return_diagnostics=True)
        if return_diagnostics:
            diagnostics.update(local_diagnostics)

        local_out_padded = local_out_windowed.reshape(batch_size, -1, self.d_model)
        local_out = local_out_padded[:, :seq_len, :]

        # --- 3. Combination ---
        # Dynamic gating based on scattering phase energy or learned gating
        if g_ii is not None:
            # Use Green's function diagonal for physics-informed gating
            # g_ii has shape (B, N, 1), take the absolute imaginary part
            energy = g_ii.imag.abs()
            g = torch.sigmoid(energy * self.gate_sensitivity)
            if return_diagnostics:
                diagnostics['scattering_energy_mean'] = energy.mean()
        else:
            # Fallback: Use learned gating based on input complexity
            # Compute input complexity as variance across feature dimension
            input_complexity = x.var(dim=-1, keepdim=True)  # (B, N, 1)
            g = torch.sigmoid(input_complexity * self.gate_sensitivity)
            if return_diagnostics:
                diagnostics['input_complexity_mean'] = input_complexity.mean()

        if return_diagnostics:
            diagnostics['hybrid_gate_mean'] = g.mean()

        # Route high-energy/complex tokens to local (hyperbolic) attention
        combined_out = g * local_out + (1 - g) * global_out
        output = self.layer_norm(combined_out)

        if return_diagnostics:
            return output, diagnostics
        else:
            return output
