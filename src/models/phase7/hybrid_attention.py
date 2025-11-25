# src/models/phase7/hybrid_attention.py
# Implementation of Hybrid Hyperbolic Attention

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .hyperbolic_attention import HyperbolicMultiHeadAttention
from ..phase1.ar_ssm_layer import AdaptiveRankSemiseparableLayer

class HybridHyperbolicAttention(nn.Module):
    """
    Implements a hybrid attention mechanism that combines local, high-resolution
    hyperbolic attention with a global, linear-complexity SSM.

    - Local Context: Processed by HyperbolicMultiHeadAttention in non-overlapping windows.
                     Captures fine-grained hierarchical relationships in a local neighborhood.
    - Global Context: Processed by AdaptiveRankSemiseparableLayer (SSM).
                      Efficiently propagates information across the entire sequence.
    """
    def __init__(self, d_model: int, num_heads: int, local_window_size: int = 64):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.local_window_size = local_window_size

        # 1. Local hyperbolic attention module
        self.local_attn = HyperbolicMultiHeadAttention(d_model, num_heads)

        # 2. Global SSM module
        # We can initialize it with default parameters for now.
        # A more robust implementation might pass a config object.
        self.global_attn = AdaptiveRankSemiseparableLayer(d_model)

        # 3. Learnable gate to combine local and global outputs
        self.gate = nn.Parameter(torch.randn(d_model))

        # 4. Final layer normalization for stability
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, return_diagnostics: bool = True):
        """
        Args:
            x (torch.Tensor): Input tensor. Shape: (batch, seq_len, d_model)
            return_diagnostics (bool): If True, returns a dictionary of monitoring metrics.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, dict]:
            - Output tensor. Shape: (batch, seq_len, d_model)
            - (Optional) Dictionary with diagnostic metrics.
        """
        batch_size, seq_len, _ = x.shape
        diagnostics = {}

        # --- 1. Global Attention Path (SSM) ---
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

        # Apply local attention within each window
        local_out_windowed, local_diagnostics = self.local_attn(x_windowed, return_diagnostics=True)
        if return_diagnostics:
            diagnostics.update(local_diagnostics)

        local_out_padded = local_out_windowed.reshape(batch_size, -1, self.d_model)
        local_out = local_out_padded[:, :seq_len, :]

        # --- 3. Combination ---
        g = torch.sigmoid(self.gate)
        if return_diagnostics:
            diagnostics['hybrid_gate_mean'] = g.mean()

        combined_out = g * global_out + (1 - g) * local_out
        output = self.layer_norm(combined_out)

        if return_diagnostics:
            return output, diagnostics
        else:
            return output
