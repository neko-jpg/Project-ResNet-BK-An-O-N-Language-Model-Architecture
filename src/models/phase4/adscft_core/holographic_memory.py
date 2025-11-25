import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math

class HolographicMemory(nn.Module):
    """
    Holographic Memory Compression via Eigenmode Condensation.

    Natively supports the [B, H, N, D_h] memory layout for efficient integration
    with Triton-optimized attention mechanisms.

    Mechanism:
    1.  Input History H (B, H, N, D_h) is reshaped to (B*H, N, D_h).
    2.  Batched SVD: H' = U * S * V^T is computed for each head.
    3.  Truncation: Keep top-K singular values/vectors per head.
    4.  Bulk State: S * V^T (B*H, K, D_h) - The "concept" vectors per head.
    5.  Reshape back to (B, H, K, D_h).
    6.  Readout: Per-head cross-attention between Query and the Bulk State.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        compression_ratio: int = 8,
        min_modes: int = 4,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.compression_ratio = compression_ratio
        self.min_modes = min_modes

        # Readout mechanism: Manual per-head Cross-Attention
        self.q_proj = nn.Linear(self.d_head, self.d_head, bias=False)
        self.k_proj = nn.Linear(self.d_head, self.d_head, bias=False)
        self.v_proj = nn.Linear(self.d_head, self.d_head, bias=False)
        self.out_proj = nn.Linear(self.d_head, self.d_head)
        self.layer_norm = nn.LayerNorm(self.d_head)

    def compress_history(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compresses hidden states into dominant eigenmodes for each head.

        Args:
            hidden_states: (B, H, N, D_h) sequence history.

        Returns:
            bulk_memory: (B, H, K, D_h) compressed representation.
        """
        B, H, N, D_h = hidden_states.shape

        # Determine number of modes K
        k = max(self.min_modes, N // self.compression_ratio)
        k = min(k, D_h, N)

        # Optimization: Skip SVD for very short sequences
        if N < self.min_modes * 2:
            return hidden_states

        # Reshape for batched SVD: treat heads as part of the batch
        history_reshaped = hidden_states.view(B * H, N, D_h)

        try:
            # Perform batched SVD
            U, S, Vh = torch.linalg.svd(history_reshaped, full_matrices=False)

            # Truncate to K
            S_k = S[..., :k]
            Vh_k = Vh[..., :k, :]

            # Bulk State: Vh represents feature composition basis, weighted by importance (S).
            bulk_memory_reshaped = Vh_k * S_k.unsqueeze(-1)  # (B*H, K, D_h)

            # Reshape back to per-head layout
            bulk_memory = bulk_memory_reshaped.view(B, H, k, D_h)

        except torch.cuda.OutOfMemoryError:
            # Fallback to simple subsampling if SVD fails (e.g. OOM)
            stride = max(1, N // k)
            bulk_memory = hidden_states[:, :, ::stride, :]
            if bulk_memory.shape[2] > k:
                bulk_memory = bulk_memory[:, :, :k, :]
        except Exception:
             # General fallback
            stride = max(1, N // k)
            bulk_memory = hidden_states[:, :, ::stride, :]
            if bulk_memory.shape[2] > k:
                 bulk_memory = bulk_memory[:, :, :k, :]


        return bulk_memory

    def read_memory(self, query: torch.Tensor, bulk_memory: torch.Tensor) -> torch.Tensor:
        """
        Retrieves information from bulk memory using per-head cross-attention.

        Args:
            query: (B, H, M, D_h) current tokens (boundary).
            bulk_memory: (B, H, K, D_h) compressed history (bulk).

        Returns:
            context: (B, H, M, D_h) retrieved context.
        """
        B, H, M, D_h = query.shape
        _B, _H, K, _Dh = bulk_memory.shape

        # Reshape for batched processing
        query_r = query.view(B * H, M, D_h)
        bulk_memory_r = bulk_memory.view(B * H, K, D_h)

        # Project Q, K, V
        q = self.q_proj(query_r)
        k = self.k_proj(bulk_memory_r)
        v = self.v_proj(bulk_memory_r)

        # Scaled Dot-Product Attention
        scale = math.sqrt(D_h)
        scores = torch.bmm(q, k.transpose(1, 2)) / scale
        attn_weights = F.softmax(scores, dim=-1)
        context_r = torch.bmm(attn_weights, v) # (B*H, M, D_h)

        # Reshape back to per-head layout
        context = context_r.view(B, H, M, D_h)

        # Final projection, residual connection, and layer norm
        output = self.out_proj(context)
        return self.layer_norm(query + output)

    def forward(self, current_state: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """
        Full Holographic Memory Step.

        Args:
            current_state: (B, H, M, D_h)
            history: (B, H, N, D_h)

        Returns:
            output: (B, H, M, D_h) enriched state
        """
        bulk = self.compress_history(history)
        output = self.read_memory(current_state, bulk)
        return output
