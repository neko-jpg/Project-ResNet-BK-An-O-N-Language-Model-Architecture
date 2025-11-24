import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class HolographicMemory(nn.Module):
    """
    Holographic Memory Compression via Eigenmode Condensation.

    Implements the "AdS/Brain" concept where the detailed boundary state (sequence of tokens)
    is compressed into a lower-dimensional "bulk" representation using singular value decomposition (SVD).

    Mechanism:
    1.  Input History H (B, N, D)
    2.  SVD: H = U * S * V^T
    3.  Truncation: Keep top-K singular values/vectors.
    4.  Bulk State: S * V^T (B, K, D) - The "concept" vectors.
    5.  Readout: Cross-Attention between Query (current state) and Bulk State.
    """

    def __init__(
        self,
        d_model: int,
        compression_ratio: int = 8,
        min_modes: int = 4,
        use_randomized_svd: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.compression_ratio = compression_ratio
        self.min_modes = min_modes
        self.use_randomized_svd = use_randomized_svd

        # Readout mechanism: Cross-Attention
        # Query: Current State
        # Key/Value: Compressed Bulk State
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)

    def compress_history(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compresses hidden states into dominant eigenmodes.

        Args:
            hidden_states: (B, N, D) sequence history.

        Returns:
            bulk_memory: (B, K, D) compressed representation.
        """
        B, N, D = hidden_states.shape

        # Determine number of modes K
        k = max(self.min_modes, N // self.compression_ratio)
        k = min(k, D, N) # Cannot exceed dimensions

        # Perform SVD
        # PyTorch SVD can be slow on CPU for large matrices.
        # Randomized SVD is faster but custom implementation needed or use low_rank approximation.
        # Here we use torch.linalg.svd (standard) or torch.svd_lowrank (better for large matrices)

        # Optimization: Skip SVD for very short sequences (less than modes needed + buffer)
        if N < self.min_modes * 2:
             stride = 1
             bulk_memory = hidden_states
             return bulk_memory

        if self.use_randomized_svd and N > 128:
            # U: (B, N, K), S: (B, K), V: (B, D, K)
            # Note: svd_lowrank returns U, S, V such that A approx U diag(S) V^T
            # We want to represent the "content".
            # The "content" is effectively the singular vectors weighted by singular values.
            # We can store V^T scaled by S, or U scaled by S.
            # Since we want to attend to it, (B, K, D) is good.
            # V is (B, D, K). V^T is (B, K, D).

            # Note: svd_lowrank does not support batched input well in older versions,
            # but usually iterates or supports it. Let's check or fallback.
            # Fallback to standard SVD for safety in this environment.
            try:
                # Standard SVD: U (B, N, N), S (B, min(N,D)), Vh (B, D, D)
                # We want top K.
                # full_matrices=False -> U (B, N, K'), S (B, K'), Vh (B, K', D)
                U, S, Vh = torch.linalg.svd(hidden_states, full_matrices=False)

                # Truncate to K
                S_k = S[:, :k]       # (B, K)
                Vh_k = Vh[:, :k, :]  # (B, K, D)

                # Bulk State: Scale Vh by S.
                # Why Vh? Vh represents the feature composition basis.
                # U represents the temporal distribution.
                # If we want "concepts", Vh is better.
                # If we want "temporal summaries", U is better.
                # We want to key off of content, so Vh is appropriate.
                # Weighted by importance (S).

                bulk_memory = Vh_k * S_k.unsqueeze(-1) # (B, K, D)

            except Exception as e:
                # Fallback to simple averaging or pooling if SVD fails (e.g. OOM)
                # Chunk pooling
                bulk_memory = F.avg_pool1d(hidden_states.transpose(1, 2), kernel_size=self.compression_ratio).transpose(1, 2)

        else:
             # Simple pooling for short sequences
             # Ensure we match dimensions roughly or just pool
             stride = max(1, N // k)
             bulk_memory = hidden_states[:, ::stride, :] # Subsampling
             if bulk_memory.shape[1] > k:
                 bulk_memory = bulk_memory[:, :k, :]

        return bulk_memory

    def read_memory(self, query: torch.Tensor, bulk_memory: torch.Tensor) -> torch.Tensor:
        """
        Retrieves information from bulk memory.

        Args:
            query: (B, M, D) current tokens (boundary).
            bulk_memory: (B, K, D) compressed history (bulk).

        Returns:
            context: (B, M, D) retrieved context.
        """
        # Cross Attention: Q=Query, K=Bulk, V=Bulk
        context, _ = self.attention(query, bulk_memory, bulk_memory)
        return self.layer_norm(query + context) # Residual connection

    def forward(self, current_state: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """
        Full Holographic Memory Step.

        Args:
            current_state: (B, M, D)
            history: (B, N, D)

        Returns:
            output: (B, M, D) enriched state
        """
        bulk = self.compress_history(history)
        output = self.read_memory(current_state, bulk)
        return output
