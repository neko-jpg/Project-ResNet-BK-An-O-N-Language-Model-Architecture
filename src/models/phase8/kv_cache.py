import torch
import torch.nn as nn
from typing import Optional, List, Tuple

class HyperbolicKVCache(nn.Module):
    """
    Implements KV Cache Compression (Logic Only - Task 24).
    Eviction Policy: Based on Hyperbolic Distance from Origin.

    Idea:
    - "Central" tokens (near origin) are fundamental/general and should be KEPT.
    - "Boundary" tokens (high norm) are specific/transient and can be EVICTED.

    Wait, usually "Attention Sink" paper says keep first token.
    "H2O" paper says keep heavy hitters.
    Here we use Geometry:
    - Keep Origin (General Context)
    - Keep Local (Recent Context - separate window)

    This class manages a fixed-size cache using this policy.
    """
    def __init__(self, d_model: int, max_cache_size: int = 128):
        super().__init__()
        self.d_model = d_model
        self.max_cache_size = max_cache_size

        # State
        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None
        self.current_seq_len = 0

    def update(self, k: torch.Tensor, v: torch.Tensor):
        """
        Updates cache with new tokens k, v.
        k, v: (Batch, Seq_New, Dim)
        """
        # 1. Concat
        if self.k_cache is None:
            self.k_cache = k
            self.v_cache = v
        else:
            self.k_cache = torch.cat([self.k_cache, k], dim=1)
            self.v_cache = torch.cat([self.v_cache, v], dim=1)

        self.current_seq_len = self.k_cache.shape[1]

        # 2. Check Capacity
        if self.current_seq_len > self.max_cache_size:
            self._evict()

    def _evict(self):
        """
        Evicts tokens to reduce size to max_cache_size.
        Policy:
        1. Always keep last window (local attention) - e.g. last 10 tokens.
        2. From the rest, keep tokens closest to Origin (lowest norm).
        """
        local_window = 10
        if self.max_cache_size <= local_window:
             # Just keep last N
             self.k_cache = self.k_cache[:, -self.max_cache_size:, :]
             self.v_cache = self.v_cache[:, -self.max_cache_size:, :]
             return

        # Split into candidates and local window
        # (Batch, N, D)
        candidates_k = self.k_cache[:, :-local_window, :]
        candidates_v = self.v_cache[:, :-local_window, :]

        local_k = self.k_cache[:, -local_window:, :]
        local_v = self.v_cache[:, -local_window:, :]

        # Calculate Scores for candidates: Norm (smaller is better/more central)
        # We want to KEEP smallest norms.
        norms = candidates_k.norm(dim=-1) # (Batch, N_cand)

        num_to_keep = self.max_cache_size - local_window

        # Top-K smallest norms
        # We use topk with largest=False
        # Note: If batch > 1, this selection might be different per batch item.
        # This complicates tensor structure (ragged).
        # For simplicity in this logic demo, we use the mean norm across batch or just process B=1 logic mostly.
        # Or we gather indices per batch.

        _, indices = norms.topk(k=num_to_keep, dim=1, largest=False, sorted=False)
        # indices: (B, NumKeep)

        # Gather
        # We need to expand indices to (B, NumKeep, D)
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, self.d_model)

        kept_k = torch.gather(candidates_k, 1, indices_expanded)
        kept_v = torch.gather(candidates_v, 1, indices_expanded)

        # Reconstruct
        # Note: Time order is lost for the kept history unless we sort indices.
        # Ideally we should sort indices to maintain temporal order.
        indices_sorted, _ = indices.sort(dim=1)
        indices_expanded_sorted = indices_sorted.unsqueeze(-1).expand(-1, -1, self.d_model)

        kept_k = torch.gather(candidates_k, 1, indices_expanded_sorted)
        kept_v = torch.gather(candidates_v, 1, indices_expanded_sorted)

        self.k_cache = torch.cat([kept_k, local_k], dim=1)
        self.v_cache = torch.cat([kept_v, local_v], dim=1)

    def get_view(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.k_cache, self.v_cache
