"""
KV Cache Compression for Hyperbolic Attention

This module implements KV cache compression with:
- 4-bit quantization with per-channel scaling
- Hyperbolic distance-based eviction
- Learned projection to lower dimension
- Fused decompression with distance computation

Requirements: 39.1, 39.2, 39.3, 39.5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
import math


@dataclass
class KVCacheConfig:
    """Configuration for KV Cache Compression."""
    d_model: int = 256
    max_cache_size: int = 128
    local_window: int = 10
    compression_ratio: float = 0.5  # Target dimension = d_model * compression_ratio
    use_quantization: bool = True
    quantization_bits: int = 4
    use_learned_projection: bool = True
    curvature: float = 1.0
    eps: float = 1e-6
    use_htt_compression: bool = False
    htt_rank: int = 16
    htt_max_tokens: int = 256


class LearnedProjection(nn.Module):
    """
    Learned projection to lower dimension for KV cache compression.
    
    Requirements: 39.3
    """
    
    def __init__(self, d_model: int, d_compressed: int):
        super().__init__()
        self.d_model = d_model
        self.d_compressed = d_compressed
        
        # Projection matrices
        self.down_proj = nn.Linear(d_model, d_compressed, bias=False)
        self.up_proj = nn.Linear(d_compressed, d_model, bias=False)
        
        # Initialize with orthogonal matrices for better reconstruction
        nn.init.orthogonal_(self.down_proj.weight)
        nn.init.orthogonal_(self.up_proj.weight)
    
    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Compress to lower dimension."""
        return self.down_proj(x)
    
    def decompress(self, x: torch.Tensor) -> torch.Tensor:
        """Decompress back to original dimension."""
        return self.up_proj(x)


class QuantizedKVCache(nn.Module):
    """
    4-bit quantized KV cache with per-channel scaling.
    
    Requirements: 39.1, 39.2
    """
    
    def __init__(self, config: KVCacheConfig):
        super().__init__()
        self.config = config
        self.bits = config.quantization_bits
        self.qmin = 0
        self.qmax = (1 << self.bits) - 1
        
        # Per-channel scales and zero points
        self.register_buffer("k_scale", None)
        self.register_buffer("k_zero", None)
        self.register_buffer("v_scale", None)
        self.register_buffer("v_zero", None)
        
        # Quantized cache
        self.k_quantized: Optional[torch.Tensor] = None
        self.v_quantized: Optional[torch.Tensor] = None
    
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize tensor with per-channel scaling."""
        # Per-channel min/max
        x_min = x.min(dim=1, keepdim=True)[0]
        x_max = x.max(dim=1, keepdim=True)[0]
        
        # Compute scale and zero point
        scale = (x_max - x_min) / (self.qmax - self.qmin)
        scale = scale.clamp(min=1e-8)
        zero_point = self.qmin - x_min / scale
        zero_point = zero_point.round().clamp(self.qmin, self.qmax)
        
        # Quantize
        x_q = (x / scale + zero_point).round().clamp(self.qmin, self.qmax)
        
        return x_q.to(torch.uint8), scale, zero_point
    
    def dequantize(self, x_q: torch.Tensor, scale: torch.Tensor, 
                   zero_point: torch.Tensor) -> torch.Tensor:
        """Dequantize tensor."""
        return scale * (x_q.float() - zero_point)
    
    def update(self, k: torch.Tensor, v: torch.Tensor):
        """Update cache with quantized values."""
        k_q, k_scale, k_zero = self.quantize(k)
        v_q, v_scale, v_zero = self.quantize(v)
        
        if self.k_quantized is None:
            self.k_quantized = k_q
            self.v_quantized = v_q
            self.k_scale = k_scale
            self.k_zero = k_zero
            self.v_scale = v_scale
            self.v_zero = v_zero
        else:
            self.k_quantized = torch.cat([self.k_quantized, k_q], dim=1)
            self.v_quantized = torch.cat([self.v_quantized, v_q], dim=1)
            # Update scales (use running average or latest)
            self.k_scale = k_scale
            self.k_zero = k_zero
            self.v_scale = v_scale
            self.v_zero = v_zero
    
    def get_decompressed(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get decompressed K, V tensors."""
        if self.k_quantized is None:
            return None, None
        k = self.dequantize(self.k_quantized, self.k_scale, self.k_zero)
        v = self.dequantize(self.v_quantized, self.v_scale, self.v_zero)
        return k, v


class HTTContextCompressor(nn.Module):
    """
    Compress long-range context into a low-rank (TT-like) core using SVD.
    """

    def __init__(self, d_model: int, rank: int = 16, max_tokens: int = 256):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.max_tokens = max_tokens
        self.register_buffer("core_left", None)
        self.register_buffer("core_right", None)
        self.stored_tokens: int = 0

    def compress(self, tokens: torch.Tensor):
        """
        tokens: (B, L, D)
        """
        B, L, D = tokens.shape
        if L == 0:
            return
        matrix = tokens.transpose(1, 2)  # (B, D, L)
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        r = min(self.rank, s.shape[-1])
        u_r = u[:, :, :r]
        s_r = s[:, :r]
        vh_r = vh[:, :r, :]
        core_left = u_r * s_r.unsqueeze(1)
        self.core_left = core_left
        self.core_right = vh_r
        self.stored_tokens = min(L, self.max_tokens)

    def decode(self) -> Optional[torch.Tensor]:
        if self.core_left is None or self.core_right is None:
            return None
        approx = torch.matmul(self.core_left, self.core_right)  # (B, D, L)
        approx_tokens = approx.transpose(1, 2)
        return approx_tokens[:, : self.stored_tokens, :]

    def reconstruction_loss(self, original: torch.Tensor) -> torch.Tensor:
        rec = self.decode()
        if rec is None:
            return torch.tensor(0.0, device=original.device, dtype=original.dtype)
        min_len = min(original.shape[1], rec.shape[1])
        return F.mse_loss(rec[:, :min_len], original[:, :min_len])


class FusedDecompressDistance(nn.Module):
    """
    Fused decompression with hyperbolic distance computation.
    
    Requirements: 39.5
    """
    
    def __init__(self, config: KVCacheConfig):
        super().__init__()
        self.config = config
        self.c = config.curvature
        self.eps = config.eps
    
    def forward(self, q: torch.Tensor, k_compressed: torch.Tensor, 
                projection: LearnedProjection,
                k_scale: Optional[torch.Tensor] = None,
                k_zero: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute hyperbolic distance with fused decompression.
        
        Args:
            q: Query tensor (B, L_q, D)
            k_compressed: Compressed key tensor (B, L_k, D_compressed) or quantized
            projection: Learned projection module
            k_scale: Quantization scale (optional)
            k_zero: Quantization zero point (optional)
            
        Returns:
            Hyperbolic distances (B, L_q, L_k)
        """
        # Dequantize if needed
        if k_scale is not None:
            k_compressed = k_scale * (k_compressed.float() - k_zero)
        
        # Decompress
        k = projection.decompress(k_compressed)
        
        # Compute hyperbolic distance
        # d_H(x, y) = 2 * arctanh(||(-x) ⊕ y||)
        # For efficiency, use approximation: d ≈ 2 * ||x - y|| / (1 - ||x||²)(1 - ||y||²)
        
        q_norm_sq = (q ** 2).sum(dim=-1, keepdim=True).clamp(max=1 - self.eps)
        k_norm_sq = (k ** 2).sum(dim=-1, keepdim=True).clamp(max=1 - self.eps)
        
        # (B, L_q, 1, D) - (B, 1, L_k, D) -> (B, L_q, L_k, D)
        diff = q.unsqueeze(2) - k.unsqueeze(1)
        diff_norm_sq = (diff ** 2).sum(dim=-1)  # (B, L_q, L_k)
        
        # Conformal factor
        lambda_q = 2 / (1 - q_norm_sq.squeeze(-1))  # (B, L_q)
        lambda_k = 2 / (1 - k_norm_sq.squeeze(-1).transpose(-1, -2))  # (B, L_k) -> (B, 1, L_k)
        
        # Hyperbolic distance approximation
        dist = torch.acosh(1 + 2 * diff_norm_sq / ((1 - q_norm_sq.squeeze(-1).unsqueeze(-1)) * 
                                                    (1 - k_norm_sq.squeeze(-1).unsqueeze(1)) + self.eps))
        
        return dist


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


class CompressedKVCache(nn.Module):
    """
    Full KV Cache with compression, quantization, and fused operations.
    
    Requirements: 39.1, 39.2, 39.3, 39.5
    """
    
    def __init__(self, config: KVCacheConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_compressed = int(config.d_model * config.compression_ratio)
        self.max_cache_size = config.max_cache_size
        self.local_window = config.local_window
        
        # Learned projection
        if config.use_learned_projection:
            self.projection = LearnedProjection(config.d_model, self.d_compressed)
        else:
            self.projection = None
        
        # Quantized cache
        if config.use_quantization:
            self.quantized_cache = QuantizedKVCache(config)
        else:
            self.quantized_cache = None
        
        # HTT-style dynamic compressor for evicted context
        if config.use_htt_compression:
            self.htt_compressor_k = HTTContextCompressor(config.d_model, rank=config.htt_rank, max_tokens=config.htt_max_tokens)
            self.htt_compressor_v = HTTContextCompressor(config.d_model, rank=config.htt_rank, max_tokens=config.htt_max_tokens)
        else:
            self.htt_compressor_k = None
            self.htt_compressor_v = None
        
        # Fused distance computation
        self.fused_distance = FusedDecompressDistance(config)
        
        # Non-quantized cache (fallback)
        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None
        self.current_seq_len = 0
    
    def update(self, k: torch.Tensor, v: torch.Tensor):
        """
        Update cache with new K, V tensors.
        
        Args:
            k: Key tensor (B, L, D)
            v: Value tensor (B, L, D)
        """
        # Compress if using learned projection
        if self.projection is not None:
            k_compressed = self.projection.compress(k)
            v_compressed = self.projection.compress(v)
        else:
            k_compressed = k
            v_compressed = v

        # Reconstruction regularization (optional)
        if getattr(self.config, "kv_use_reconstruction_loss", False) and self.config.kv_reconstruction_weight > 0:
            recon_k = self.projection.decompress(k_compressed) if self.projection is not None else k_compressed
            recon_v = self.projection.decompress(v_compressed) if self.projection is not None else v_compressed
            self.last_reconstruction_loss = self.config.kv_reconstruction_weight * (
                F.mse_loss(recon_k, k) + F.mse_loss(recon_v, v)
            )
        else:
            self.last_reconstruction_loss = None
        
        # Quantize if enabled
        if self.quantized_cache is not None:
            self.quantized_cache.update(k_compressed, v_compressed)
        else:
            if self.k_cache is None:
                self.k_cache = k_compressed
                self.v_cache = v_compressed
            else:
                self.k_cache = torch.cat([self.k_cache, k_compressed], dim=1)
                self.v_cache = torch.cat([self.v_cache, v_compressed], dim=1)
        
        self.current_seq_len = self._get_cache_len()
        
        # Evict if over capacity
        if self.current_seq_len > self.max_cache_size:
            self._evict()
    
    def _get_cache_len(self) -> int:
        """Get current cache length."""
        if self.quantized_cache is not None and self.quantized_cache.k_quantized is not None:
            return self.quantized_cache.k_quantized.shape[1]
        elif self.k_cache is not None:
            return self.k_cache.shape[1]
        return 0
    
    def _evict(self):
        """Evict tokens based on hyperbolic distance from origin."""
        if self.quantized_cache is not None:
            k, v = self.quantized_cache.get_decompressed()
        else:
            k, v = self.k_cache, self.v_cache
        
        if k is None:
            return
        
        # Keep local window
        if self.max_cache_size <= self.local_window:
            k = k[:, -self.max_cache_size:, :]
            v = v[:, -self.max_cache_size:, :]
        else:
            candidates_k = k[:, :-self.local_window, :]
            candidates_v = v[:, :-self.local_window, :]
            local_k = k[:, -self.local_window:, :]
            local_v = v[:, -self.local_window:, :]

            # Compress the long-tail context into TT-like cores
            if self.htt_compressor_k is not None:
                self.htt_compressor_k.compress(candidates_k)
                self.htt_compressor_v.compress(candidates_v)
            
            # Score by norm (keep smallest = most central)
            norms = candidates_k.norm(dim=-1)
            num_to_keep = self.max_cache_size - self.local_window
            
            _, indices = norms.topk(k=min(num_to_keep, norms.shape[1]), dim=1, largest=False)
            indices_sorted, _ = indices.sort(dim=1)
            indices_expanded = indices_sorted.unsqueeze(-1).expand(-1, -1, k.shape[-1])
            
            kept_k = torch.gather(candidates_k, 1, indices_expanded)
            kept_v = torch.gather(candidates_v, 1, indices_expanded)
            
            k = torch.cat([kept_k, local_k], dim=1)
            v = torch.cat([kept_v, local_v], dim=1)
        
        # Update cache
        if self.quantized_cache is not None:
            self.quantized_cache.k_quantized = None
            self.quantized_cache.v_quantized = None
            self.quantized_cache.update(k, v)
        else:
            self.k_cache = k
            self.v_cache = v
        # Compress evicted context into TT-style cores for long-term reference
        if self.htt_compressor_k is not None:
            self.htt_compressor_k.compress(k)
            self.htt_compressor_v.compress(v)
    
    def compute_attention_with_fused_decompress(
        self, 
        q: torch.Tensor,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention with fused decompression and distance computation.
        
        Args:
            q: Query tensor (B, L_q, D)
            temperature: Softmax temperature
            
        Returns:
            attention_output: (B, L_q, D)
            attention_weights: (B, L_q, L_k)
        """
        # Get compressed cache
        if self.quantized_cache is not None:
            k_compressed = self.quantized_cache.k_quantized
            v_compressed = self.quantized_cache.v_quantized
            k_scale = self.quantized_cache.k_scale
            k_zero = self.quantized_cache.k_zero
        else:
            k_compressed = self.k_cache
            v_compressed = self.v_cache
            k_scale = None
            k_zero = None

        htt_k = self.htt_compressor_k.decode() if self.htt_compressor_k is not None else None
        htt_v = self.htt_compressor_v.decode() if self.htt_compressor_v is not None else None
        if htt_k is not None and htt_v is not None:
            if k_compressed is None:
                k_compressed = htt_k
                v_compressed = htt_v
            else:
                k_compressed = torch.cat([htt_k, k_compressed], dim=1)
                v_compressed = torch.cat([htt_v, v_compressed], dim=1)
        
        if k_compressed is None:
            return q, torch.ones(q.shape[0], q.shape[1], 1, device=q.device)
        
        # Compress query
        if self.projection is not None:
            q_compressed = self.projection.compress(q)
        else:
            q_compressed = q
        
        # Fused distance computation
        if self.projection is not None:
            distances = self.fused_distance(
                q_compressed, k_compressed, self.projection, k_scale, k_zero
            )
        else:
            # Simple dot product attention
            k_decompressed = k_compressed.float() if k_scale is None else \
                            k_scale * (k_compressed.float() - k_zero)
            distances = -torch.bmm(q_compressed, k_decompressed.transpose(-1, -2))
        
        # Softmax
        attn_weights = F.softmax(-distances / temperature, dim=-1)
        
        # Get values
        if self.quantized_cache is not None:
            _, v = self.quantized_cache.get_decompressed()
        else:
            v = self.v_cache
        
        # Decompress values
        if self.projection is not None:
            v_full = self.projection.decompress(v)
        else:
            v_full = v
        
        # Compute output
        output = torch.bmm(attn_weights, v_full)
        
        return output, attn_weights
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        original_size = self.d_model * self.current_seq_len * 4  # FP32
        
        if self.quantized_cache is not None:
            compressed_size = self.d_compressed * self.current_seq_len * (self.config.quantization_bits / 8)
        elif self.projection is not None:
            compressed_size = self.d_compressed * self.current_seq_len * 4
        else:
            compressed_size = original_size
        
        return {
            "original_size_bytes": original_size,
            "compressed_size_bytes": compressed_size,
            "compression_ratio": original_size / max(compressed_size, 1),
            "cache_length": self.current_seq_len,
            "d_model": self.d_model,
            "d_compressed": self.d_compressed,
        }


def create_compressed_kv_cache(
    d_model: int = 256,
    max_cache_size: int = 128,
    compression_ratio: float = 0.5,
    use_quantization: bool = True,
    **kwargs
) -> CompressedKVCache:
    """Factory function to create compressed KV cache."""
    config = KVCacheConfig(
        d_model=d_model,
        max_cache_size=max_cache_size,
        compression_ratio=compression_ratio,
        use_quantization=use_quantization,
        **kwargs
    )
    return CompressedKVCache(config)
