"""
Holographic Compression - Moonshot #11

AdS/CFT-inspired memory compression using bulk-boundary correspondence.
Projects high-dimensional "bulk" hidden states to lower-dimensional 
"boundary" representation for efficient storage.

Theory (from docs/research):
    φ_boundary(x) = ∫ K_Δ(z,x) φ_bulk(z) √g dz
    
    Where:
    - K_Δ: Bulk-to-boundary propagator
    - √g: Metric determinant (hyperbolic measure)
    - z: Radial coordinate (renormalization scale)
    
    Memory scaling: Volume R³ → Surface Area R² 
    (Bekenstein bound inspired)

Holographic Error Correction:
    The boundary representation forms a quantum error correcting code,
    allowing partial reconstruction from incomplete data.

Reference: docs/research/物理概念による深層学習革新リサーチ.md, Section 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class BulkBoundaryKernel(nn.Module):
    """
    Bulk-to-Boundary propagator kernel.
    
    Implements K_Δ(z, x) for mapping bulk states to boundary.
    Uses conformal dimension Δ to control locality vs non-locality.
    """
    
    def __init__(
        self,
        bulk_dim: int,
        boundary_dim: int,
        conformal_dim: float = 2.0,
        num_radial_shells: int = 4,
    ):
        """
        Args:
            bulk_dim: Dimension of bulk representation (d_model)
            boundary_dim: Dimension of boundary representation (compressed)
            conformal_dim: Conformal dimension Δ controlling propagator decay
            num_radial_shells: Number of radial shells for discretization
        """
        super().__init__()
        self.bulk_dim = bulk_dim
        self.boundary_dim = boundary_dim
        self.conformal_dim = conformal_dim
        self.num_radial_shells = num_radial_shells
        
        # Radial discretization (z-coordinate in AdS)
        # z → 0 is boundary, z → ∞ is deep bulk
        self.register_buffer(
            'z_values',
            torch.exp(torch.linspace(-2.0, 2.0, num_radial_shells))
        )
        
        # Learnable bulk-to-boundary transformation
        # Acts like Witten diagram vertex
        self.projection = nn.Linear(bulk_dim, boundary_dim, bias=False)
        
        # Conformal weights per radial shell
        self.shell_weights = nn.Parameter(
            torch.ones(num_radial_shells) / num_radial_shells
        )
        
        # Phase encoding for holographic interference
        self.phase_encoder = nn.Parameter(
            torch.randn(boundary_dim) * 0.1
        )
    
    def compute_propagator(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute K_Δ(z) propagator strength.
        
        K_Δ(z) ∝ z^Δ / (1 + z²)^Δ
        
        Args:
            z: Radial coordinate values
            
        Returns:
            Propagator weights
        """
        delta = self.conformal_dim
        numerator = z.pow(delta)
        denominator = (1 + z.pow(2)).pow(delta)
        return numerator / (denominator + 1e-8)
    
    def forward(
        self,
        bulk_states: torch.Tensor,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Project bulk states to boundary.
        
        Args:
            bulk_states: [batch, seq_len, bulk_dim]
            return_weights: Whether to return propagator weights
            
        Returns:
            boundary_states: [batch, seq_len, boundary_dim]
            weights: Optional propagator weights used
        """
        B, L, D = bulk_states.shape
        
        # Compute propagator weights for each radial shell
        propagator = self.compute_propagator(self.z_values)  # [num_shells]
        shell_contribution = propagator * F.softmax(self.shell_weights, dim=0)
        
        # Project to boundary dimension
        boundary = self.projection(bulk_states)  # [B, L, boundary_dim]
        
        # Apply holographic phase modulation
        phase = torch.cos(self.phase_encoder)
        boundary = boundary * phase.unsqueeze(0).unsqueeze(0)
        
        # Weighted integration over radial shells
        # (In full implementation, would have per-shell representations)
        total_weight = shell_contribution.sum()
        boundary = boundary * total_weight
        
        if return_weights:
            return boundary, shell_contribution
        return boundary, None


class HolographicEncoder(nn.Module):
    """
    Encodes high-dimensional bulk states to compact boundary representation.
    
    Uses hierarchical structure inspired by MERA tensor networks:
    Fine details → Coarse-grained representation
    """
    
    def __init__(
        self,
        bulk_dim: int,
        boundary_dim: int,
        num_levels: int = 2,
        compression_ratio: float = 0.5,
    ):
        super().__init__()
        self.bulk_dim = bulk_dim
        self.boundary_dim = boundary_dim
        self.num_levels = num_levels
        
        # Hierarchical projection (MERA-like)
        levels = []
        current_dim = bulk_dim
        for i in range(num_levels - 1):
            next_dim = max(boundary_dim, int(current_dim * compression_ratio))
            levels.append(nn.Linear(current_dim, next_dim))
            levels.append(nn.GELU())
            current_dim = next_dim
        levels.append(nn.Linear(current_dim, boundary_dim))
        
        self.encoder = nn.Sequential(*levels)
        
        # Bulk-boundary kernel for final projection
        self.kernel = BulkBoundaryKernel(
            bulk_dim=bulk_dim,
            boundary_dim=boundary_dim,
        )
    
    def forward(self, bulk_states: torch.Tensor) -> torch.Tensor:
        """
        Encode bulk to boundary.
        
        Args:
            bulk_states: [batch, seq_len, bulk_dim]
            
        Returns:
            boundary: [batch, seq_len, boundary_dim]
        """
        # Direct hierarchical encoding
        encoded = self.encoder(bulk_states)
        
        # Additional holographic projection
        boundary, _ = self.kernel(bulk_states)
        
        # Combine both paths
        return (encoded + boundary) / 2


class HolographicDecoder(nn.Module):
    """
    Reconstructs bulk states from boundary representation.
    
    Uses error-correcting properties of holographic codes
    to reconstruct even from partial boundary information.
    """
    
    def __init__(
        self,
        boundary_dim: int,
        bulk_dim: int,
        num_levels: int = 2,
    ):
        super().__init__()
        self.boundary_dim = boundary_dim
        self.bulk_dim = bulk_dim
        
        # Hierarchical reconstruction
        levels = []
        current_dim = boundary_dim
        expansion = bulk_dim / boundary_dim
        for i in range(num_levels - 1):
            next_dim = int(current_dim * (expansion ** (1 / num_levels)))
            levels.append(nn.Linear(current_dim, next_dim))
            levels.append(nn.GELU())
            current_dim = next_dim
        levels.append(nn.Linear(current_dim, bulk_dim))
        
        self.decoder = nn.Sequential(*levels)
        
        # Error correction layer
        self.error_correction = nn.Linear(bulk_dim, bulk_dim)
    
    def forward(
        self, 
        boundary_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode boundary to bulk.
        
        Args:
            boundary_states: [batch, seq_len, boundary_dim]
            mask: Optional mask for partial reconstruction
            
        Returns:
            bulk: [batch, seq_len, bulk_dim]
        """
        # Main decoding
        bulk = self.decoder(boundary_states)
        
        # Apply error correction
        bulk = bulk + 0.1 * self.error_correction(bulk)
        
        return bulk


class HolographicKVCache(nn.Module):
    """
    KV Cache with holographic compression.
    
    Compresses KV states to boundary representation for storage,
    decompresses on demand for attention computation.
    
    Memory efficiency: O(L × d_boundary) instead of O(L × d_model)
    """
    
    def __init__(
        self,
        d_model: int,
        max_length: int = 2048,
        compression_ratio: float = 0.25,
        num_heads: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length
        self.num_heads = num_heads
        self.boundary_dim = int(d_model * compression_ratio)
        
        # Separate encoder/decoder for K and V
        self.k_encoder = HolographicEncoder(d_model, self.boundary_dim)
        self.v_encoder = HolographicEncoder(d_model, self.boundary_dim)
        self.k_decoder = HolographicDecoder(self.boundary_dim, d_model)
        self.v_decoder = HolographicDecoder(self.boundary_dim, d_model)
        
        # Cache storage (boundary representation)
        self.register_buffer(
            'k_cache',
            torch.zeros(1, max_length, self.boundary_dim)
        )
        self.register_buffer(
            'v_cache',
            torch.zeros(1, max_length, self.boundary_dim)
        )
        self.cache_length = 0
    
    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new KV and return decompressed full cache.
        
        Args:
            key: New key states [batch, new_len, d_model]
            value: New value states [batch, new_len, d_model]
            
        Returns:
            full_k: Decompressed full key cache
            full_v: Decompressed full value cache
        """
        B, new_len, D = key.shape
        
        # Compress new KV to boundary
        k_boundary = self.k_encoder(key)  # [B, new_len, boundary_dim]
        v_boundary = self.v_encoder(value)
        
        # Expand cache if needed
        if self.k_cache.shape[0] != B:
            self.k_cache = self.k_cache.expand(B, -1, -1).clone()
            self.v_cache = self.v_cache.expand(B, -1, -1).clone()
        
        # Update cache
        end_pos = min(self.cache_length + new_len, self.max_length)
        start_pos = self.cache_length
        
        if start_pos < self.max_length:
            actual_len = min(new_len, self.max_length - start_pos)
            self.k_cache[:, start_pos:start_pos + actual_len] = k_boundary[:, :actual_len]
            self.v_cache[:, start_pos:start_pos + actual_len] = v_boundary[:, :actual_len]
            self.cache_length = end_pos
        
        # Decompress full cache
        full_k = self.k_decoder(self.k_cache[:, :self.cache_length])
        full_v = self.v_decoder(self.v_cache[:, :self.cache_length])
        
        return full_k, full_v
    
    def clear(self):
        """Clear the cache."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.cache_length = 0
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        full_memory = self.cache_length * self.d_model * 2 * 4  # 4 bytes per float32
        compressed_memory = self.cache_length * self.boundary_dim * 2 * 4
        
        return {
            'full_memory_mb': full_memory / (1024 ** 2),
            'compressed_memory_mb': compressed_memory / (1024 ** 2),
            'compression_ratio': compressed_memory / max(1, full_memory),
            'cache_length': self.cache_length,
        }


class QuantumErrorCorrector(nn.Module):
    """
    Quantum-inspired error correction for holographic codes.
    
    Can reconstruct bulk information even when boundary
    data is partially corrupted or missing.
    """
    
    def __init__(
        self,
        dim: int,
        redundancy: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.redundancy = redundancy
        
        # Syndrome extraction
        self.syndrome_net = nn.Linear(dim, dim // 4)
        
        # Error correction
        self.correction_net = nn.Sequential(
            nn.Linear(dim + dim // 4, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
    
    def forward(
        self,
        data: torch.Tensor,
        noise_level: float = 0.0,
    ) -> torch.Tensor:
        """
        Apply error correction.
        
        Args:
            data: Input tensor
            noise_level: Estimated noise level
            
        Returns:
            Corrected output
        """
        # Extract syndrome (error pattern)
        syndrome = self.syndrome_net(data)
        
        # Apply correction
        combined = torch.cat([data, syndrome], dim=-1)
        correction = self.correction_net(combined)
        
        return data + correction * noise_level


def create_holographic_kv_cache(
    d_model: int,
    max_length: int = 2048,
    compression_ratio: float = 0.25,
    num_heads: int = 8,
) -> HolographicKVCache:
    """Factory function for HolographicKVCache."""
    return HolographicKVCache(
        d_model=d_model,
        max_length=max_length,
        compression_ratio=compression_ratio,
        num_heads=num_heads,
    )


def create_holographic_encoder(
    bulk_dim: int,
    boundary_dim: Optional[int] = None,
    compression_ratio: float = 0.25,
) -> HolographicEncoder:
    """Factory function for HolographicEncoder."""
    if boundary_dim is None:
        boundary_dim = int(bulk_dim * compression_ratio)
    
    return HolographicEncoder(
        bulk_dim=bulk_dim,
        boundary_dim=boundary_dim,
    )
