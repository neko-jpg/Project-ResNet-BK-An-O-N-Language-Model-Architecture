import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, Any
from src.models.phase4.adscft_core.geodesic_search import fast_marching_method_cpu
from src.models.phase4.memory_monitor import MemoryMonitor
from src.models.phase4.stability import NumericalStability
from src.models.phase4.adscft_core.holographic_memory import HolographicMemory

class BulkSpaceGenerator(nn.Module):
    """
    Holographic Dual: Generates Bulk Space from Boundary (Token Sequence).

    AdS/CFT Correspondence:
    - Boundary: 1D Token Sequence
    - Bulk: High-dimensional semantic space (extra dimension z)
    - Geodesics: Optimal paths in curved AdS space

    Args:
        d_model: Model dimension.
        bulk_dim: Size of extra dimension (default: log2(d_model)).
        ads_radius: Radius of curvature (L).
        enable_compression: Whether to use holographic memory compression for long contexts.
    """

    def __init__(
        self,
        d_model: int,
        bulk_dim: Optional[int] = None,
        ads_radius: float = 1.0,
        monitor: Optional[MemoryMonitor] = None,
        enable_compression: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.bulk_dim = bulk_dim or int(np.log2(d_model)) if d_model > 1 else 2
        self.ads_radius = ads_radius
        self.monitor = monitor or MemoryMonitor()
        self.dynamic_adjustment = True
        self.enable_compression = enable_compression

        # Boundary to Bulk Projection
        # We project to (bulk_dim, d_model) space
        self.boundary_to_bulk = nn.Linear(d_model, d_model * self.bulk_dim)

        # Potential in Bulk
        self.bulk_potential = nn.Sequential(
             nn.Linear(d_model * self.bulk_dim, d_model),
             nn.GELU(),
             nn.Linear(d_model, 1)
        )

        # Holographic Memory Compression
        if enable_compression:
            self.holographic_memory = HolographicMemory(d_model)
        else:
            self.holographic_memory = None

    def forward(
        self,
        boundary_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            boundary_tokens: (B, N, D)

        Returns:
            bulk_features: (B, N, D) projected back from geodesic
            diagnostics: dict
        """
        B, N, D = boundary_tokens.shape

        # 1. Dynamic Dimension Adjustment (Task 8.2)
        active_bulk_dim = self.bulk_dim
        low_memory_flag = False

        if self.dynamic_adjustment:
            free_mem = self.monitor.get_free_memory()
            # Threshold: 2GB (approx)
            if free_mem < 2 * 1024**3:
                active_bulk_dim = max(2, self.bulk_dim // 2)
                low_memory_flag = True

        # 2. Generate Bulk Coordinates (Initial Guess / Potential Field)
        # (B, N, D) -> (B, N, bulk_dim * D) -> (B, N, bulk_dim, D)
        bulk_coords = self.boundary_to_bulk(boundary_tokens)
        bulk_coords = bulk_coords.view(B, N, self.bulk_dim, D)

        # Slice if dimension is reduced
        if active_bulk_dim < self.bulk_dim:
            bulk_coords = bulk_coords[:, :, :active_bulk_dim, :]

        # 3. Geodesic Search (Fast Marching Method)
        # Finds optimal path through the bulk dimensions
        geodesics = fast_marching_method_cpu(bulk_coords, self.ads_radius)

        # 4. Project back to Boundary (Integrate or Mean)
        # We take the mean of the geodesic path as the "holographic dual" feature
        bulk_features = geodesics.mean(dim=2) # (B, N, D)

        # 5. Holographic Compression (Optional)
        # If enabled, enrich the features with compressed history from memory
        if self.enable_compression and self.holographic_memory is not None:
            # Treat the bulk features as both current state and history for self-enrichment
            # Or use them to retrieve from a larger history if stateful.
            # Here we demonstrate self-compression/refinement within the batch.
            compressed_context = self.holographic_memory(bulk_features, bulk_features)
            bulk_features = bulk_features + compressed_context # Residual

        # Energy Conservation Check (Task 9.2)
        energy_in = NumericalStability.compute_bulk_energy(boundary_tokens)
        energy_out = NumericalStability.compute_bulk_energy(bulk_features)
        energy_stats = NumericalStability.check_energy_conservation(energy_in, energy_out, threshold=0.50) # Relaxed threshold for projection

        diagnostics = {
            'bulk_coords_sample': bulk_coords[:, :1, :, :].detach().cpu(),
            'geodesic_sample': geodesics[:, :1, :, :].detach().cpu(),
            'active_bulk_dim': active_bulk_dim,
            'low_memory_mode': low_memory_flag,
            'energy_stats': energy_stats
        }

        return bulk_features, diagnostics

    def cleanup_bulk_space(self):
        """Clear caches if any"""
        pass
