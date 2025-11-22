"""
Sparse Knot Representation for Topological Memory

Implements efficient storage and retrieval of knot-based concepts
using Sparse Matrix Product States (MPS) and Zarr for persistence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from collections import OrderedDict
import os
import warnings

try:
    import zarr
    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False
    warnings.warn("zarr not found. Persistence will be disabled.")

try:
    import aiofiles
    import asyncio
    HAS_ASYNCIO = True
except ImportError:
    HAS_ASYNCIO = False

from src.models.phase4.topological_memory.knot_invariants import KnotInvariantCalculator

class SparseKnotRepresentation:
    """
    Sparse representation of knots for memory efficiency.

    Strategy:
    - Concepts -> Knots via dimensionality reduction/projection
    - Storage: Sparse tensors / Zarr chunks
    - Retrieval: Topological similarity

    Args:
        d_model: Model dimension
        max_knots: Maximum number of knots to store (default: 1000)
        compression_ratio: Ratio for knot coordinate dimensionality (default: 0.1)
        storage_path: Path to Zarr storage
    """

    def __init__(
        self,
        d_model: int,
        max_knots: int = 1000,
        compression_ratio: float = 0.1,
        storage_path: str = 'data/phase4_knot_memory.zarr',
        cache_capacity: int = 100
    ):
        self.d_model = d_model
        self.max_knots = max_knots
        self.compression_ratio = compression_ratio
        self.storage_path = storage_path
        self.cache_capacity = cache_capacity

        # In-memory sparse index
        self.knot_indices = []
        self.knot_values = {} # Changed to dict for ID-based fallback
        self.metadata_store = {}

        # LRU Cache for tensors (Task 8.2)
        self.cache = OrderedDict()

        # Initialize Zarr store
        if HAS_ZARR:
            try:
                self.zarr_store = zarr.open(storage_path, mode='a')
            except Exception as e:
                warnings.warn(f"Failed to open Zarr store: {e}")
                self.zarr_store = None
        else:
            self.zarr_store = None

        # Calculator for similarity
        self.calculator = KnotInvariantCalculator()

        # Projection matrix for concept encoding (random orthogonal)
        n_coords = int(d_model * compression_ratio) * 3
        self.projection = torch.randn(d_model, n_coords) / np.sqrt(d_model)

    def encode_concept_to_knot(
        self,
        concept: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode a concept vector into a knot (3D coordinates).

        Args:
            concept: (D,) concept vector

        Returns:
            knot_coords: (N, 3) knot coordinates
        """
        device = concept.device
        if self.projection.device != device:
            self.projection = self.projection.to(device)

        # Project concept to lower dimension
        # (D,) @ (D, N*3) -> (N*3,)
        flat_coords = torch.matmul(concept, self.projection)

        # Reshape to (N, 3)
        n_points = flat_coords.shape[0] // 3
        knot_coords = flat_coords.reshape(n_points, 3)

        # Normalize to unit sphere to keep coordinates bounded
        knot_coords = F.normalize(knot_coords, dim=-1)

        # Add topological structure (braiding/twisting) based on value magnitude
        # We modulate the Z-coordinate with a sine wave whose frequency/phase depends on the concept
        # This helps create distinct crossing patterns for different concepts
        t = torch.linspace(0, 2 * np.pi, n_points, device=device)

        # Use concept statistics to vary the topology
        # Increase frequency range to create more complex knots (more crossings)
        # Use modulo to keep it in a reasonable range but varied
        # Make dependence on concept stronger to ensure distinct knots
        val_sum = concept.sum().item()
        freq = 8.0 + (abs(val_sum) * 10.0) % 12.0
        phase = concept.norm().item() * 10.0

        # Apply perturbation to Z (twist) and also X/Y modulation to ensure 2D projection has crossings
        # Use higher amplitude for distinctness
        twist_z = 1.2 * torch.sin(freq * t + phase)
        twist_xy_1 = 0.5 * torch.cos(freq * 0.7 * t + phase)
        twist_xy_2 = 0.5 * torch.sin(freq * 0.3 * t)

        knot_coords[:, 0] += twist_xy_1
        knot_coords[:, 1] -= twist_xy_1
        knot_coords[:, 2] += twist_z + twist_xy_2

        # Re-normalize
        knot_coords = F.normalize(knot_coords, dim=-1)

        return knot_coords

    def compute_knot_similarity(
        self,
        knot1_coords: torch.Tensor,
        knot2_coords: torch.Tensor
    ) -> float:
        """
        Compute topological similarity between two knots.

        Args:
            knot1_coords: (N1, 3)
            knot2_coords: (N2, 3)

        Returns:
            similarity: 0.0 to 1.0
        """
        # Compute invariants
        # Note: Jones coeffs are tensors
        jones1 = self.calculator.compute_jones_polynomial(knot1_coords)
        jones2 = self.calculator.compute_jones_polynomial(knot2_coords)

        # Pad to same length if necessary
        len1 = jones1.shape[0]
        len2 = jones2.shape[0]
        max_len = max(len1, len2)

        j1 = F.pad(jones1, (0, max_len - len1))
        j2 = F.pad(jones2, (0, max_len - len2))

        # Distance
        distance = torch.norm(j1 - j2)

        # Convert to similarity
        similarity = 1.0 / (1.0 + distance.item())

        return similarity

    def add_knot(
        self,
        knot_coords: torch.Tensor,
        metadata: Dict[str, Any]
    ):
        """
        Add a knot to memory.
        Uses Zarr for persistence and avoids permanent RAM storage (Lazy Loading).

        Args:
            knot_coords: (N, 3)
            metadata: dict
        """
        knot_id = len(self.knot_indices)
        self.knot_indices.append(knot_id)
        self.metadata_store[knot_id] = metadata

        # Persist
        if HAS_ZARR and self.zarr_store is not None:
            # Trigger async write or sync write
            if HAS_ASYNCIO:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(self._async_write_to_zarr(knot_id, knot_coords, metadata))
                    else:
                        self._sync_write_to_zarr(knot_id, knot_coords, metadata)
                except RuntimeError:
                     self._sync_write_to_zarr(knot_id, knot_coords, metadata)
            else:
                self._sync_write_to_zarr(knot_id, knot_coords, metadata)
        else:
            # Fallback: Store in memory if Zarr not available
            # This is NOT memory efficient but required if no Zarr
            self.knot_values[knot_id] = knot_coords.detach().cpu()

    def get_knot(self, knot_id: int) -> Optional[torch.Tensor]:
        """
        Retrieve a knot tensor with LRU caching.

        Args:
            knot_id: ID of the knot

        Returns:
            knot_coords: (N, 3) or None
        """
        # 1. Check Cache
        if knot_id in self.cache:
            self.cache.move_to_end(knot_id)
            return self.cache[knot_id]

        # 2. Check Fallback Memory
        if knot_id in self.knot_values:
            tensor = self.knot_values[knot_id]
            self._update_cache(knot_id, tensor)
            return tensor

        # 3. Load from Zarr
        if HAS_ZARR and self.zarr_store is not None:
            key = f'knot_{knot_id}'
            if key in self.zarr_store:
                try:
                    arr = self.zarr_store[key][:] # Load numpy array
                    tensor = torch.from_numpy(arr)
                    self._update_cache(knot_id, tensor)
                    return tensor
                except Exception as e:
                    print(f"Error reading knot {knot_id} from Zarr: {e}")
                    return None

        return None

    def _update_cache(self, knot_id: int, tensor: torch.Tensor):
        """Update LRU cache with new item."""
        if len(self.cache) >= self.cache_capacity:
            self.cache.popitem(last=False) # Remove LRU
        self.cache[knot_id] = tensor
        self.cache.move_to_end(knot_id)

    async def _async_write_to_zarr(
        self,
        knot_id: int,
        knot_coords: torch.Tensor,
        metadata: Dict[str, Any]
    ):
        """Async write to Zarr."""
        # Convert to numpy
        arr = knot_coords.detach().cpu().numpy()

        # Use thread pool for blocking I/O
        # zarr is not natively async
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._write_zarr_chunk, knot_id, arr, metadata)

    def _sync_write_to_zarr(
        self,
        knot_id: int,
        knot_coords: torch.Tensor,
        metadata: Dict[str, Any]
    ):
        """Synchronous fallback."""
        arr = knot_coords.detach().cpu().numpy()
        self._write_zarr_chunk(knot_id, arr, metadata)

    def _write_zarr_chunk(self, knot_id, arr, metadata):
        """Worker function for writing."""
        try:
            key = f'knot_{knot_id}'
            self.zarr_store[key] = arr
            self.zarr_store[key].attrs.update(metadata)
        except Exception as e:
            print(f"Error writing to Zarr: {e}")
