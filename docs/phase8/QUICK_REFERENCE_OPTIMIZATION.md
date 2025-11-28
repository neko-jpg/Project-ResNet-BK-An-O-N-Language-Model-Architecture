# Quick Reference: Adaptive Computation & Optimization

## 1. Adaptive Computation
Dynamically adjusts network depth per token based on complexity.

- **Signal**: Hyperbolic Radius (Distance from Origin).
- **Logic**:
  - **Near Origin (Root/General)**: High probability of early exit.
  - **Near Boundary (Leaf/Specific)**: Lower probability of early exit (requires full processing).
- **Class**: `src.models.phase8.adaptive.AdaptiveComputation`

## 2. Sparse Hyperbolic Attention
Reduces $O(N^2)$ attention to $O(N \cdot K)$ by exploiting geometric locality.

- **Logic**: Divides sequence into blocks. Computes distance between block centroids. Only attends to closest $K$ blocks.
- **Class**: `src.models.phase8.sparse_attention.SparseHyperbolicAttention`

## 3. KV Cache Compression
Maintains infinite context with finite memory by intelligent eviction.

- **Policy**:
  1. **Keep Local**: Always keep the most recent window (e.g., last 10 tokens).
  2. **Keep Central**: From history, keep tokens closest to the origin (General Context).
- **Class**: `src.models.phase8.kv_cache.HyperbolicKVCache`
