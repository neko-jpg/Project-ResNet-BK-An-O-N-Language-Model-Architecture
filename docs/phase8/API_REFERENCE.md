# Phase 8 API Reference

## `src.models.phase8`

### `config`
**`Phase8Config`**
- `enable_entailment_cones`: bool
- `enable_topological_norm`: bool
- ... (and other flags)

**`Phase8Diagnostics`**
- `entailment_violation_rate`: float
- `persistent_entropy`: float
- `curvature_value`: float

### `integrated_model`
**`Phase8IntegratedModel(d_model, n_layers, config=None)`**
- Main entry point. Wraps all sub-components.
- `forward(x)`: Returns `(output, diagnostics_dict)`.

### `entailment`
**`EntailmentCone(d_model)`**
- `forward(u, v)`: Returns `(penalty, aperture)`.
- Checks if $u \implies v$ (i.e., is $v$ in $u$'s cone?).

### `topology`
**`TopologicalNorm(d_model)`**
- `forward(x)`: Applies normalization modulated by topological complexity.
- `_approximate_persistence(x)`: Computes variance of pairwise distances.

### `adaptive`
**`AdaptiveComputation(d_model)`**
- `forward(x, layer_idx, total_layers)`: Returns `(should_exit, probability)`.
- Uses distance from origin as signal.

### `koopman_bridge`
**`KoopmanBridge(d_model)`**
- `forward(eigenfunctions, eigenvalues)`: Maps inputs to Poincaré ball.
- Stable modes -> Boundary. Transient modes -> Origin.

### `sparse_attention`
**`SparseHyperbolicAttention(d_model)`**
- `forward(q, k, v)`: Applies block-sparse masking logic.
- `create_sparse_mask(q, k)`: Generates mask based on centroid distance.

### `kv_cache`
**`HyperbolicKVCache(d_model)`**
- `update(k, v)`: Adds new tokens.
- `_evict()`: Removes tokens based on policy (Keep Local + Keep Origin).

### `curvature`
**`CurvatureAdapter(d_model)`**
- `forward(x)`: Returns adapted curvature $c$.
- Based on hierarchy depth estimation (Norm Variance).
