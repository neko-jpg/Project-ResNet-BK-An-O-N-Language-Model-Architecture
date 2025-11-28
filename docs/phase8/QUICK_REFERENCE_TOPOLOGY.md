# Quick Reference: Topological Normalization

## Concept
Standard LayerNorm normalizes statistics (mean/var) but ignores the geometric/topological structure of the embedding cloud. Topological Norm preserves (or adapts to) the manifold structure.

## Mechanism
1. **Input**: Batch of token embeddings $X \in \mathbb{R}^{B \times N \times D}$.
2. **Metric**: Calculates "Variance of Pairwise Distances".
   - **Clustered (High Structure)**: High variance (Small intra-cluster dists, Large inter-cluster dists).
   - **Uniform (No Structure)**: Low variance.
3. **Modulation**: Adjusts the scale $\gamma$ of the normalization layer.
   - High Variance $\to$ Increase scale (allow clusters to separate).
   - Low Variance $\to$ Decrease scale (keep compact).

## Implementation
Class: `src.models.phase8.topology.TopologicalNorm`

### Usage
```python
norm = TopologicalNorm(d_model=64)
x_normalized = norm(x_input)
diagnostics = norm.get_diagnostics()
print(diagnostics['topo_metric'])
```
