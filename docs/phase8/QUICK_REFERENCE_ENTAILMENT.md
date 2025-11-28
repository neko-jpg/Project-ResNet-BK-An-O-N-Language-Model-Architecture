# Quick Reference: Entailment Cones

## Concept
In Hyperbolic Entailment Cones (Ganea et al., 2018), hierarchical relationships are modeled by containment regions in the Poincaré ball.

- **Root**: Origin (0,0)
- **Hierarchy**: Depth $\approx$ Distance from origin
- **Entailment**: $u \implies v$ if $v$ lies within the cone of $u$.

## Implementation
Class: `src.models.phase8.entailment.EntailmentCone`

### Key Methods
- `forward(u, v)`: Calculates logical violation penalty.
  - Returns `0.0` if $u$ logically entails $v$.
  - Returns positive value if $v$ is outside $u$'s cone.

### Usage
```python
cone = EntailmentCone(d_model=64)
penalty, angle = cone(parent_vec, child_vec)
loss = penalty.mean()
```

### Parameters
- `aperture`: Learnable parameter controlling the width of the cone.
  - Wide aperture = General concept (entails many things).
  - Narrow aperture = Specific concept.
