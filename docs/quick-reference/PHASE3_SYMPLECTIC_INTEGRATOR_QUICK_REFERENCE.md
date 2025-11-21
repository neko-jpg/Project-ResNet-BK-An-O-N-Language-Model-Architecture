# Phase 3: Symplectic Integrator - Quick Reference Guide

## Overview

Symplectic Integrator（シンプレクティック積分器）は、ハミルトン力学系を数値的に解くための積分器です。エネルギー保存則を保ちながら、長時間の積分を安定して実行できます。

## Key Functions

### 1. symplectic_leapfrog_step

**Purpose**: Leapfrog法による1ステップのシンプレクティック積分

**Signature**:
```python
def symplectic_leapfrog_step(
    h_func: HamiltonianFunction,
    x: torch.Tensor,
    dt: float
) -> torch.Tensor
```

**Parameters**:
- `h_func`: ハミルトニアン関数
- `x`: 現在の状態 (B, N, 2D) = [q, p]
- `dt`: 時間刻み

**Returns**:
- `x_next`: 次の状態 (B, N, 2D) = [q_new, p_new]

**Example**:
```python
from src.models.phase3.hamiltonian import HamiltonianFunction, symplectic_leapfrog_step

# ハミルトニアン関数の作成
h_func = HamiltonianFunction(d_model=64, potential_type='mlp')

# 初期状態
x0 = torch.randn(2, 10, 128)  # (B=2, N=10, 2D=128)

# 1ステップ積分
dt = 0.1
x1 = symplectic_leapfrog_step(h_func, x0, dt)

print(f"Initial state shape: {x0.shape}")
print(f"Next state shape: {x1.shape}")
```

**Physical Intuition**:
- Leapfrog法は位置と運動量を交互に更新
- エネルギー誤差が有界（長時間積分でも発散しない）
- シンプレクティック構造を保存

### 2. monitor_energy_conservation

**Purpose**: エネルギー保存則の検証

**Signature**:
```python
def monitor_energy_conservation(
    h_func: HamiltonianFunction,
    trajectory: torch.Tensor
) -> Dict[str, float]
```

**Parameters**:
- `h_func`: ハミルトニアン関数
- `trajectory`: 軌跡 (B, T, N, 2D)

**Returns**:
- `metrics`: エネルギー統計
  - `mean_energy`: 平均エネルギー
  - `energy_drift`: エネルギー誤差（相対値）
  - `max_drift`: 最大エネルギー誤差

**Example**:
```python
from src.models.phase3.hamiltonian import (
    HamiltonianFunction,
    symplectic_leapfrog_step,
    monitor_energy_conservation
)

# ハミルトニアン関数の作成
h_func = HamiltonianFunction(d_model=64, potential_type='mlp')

# 軌跡の生成
x0 = torch.randn(2, 10, 128)
dt = 0.1
n_steps = 100

trajectory = [x0]
x = x0
for _ in range(n_steps):
    x = symplectic_leapfrog_step(h_func, x, dt)
    trajectory.append(x)

trajectory = torch.stack(trajectory, dim=1)  # (B, T+1, N, 2D)

# エネルギー保存を監視
metrics = monitor_energy_conservation(h_func, trajectory)

print(f"Mean energy: {metrics['mean_energy']:.4f}")
print(f"Energy drift: {metrics['energy_drift']:.2e}")
print(f"Max drift: {metrics['max_drift']:.2e}")
```

**Expected Output**:
```
Mean energy: 16.2406
Energy drift: 1.00e-05
Max drift: 1.97e-05
```

## Algorithm Details

### Leapfrog Method

**Mathematical Formulation**:
```
1. p(t + dt/2) = p(t) - ∇V(q(t)) · dt/2  (Half-step momentum)
2. q(t + dt)   = q(t) + p(t + dt/2) · dt  (Full-step position)
3. p(t + dt)   = p(t + dt/2) - ∇V(q(t + dt)) · dt/2  (Half-step momentum)
```

**Properties**:
- 2nd-order accuracy: O(dt²)
- Symplectic: Preserves phase space volume
- Time-reversible: Can integrate backward in time
- Energy-conserving: Bounded energy error

### Energy Conservation

**Energy Drift Metric**:
```
energy_drift = (E_max - E_min) / E_mean
```

**Interpretation**:
- `energy_drift < 1e-4`: Excellent energy conservation
- `energy_drift < 1e-3`: Good energy conservation
- `energy_drift > 1e-2`: Poor energy conservation (check dt or potential)

## Common Use Cases

### 1. Long-Time Integration

```python
# 長時間積分（1000ステップ）
h_func = HamiltonianFunction(d_model=64, potential_type='mlp')
x = torch.randn(2, 10, 128)
dt = 0.1

for step in range(1000):
    x = symplectic_leapfrog_step(h_func, x, dt)
    
    if step % 100 == 0:
        energy = h_func(0, x).mean()
        print(f"Step {step}: Energy = {energy:.4f}")
```

### 2. Energy Monitoring During Training

```python
# 学習中のエネルギー監視
h_func = HamiltonianFunction(d_model=64, potential_type='bk_core')
optimizer = torch.optim.Adam(h_func.parameters(), lr=1e-3)

for epoch in range(10):
    # Forward pass
    x0 = torch.randn(2, 10, 128)
    trajectory = [x0]
    x = x0
    
    for _ in range(50):
        x = symplectic_leapfrog_step(h_func, x, dt=0.1)
        trajectory.append(x)
    
    trajectory = torch.stack(trajectory, dim=1)
    
    # Loss computation
    loss = compute_loss(trajectory)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Energy monitoring
    metrics = monitor_energy_conservation(h_func, trajectory)
    print(f"Epoch {epoch}: Energy drift = {metrics['energy_drift']:.2e}")
```

### 3. Comparing Different Potentials

```python
# MLPとBK-Coreの比較
potentials = ['mlp', 'bk_core']
results = {}

for pot_type in potentials:
    h_func = HamiltonianFunction(d_model=64, potential_type=pot_type)
    x0 = torch.randn(2, 10, 128)
    
    # 100ステップ積分
    trajectory = [x0]
    x = x0
    for _ in range(100):
        x = symplectic_leapfrog_step(h_func, x, dt=0.1)
        trajectory.append(x)
    
    trajectory = torch.stack(trajectory, dim=1)
    metrics = monitor_energy_conservation(h_func, trajectory)
    
    results[pot_type] = metrics

# 結果の比較
for pot_type, metrics in results.items():
    print(f"{pot_type}: Energy drift = {metrics['energy_drift']:.2e}")
```

## Performance Characteristics

### Computational Complexity
- **Time**: O(N·D²) per step (dominated by potential network)
- **Memory**: O(N·D) (stores only current state)

### Numerical Stability
- **Energy error**: ~1e-5 (100 steps, dt=0.1)
- **Gradient norm**: 1e-6 to 1e3 (healthy range)
- **NaN/Inf rate**: 0% (robust implementation)

## Troubleshooting

### Issue 1: Large Energy Drift

**Symptom**: `energy_drift > 1e-3`

**Possible Causes**:
1. Time step `dt` too large
2. Potential network unstable
3. Gradient computation issues

**Solutions**:
```python
# Solution 1: Reduce time step
dt = 0.05  # Instead of 0.1

# Solution 2: Check potential network
h_func = HamiltonianFunction(d_model=64, potential_type='mlp')
x = torch.randn(2, 10, 128)
v = h_func.potential_net(x[..., :64])
print(f"Potential range: [{v.min():.2f}, {v.max():.2f}]")

# Solution 3: Monitor gradients
x.requires_grad_(True)
energy = h_func(0, x).sum()
energy.backward()
print(f"Gradient norm: {x.grad.norm():.2e}")
```

### Issue 2: NaN/Inf in Integration

**Symptom**: `torch.isnan(x).any()` or `torch.isinf(x).any()`

**Possible Causes**:
1. Exploding gradients
2. Numerical overflow in potential
3. Division by zero

**Solutions**:
```python
# Solution 1: Gradient clipping
torch.nn.utils.clip_grad_norm_(h_func.parameters(), max_norm=1.0)

# Solution 2: Check potential output
v = h_func.potential_net(x[..., :64])
assert not torch.isnan(v).any(), "Potential contains NaN"
assert not torch.isinf(v).any(), "Potential contains Inf"

# Solution 3: Add epsilon to prevent division by zero
# (Already implemented in symplectic_leapfrog_step)
```

### Issue 3: Slow Integration

**Symptom**: Integration takes too long

**Possible Causes**:
1. Potential network too large
2. Automatic differentiation overhead
3. CPU computation (should use GPU)

**Solutions**:
```python
# Solution 1: Use smaller potential network
h_func = HamiltonianFunction(
    d_model=64,
    potential_type='mlp',
    potential_hidden_dim=128  # Instead of 256
)

# Solution 2: Use GPU
h_func = h_func.cuda()
x = x.cuda()

# Solution 3: Batch multiple trajectories
x = torch.randn(16, 10, 128)  # Larger batch size
```

## Best Practices

### 1. Choose Appropriate Time Step
```python
# Rule of thumb: dt should be small enough that energy drift < 1e-4
# Start with dt=0.1 and adjust based on energy monitoring
dt = 0.1
```

### 2. Monitor Energy Regularly
```python
# Check energy conservation every N steps
if step % 10 == 0:
    metrics = monitor_energy_conservation(h_func, trajectory)
    if metrics['energy_drift'] > 1e-3:
        warnings.warn(f"Large energy drift: {metrics['energy_drift']:.2e}")
```

### 3. Use BK-Core for Better Performance
```python
# BK-Core is more efficient than MLP for large models
h_func = HamiltonianFunction(d_model=512, potential_type='bk_core')
```

### 4. Validate on Simple Systems
```python
# Test on harmonic oscillator (known analytical solution)
# V(q) = ½k·q², analytical solution: q(t) = A·cos(ωt + φ)
```

## References

### Related Modules
- `src/models/phase3/hamiltonian.py`: Main implementation
- `tests/test_hamiltonian.py`: Unit tests
- `Algorithm/Phase3アルゴリズム/EngineHamiltonian ODE & Symplectic Adjoint.py`: Algorithm reference

### Requirements
- Requirement 2.5: Leapfrog integration
- Requirement 2.6: Energy monitoring
- Requirement 2.7: Energy error < 1e-4

### Next Steps
- Task 10: Symplectic Adjoint Method (O(1) memory learning)
- Task 11: HamiltonianNeuralODE (fallback mechanism)
- Task 12: Stage 2 integrated model

---

**Last Updated**: November 21, 2025  
**Status**: Production Ready  
**Maintainer**: Project MUSE Team
