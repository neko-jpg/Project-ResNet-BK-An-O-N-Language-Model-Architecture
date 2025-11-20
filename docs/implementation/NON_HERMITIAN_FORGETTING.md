# Non-Hermitian Forgetting Implementation

**Status**: ✅ Completed  
**Phase**: Phase 2 - Breath of Life  
**Date**: 2025-11-20

## Overview

Non-Hermitian Forgetting機構は、開放量子系の散逸を模倣した自然な忘却メカニズムです。複素ポテンシャル V - iΓ を用いて、情報の時間発展と減衰を物理的に正しく表現します。

## Physical Background

### Open Quantum Systems

開放量子系では、系が環境と相互作用することでエネルギーが散逸します。この現象は非エルミート演算子で記述されます：

```
H_eff = H_0 + V - iΓ
```

ここで：
- `H_0`: 基底ハミルトニアン
- `V`: 実ポテンシャル（意味的エネルギー）
- `Γ`: 散逸率（常に正、情報の減衰速度）

### Time Evolution

時間発展は以下の式で記述されます：

```
||ψ(t)||² = exp(-2Γt) ||ψ(0)||²
```

これは情報の振幅が指数関数的に減衰することを示しています。

## Implementation

### 1. NonHermitianPotential

複素ポテンシャルを生成するモジュール。

```python
from src.models.phase2 import NonHermitianPotential

potential = NonHermitianPotential(
    d_model=512,
    n_seq=1024,
    base_decay=0.01,        # 最小減衰率
    adaptive_decay=True,    # 入力依存の減衰
    schatten_p=1.0,         # Schatten Norm (1.0 = Nuclear Norm)
    stability_threshold=1e-3
)

# Forward pass
x = torch.randn(batch_size, n_seq, d_model)
V_complex = potential(x)  # (B, N) complex64

# Extract components
V_real = V_complex.real   # Semantic potential
gamma = -V_complex.imag   # Decay rate (positive)
```

#### Key Features

1. **Adaptive Decay**: Γは入力特徴量に依存して動的に変化
2. **Positivity Guarantee**: Softplus活性化により Γ > 0 を保証
3. **Base Decay**: 最小減衰率 `base_decay` を設定可能
4. **Stability Monitoring**: 過減衰（Γ >> |V|）を自動検出

### 2. DissipativeBKLayer

NonHermitianPotentialをBK-Coreに統合するラッパー。

```python
from src.models.phase2 import DissipativeBKLayer

layer = DissipativeBKLayer(
    d_model=512,
    n_seq=1024,
    use_triton=True,        # Tritonカーネルを使用
    base_decay=0.01,
    adaptive_decay=True
)

# Forward pass
x = torch.randn(batch_size, n_seq, d_model)
features, V_complex = layer(x, return_potential=True)

# features: (B, N, 2) [Re(G_ii), Im(G_ii)]
# V_complex: (B, N) complex potential

# Extract decay rate
gamma = layer.get_gamma(x)  # (B, N)
```

#### Integration with BK-Core

DissipativeBKLayerは以下の処理を行います：

1. **Potential Generation**: 入力から複素ポテンシャルを生成
2. **Component Separation**: 実部と虚部を分離
3. **BK-Core Computation**: O(N)で Green関数の対角要素を計算
4. **Gradient Flow**: 複素勾配を安全に伝播

### 3. Stability Monitoring

学習中、システムの安定性を自動監視します。

```python
# Get statistics
stats = potential.get_statistics()

print(f"Mean Γ: {stats['mean_gamma']:.4f}")
print(f"Std Γ: {stats['std_gamma']:.4f}")
print(f"Energy ratio (Γ/|V|): {stats['mean_energy_ratio']:.4f}")
print(f"Max energy ratio: {stats['max_energy_ratio']:.4f}")
```

#### Overdamping Detection

過減衰条件 `Γ/|V| > 10` が検出されると、警告が発行されます：

```
UserWarning: Overdamped system detected: Γ/|V| = 12.34.
Information may vanish too quickly.
Consider reducing base_decay or checking input features.
```

## Architecture Integration

### Phase 2 Block Structure

```
Input (B, N, D)
    ↓
[NonHermitianPotential] → V - iΓ
    ↓
[BK-Core] → G_ii = diag((H - zI)^-1)
    ↓
[DissipativeHebbian] → Fast Weights with decay
    ↓
Output (B, N, D)
```

### Data Flow

1. **Input Features** → NonHermitianPotential
2. **Complex Potential** → BK-Core (実部のみ使用)
3. **Decay Rate Γ** → DissipativeHebbian (Fast Weights減衰)
4. **BK Features** → 次の層へ

## Performance Characteristics

### Computational Complexity

| Component | Complexity | Notes |
|-----------|-----------|-------|
| NonHermitianPotential | O(N·D) | 線形射影 |
| BK-Core | O(N) | Triton最適化済み |
| **Total** | **O(N·D)** | Phase 1と同等 |

### Memory Usage

| Component | Memory | Notes |
|-----------|--------|-------|
| Potential Parameters | 2·D | v_proj + gamma_proj |
| BK-Core Parameters | 2·(N-1) + 1 | h0_super + h0_sub + z |
| Activations | B·N·2 | Complex potential |
| **Total** | **~2D + 2N + B·N·2** | 軽量 |

## Testing

### Unit Tests

```bash
# Run basic tests
python tests/test_non_hermitian_basic.py

# Run comprehensive tests (if available)
pytest tests/test_non_hermitian.py -v
```

### Demo

```bash
# Run interactive demo
python examples/non_hermitian_demo.py
```

デモでは以下を確認できます：
1. 基本的なポテンシャル生成
2. 時間発展シミュレーション
3. BK-Core統合
4. 安定性監視
5. Adaptive vs Fixed decay比較

## Validation Results

### Test Results (2025-11-20)

```
✓ NonHermitianPotential basic test passed
✓ NonHermitianPotential non-adaptive test passed
✓ DissipativeBKLayer basic test passed
✓ DissipativeBKLayer gradient test passed
✓ Stability monitoring test passed

✅ All Non-Hermitian tests passed!
```

### Key Metrics

| Metric | Value | Status |
|--------|-------|--------|
| Γ Positivity | 100% | ✅ Pass |
| Base Decay Guarantee | Γ ≥ 0.01 | ✅ Pass |
| Gradient Flow | No NaN/Inf | ✅ Pass |
| Stability Monitoring | Functional | ✅ Pass |
| BK-Core Integration | Working | ✅ Pass |

## Usage Examples

### Example 1: Basic Usage

```python
import torch
from src.models.phase2 import NonHermitianPotential

# Create module
potential = NonHermitianPotential(
    d_model=512,
    n_seq=1024,
    base_decay=0.01,
    adaptive_decay=True
)

# Generate potential
x = torch.randn(4, 1024, 512)
V_complex = potential(x)

# Use in training
V_complex.real.mean().backward()  # Gradient flows correctly
```

### Example 2: Integration with BK-Core

```python
from src.models.phase2 import DissipativeBKLayer

# Create layer
layer = DissipativeBKLayer(
    d_model=512,
    n_seq=1024,
    use_triton=True
)

# Forward pass
x = torch.randn(4, 1024, 512)
features, potential = layer(x, return_potential=True)

# Extract decay rate for downstream use
gamma = layer.get_gamma(x)
```

### Example 3: Monitoring Stability

```python
# Enable training mode for monitoring
potential.train()

# Run forward passes
for i in range(100):
    x = torch.randn(4, 1024, 512)
    V_complex = potential(x)

# Check statistics
stats = potential.get_statistics()
if stats['max_energy_ratio'] > 10.0:
    print("Warning: Overdamping detected!")
```

## Design Decisions

### 1. Why Softplus for Γ?

Softplus活性化 `softplus(x) = log(1 + exp(x))` を使用する理由：

- **Smooth**: 微分可能で勾配が滑らか
- **Positive**: 常に正の値を出力
- **Unbounded**: 上限がないため柔軟性が高い
- **Numerical Stability**: exp(x)の爆発を防ぐ

代替案（ReLU, ELU）と比較して、Softplusが最も安定しています。

### 2. Why Separate V and Γ?

実部と虚部を別々の線形層で生成する理由：

- **Physical Interpretation**: V（意味）とΓ（忘却）は独立した概念
- **Flexibility**: 異なる学習率や正則化を適用可能
- **Stability**: Γの正値性を独立に保証できる

### 3. Why Monitor Γ/|V| Ratio?

エネルギー比 `Γ/|V|` を監視する理由：

- **Overdamping Detection**: Γ >> |V| の時、情報が即座に消失
- **Physical Meaning**: 減衰支配 vs 振動支配の判定
- **Training Guidance**: 過減衰を検出して学習を調整

## Future Enhancements

### Planned Improvements

1. **Complex BK-Core**: 虚部も含めた完全な複素ポテンシャル対応
2. **Adaptive Base Decay**: base_decayを学習可能パラメータに
3. **Multi-Scale Γ**: 異なる時間スケールの減衰率
4. **Γ Regularization**: 過度な減衰を防ぐ正則化項

### Research Directions

1. **Optimal Decay Schedule**: 学習段階に応じた最適なΓ値の探索
2. **Γ-Attention Coupling**: AttentionとΓの相互作用
3. **Hierarchical Forgetting**: 層ごとに異なる忘却率

## References

### Theoretical Background

1. **Moiseyev, N. (2011)**. "Non-Hermitian Quantum Mechanics"
   - 開放量子系の理論的基礎

2. **Rotter, I. (2009)**. "A non-Hermitian Hamilton operator and the physics of open quantum systems"
   - 非エルミート演算子の物理的解釈

### Implementation References

1. **Phase 1 BK-Core**: `src/models/bk_core.py`
2. **Gradient Safety**: `src/models/phase2/gradient_safety.py`
3. **Design Document**: `.kiro/specs/phase2-breath-of-life/design.md`

## Troubleshooting

### Issue: Γ値が大きすぎる

**症状**: `mean_gamma > 1.0` で情報が急速に消失

**解決策**:
```python
# base_decayを減らす
potential = NonHermitianPotential(
    d_model=512,
    n_seq=1024,
    base_decay=0.001,  # 0.01 → 0.001
    adaptive_decay=True
)
```

### Issue: 過減衰警告が頻発

**症状**: `Overdamped system detected` 警告が多数

**解決策**:
1. 入力特徴量の正規化を確認
2. `base_decay`を減らす
3. `adaptive_decay=False`で固定減衰を試す

### Issue: 勾配が消失

**症状**: `x.grad.norm()` が非常に小さい

**解決策**:
```python
# Gradient clippingを使用
from src.models.phase2 import clip_grad_norm_safe

clip_grad_norm_safe(model.parameters(), max_norm=1.0)
```

## Conclusion

Non-Hermitian Forgetting機構は、物理的に正しい忘却メカニズムを提供します。開放量子系の理論に基づき、情報の自然な減衰を実現しながら、O(N)複雑度とVRAM効率を維持します。

**Key Achievements**:
- ✅ 複素ポテンシャル V - iΓ の実装
- ✅ Γ > 0 の保証（Softplus活性化）
- ✅ BK-Coreとの統合（O(N)維持）
- ✅ 安定性監視（過減衰検出）
- ✅ 勾配フローの安全性確保

Phase 2の基盤として、この機構は後続のDissipative HebbianやMemory Resonanceと統合され、動的な記憶システムを構築します。
