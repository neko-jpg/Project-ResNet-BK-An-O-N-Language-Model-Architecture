# Phase 3: Symplectic Adjoint Method - Quick Reference

## 概要

**Symplectic Adjoint Method**は、O(1)メモリでハミルトニアンODEを学習するための手法です。

### 主な特徴

- **メモリ効率**: O(1)メモリ（ステップ数Tに依存しない）
- **数値安定性**: 再構成誤差を監視し、閾値を超えた場合は自動フォールバック
- **物理的整合性**: シンプレクティック構造を保存

## 実装ファイル

- **実装**: `src/models/phase3/symplectic_adjoint.py`
- **テスト**: `tests/test_symplectic_adjoint.py`

## 基本的な使い方

### 1. Symplectic Adjointの使用

```python
from src.models.phase3.hamiltonian import HamiltonianFunction
from src.models.phase3.symplectic_adjoint import SymplecticAdjoint

# ハミルトニアン関数の作成
h_func = HamiltonianFunction(d_model=512, potential_type='bk_core')

# 初期状態
B, N, D = 4, 128, 512
q0 = torch.randn(B, N, D)
p0 = torch.randn(B, N, D)
x0 = torch.cat([q0, p0], dim=-1).requires_grad_(True)

# Symplectic Adjointで時間発展
x_final = SymplecticAdjoint.apply(
    h_func,
    x0,
    t_span=(0.0, 1.0),
    dt=0.1,
    recon_threshold=1e-5,
    *h_func.parameters()
)

# 損失計算と逆伝播
loss = x_final.sum()
loss.backward()  # O(1)メモリで勾配計算

print(f"x0.grad: {x0.grad.shape}")  # (B, N, 2D)
```

### 2. 再構成誤差の処理

```python
from src.models.phase3.symplectic_adjoint import ReconstructionError

try:
    x_final = SymplecticAdjoint.apply(
        h_func, x0, (0.0, 5.0), dt=0.1, recon_threshold=1e-5,
        *h_func.parameters()
    )
    loss = x_final.sum()
    loss.backward()
except ReconstructionError as e:
    print(f"Reconstruction error: {e.error:.2e} > {e.threshold:.2e}")
    print(f"Failed at step: {e.step}")
    # フォールバック処理（例: Gradient Checkpointing）
```

## アルゴリズム

### Forward Pass（順伝播）

```
1. 初期状態 x₀ = [q₀, p₀] を保存
2. Leapfrog積分で x₁, x₂, ..., x_T を計算
3. 最終状態 x_T のみを保存（中間状態は破棄）
```

**メモリ使用量**: O(1)

### Backward Pass（逆伝播）

```
1. 随伴状態の初期化: a_T = ∂L/∂x_T
2. 時間を逆再生: t = T → T-1 → ... → 0
3. 各ステップで:
   - 状態を逆積分: x_{t-1} = Leapfrog⁻¹(x_t)
   - 随伴状態を更新: a_{t-1} = a_t + a_t·∂f/∂x·dt
   - パラメータ勾配を累積: ∂L/∂θ += a_t·∂f/∂θ·dt
4. 再構成誤差を監視（10ステップごと）
```

**メモリ使用量**: O(1)

## 物理的直観

### エネルギー保存則

ハミルトン系は時間反転対称性を持つため、逆時間積分により元の状態を再構成できます。

```
Forward:  x₀ → x₁ → x₂ → ... → x_T
Backward: x₀ ← x₁ ← x₂ ← ... ← x_T
```

### 再構成誤差

逆時間積分時の数値誤差を監視します。

```
再構成誤差 = |x_check - x_t|
```

- 誤差が小さい: 数値積分が安定
- 誤差が大きい: カオス的挙動または数値不安定性

## テスト結果

### 1. Forward Pass Test

```
✓ Forward pass: shape=torch.Size([2, 8, 64]), mean=0.0424, std=1.2754
```

### 2. Backward Pass Test

```
✓ Backward pass: x0.grad mean=1.5033e+00, std=4.9921e-01
```

### 3. Memory Efficiency Test

```
  t_end=0.5: 17.37 MB
  t_end=1.0: 17.37 MB
  t_end=2.0: 17.37 MB
✓ Memory efficiency: O(1) confirmed (increase ratio: 1.00x)
```

### 4. Reconstruction Error Monitoring Test

```
✓ Reconstruction error monitoring: error=2.38e-07 > threshold=1.00e-07 at step 10
```

### 5. Comparison with Full Backprop Test

```
✓ Comparison with Full Backprop:
  Mean relative error: 7.3664e-03
  Max relative error: 4.0865e-02
```

## パラメータ

### SymplecticAdjoint.apply()

| パラメータ | 型 | 説明 | デフォルト |
|-----------|-----|------|-----------|
| `h_func` | `HamiltonianFunction` | ハミルトニアン関数 | 必須 |
| `x0` | `torch.Tensor` | 初期状態 (B, N, 2D) | 必須 |
| `t_span` | `Tuple[float, float]` | 時間範囲 (t0, t1) | 必須 |
| `dt` | `float` | 時間刻み | 必須 |
| `recon_threshold` | `float` | 再構成誤差の閾値 | 必須 |
| `*params` | `torch.Tensor` | h_funcのパラメータ | 必須 |

### ReconstructionError

| 属性 | 型 | 説明 |
|------|-----|------|
| `error` | `float` | 再構成誤差の値 |
| `threshold` | `float` | 閾値 |
| `step` | `int` | エラーが発生したステップ |

## 数値安定性

### 再構成誤差の閾値

推奨値: `1e-5`

- **厳しすぎる閾値** (< 1e-6): ReconstructionErrorが頻発
- **緩すぎる閾値** (> 1e-4): 勾配の精度が低下

### 時間刻み

推奨値: `0.1`

- **小さすぎる刻み** (< 0.01): 計算コストが増加
- **大きすぎる刻み** (> 0.5): 数値誤差が増加

## メモリ使用量の比較

| 手法 | メモリ使用量 | 計算時間 |
|------|-------------|---------|
| Full Backprop | O(T) | 1x |
| Gradient Checkpointing | O(√T) | 1.5x |
| Symplectic Adjoint | O(1) | 2x |

**T**: 積分ステップ数

## トラブルシューティング

### ReconstructionErrorが頻発する

**原因**: 数値不安定性またはカオス的挙動

**解決策**:
1. 時間刻み `dt` を小さくする（例: 0.1 → 0.05）
2. 再構成誤差の閾値を緩める（例: 1e-5 → 1e-4）
3. Gradient Checkpointingにフォールバック

### 勾配がNaNになる

**原因**: ポテンシャルネットワークの出力が発散

**解決策**:
1. ポテンシャルネットワークの初期化を確認
2. 学習率を下げる
3. Gradient Clippingを使用

### メモリ使用量が増加する

**原因**: 中間状態が保存されている

**解決策**:
1. `torch.no_grad()` を使用していないか確認
2. `retain_graph=True` を使用していないか確認

## 関連ドキュメント

- [Phase 3 Hamiltonian Quick Reference](PHASE3_HAMILTONIAN_QUICK_REFERENCE.md)
- [Phase 3 Symplectic Integrator Quick Reference](PHASE3_SYMPLECTIC_INTEGRATOR_QUICK_REFERENCE.md)
- [Phase 3 Implementation Guide](../PHASE3_IMPLEMENTATION_GUIDE.md)

## Requirements

- Requirements: 2.8, 2.9, 2.10, 2.11, 2.12

## 作成日

2025-11-21

## ステータス

✅ 実装完了・テスト済み
