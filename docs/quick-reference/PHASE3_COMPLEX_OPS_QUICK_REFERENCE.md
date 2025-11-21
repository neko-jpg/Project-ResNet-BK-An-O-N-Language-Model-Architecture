# Phase 3: Complex Operations - クイックリファレンス

## 概要

Task 2「ComplexLinear層の実装」が完了しました。このドキュメントは、ComplexLinear、ModReLU、ComplexLayerNormの使用方法をまとめたクイックリファレンスです。

## 実装完了日

2025-11-21

## 実装内容

### 1. ComplexLinear（複素線形層）

複素重み行列を使用して複素入力を変換する層です。

**特徴**:
- Planar形式のメモリレイアウト（実部と虚部を分離）
- complex32（float16）対応でメモリ効率50%削減
- Xavier初期化（複素数版）
- ComplexTensorとcomplex64の両方に対応

**使用例**:

```python
from src.models.phase3 import ComplexLinear, ComplexTensor

# ComplexLinear層の作成
layer = ComplexLinear(
    in_features=64,
    out_features=128,
    bias=True,
    use_complex32=True  # float16を使用（メモリ効率化）
)

# ComplexTensor入力
real = torch.randn(4, 10, 64, dtype=torch.float16)
imag = torch.randn(4, 10, 64, dtype=torch.float16)
x = ComplexTensor(real, imag)

# Forward pass
y = layer(x)  # (4, 10, 128)

# complex64入力（互換性）
x_complex64 = torch.randn(4, 10, 64, dtype=torch.complex64)
y_complex64 = layer(x_complex64)  # (4, 10, 128)
```

**数学的定式化**:
```
W = A + iB (複素重み行列)
z = x + iy (複素入力)
Wz = (Ax - By) + i(Bx + Ay)
```

**メモリ使用量**:
- use_complex32=True: 4 bytes/element (50%削減)
- use_complex32=False: 8 bytes/element (complex64相当)

### 2. ModReLU（位相保存活性化関数）

複素数の振幅をフィルタリングしながら、位相を保存する活性化関数です。

**特徴**:
- 振幅（情報の強さ）をReLUでフィルタリング
- 位相（情報の方向性）を保存
- 学習可能なバイアスパラメータ

**使用例**:

```python
from src.models.phase3 import ModReLU, ComplexTensor

# ModReLU層の作成
modrelu = ModReLU(
    features=64,
    use_half=True  # float16を使用
)

# ComplexTensor入力
x = ComplexTensor(
    torch.randn(4, 10, 64, dtype=torch.float16),
    torch.randn(4, 10, 64, dtype=torch.float16)
)

# Forward pass
y = modrelu(x)  # 位相が保存される

# 位相の確認
phase_before = x.angle()
phase_after = y.angle()
# phase_before ≈ phase_after（振幅がゼロでない場合）
```

**数学的定式化**:
```
z' = ReLU(|z| + b) · z / |z|

where:
    |z| = √(real² + imag²)  (振幅)
    z / |z|  (位相を保存)
    b: 学習可能なバイアス
```

**物理的直観**:
- 振幅: 情報の「強さ」を表す → ReLUでフィルタリング
- 位相: 情報の「方向性」を表す → 保存

### 3. ComplexLayerNorm（複素正規化層）

複素平面上で正規化を行う層です。

**特徴**:
- 複素平均と複素分散を計算
- 学習可能なアフィン変換パラメータ（実数）
- ComplexTensorとcomplex64の両方に対応

**使用例**:

```python
from src.models.phase3 import ComplexLayerNorm, ComplexTensor

# ComplexLayerNorm層の作成
norm = ComplexLayerNorm(
    normalized_shape=64,
    eps=1e-5,
    elementwise_affine=True  # アフィン変換を使用
)

# ComplexTensor入力
x = ComplexTensor(
    torch.randn(4, 10, 64, dtype=torch.float16),
    torch.randn(4, 10, 64, dtype=torch.float16)
)

# Forward pass
y = norm(x)  # 正規化後の平均≈0、分散≈1

# 正規化の確認
mean = y.mean(dim=(0, 1))
print(mean.abs().mean())  # ~0.0
```

**数学的定式化**:
```
z' = γ · (z - μ) / √(σ² + ε) + β

where:
    μ = E[z]  (複素平均)
    σ² = E[|z - μ|²]  (複素分散)
    γ, β: 学習可能なアフィン変換パラメータ（実数）
```

## 統合使用例

ComplexLinear、ModReLU、ComplexLayerNormを組み合わせた例：

```python
import torch
import torch.nn as nn
from src.models.phase3 import ComplexTensor, ComplexLinear, ModReLU, ComplexLayerNorm

class ComplexBlock(nn.Module):
    """複素数ニューラルネットワークのブロック"""
    
    def __init__(self, d_model):
        super().__init__()
        self.norm = ComplexLayerNorm(d_model)
        self.linear = ComplexLinear(d_model, d_model)
        self.activation = ModReLU(d_model)
    
    def forward(self, x):
        # x: ComplexTensor (B, N, D)
        
        # 正規化
        x = self.norm(x)
        
        # 線形変換
        x = self.linear(x)
        
        # 活性化
        x = self.activation(x)
        
        return x

# 使用例
block = ComplexBlock(d_model=64)

# 入力
x = ComplexTensor(
    torch.randn(4, 10, 64, dtype=torch.float16),
    torch.randn(4, 10, 64, dtype=torch.float16)
)

# Forward pass
y = block(x)  # (4, 10, 64)
```

## テスト結果

すべてのテストが成功しました：

```bash
$ python -m pytest tests/test_complex_ops.py -v
=========================== 13 passed in 4.09s ============================

TestComplexLinear:
  ✓ test_output_shape_complex_tensor
  ✓ test_output_shape_complex64
  ✓ test_complex_matrix_multiplication
  ✓ test_gradient_computation
  ✓ test_xavier_initialization
  ✓ test_bias_disabled

TestModReLU:
  ✓ test_phase_preservation
  ✓ test_amplitude_filtering
  ✓ test_gradient_computation

TestComplexLayerNorm:
  ✓ test_normalization_accuracy
  ✓ test_affine_transformation
  ✓ test_gradient_computation
  ✓ test_complex_tensor_input
```

## 要件達成状況

| 要件 | 内容 | 状態 |
|------|------|------|
| Requirement 1.5 | ComplexLinearの基本構造 | ✅ 完了 |
| Requirement 1.6 | 複素行列積の実装 | ✅ 完了 |
| Requirement 1.7 | Xavier初期化（複素数版） | ✅ 完了 |
| Requirement 1.8 | ComplexLinear単体テスト | ✅ 完了 |
| Requirement 1.9 | ModReLU数式の実装 | ✅ 完了 |
| Requirement 1.10 | ModReLU単体テスト | ✅ 完了 |
| Requirement 1.11 | ComplexLayerNorm実装 | ✅ 完了 |
| Requirement 1.12 | ComplexLayerNorm単体テスト | ✅ 完了 |

## ファイル構成

```
src/models/phase3/
├── __init__.py                 # エクスポート定義（更新済み）
├── complex_tensor.py           # ComplexTensorクラス（既存）
└── complex_ops.py              # ComplexLinear, ModReLU, ComplexLayerNorm（新規）

tests/
└── test_complex_ops.py         # 単体テスト（新規）

docs/quick-reference/
└── PHASE3_COMPLEX_OPS_QUICK_REFERENCE.md  # このファイル
```

## 次のステップ

Task 2が完了したので、次はTask 3「ModReLU活性化関数の実装」に進みます。

ただし、ModReLUは既にTask 2で実装済みなので、Task 3はスキップして、Task 4「ComplexLayerNormの実装」に進むことができます。

実際には、Task 2でComplexLinear、ModReLU、ComplexLayerNormの3つをまとめて実装したため、Task 3とTask 4もすでに完了しています。

次は**Task 5「Complex Embedding層の実装」**に進むことをお勧めします。

## 参考資料

- [Phase 3 Requirements](.kiro/specs/phase3-physics-transcendence/requirements.md)
- [Phase 3 Design](.kiro/specs/phase3-physics-transcendence/design.md)
- [Phase 3 Tasks](.kiro/specs/phase3-physics-transcendence/tasks.md)
- [ComplexTensor Quick Reference](PHASE3_COMPLEX_TENSOR_QUICK_REFERENCE.md)

## 物理的直観

### 複素数ニューラルネットワークの意義

複素数ニューラルネットワークは、実部（振幅）と虚部（位相）を独立かつ相互作用させながら処理します。

**言語処理への応用**:
- **振幅**: 単語の「意味の強さ」を表現
- **位相**: 単語の「文脈的な方向性」を表現

**例**: 否定形の処理
- "good"（良い）: 振幅=大、位相=0°
- "not good"（良くない）: 振幅=大、位相=180°（位相反転）

複素数の干渉効果により、否定形・皮肉・多義語などの複雑な言語現象を自然にモデリングできます。

### メモリ効率化の重要性

Phase 3では、複素数化によりパラメータ数が2倍になります。これを補うため、complex32（float16）を使用してメモリ使用量を50%削減します。

**メモリ比較**:
- 実数（float32）: 4 bytes/element
- 複素数（complex64）: 8 bytes/element（2倍）
- 複素数（complex32）: 4 bytes/element（実数と同じ）

これにより、8GB VRAM制約を守りながら、複素数ニューラルネットワークを実現できます。

---

**作成日**: 2025-11-21  
**最終更新**: 2025-11-21  
**ステータス**: 完了
