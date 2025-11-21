# Phase 3: Complex Embedding - Quick Reference

## 概要

ComplexEmbeddingは、Token EmbeddingとPosition Embeddingを複素数化し、Phase 2のZetaEmbeddingを継承して統合します。

## 主要機能

### 1. ComplexEmbedding

複素数埋め込み層。実部と虚部を独立に学習します。

```python
from src.models.phase3 import ComplexEmbedding

# 基本的な使用方法
embedding = ComplexEmbedding(
    vocab_size=50000,
    d_model=512,
    max_seq_len=2048,
    use_complex32=True  # メモリ効率優先
)

# Forward pass
input_ids = torch.randint(0, 50000, (4, 128))
z = embedding(input_ids)  # ComplexTensor(4, 128, 512)

# カスタム位置インデックス
positions = torch.arange(128).unsqueeze(0).expand(4, -1)
z = embedding(input_ids, positions=positions)
```

### 2. Phase 2互換性

Phase 2のEmbeddingをPhase 3のComplexEmbeddingに変換できます。

```python
from src.models.phase3 import convert_phase2_embedding_to_complex
import torch.nn as nn

# Phase 2のEmbedding
phase2_emb = nn.Embedding(50000, 512)

# Phase 3のComplexEmbeddingに変換
phase3_emb = convert_phase2_embedding_to_complex(phase2_emb)
```

### 3. 干渉効果の分析

複素数埋め込みの干渉効果を分析できます。

```python
from src.models.phase3 import analyze_complex_embedding_interference

# 干渉効果の分析
token_ids = torch.tensor([10, 20, 30])
analysis = analyze_complex_embedding_interference(embedding, token_ids)

print(f"Magnitude: {analysis['magnitude']}")  # 各トークンの振幅
print(f"Phase: {analysis['phase']}")  # 各トークンの位相
print(f"Interference: {analysis['interference']}")  # トークン間の干渉強度
```

## アーキテクチャ

```
Input: Token IDs (B, N)
    ↓
[Token Embedding (Real)] → real_emb (B, N, D)
[Token Embedding (Imag)] → imag_emb (B, N, D)
    ↓
[Zeta Position Embedding] → pos_emb (B, N, D) [real only]
    ↓
Output: ComplexTensor(real_emb + pos_emb, imag_emb) (B, N, D)
```

## メモリ効率

### complex32 vs complex64

| モード | メモリ使用量 | 精度 | 推奨用途 |
|--------|-------------|------|---------|
| complex32 | 4 bytes/element | float16 | 学習・推論（メモリ制約あり） |
| complex64 | 8 bytes/element | float32 | 高精度が必要な場合 |

### メモリ削減率

- **約50%削減**: complex32はcomplex64の半分のメモリで動作
- **8GB VRAM制約**: complex32を使用することで、より大きなモデルを学習可能

## 物理的解釈

### 複素数表現

- **実部**: トークンの基本的な意味表現
- **虚部**: トークンの文脈依存の位相情報

### 言語現象のモデリング

- **否定形**: 位相反転（π回転）
- **皮肉**: 位相のずれ
- **多義語**: 複数の位相成分の重ね合わせ

### Zeta Position Embedding

Phase 2のZetaEmbeddingを継承し、リーマンゼータ関数の零点を周波数として使用します。

```
PE(pos, 2i) = sin(pos * gamma_i / (2π))
PE(pos, 2i+1) = cos(pos * gamma_i / (2π))
```

ここで gamma_i はi番目のゼータ零点の虚部です。

## パラメータ

### ComplexEmbedding

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| vocab_size | int | - | 語彙サイズ |
| d_model | int | - | モデル次元 |
| max_seq_len | int | 2048 | 最大シーケンス長 |
| use_complex32 | bool | True | complex32を使用するか |
| zeta_scale | float | 1.0 | Zeta初期化のスケール |
| trainable_pos | bool | False | 位置埋め込みを学習可能にするか |
| dropout | float | 0.1 | ドロップアウト率 |

## 統計情報

### get_statistics()

埋め込みの統計情報を取得します。

```python
stats = embedding.get_statistics()

print(f"Call count: {stats['call_count']}")
print(f"Real norm: {stats['real_norm']:.3f}")
print(f"Imag norm: {stats['imag_norm']:.3f}")
print(f"Real mean: {stats['real_mean']:.3f}")
print(f"Imag mean: {stats['imag_mean']:.3f}")
print(f"Real std: {stats['real_std']:.3f}")
print(f"Imag std: {stats['imag_std']:.3f}")
```

### get_embedding_weight()

埋め込み重みを取得します。

```python
# 実部の重み
real_weight = embedding.get_embedding_weight('real')  # (vocab_size, d_model)

# 虚部の重み
imag_weight = embedding.get_embedding_weight('imag')  # (vocab_size, d_model)
```

## テスト結果

### 全テスト合格

```
tests/test_complex_embedding.py::TestComplexEmbeddingBasic::test_initialization PASSED
tests/test_complex_embedding.py::TestComplexEmbeddingBasic::test_forward_pass_complex32 PASSED
tests/test_complex_embedding.py::TestComplexEmbeddingBasic::test_forward_pass_complex64 PASSED
tests/test_complex_embedding.py::TestComplexEmbeddingBasic::test_custom_positions PASSED
tests/test_complex_embedding.py::TestComplexEmbeddingBasic::test_sequence_length_validation PASSED
tests/test_complex_embedding.py::TestComplexEmbeddingBasic::test_input_shape_validation PASSED
tests/test_complex_embedding.py::TestComplexEmbeddingMemory::test_memory_usage_complex32_vs_complex64 PASSED
tests/test_complex_embedding.py::TestComplexEmbeddingMemory::test_parameter_count PASSED
tests/test_complex_embedding.py::TestComplexEmbeddingMemory::test_memory_efficiency_report PASSED
tests/test_complex_embedding.py::TestComplexEmbeddingStatistics::test_get_statistics PASSED
tests/test_complex_embedding.py::TestComplexEmbeddingStatistics::test_get_embedding_weight PASSED
tests/test_complex_embedding.py::TestComplexEmbeddingPhase2Compatibility::test_convert_phase2_to_complex PASSED
tests/test_complex_embedding.py::TestComplexEmbeddingInterference::test_analyze_interference PASSED
tests/test_complex_embedding.py::TestComplexEmbeddingGradient::test_backward_pass PASSED
tests/test_complex_embedding.py::TestComplexEmbeddingGradient::test_gradient_flow_complex32 PASSED
tests/test_complex_embedding.py::TestComplexEmbeddingCUDA::test_cuda_forward PASSED
tests/test_complex_embedding.py::TestComplexEmbeddingCUDA::test_cuda_backward PASSED

=========================== 17 passed in 10.49s ===========================
```

### メモリ効率検証

- **complex32**: 約50%のメモリ削減を達成
- **勾配伝播**: 正常に動作（NaN/Inf なし）
- **CUDA対応**: GPU環境で正常に動作

## トラブルシューティング

### シーケンス長エラー

```python
# エラー: Sequence length exceeds max_seq_len
# 解決策: max_seq_lenを増やす
embedding = ComplexEmbedding(
    vocab_size=50000,
    d_model=512,
    max_seq_len=4096  # 増やす
)
```

### メモリ不足

```python
# 解決策1: complex32を使用
embedding = ComplexEmbedding(
    vocab_size=50000,
    d_model=512,
    use_complex32=True  # メモリ効率優先
)

# 解決策2: バッチサイズを削減
# 解決策3: d_modelを削減
```

### 勾配消失/爆発

```python
# 解決策: zeta_scaleを調整
embedding = ComplexEmbedding(
    vocab_size=50000,
    d_model=512,
    zeta_scale=0.5  # 小さくする
)
```

## 次のステップ

1. **Stage 1統合モデル**: ComplexEmbeddingを使用したPhase3Stage1Modelの実装
2. **Hamiltonian ODE統合**: エネルギー保存思考機構の追加
3. **ベンチマーク**: WikiText-2でのPerplexity測定

## 関連ドキュメント

- [Complex Tensor Quick Reference](PHASE3_COMPLEX_TENSOR_QUICK_REFERENCE.md)
- [Complex Ops Quick Reference](PHASE3_COMPLEX_OPS_QUICK_REFERENCE.md)
- [Phase 3 Design Document](../../.kiro/specs/phase3-physics-transcendence/design.md)
- [Phase 3 Requirements](../../.kiro/specs/phase3-physics-transcendence/requirements.md)

---

**作成日**: 2025-11-21  
**ステータス**: Complete  
**テスト**: 17/17 Passed
