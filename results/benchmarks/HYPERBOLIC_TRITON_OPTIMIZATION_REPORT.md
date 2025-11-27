# Hyperbolic Attention Triton Kernel 最適化レポート

## 概要

Phase 7の双曲アテンション（Hyperbolic Attention）のTritonカーネルを大幅に最適化し、PyTorch実装比で**最大36倍の高速化**を達成。

## 最適化内容

### 新規カーネル: `fast` (hyperbolic_attention_fast.py)

**最適化ポイント:**
1. **近似双曲距離**: 完全なPoincaré距離計算（acosh, atanh）を高速近似に置換
   - `d_approx(q,k) = ||q-k|| * sqrt(1 + c*(||q||^2 + ||k||^2))`
   - 双曲幾何の本質（境界効果）を維持しつつ計算を簡略化
2. **Flash Attentionスタイル**: Online softmaxによるメモリ効率化
3. **Autotune**: RTX 3080向けに最適化されたブロックサイズ自動選択
4. **Autograd対応**: Forward=Triton, Backward=PyTorch参照実装

### カーネル比較

| カーネル | 説明 | 速度 |
|---------|------|------|
| `fast` | 近似双曲距離、最速 | ★★★★★ |
| `v1` | 従来版、完全な双曲距離 | ★★★ |
| `v2` | 事前計算版（autotuneオーバーヘッド） | ★★ |
| `pytorch` | 参照実装 | ★ |

## ベンチマーク結果

### seq_len=1024, batch=4, d_model=256, heads=8

| カーネル | 平均時間 | スループット | PyTorch比 |
|---------|---------|-------------|-----------|
| fast | 1.05ms | 3.89M tok/s | **19.2x** |
| v1 | 2.30ms | 1.78M tok/s | 8.8x |
| pytorch | 20.21ms | 0.20M tok/s | 1.0x |

### seq_len=2048, batch=2, d_model=256, heads=8

| カーネル | 平均時間 | スループット | PyTorch比 |
|---------|---------|-------------|-----------|
| fast | 0.73ms | 5.59M tok/s | **36.5x** |
| v1 | 2.31ms | 1.78M tok/s | 11.6x |
| pytorch | 26.71ms | 0.15M tok/s | 1.0x |

## 物理的正当性

近似双曲距離は以下の性質を維持:
- **境界効果**: `||x||`が大きいほど距離が増大（Poincaré球の本質）
- **曲率依存性**: 曲率`c`が大きいほど双曲的効果が強まる
- **対称性**: `d(x,y) = d(y,x)`

## 使用方法

```python
from src.models.phase7.hyperbolic_attention import HyperbolicMultiHeadAttention

# 最速版（デフォルト）
attn = HyperbolicMultiHeadAttention(
    d_model=256,
    num_heads=8,
    use_triton_kernel=True,
    kernel_version='fast'  # 'fast', 'v2', 'v1'
)
```

## Makefileターゲット

```bash
make triton-attn   # smoke test (fast kernel)
make triton-bench  # 全カーネル比較ベンチマーク
make triton-fast   # fast kernel詳細テスト
```

## 結論

- **19-36倍の高速化**を達成（シーケンス長に依存）
- 双曲幾何の本質的な性質は維持
- RTX 3080 (10GB VRAM) で効率的に動作
- O(N)アーキテクチャの目標に貢献
