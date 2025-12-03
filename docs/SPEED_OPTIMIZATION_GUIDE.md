# Phase 8 Speed Optimization - Quick Guide

## 🚀 速度最適化の使い方

### 1. Flash Attention 2インストール (オプションだが推奨)
```bash
# CUDA 11.8/12.1の場合
pip install flash-attn --no-build-isolation

# インストールできない場合は自動的にfallbackされます
```

### 2. torch.compile使用方法

#### デフォルトモード(バランス型)
```python
config = Phase8Config(
    use_torch_compile=True,
    compile_mode="default",  # バランス
)
```

#### 最大速度モード
```python
config = Phase8Config(
    use_torch_compile=True,
    compile_mode="max-autotune",  # 最速
)
```

### 3. 10B Ultra設定で訓練
```bash
# WSL Ubuntu環境
cd /mnt/c/dev/Project-ResNet-BK-An-O-N-Language-Model-Architecture
source venv_ubuntu/bin/activate

# Dry run test
python scripts/train_phase8.py --config configs/phase8_10b_ultra.yaml --dry-run

# Full training (config already has all optimizations)
python scripts/train_phase8.py --config configs/phase8_10b_ultra.yaml --dataset configs/dataset_mixing.yaml
```

### 4. 速度ベンチマーク
```bash
# Baseline (最適化なし)
python scripts/benchmark_phase8_speed.py --d-model 512 --n-layers 8 --n-seq 512 --low-rank-rank 16

# With torch.compile
python scripts/benchmark_phase8_speed.py --d-model 512 --n-layers 8 --n-seq 512 --low-rank-rank 16 --use-compile

# With torch.compile + Flash Attention 2
python scripts/benchmark_phase8_speed.py --d-model 512 --n-layers 8 --n-seq 512 --low-rank-rank 16 --use-compile --use-flash-attn2
```

## ⚡ 期待される速度向上

| 最適化 | 速度向上 | 累積向上 |
|--------|----------|----------|
| Baseline | 1x | 1x |
| Data loading最適化 | +20% | 1.2x |
| bfloat16 mixed precision | +50% | 1.8x|
| torch.compile (default) | +100% | 3.6x |
| torch.compile (max-autotune) | +150% | 4.5x |
| Flash Attention 2 | +50% | 6.7x |
| Fused kernels (計画中) | +50% | 10x |

**目標: >1000 tokens/秒**
- Baselineが~100-150 tokens/秒の場合
- 現在の最適化で600-800 tokens/秒達成可能
- Fused kernels実装後に1000+ tokens/秒達成

## 🗜️ メモリとの兼ね合い

### RTX 3080 8GB推奨設定

**高速優先 (Medium model)**:
```yaml
d_model: 1024
n_layers: 16
low_rank_rank: 32
use_torch_compile: true
use_flash_attention_2: true
gradient_checkpointing: false  # 速度優先
```

**メモリ優先 (10B ultra)**:
```yaml
d_model: 4096
n_layers: 48
low_rank_rank: 16
use_torch_compile: true  # compile自体はメモリ節約
use_flash_attention_2: true  # メモリ効率も良い
gradient_checkpointing: true  # 必須
gradient_accumulation_steps: 32
```

## トラブルシューティング

### torch.compileエラー
```
# エラー: "Triton kernel failed"
→ compile_fullgraph: false に設定 (デフォルト)
→ または compile_mode: "default" に変更
```

### Flash Attention 2がinstallできない
```
# 問題ない、自動的にfallbackします
# wrapper が標準attention を使用
```

### OOM (Out of Memory)
```
# gradient_checkpointing を有効化
use_gradient_checkpointing: true

# または gradient accumulation を増やす
gradient_accumulation_steps: 64
```

## 📊 設定ファイル

すべての最適化は`configs/phase8_10b_ultra.yaml`に含まれています:
- ✅ torch.compile (max-autotune)
- ✅ Flash Attention 2
- ✅ Data loading optimizations
- ✅ Gradient accumulation (32 steps)
- ✅ Mixed precision (bfloat16)
- ✅ Gradient checkpointing

そのまま使用可能です！
