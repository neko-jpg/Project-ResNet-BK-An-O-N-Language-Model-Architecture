# 詳細な比較テーブル

**日付**: 2025-11-19
**構成**: vocab=10000, d=512, layers=6, batch=2, seq=512

## Table 1: パラメータ数の比較

| Component | Baseline | Ultra Optimized | Reduction |
|-----------|----------|-----------------|----------|
| Embedding | 5.12M | 18.40K | 99.6% |
| Transformer Layers | 14.19M | 544.61K | 96.2% |
| Output Head | 5.13M | 79.70K | 98.4% |
| **Total** | **24.44M** | **616.09K** | **97.5%** |

## Table 2: VRAM使用量の比較（学習時）

| Metric | Baseline (FP32) | Baseline (FP16) | Ultra Optimized (FP16) | Reduction |
|--------|-----------------|-----------------|------------------------|----------|
| Parameter Memory | 58.1 MB | 194.5 MB | 257.3 MB | -342.5% |
| Forward Memory | 229.2 MB | 355.0 MB | 289.3 MB | -26.2% |
| Peak Memory (Training) | 264.1 MB | 382.8 MB | 309.4 MB | -17.2% |
| Activation Memory | 206.0 MB | 188.3 MB | 52.1 MB | 74.7% |

## Table 3: Ultra Optimized Model の内訳

| Component | Parameters | Memory (FP16) |
|-----------|------------|---------------|
| AR-SSM Layers | 486.19K | 0.93 MB |
| HTT Embedding | 18.40K | 0.04 MB |
| Low-Rank FFN | 52.27K | 0.10 MB |
| Normalization | 7.17K | 0.01 MB |
| Output Head | 52.05K | 0.10 MB |
| **Total Parameters** | **616.09K** | **257.26 MB** |
| **Activation Memory** | - | **52.13 MB** |
| **Peak Memory (Training)** | - | **309.39 MB** |
