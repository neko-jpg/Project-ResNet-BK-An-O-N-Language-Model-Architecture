# Phase 3 Stage 2 Benchmark - Quick Reference

## 概要

Phase 3 Stage 2（Hamiltonian ODE Integration）のベンチマークスクリプトです。

## 測定項目

### 1. Perplexity（WikiText-2）
- **測定条件**: Batch=4, Seq=1024, fp16, ODE steps=10
- **目標**: Stage 1比 +2%以内
- **記録項目**: PPL、PPL_stage1、PPL_ratio、pass/fail

### 2. Energy Drift
- **測定条件**: Batch=4, Seq=512, dt=0.1, 100 steps
- **目標**: < 5e-5（閾値1e-4の半分）
- **追加検証**: エネルギーが単調増加/減少していないこと（振動許容範囲 ±10%）
- **記録項目**: mean_energy、max_drift、mean_drift、pass/fail

### 3. VRAM使用量
- **測定条件**: Batch=2, Seq=2048、Forward + Backward pass
- **比較**: Symplectic Adjoint vs Full Backprop
- **目標**: 
  - Symplectic Adjoint: < 7.5GB（8GBの93.75%）
  - 削減率: Full Backprop比 70%以上削減
- **記録項目**: vram_symplectic_gb、vram_full_backprop_gb、reduction_ratio、pass/fail

## 使用方法

### 基本実行
```bash
python scripts/benchmark_phase3_stage2.py
```

### オプション付き実行
```bash
# クイックテスト（最初の10バッチのみ）
python scripts/benchmark_phase3_stage2.py --max-ppl-batches 10

# Stage 1ベースラインをスキップ
python scripts/benchmark_phase3_stage2.py --skip-stage1

# カスタム出力パス
python scripts/benchmark_phase3_stage2.py --output results/my_benchmark.json

# 全オプション
python scripts/benchmark_phase3_stage2.py \
    --device cuda \
    --seed 42 \
    --ppl-batch-size 4 \
    --ppl-seq-length 1024 \
    --energy-batch-size 4 \
    --energy-seq-length 512 \
    --vram-batch-size 2 \
    --vram-seq-length 2048 \
    --max-ppl-batches 50 \
    --output results/benchmarks/phase3_stage2_comparison.json
```

## 出力形式

### JSON出力例
```json
{
  "benchmark_name": "Phase 3 Stage 2 Benchmark",
  "timestamp": "2025-11-21 12:00:00",
  "device": "cuda",
  "seed": 42,
  
  "stage2_ppl": 30.8,
  "stage1_ppl": 30.5,
  "ppl_ratio": 1.010,
  "ppl_diff_pct": 1.0,
  "ppl_target": 1.02,
  "ppl_pass": true,
  
  "mean_energy": 0.0123,
  "max_drift": 3.2e-5,
  "mean_drift": 1.5e-5,
  "monotonic_violation": false,
  "energy_pass": true,
  
  "vram_symplectic_gb": 7.2,
  "vram_full_backprop_gb": 24.5,
  "vram_reduction_ratio": 0.294,
  "vram_reduction_pct": 70.6,
  "vram_pass": true,
  
  "all_pass": true
}
```

## 完了条件

すべての項目で目標を達成すること：

1. ✅ **Perplexity**: Stage 1比 +2%以内
2. ✅ **Energy Drift**: < 5e-5
3. ✅ **VRAM**: < 7.5GB
4. ✅ **VRAM削減率**: 70%以上

## トラブルシューティング

### CUDA Out of Memory
```bash
# バッチサイズを削減
python scripts/benchmark_phase3_stage2.py --vram-batch-size 1 --vram-seq-length 1024
```

### Energy Drift測定失敗
- Hamiltonian関数がモデルに存在するか確認
- `model.blocks[0].ode.h_func`が正しく設定されているか確認

### Perplexity測定が遅い
```bash
# バッチ数を制限
python scripts/benchmark_phase3_stage2.py --max-ppl-batches 20
```

## 関連ファイル

- **ベンチマークスクリプト**: `scripts/benchmark_phase3_stage2.py`
- **Stage 2モデル**: `src/models/phase3/stage2_model.py`
- **Hamiltonian ODE**: `src/models/phase3/hamiltonian_ode.py`
- **Symplectic Adjoint**: `src/models/phase3/symplectic_adjoint.py`
- **出力**: `results/benchmarks/phase3_stage2_comparison.json`

## Requirements

- Requirements 2.21: Perplexity測定
- Requirements 2.22: Energy Drift測定
- Requirements 2.23: VRAM測定

## 次のステップ

Stage 2のベンチマークが完了したら：

1. 結果を確認: `cat results/benchmarks/phase3_stage2_comparison.json`
2. 論文に追記: `paper/main.tex`に実験結果を記載
3. Stage 3へ進む: 全機能統合（Koopman、MERA、Dialectic、Entropic Selection）

---

**作成日**: 2025-11-21  
**更新日**: 2025-11-21  
**ステータス**: Completed
