# Project-ResNet-BK-An-O-N-Language-Model-Architecture
(AI Learning Cost "One Millionth" Plan - Step 1/4 Achieved)
?? Repository Summary (README.md)

Project ResNet-BK: An O(N) Language Model Architecture
(1,000,000x AI Training Cost Reduction Plan ? Step 1/4 Achieved)

?? Overview (Elevator Pitch)

This repository documents the research and development of ResNet-BK, a new O(N) language model architecture designed to overcome the dominant bottleneck in modern AI: the O(N?) computational cost of Transformers.

This work represents a successful proof-of-concept for Step 1 (Architectural Overhaul) and Step 3 (Sparsification) of the long-term 1,000,000x Cost Reduction Plan.

?? Latest Status (Dec 2025, research prototype)
- Colab small-scale benchmark (WikiText-2, seq_len=512, batch=4, 2000 steps) shows ResNet-BK beating a Transformer baseline: val ppl ~590 vs ~1288 (no OOM, CUDA).
- Fairness note: Transformer uses autocast; ResNet-BK currently not. Speed numbers are therefore not comparable yet (throughput ~3.4k tok/s vs ~71k tok/s). Accuracy advantage remains.
- Long-context bench script builds models per sequence length to avoid n_seq mismatch; use `notebooks/long_context_benchmark_colab.py`.
- Target audience: research users. Large-scale recipes/CI are not production-ready yet.

?? Earlier Results: Faster & Trainable
🚀 Final Results: 6.7× Faster & Demonstrated Learning Ability

1. Speed: 6.7× Faster than Attention at N=2048 (CPU)

The final integrated architecture — combining:

the O(N) core algorithm

analytic gradient (manual backward pass)

sparse MoE

surpasses standard Attention as sequence length increases.

At N = 2048, it achieves ~6.7× speedup over Autograd-based Attention.
(From TeppeiArai_ONResNetBK_MoE_FinalScaling_Report.pdf)

2. Intelligence: Fully Trainable as a Language Model (GPU)

ResNet-BK is not only fast — it can learn.

Using BK-MoE_Language_Model.py, stable learning was observed on GPU:

Parameters: 10.16M

Task: WikiText-2

Result: Perplexity 428.84 after 3 epochs

Notes (Transformer baseline clarity):
- Uses pre-norm blocks and learned absolute positional embeddings (swap to sinusoidal / RoPE for ablations if desired).
- Small benchmark settings (vocab≈20k, seq_len≈256, d_model=256, L=6) can show higher initial loss; this is expected, not a bug.
- LayerNorm is applied before each sublayer for stability in both baselines.

This confirms that the architecture is viable as a language model.

🔬 Technical Milestones

Each result was achieved through the following PoCs:

1. O(N) Core vs O(N²) Attention

Benchmarking pure compute throughput

Finding: Around N ≈ 1000, O(N) computation becomes superior.

2. Analytic Gradient Implementation

Manual backward pass without Autograd

Finding: ~1.6× faster in PoC; integrated version yields 2.5× speedup at N=2048.

3. Sparse MoE Integration

Replaced dense MLP with sparse Mixture of Experts

Finding: Faster than dense FFN while maintaining accuracy.

🗂️ Repository Structure
/1_BK_Language_Model_PoC/

Contains the final integrated model (BK-MoE_Language_Model.py) and training results
(including PPL 428).

/2_Scaling_Benchmarks/

Time-ordered benchmarks, reports, and source code demonstrating:

O(N) vs O(N²)

Analytic Gradient speedups

Sparse MoE

Final 6.7× speed benchmark

🔮 Future Work (What Comes Next)

This project completes Step 1 + Step 3 of the plan.

The next frontier is Step 2: Replacing Backpropagation.

Future research will explore:

operator-based learning (e.g., Koopman theory)

physics-informed optimization

gradient-free or hybrid training mechanisms

ResNet-BK now provides the O(N) “vessel” needed to host these new learning paradigms.


最新のモデルのコード：BK-MoE_Ultra_v2_Stable.py
Running on CUDA
Vocabulary Size: 30000
Train tokens: 500000 (after batchify)
--- ResNet-BK Ultra v2: O(N) + Hybrid Analytic Grad + Sparse MoE ---
Model Parameters: 4.15M
Total Steps (approx): 585
BKCore GRAD_BLEND = 0.5
  [Step 50] Epoch 1 | Loss: 7.4817 | LR: 0.000984
  [Step 100] Epoch 1 | Loss: 7.1682 | LR: 0.000937
  [Step 150] Epoch 1 | Loss: 7.2618 | LR: 0.000862
============================================================
Epoch 1/3 | Time: 28.82s | Avg Loss: 7.6057 | Perplexity: 2009.60
============================================================
  [Step 200] Epoch 2 | Loss: 7.0199 | LR: 0.000764
  [Step 250] Epoch 2 | Loss: 7.0463 | LR: 0.000652
  [Step 300] Epoch 2 | Loss: 7.0798 | LR: 0.000532
  [Step 350] Epoch 2 | Loss: 7.1368 | LR: 0.000413
============================================================
Epoch 2/3 | Time: 24.11s | Avg Loss: 7.0517 | Perplexity: 1154.78
============================================================
  [Step 400] Epoch 3 | Loss: 7.0109 | LR: 0.000304
  [Step 450] Epoch 3 | Loss: 6.9486 | LR: 0.000213
  [Step 500] Epoch 3 | Loss: 7.0623 | LR: 0.000146
  [Step 550] Epoch 3 | Loss: 6.9950 | LR: 0.000108
============================================================
Epoch 3/3 | Time: 24.25s | Avg Loss: 7.0229 | Perplexity: 1122.06
============================================================


---

## 🎯 Google Colab で今すぐ試す！

Step 2 Phase 1の実装をGoogle Colabで簡単に実行できます：

### クイックスタート（5分）

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/blob/main/notebooks/step2_phase1_colab.ipynb)

**実行手順:**
1. 上のバッジをクリック
2. GPU設定: ランタイム → T4 GPU を選択
3. すべてのセルを実行
4. 20-30分で完了！

**実装内容:**
- ✅ Mixed-precision gradient computation (2× speedup)
- ✅ Batched analytic gradient with vmap (2.5× speedup)
- ✅ GRAD_BLEND grid search (最適なα値の発見)
- ✅ 3-epoch training with numerical stability

詳細は [COLAB_QUICK_START.md](COLAB_QUICK_START.md) を参照してください。

---


---

## 🎊 Step 4: Advanced Model Compression 完了！（NEW）

**実装完了:**

Step 4の完全な圧縮パイプラインを実装しました！

### 主な成果

- ✅ **Quantization-Aware Training (QAT)** - INT8量子化で4×圧縮
- ✅ **Complex Number Quantization** - 実部・虚部の個別量子化
- ✅ **INT4 MoE Quantization** - グループワイズ量子化で8×圧縮
- ✅ **Structured Pruning** - 使用率5%未満のエキスパートを自動削除
- ✅ **Knowledge Distillation** - 教師モデルから小型学生モデルへ知識転移
- ✅ **Compression Pipeline** - 自動化された3段階パイプライン
- ✅ **Target: 100× compression** with <15% perplexity degradation

### 圧縮パイプライン

```
Original Model (4.15M params)
    ↓
[Stage 1: QAT] → 4× compression
    ↓
[Stage 2: Pruning] → 4× compression
    ↓
[Stage 3: Distillation] → 6× compression
    ↓
Final Model (~42K params) = 96× ≈ 100× compression
```

### Google Colabで試す

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/blob/main/notebooks/step4_compression.ipynb)

詳細は以下を参照:
- [STEP4_COMPRESSION_IMPLEMENTATION.md](STEP4_COMPRESSION_IMPLEMENTATION.md) - 詳細な実装ドキュメント
- [STEP4_QUICK_REFERENCE.md](STEP4_QUICK_REFERENCE.md) - クイックリファレンス

---

## 🎊 Step 2 Phase 1 完了！

**Google Colab実行結果:**

Step 2 Phase 1の実装がGoogle Colab（T4 GPU）で正常に完了しました！

### 主な成果

- ✅ **GRAD_BLEND最適化完了** - α = 0.0（純粋な理論的勾配）が最適
- ✅ **Mixed-precision実装** - 1.5-2.0× speedup達成
- ✅ **Batched gradient実装** - 2.0-2.5× speedup達成
- ✅ **数値安定性確認** - NaN/Infなしで学習完了
- ✅ **Best Perplexity: 309.90** on WikiText-2

### Grid Search結果

| GRAD_BLEND (α) | Perplexity | Gradient Variance | Status |
|----------------|------------|-------------------|--------|
| **0.0** | **309.90** | 0.0216 | ✅ Best |
| 0.3 | 341.95 | 0.1778 | ⚠️ |
| 0.5 | 322.15 | 0.0742 | ⚠️ |
| 0.7 | 495.04 | 427.32 | ❌ Unstable |
| 1.0 | 494.01 | 437.88 | ❌ Unstable |

**重要な発見:** 理論的勾配（α=0.0）がHypothesis-7勾配よりも優れていることが実証されました。

詳細は [STEP2_PHASE1_COLAB_RESULTS.md](STEP2_PHASE1_COLAB_RESULTS.md) を参照してください。

---
