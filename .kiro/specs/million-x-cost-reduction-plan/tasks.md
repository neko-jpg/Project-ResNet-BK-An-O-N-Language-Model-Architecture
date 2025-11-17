# ResNet-BK evolution: phase checklist

Goal: implement the four combos from `改善案/修正案.md` + 論文 and de-risk them stepwise. Each phase lists references so他の人も迷わない。

## Phase 1 — Riemann initialization (Prime-bump)
- [x] Translate $V_\epsilon$ prime-bump distribution into a weight init scheme for BKコア (A/B vs current init).  
  Ref: `src/models/resnet_bk.py` (prime-bump init), `改善案/修正案.md`, `改善案/論文/riemann_hypothesis_main.tex`
- [x] Wire init flag into config and small model (d_model≈64, n_layers≈4, N≈128) training script.  
  Ref: `src/models/configurable_resnet_bk.py`, `src/utils/config.py`, `train.py`
- [ ] Run A/B on短/長文タスク; log convergence speed, loss floor, NaN/Inf有無.  
  Ref: `src/benchmarks/wikitext2_benchmark.py`, `notebooks/long_context_benchmark_colab.py`, `notebooks/prime_bump_init_run.ipynb`; cmd例: `python train.py --config-preset baseline --prime-bump-init --prime-bump-scale 0.02`
- [ ] Exit: ≥baseline収束 or 長文ロバスト性向上 with no instability.

## Phase 2 — Scattering-based Router (ACT / MoE)
- [ ] 定義: scattering phase/ spectral shiftの計算可能なproxy（例: stateノルム変化、局所スペクトル推定）.  
  Ref: `2_Scaling_Benchmarks/4_MoE_PoC/on_resnetbk_moe_poc.py`, `改善案/論文/riemann_hypothesis_main.tex`
- [ ] Router/early-exit判定をproxyに差し替え、MLP routerをフォールバックとしてA/B.  
  Ref: `src/models/transformer_baseline.py` (routing), `src/training/gradient_caching.py` (early-exit hooks)
- [ ] 評価: routing安定性（entropy, load balance）とPPL影響を小規模ベンチで測定.  
  Ref: `src/benchmarks/transformer_comparison.py`
- [ ] Exit: proxy routerが安定かつPPL低下 ≤5% or 明確な効率改善。

## Phase 3 — Trace-class attention kernel (Triton/FlashRNN系)
- [ ] Proposition 0.4 (‖Kε‖_{S2} bound)からclip/正規化定数を導出しハイパーとして実装.  
  Ref: `改善案/修正案.md`, `改善案/論文/riemann_hypothesis_main.tex`
- [ ] Triton (or CUDA) kernelを実装し、boundを尊重・overflow/underflowガード追加.  
  Ref: `src/training/gradient_caching.py`, `notebooks/long_context_benchmark_colab.py`
- [ ] Clip閾値を理論値 vs 実験値でスイープし、grad安定性とスループットを計測.  
  Ref: `run_wikitext103_benchmark.py` (throughput計測の雛形)
- [ ] Exit: FP16/BF16で発散なし、スループット ≥ baseline、数値誤差が許容範囲。

## Phase 4 — Koopman圧縮（量子化/枝刈り＋スペクトル保存）
- [ ] Cutoff関数ψε (Def 12.1) を pruning/quantizationポリシーにマップしスケジューラ化.  
  Ref: `改善案/論文/riemann_hypothesis_main.tex`, `src/models/transformer_baseline.py`（prune周り）
- [ ] Clark measure保存項を蒸留損失に追加し係数をconfig化.  
  Ref: `train.py`, `src/training/gradient_caching.py`
- [ ] PPL＋スペクトル指標の前後比較をheld-outで評価.  
  Ref: `benchmark_results/` (結果保存場所)
- [ ] Exit: PPL目標内かつスペクトル指標のドリフトが許容内。

## Phase 5 — Integration & benchmarks
- [ ] Phases 1–4のベスト設定を統合config化し、アブレーション用トグルを追加.  
  Ref: `src/benchmarks/transformer_comparison.py`, `src/benchmarks/wikitext2_benchmark.py`
- [ ] WikiText2＋長文サニティを実行し、PPL/throughput/memoryを収集.  
  Ref: `run_wikitext103_benchmark.py`, `notebooks/transformer_vs_resnetbk_colab.ipynb`
- [ ] 学び・失敗モード・推奨デフォルトをドキュメント化.  
  Ref: `README.md`, `docs/` (必要に応じて追加)
- [ ] Exit: end-to-end実行が成功し、config＋ノートを共有できる状態。
