# Requirements Document: Mamba-Killer Ultra-Scale ResNet-BK

## Introduction

本ドキュメントは、ResNet-BKアーキテクチャをMambaを超える世界最高のO(N)言語モデルに進化させるための要件を定義します。修正案で提案された**リーマン・ゼータ関数の零点をスペクトルとして持つ量子ハミルトニアンの離散化近似**という数学的基盤に基づき、Google Colab無料枠で10億〜100億パラメータの学習を実現します。

**現状:**
- O(N) BK-Coreアーキテクチャ完成（6.7×高速化 @ N=2048）
- Sparse MoE統合完了
- WikiText-2でPerplexity 1122達成（4.15Mパラメータ）
- 既存の100万倍コスト削減計画進行中

**新たな目標:**
1. **Mamba超え**: 長文安定性、量子化耐性、動的計算効率の3軸でMambaを圧倒
2. **超大規模学習**: Google Colab無料枠（T4 GPU, 15GB RAM）で10億〜100億パラメータを学習可能に
3. **数学的革新**: リーマン初期化、散乱ベースルーティング、トレースクラスAttentionの実装
4. **決定的な1枚グラフ**: 誰が見ても「BKの方がヤベぇ」と分かる比較結果

## Glossary

- **Mamba**: 状態空間モデル（SSM）ベースのO(N)言語モデル、現在のSOTA候補
- **Riemann Initialization (リーマン初期化)**: 素数位置にポテンシャルの山を配置し、ゼータ関数の零点分布（GUE統計）に基づいた初期化
- **Prime-Bump Potential**: 素数インデックスにバンプ（山）を持つポテンシャル関数
- **Scattering Phase (散乱位相)**: 量子散乱理論における位相シフト、トークンの「難しさ」の指標
- **Scattering-Based Router**: 散乱位相を用いたMoEルーティング機構
- **Trace-Class Attention**: トレースクラス条件を満たす数値安定なAttention機構
- **Birman-Schwinger Kernel**: 論文中の式(9) K_ε(z) = |V_ε|^{1/2}R_0(z)|V_ε|^{1/2}
- **Semiseparable Structure**: 半可分行列構造、O(N)計算を可能にする数学的性質
- **GUE Statistics**: ガウス直交アンサンブル統計、ランダム行列理論の基礎
- **Spectral Shift Function**: スペクトルシフト関数、ポテンシャルによる固有値の変化
- **Clark Measure**: クラーク測度、スペクトル分布の保存を保証
- **Cutoff Function ψ_ε**: 正則化カットオフ関数、圧縮時のスペクトル保存に使用
- **Gradient Checkpointing**: メモリ効率化のため中間活性化を再計算する手法
- **ZeRO Optimizer**: DeepSpeedのメモリ最適化技術、オプティマイザ状態を分散
- **Mixed Precision Training**: FP16/BF16とFP32を混在させた学習
- **Activation Checkpointing**: 活性化の一部のみを保存し、必要時に再計算
- **CPU Offloading**: GPU メモリ不足時にCPUメモリを活用
- **Long Context Stability**: 長文（128k〜1M tokens）での学習安定性
- **Quantization Robustness**: INT8/INT4量子化後の性能維持能力
- **Dynamic Compute Efficiency**: 入力に応じた動的な計算量調整

## Requirements

### Requirement 1: Birman-Schwinger Kernel と Prime-Bump Potential の実装

**User Story:** 研究者として、論文で証明されたBirman-Schwinger核とPrime-Bump potentialを正確に実装し、数学的に保証された数値安定性とMambaを超える性能を実現したい。

#### Acceptance Criteria

1. THE System SHALL implement Birman-Schwinger operator: K_ε(z) = |V_ε|^{1/2} R_0(z) |V_ε|^{1/2} where R_0(z) = (H_0 - z)^{-1}
2. WHEN computing resolvent kernel, THE System SHALL use R_0(z; u,v) = (i/2) exp(iz(u-v)) sgn(u-v) with bound |R_0| ≤ (1/2) exp(-Im(z)|u-v|)
3. THE System SHALL implement Prime-Bump potential: V_ε(x) = Σ_p α_{p,k}(ε) ψ_ε(x - log p) where α_{p,k}(ε) = (log p) p^{-k(1/2+ε)}
4. WHEN initializing potential, THE System SHALL place Gaussian bumps at prime positions: ψ_ε(x) = ε^{-1/2} exp(-x²/(2ε))
5. THE System SHALL verify Hilbert-Schmidt bound: ||K_ε(z)||_S2 ≤ (1/2)(Im z)^{-1/2} ||V_ε||_L2
6. WHEN ε > 1/2, THE System SHALL verify trace-class bound: ||K_ε(z)||_S1 ≤ (1/2)(Im z)^{-1} ||V_ε||_L1
7. THE System SHALL implement Schatten norm monitoring: track ||K_ε||_S2 during forward pass and clip if exceeds threshold
8. WHEN Schatten norm exceeds C * ε^{-1/2}, THE System SHALL apply spectral clipping to maintain trace-class property
9. THE System SHALL implement canonical coefficients: α_{p,k}(ε) = (log p) / p^{k(1/2+ε)} for all primes p and k ≥ 1
10. THE System SHALL verify that potential satisfies finite overlap condition: supp(ψ_ε(· - log p)) ∩ supp(ψ_ε(· - log q)) = ∅ for |log p - log q| > 2√ε
11. THE System SHALL implement regularization parameter schedule: start with ε = 1.0, gradually decrease to ε = 0.5 during training
12. WHEN ε → 0, THE System SHALL monitor numerical stability and halt if condition number exceeds 10^6
13. THE System SHALL compare Prime-Bump initialization vs. random initialization: measure convergence speed, final perplexity, gradient stability
14. WHEN training for 1000 steps, THE System SHALL achieve at least 30% faster convergence with Prime-Bump initialization
15. THE System SHALL visualize learned potential: plot V_ε(x) and verify that peaks align with prime positions
16. THE System SHALL compute spectral shift function: ξ(λ) = (1/π) arg det(I + K_ε(λ + i0)) and verify it matches prime counting function
17. THE System SHALL implement GUE eigenvalue spacing analysis: compute nearest-neighbor spacing distribution and verify Wigner surmise
18. WHEN analyzing eigenvalues of H_ε, THE System SHALL verify that spacing follows s * exp(-πs²/4) (GUE statistics)
19. THE System SHALL measure long-context stability: train on N ∈ {512, 1024, 2048, 4096, 8192, 16384} and verify no NaN/Inf
20. THE System SHALL demonstrate that Prime-Bump initialization provides 2× better gradient stability (lower variance) than random initialization

### Requirement 2: Scattering Phase Router と Spectral Shift Function の実装

**User Story:** 研究者として、論文の散乱理論に基づいた完全に学習不要なMoEルーティングを実装し、MLPルーターを完全に置き換えたい。

#### Acceptance Criteria

1. THE System SHALL implement scattering phase: δ_ε(λ) = arg(det_2(I + K_ε(λ + i0))) using boundary limit from Corollary BK-boundary
2. WHEN computing phase, THE System SHALL use Birman-Krein formula: d/dλ log D_ε(λ) = -Tr((H_ε - λ)^{-1} - (H_0 - λ)^{-1})
3. THE System SHALL implement spectral shift function: ξ(λ; H_ε, H_0) = (1/π) Im log D_ε(λ + i0)
4. WHEN computing ξ(λ), THE System SHALL verify that ∫ ξ(λ) dλ = Tr(f(H_ε) - f(H_0)) for test functions f
5. THE System SHALL implement phase-based routing: route token i to expert e if δ_ε(λ_i) ∈ [(e-1)π/E, eπ/E] where E = num_experts
6. WHEN phase is near 0 or π (resonance), THE System SHALL route to multiple experts (top-2 or top-3)
7. WHEN phase is in middle range, THE System SHALL route to single expert (top-1)
8. THE System SHALL eliminate all learnable parameters in routing: purely physics-based, zero training cost
9. WHEN measuring routing overhead, THE System SHALL achieve at least 10× faster routing than MLP gating (no forward pass needed)
10. THE System SHALL implement Clark measure: μ_ε(E) = (1/2π) ∫_E |D_ε(λ + i0)|^{-2} dλ for Borel sets E ⊂ ℝ
11. WHEN using Clark measure, THE System SHALL verify that μ_ε is a probability measure: μ_ε(ℝ) = 1
12. THE System SHALL implement resonance detection: identify λ where |D_ε(λ + i0)| is small (near-zero of determinant)
13. WHEN resonance detected, THE System SHALL increase computation budget for that token (ACT-style early exit disabled)
14. THE System SHALL implement Weil explicit formula verification: -Σ_p Σ_k (log p / p^{k(1/2+ε)}) φ̂(k log p) = (1/2πi) ∫ φ(λ) d log D_ε(λ)
15. WHEN verifying Weil formula, THE System SHALL use band-limited test functions φ ∈ C_c^∞(ℝ)
16. THE System SHALL provide interpretability: visualize scattering phase δ_ε(λ_i) for each token and correlate with linguistic difficulty
17. WHEN analyzing difficult tokens (high perplexity), THE System SHALL verify they have high |δ_ε| (strong scattering)
18. THE System SHALL implement adaptive expert allocation: allocate more experts to frequency bands with high spectral density
19. WHEN spectral density ρ(λ) = dξ/dλ is high, THE System SHALL use more experts in that λ range
20. THE System SHALL demonstrate that scattering-based routing achieves equal or better performance than learned MLP routing with zero training cost

### Requirement 3: Mourre Estimate と Limiting Absorption Principle に基づく数値安定カーネル

**User Story:** 研究者として、論文で証明されたMourre estimateとLAPを実装し、数学的に保証された数値安定性を持つCUDA/Tritonカーネルを実現したい。

#### Acceptance Criteria

1. THE System SHALL implement Mourre estimate verification: [H_0, iA] = I where A = x (position operator)
2. WHEN computing commutator, THE System SHALL verify that 1_I(H_0)[H_0, iA]1_I(H_0) = 1_I(H_0) (optimal estimate with c_I = 1)
3. THE System SHALL implement Limiting Absorption Principle: ⟨x⟩^{-s}(H_0 - λ ∓ iη)^{-1}⟨x⟩^{-s} extends continuously to η = 0 for s > 1/2
4. WHEN η → 0, THE System SHALL verify uniform bound: ||(H_ε - λ - iη)^{-1}||_{B(L²_s, L²_{-s})} ≤ C_{I,η₀} uniformly in ε
5. THE System SHALL implement weighted resolvent computation: use weight ⟨x⟩^{-s} = (1 + x²)^{-s/2} with s = 1 (default)
6. WHEN computing resolvent near real axis, THE System SHALL use LAP to ensure numerical stability (no divergence)
7. THE System SHALL implement Birman-Schwinger invertibility check: verify (I + K_ε(z))^{-1} remains bounded as Im z → 0
8. WHEN ||K_ε(λ + iη)||_S2 ≤ C||V_ε||_L2 uniformly as η → 0, THE System SHALL guarantee invertibility
9. THE System SHALL implement Schatten norm monitoring: track ||K_ε(z)||_S1 and ||K_ε(z)||_S2 during forward pass
10. WHEN Schatten norm exceeds theoretical bound C_p η^{-1/p}, THE System SHALL apply spectral regularization
11. THE System SHALL implement fused CUDA kernel for theta/phi recursions with LAP-based stability guarantees
12. WHEN using fused kernel, THE System SHALL achieve at least 15× speedup over sequential PyTorch (論文の理論保証付き)
13. THE System SHALL implement mixed-precision strategy: complex64 for recursions, complex128 for critical accumulations
14. WHEN condition number κ(H_ε - zI) > 10^6, THE System SHALL automatically upgrade precision to complex128
15. THE System SHALL implement Triton kernel with explicit Mourre estimate enforcement
16. WHEN benchmarking on T4 GPU, THE System SHALL achieve at least 85% of theoretical peak FLOPs
17. THE System SHALL implement batched tridiagonal solve using cuSPARSE gtsv2 and compare to custom LAP-based kernel
18. WHEN comparing kernels, THE System SHALL demonstrate that LAP-based kernel has 10× better numerical stability (lower error)
19. THE System SHALL provide real-time stability dashboard: condition numbers, Schatten norms, LAP bounds, Mourre constants
20. THE System SHALL verify that all numerical operations satisfy trace-class conditions from Propositions BS-HS and BS-trace

### Requirement 4: ε→0 極限による Koopman 圧縮と Clark 測度保存

**User Story:** 研究者として、論文の ε→0 極限操作を圧縮理論として実装し、Clark測度（スペクトル分布）を保存しながらモデルを圧縮したい。

#### Acceptance Criteria

1. THE System SHALL implement ε-parametrized model family: train models with ε ∈ {1.0, 0.75, 0.5, 0.25, 0.1}
2. WHEN ε decreases, THE System SHALL verify that model becomes more compressed (fewer effective parameters)
3. THE System SHALL implement cutoff function: ψ_ε(x) = ε^{-1/2} exp(-x²/(2ε)) with support shrinking as ε → 0
4. WHEN applying cutoff, THE System SHALL verify that ||V_ε||_L2 remains bounded uniformly in ε
5. THE System SHALL implement Clark measure preservation: μ_ε(E) = (1/2π) ∫_E |D_ε(λ + i0)|^{-2} dλ
6. WHEN compressing from ε = 1.0 to ε = 0.1, THE System SHALL verify that ||μ_1.0 - μ_0.1||_TV < 0.1 (total variation distance)
7. THE System SHALL implement spectral distribution matching: ensure compressed model has same eigenvalue distribution as full model
8. WHEN measuring eigenvalue spacing, THE System SHALL verify that both models follow GUE statistics
9. THE System SHALL implement knowledge distillation with Clark measure loss: L_Clark = ||μ_teacher - μ_student||²
10. WHEN distilling, THE System SHALL use soft targets (teacher logits) + Clark measure matching
11. THE System SHALL implement progressive compression: ε = 1.0 → 0.75 → 0.5 → 0.25 → 0.1 with retraining at each step
12. WHEN compressing progressively, THE System SHALL achieve at least 10× parameter reduction with < 15% perplexity degradation
13. THE System SHALL implement Koopman operator compression: use ε → 0 limit to identify essential Koopman modes
14. WHEN analyzing Koopman eigenvalues, THE System SHALL prune modes with |λ| < ε (vanishing in limit)
15. THE System SHALL implement trace-class compression: quantize only operators that remain in S_1 as ε → 0
16. WHEN quantizing, THE System SHALL verify that ||K_ε||_S1 ≤ (1/2)(Im z)^{-1}||V_ε||_L1 is maintained
17. THE System SHALL implement semiseparable structure preservation: ensure compressed matrix maintains O(N) complexity
18. WHEN compressing, THE System SHALL verify that tridiagonal + low-rank structure is preserved
19. THE System SHALL implement archimedean contribution preservation: W_∞(φ) term in Weil formula must be maintained
20. THE System SHALL demonstrate that ε-based compression achieves better accuracy-size trade-off than standard pruning/quantization

### Requirement 5: Google Colab無料枠での超大規模学習（Semiseparable構造の活用）

**User Story:** 研究者として、論文のsemiseparable（半可分）行列構造を活用し、Google Colab無料枠で10億〜100億パラメータを学習したい。

#### Acceptance Criteria

1. THE System SHALL implement semiseparable matrix factorization: H = tridiag + low_rank where low_rank has rank r << N
2. WHEN factorizing, THE System SHALL verify that r ≤ log N (logarithmic rank growth)
3. THE System SHALL implement O(N) matrix-vector multiplication using semiseparable structure
4. WHEN computing Hx, THE System SHALL achieve O(N) complexity instead of O(N²) for dense matrices
5. THE System SHALL implement gradient checkpointing with semiseparable-aware recomputation
6. WHEN recomputing activations, THE System SHALL use O(N) semiseparable structure instead of O(N²) full matrix
7. THE System SHALL reduce memory usage by at least 70% compared to dense attention (O(N²) → O(N log N))
8. THE System SHALL implement ZeRO Stage 1 + semiseparable partitioning: partition low-rank factors across GPUs
9. WHEN using ZeRO on 2 GPUs, THE System SHALL train models 3× larger than single GPU (better than standard 2×)
10. THE System SHALL implement CPU offloading for low-rank factors: keep tridiagonal on GPU, offload low-rank to CPU
11. WHEN using CPU offloading, THE System SHALL train models up to 8× larger with <25% slowdown
12. THE System SHALL implement activation checkpointing: store only tridiagonal part, recompute low-rank during backward
13. WHEN k=4 checkpointing, THE System SHALL reduce activation memory by 85% (better than standard 75%)
14. THE System SHALL implement dynamic batch sizing with semiseparable memory estimation
15. WHEN OOM detected, THE System SHALL adjust batch size based on O(N) memory model instead of O(N²)
16. THE System SHALL implement mixed-precision with structure-aware precision: FP16 for low-rank, FP32 for tridiagonal
17. WHEN using mixed-precision, THE System SHALL achieve 2.5× memory reduction (better than standard 2×)
18. THE System SHALL implement model parallelism: split sequence dimension using semiseparable block structure
19. WHEN using 2-way parallelism, THE System SHALL train models 1.9× larger (better than standard 1.8×)
20. THE System SHALL implement parameter sharing: share tridiagonal structure across layers
21. WHEN sharing structure, THE System SHALL reduce parameters by 40-50% (better than standard 20-30%)
22. THE System SHALL implement hierarchical semiseparable structure: nested low-rank approximations
23. WHEN using hierarchical structure, THE System SHALL reduce memory from O(N log N) to O(N log log N)
24. THE System SHALL provide memory profiling: show breakdown by tridiagonal, low-rank, activations, optimizer
25. THE System SHALL achieve training of 1B parameters on single T4 GPU and 10B parameters on 4× T4 GPUs
26. THE System SHALL achieve training of 100B parameters on 8× A100 GPUs (stretch goal) using full semiseparable optimization

### Requirement 5: Mamba超え - 長文安定性

**User Story:** 研究者として、128k〜1M tokensの超長文でMambaより安定した学習を実現し、決定的な優位性を示したい。

#### Acceptance Criteria

1. THE System SHALL train on sequence lengths N ∈ {128, 512, 2048, 8192, 32768, 131072, 524288, 1048576}
2. WHEN training on N=131072 (128k tokens), THE System SHALL complete training without NaN/Inf/divergence
3. THE System SHALL compare to Mamba baseline: train identical model size and hyperparameters on same data
4. WHEN Mamba diverges at N=32768, THE System SHALL demonstrate stable training up to N=131072 or beyond
5. THE System SHALL implement gradient norm tracking: plot gradient norm vs. training step for each sequence length
6. WHEN plotting gradient norms, THE System SHALL show that ResNet-BK maintains stable gradients while Mamba exhibits spikes
7. THE System SHALL implement loss spike detection: count number of loss spikes (loss increase > 2× previous value)
8. WHEN comparing spike counts, THE System SHALL demonstrate at least 10× fewer spikes than Mamba
9. THE System SHALL measure perplexity degradation: compare PPL at N=1024 vs. N=131072
10. WHEN sequence length increases 128×, THE System SHALL maintain PPL degradation < 30% while Mamba degrades > 100%
11. THE System SHALL implement numerical stability analysis: measure condition numbers of state matrices
12. WHEN analyzing condition numbers, THE System SHALL show that BK-Core maintains condition number < 10^4 while Mamba exceeds 10^8
13. THE System SHALL provide long-context benchmark script: automated training and evaluation on multiple sequence lengths
14. WHEN running benchmark script, THE System SHALL generate comparison graphs: loss curves, gradient norms, PPL vs. N
15. THE System SHALL implement streaming evaluation: evaluate on ultra-long sequences (1M tokens) without loading entire sequence into memory

### Requirement 6: Mamba超え - 量子化耐性

**User Story:** 研究者として、INT8/INT4量子化後もMambaより高い性能を維持し、エッジデバイス展開での優位性を示したい。

#### Acceptance Criteria

1. THE System SHALL implement post-training quantization (PTQ): quantize trained model to INT8 without retraining
2. WHEN applying INT8 PTQ, THE System SHALL maintain perplexity degradation < 5% on WikiText-2
3. THE System SHALL implement quantization-aware training (QAT): simulate INT8 operations during training
4. WHEN using INT8 QAT, THE System SHALL achieve perplexity within 2% of FP32 baseline
5. THE System SHALL implement INT4 quantization with group-wise quantization (group size = 128)
6. WHEN using INT4 quantization, THE System SHALL maintain perplexity degradation < 15% on WikiText-2
7. THE System SHALL compare to Mamba quantization: apply identical quantization schemes to both models
8. WHEN comparing INT8 performance, THE System SHALL demonstrate at least 10% lower perplexity degradation than Mamba
9. WHEN comparing INT4 performance, THE System SHALL demonstrate at least 20% lower perplexity degradation than Mamba
10. THE System SHALL implement mixed-precision quantization: INT4 for MoE, INT8 for BK-Core, FP16 for output layers
11. WHEN using mixed-precision quantization, THE System SHALL achieve 6× model size reduction with < 8% PPL degradation
12. THE System SHALL implement dynamic quantization: adjust quantization precision based on layer importance
13. WHEN using dynamic quantization, THE System SHALL achieve better accuracy-size trade-off than uniform quantization
14. THE System SHALL provide quantization sweep script: evaluate PPL across bit widths {FP32, FP16, INT8, INT4, INT2}
15. WHEN running quantization sweep, THE System SHALL generate comparison graph: PPL vs. bit width for ResNet-BK and Mamba

### Requirement 7: Mamba超え - 動的計算効率

**User Story:** 研究者として、散乱ベースルーティングとACTを組み合わせ、同じ精度をMambaより少ないFLOPsで達成したい。

#### Acceptance Criteria

1. THE System SHALL implement adaptive computation time (ACT) with scattering-phase-based halting
2. WHEN scattering phase is low (< 0.2), THE System SHALL halt computation early (exit after 2-3 layers)
3. WHEN scattering phase is high (> 0.8), THE System SHALL use full depth (all 8-12 layers)
4. THE System SHALL measure average FLOPs per token: track actual computation performed for each token
5. WHEN using ACT, THE System SHALL reduce average FLOPs by at least 40% while maintaining PPL within 5%
6. THE System SHALL compare to Mamba: measure FLOPs required to achieve target PPL (e.g., PPL=30 on WikiText-2)
7. WHEN comparing at equal PPL, THE System SHALL demonstrate at least 2× lower FLOPs than Mamba
8. THE System SHALL implement learned sparsity: predict which G_ii elements are important, compute only those
9. WHEN achieving 60% sparsity, THE System SHALL reduce BK-Core FLOPs by 2.5× with < 3% PPL degradation
10. THE System SHALL implement multi-scale processing: downsample sequence at middle layers
11. WHEN using 2× downsampling at middle layers, THE System SHALL reduce FLOPs by 30% with < 5% PPL degradation
12. THE System SHALL provide FLOPs counter: track forward FLOPs, backward FLOPs, total FLOPs per example
13. WHEN measuring FLOPs, THE System SHALL account for all operations: matrix multiplies, activations, routing, BK-Core recursions
14. THE System SHALL generate efficiency graph: plot PPL vs. average FLOPs for ResNet-BK and Mamba
15. WHEN plotting efficiency graph, THE System SHALL demonstrate that ResNet-BK achieves lower PPL at every FLOPs budget

### Requirement 8: 決定的な1枚グラフの生成（修正案の3軸戦略）

**User Story:** 研究者として、修正案で提案された「長文安定性」「量子化耐性」「動的計算効率」の3軸でMambaを圧倒するグラフを生成したい。

#### Acceptance Criteria

1. THE System SHALL generate "Long-Context Stability Graph": plot loss vs. training step for N ∈ {8k, 32k, 128k, 512k, 1M}
2. WHEN plotting stability graph, THE System SHALL show Mamba diverging (loss → ∞, NaN spikes) while ResNet-BK converges smoothly
3. WHEN Mamba diverges at N=32k, THE System SHALL show ResNet-BK maintaining stable loss up to N=1M
4. THE System SHALL annotate graph with "Mamba divergence point" and "ResNet-BK stable region"
5. THE System SHALL generate "Quantization Robustness Graph": plot PPL vs. bit width {FP32, FP16, INT8, INT4, INT2}
6. WHEN plotting quantization graph, THE System SHALL show ResNet-BK maintaining PPL < 50 at INT4 while Mamba exceeds PPL > 200
7. WHEN comparing INT4 performance, THE System SHALL show ResNet-BK has 4× lower PPL than Mamba
8. THE System SHALL annotate graph with "practical deployment threshold" (PPL < 100) and show only ResNet-BK meets it at INT4
9. THE System SHALL generate "Dynamic Efficiency Graph": plot PPL vs. average FLOPs per token
10. WHEN plotting efficiency graph, THE System SHALL show ResNet-BK achieving PPL=30 with 2× fewer FLOPs than Mamba
11. WHEN comparing at equal FLOPs budget, THE System SHALL show ResNet-BK has 30% lower PPL than Mamba
12. THE System SHALL annotate graph with "Pareto frontier" and show ResNet-BK dominates Mamba at all points
13. THE System SHALL implement automated benchmark pipeline: `python scripts/mamba_vs_bk_killer_graphs.py --all`
14. WHEN running pipeline, THE System SHALL complete in < 48 hours on Google Colab (4× T4 GPUs)
15. THE System SHALL generate publication-quality figures: 300 DPI, vector graphics (PDF/SVG), clear labels, error bars (±std over 5 runs)
16. THE System SHALL use consistent color scheme: ResNet-BK in blue, Mamba in red, with clear legend
17. THE System SHALL provide statistical significance testing: compute p-values (paired t-test) for all comparisons
18. WHEN reporting results, THE System SHALL include confidence intervals and ensure p < 0.001 for key claims
19. THE System SHALL generate summary table: compare ResNet-BK vs. Mamba on 15+ metrics
20. THE System SHALL include metrics: PPL, FLOPs, memory, speed, gradient stability, condition number, quantization error, etc.
21. THE System SHALL provide reproducibility package: Docker container + scripts + data + checkpoints
22. WHEN running reproducibility package, THE System SHALL generate identical results (within 2% variance)
23. THE System SHALL generate interactive dashboard: web-based visualization with zoom, filter, and comparison tools
24. THE System SHALL provide "one-click comparison": load ResNet-BK and Mamba checkpoints, run all benchmarks, generate graphs
25. THE System SHALL save all results in standardized JSON format: `results/{longcontext,quantization,efficiency}_comparison.json`

### Requirement 9: 再現性フルセットの提供

**User Story:** 研究者として、他の研究者が結果を完全に再現できるよう、スクリプト・データ・環境を整備したい。

#### Acceptance Criteria

1. THE System SHALL provide single-command benchmark script: `python scripts/mamba_vs_bk_longcontext.py --model {mamba,bk} --seq_len N --bits B`
2. WHEN running benchmark script, THE System SHALL automatically download datasets, train models, evaluate, and save results
3. THE System SHALL save results in standardized JSON format: `results/longcontext_loss_curves.json`, `results/quantization_sweep.json`
4. THE System SHALL provide visualization notebook: `notebooks/generate_killer_graphs.ipynb` to create all figures from JSON
5. WHEN running visualization notebook, THE System SHALL generate all three "killer graphs" in < 5 minutes
6. THE System SHALL provide Docker container: `docker pull resnetbk/mamba-killer:latest` with all dependencies
7. WHEN running Docker container, THE System SHALL provide Jupyter environment with all scripts and notebooks pre-installed
8. THE System SHALL provide Google Colab notebook: one-click execution of full benchmark pipeline
9. WHEN running Colab notebook, THE System SHALL complete training and evaluation in < 24 hours on free tier
10. THE System SHALL provide dataset preparation script: download and preprocess WikiText-2, WikiText-103, C4, The Pile
11. THE System SHALL provide model checkpoint sharing: upload trained models to Hugging Face Hub
12. WHEN downloading checkpoints, THE System SHALL provide models for all configurations: {1M, 10M, 100M, 1B, 10B} parameters
13. THE System SHALL provide hyperparameter configuration files: YAML files for all experiments
14. THE System SHALL provide logging infrastructure: automatic logging to Weights & Biases or TensorBoard
15. THE System SHALL provide troubleshooting guide: common errors and solutions for Google Colab execution

### Requirement 10: 理論的正当性の証明

**User Story:** 研究者として、なぜResNet-BKがMambaより優れているのかを数学的に説明し、論文として発表可能な理論的基盤を構築したい。

#### Acceptance Criteria

1. THE System SHALL provide mathematical proof: BK-Core can approximate SSM (Mamba) as special case
2. WHEN setting specific parameters, THE System SHALL demonstrate that BK-Core reduces to structured SSM
3. THE System SHALL prove expressiveness: BK-Core can represent any linear time-invariant system
4. THE System SHALL analyze spectral properties: prove that Riemann initialization leads to optimal eigenvalue distribution
5. WHEN analyzing eigenvalues, THE System SHALL show that GUE statistics maximize information propagation efficiency
6. THE System SHALL prove numerical stability: derive condition number bounds for theta/phi recursions
7. WHEN deriving bounds, THE System SHALL show that trace-class condition guarantees stability
8. THE System SHALL analyze long-context behavior: prove that BK-Core has bounded error accumulation
9. WHEN sequence length increases, THE System SHALL show that error grows as O(√N) vs. O(N) for naive methods
10. THE System SHALL prove quantization robustness: show that BK-Core has lower Lipschitz constant than Mamba
11. WHEN quantizing, THE System SHALL demonstrate that lower Lipschitz constant leads to smaller quantization error
12. THE System SHALL analyze computational complexity: prove that all operations are O(N) or better
13. THE System SHALL provide complexity breakdown: forward O(N), backward O(N), routing O(1) per token
14. THE System SHALL prove convergence guarantees: show that hybrid analytic gradient converges under standard assumptions
15. THE System SHALL provide theoretical FLOPs analysis: derive exact formulas and compare to Mamba
16. WHEN comparing FLOPs, THE System SHALL show that ResNet-BK has lower constant factors in O(N) complexity
17. THE System SHALL provide interpretability analysis: explain what each component learns (potential, scattering phase, etc.)
18. THE System SHALL generate theory document: LaTeX paper with all proofs, theorems, and experimental validation
19. WHEN submitting to conference, THE System SHALL provide camera-ready paper with all required sections
20. THE System SHALL provide supplementary material: code, data, and extended proofs for reviewers

---

**Overall Success Criteria:**

THE System SHALL be considered successful when:
1. All three "killer graphs" demonstrate clear superiority over Mamba (p < 0.01)
2. 10B parameter model trains successfully on Google Colab free tier (4× T4 GPUs with ZeRO)
3. Riemann initialization shows at least 20% faster convergence than standard initialization
4. Scattering-based routing achieves at least 5× faster routing than MLP-based gating
5. Trace-class CUDA kernel achieves at least 10× speedup over PyTorch implementation
6. Full reproducibility package allows independent verification of all results
7. Theoretical analysis provides rigorous mathematical justification for empirical results
8. Research paper is accepted at top-tier conference (NeurIPS, ICML, ICLR)


### Requirement 11: Mamba との直接比較ベンチマーク

**User Story:** 研究者として、Mambaと完全に同一条件で比較し、公平性を保証された結果を論文として発表したい。

#### Acceptance Criteria

1. THE System SHALL implement Mamba baseline: use official Mamba implementation from state-spaces/mamba repository
2. WHEN comparing models, THE System SHALL use identical hyperparameters: learning rate, batch size, optimizer, warmup steps
3. THE System SHALL use identical tokenization: same tokenizer, vocabulary size, sequence length
4. WHEN training both models, THE System SHALL use same random seeds for reproducibility
5. THE System SHALL implement fair FLOPs counting: count all operations including state updates, gating, normalization
6. WHEN measuring FLOPs, THE System SHALL verify that Mamba's SSM state update is correctly counted as O(N*D*state_dim)
7. THE System SHALL implement fair memory measurement: include all buffers, activations, optimizer states
8. WHEN measuring memory, THE System SHALL use identical batch size and sequence length for both models
9. THE System SHALL implement convergence analysis: plot loss curves with identical x-axis (tokens seen, not steps)
10. WHEN comparing convergence, THE System SHALL normalize by total compute (FLOPs) not wall-clock time
11. THE System SHALL implement ablation studies: disable each ResNet-BK component and measure impact
12. WHEN performing ablation, THE System SHALL test: Prime-Bump init, scattering router, LAP stability, semiseparable structure
13. THE System SHALL implement Mamba variants comparison: compare to Mamba-1, Mamba-2, Mamba-2.5 (if available)
14. WHEN comparing variants, THE System SHALL use latest official checkpoints and configurations
15. THE System SHALL implement multi-dataset evaluation: WikiText-2, WikiText-103, Penn Treebank, C4, The Pile
16. WHEN evaluating on multiple datasets, THE System SHALL report mean and std across all datasets
17. THE System SHALL implement downstream task evaluation: GLUE, SuperGLUE, SQuAD, MMLU
18. WHEN evaluating downstream tasks, THE System SHALL use identical fine-tuning protocol for both models
19. THE System SHALL implement statistical significance testing: bootstrap confidence intervals, permutation tests
20. WHEN reporting results, THE System SHALL ensure p < 0.01 for all "Mamba超え" claims with Bonferroni correction

### Requirement 12: 失敗モード分析と自動リカバリ

**User Story:** 研究者として、学習中の失敗を自動検出・回復し、Google Colab無料枠の制限時間内で確実に結果を得たい。

#### Acceptance Criteria

1. THE System SHALL implement NaN/Inf detection: check all tensors (activations, gradients, parameters) every N steps
2. WHEN NaN/Inf detected, THE System SHALL automatically rollback to last stable checkpoint
3. THE System SHALL implement gradient explosion detection: monitor gradient norm and detect sudden spikes (>10× median)
4. WHEN gradient explosion detected, THE System SHALL reduce learning rate by 10× and retry from checkpoint
5. THE System SHALL implement loss divergence detection: detect when loss increases >50% over 100 steps
6. WHEN loss divergence detected, THE System SHALL try: (1) reduce LR, (2) increase ε, (3) reduce batch size
7. THE System SHALL implement OOM recovery: catch CUDA OOM errors and automatically reduce batch size
8. WHEN OOM occurs, THE System SHALL save current state, clear cache, reduce batch size by 50%, and resume
9. THE System SHALL implement checkpoint corruption detection: verify checkpoint integrity before loading
10. WHEN checkpoint corrupted, THE System SHALL fallback to previous checkpoint (keep last 5 checkpoints)
11. THE System SHALL implement Colab timeout handling: detect when <30 min remaining and save emergency checkpoint
12. WHEN timeout imminent, THE System SHALL save: model, optimizer, scheduler, training state, metrics history
13. THE System SHALL implement automatic resume: detect incomplete training and resume from last checkpoint
14. WHEN resuming, THE System SHALL verify: epoch number, step number, random state, optimizer state
15. THE System SHALL implement numerical stability monitoring: track condition numbers, Schatten norms, eigenvalues
16. WHEN stability metrics exceed thresholds, THE System SHALL apply: spectral clipping, precision upgrade, ε adjustment
17. THE System SHALL implement training health dashboard: real-time monitoring of 20+ health metrics
18. WHEN health metrics degrade, THE System SHALL send alerts and suggest corrective actions
19. THE System SHALL implement automatic hyperparameter adjustment: tune ε, learning rate, batch size based on stability
20. WHEN automatic adjustment fails 3 times, THE System SHALL halt training and generate diagnostic report

### Requirement 13: 解釈可能性と可視化

**User Story:** 研究者として、なぜResNet-BKが優れているのかを直感的に理解し、論文の図として使用できる可視化を生成したい。

#### Acceptance Criteria

1. THE System SHALL visualize Prime-Bump potential: plot V_ε(x) with prime positions highlighted
2. WHEN visualizing potential, THE System SHALL show evolution over training: initial → epoch 1 → epoch 5 → final
3. THE System SHALL visualize scattering phase: plot δ_ε(λ) for each token with color-coded difficulty
4. WHEN visualizing scattering, THE System SHALL correlate with linguistic features: syntax, semantics, rare words
5. THE System SHALL visualize eigenvalue distribution: plot eigenvalues of H_ε and compare to GUE prediction
6. WHEN visualizing eigenvalues, THE System SHALL show spacing distribution and verify Wigner surmise
7. THE System SHALL visualize Schatten norms: plot ||K_ε||_S1 and ||K_ε||_S2 over training
8. WHEN Schatten norms approach theoretical bounds, THE System SHALL highlight and explain
9. THE System SHALL visualize Clark measure: plot μ_ε(λ) and show how it changes with ε
10. WHEN compressing (ε → 0), THE System SHALL show that μ_ε is preserved (low total variation distance)
11. THE System SHALL visualize attention patterns: show which tokens attend to which using G_ii coupling
12. WHEN visualizing attention, THE System SHALL compare to Transformer attention and highlight differences
13. THE System SHALL visualize expert specialization: cluster tokens by assigned expert and analyze patterns
14. WHEN analyzing experts, THE System SHALL identify: syntax expert, semantics expert, rare word expert, etc.
15. THE System SHALL visualize gradient flow: plot gradient magnitudes at each layer over training
16. WHEN comparing to Mamba, THE System SHALL show that ResNet-BK has more stable gradient flow
17. THE System SHALL visualize memory usage: breakdown by component (tridiagonal, low-rank, activations, optimizer)
18. WHEN using semiseparable structure, THE System SHALL show memory savings compared to dense matrices
19. THE System SHALL visualize computational graph: show O(N) complexity visually with operation counts
20. WHEN comparing to Mamba, THE System SHALL highlight where ResNet-BK saves computation
21. THE System SHALL generate publication-quality figures: vector graphics (PDF/SVG), consistent style, clear labels
22. WHEN generating figures, THE System SHALL follow academic standards: Nature/Science figure guidelines
23. THE System SHALL implement interactive visualization: web-based dashboard with zoom, pan, filter
24. WHEN using dashboard, THE System SHALL allow real-time exploration of training dynamics
25. THE System SHALL provide visualization export: save all figures in multiple formats (PNG, PDF, SVG, EPS)

### Requirement 14: コミュニティ連携とオープンソース戦略

**User Story:** 研究者として、ResNet-BKをコミュニティに広め、Mambaに代わる標準として確立したい。

#### Acceptance Criteria

1. THE System SHALL provide Hugging Face integration: implement transformers-compatible model class
2. WHEN using Hugging Face, THE System SHALL support: AutoModel, AutoTokenizer, Trainer API
3. THE System SHALL provide pre-trained checkpoints: upload models to Hugging Face Hub for all sizes {1M, 10M, 100M, 1B, 10B}
4. WHEN downloading checkpoints, THE System SHALL provide: model weights, config, tokenizer, training logs
5. THE System SHALL provide PyTorch Hub integration: enable `torch.hub.load('user/resnet-bk', 'resnet_bk_1b')`
6. WHEN using PyTorch Hub, THE System SHALL automatically download and cache models
7. THE System SHALL provide ONNX export: convert trained models to ONNX format for deployment
8. WHEN exporting to ONNX, THE System SHALL verify numerical equivalence (max error < 1e-5)
9. THE System SHALL provide TensorRT optimization: generate optimized engines for NVIDIA GPUs
10. WHEN using TensorRT, THE System SHALL achieve at least 3× inference speedup over PyTorch
11. THE System SHALL provide comprehensive documentation: README, tutorials, API reference, examples
12. WHEN writing documentation, THE System SHALL include: quick start, installation, training, inference, fine-tuning
13. THE System SHALL provide Colab tutorials: interactive notebooks for all major use cases
14. WHEN using tutorials, THE System SHALL complete in < 30 minutes on Colab free tier
15. THE System SHALL provide benchmark scripts: easy-to-run comparisons with Mamba, Transformer, RWKV
16. WHEN running benchmarks, THE System SHALL generate comparison tables and graphs automatically
17. THE System SHALL provide community forum: GitHub Discussions or Discord for Q&A and collaboration
18. WHEN users report issues, THE System SHALL provide: issue templates, debugging guides, FAQ
19. THE System SHALL provide citation information: BibTeX entry, DOI, arXiv link
20. WHEN paper is published, THE System SHALL update README with publication details and citation count
21. THE System SHALL provide contribution guidelines: code style, testing requirements, PR process
22. WHEN accepting contributions, THE System SHALL require: tests, documentation, benchmark results
23. THE System SHALL provide continuous integration: GitHub Actions for testing, benchmarking, deployment
24. WHEN CI runs, THE System SHALL test on: multiple Python versions, PyTorch versions, CUDA versions
25. THE System SHALL provide release process: semantic versioning, changelog, migration guides

### Requirement 15: 論文執筆サポート

**User Story:** 研究者として、トップカンファレンス（NeurIPS/ICML/ICLR）に採択される論文を執筆したい。

#### Acceptance Criteria

1. THE System SHALL generate LaTeX paper template: use NeurIPS/ICML style with all required sections
2. WHEN generating template, THE System SHALL include: abstract, introduction, related work, method, experiments, conclusion
3. THE System SHALL auto-generate method section: convert implementation to mathematical notation
4. WHEN describing method, THE System SHALL include: algorithm pseudocode, complexity analysis, theoretical guarantees
5. THE System SHALL auto-generate experiment section: convert benchmark results to tables and figures
6. WHEN presenting results, THE System SHALL include: main results table, ablation studies, statistical tests
7. THE System SHALL auto-generate related work: compare to Mamba, Transformer, RWKV, S4, Hyena, etc.
8. WHEN writing related work, THE System SHALL highlight: key differences, advantages, limitations
9. THE System SHALL generate supplementary material: extended proofs, additional experiments, implementation details
10. WHEN writing supplementary, THE System SHALL include: hyperparameters, training curves, failure cases
11. THE System SHALL provide theorem/proof templates: formal statements with LaTeX formatting
12. WHEN stating theorems, THE System SHALL include: assumptions, main result, proof sketch
13. THE System SHALL generate bibliography: auto-collect citations from code comments and documentation
14. WHEN generating bibliography, THE System SHALL use: BibTeX format, consistent style, complete metadata
15. THE System SHALL provide rebuttal templates: address common reviewer concerns
16. WHEN writing rebuttal, THE System SHALL include: additional experiments, clarifications, revised text
17. THE System SHALL generate camera-ready version: incorporate reviewer feedback and format for publication
18. WHEN preparing camera-ready, THE System SHALL verify: page limits, figure quality, citation format
19. THE System SHALL provide arXiv submission package: PDF, source files, supplementary material
20. WHEN submitting to arXiv, THE System SHALL include: abstract, keywords, categories, license

---

## Overall Success Criteria (Updated)

THE System SHALL be considered successful when:

1. **Mathematical Rigor**: All implementations satisfy theoretical guarantees from Birman-Schwinger paper
2. **Mamba超え**: Demonstrate superiority on all 3 axes (long-context, quantization, efficiency) with p < 0.01
3. **Scalability**: Train 10B parameters on Google Colab free tier (4× T4 GPUs)
4. **Reproducibility**: Independent researchers can reproduce all results within 2% variance
5. **Stability**: Zero training failures (NaN/Inf/divergence) across 100+ training runs
6. **Community Adoption**: 1000+ GitHub stars, 100+ citations within 1 year of publication
7. **Publication**: Paper accepted at NeurIPS/ICML/ICLR with strong reviews (>6/10 average)
8. **Impact**: ResNet-BK becomes reference implementation for O(N) language models

## Risk Mitigation

**High-Risk Areas:**
1. **Numerical Stability**: Mitigated by LAP, Mourre estimate, trace-class conditions
2. **Implementation Complexity**: Mitigated by modular design, comprehensive testing, CI/CD
3. **Fair Comparison**: Mitigated by using official Mamba implementation, identical hyperparameters
4. **Colab Limitations**: Mitigated by semiseparable structure, gradient checkpointing, automatic recovery
5. **Community Skepticism**: Mitigated by rigorous benchmarks, open source, reproducibility package

**Contingency Plans:**
- If Mamba超え fails on 1 axis: Focus on other 2 axes and position as complementary approach
- If 10B training fails: Demonstrate 1B on single GPU and provide scaling analysis
- If paper rejected: Incorporate feedback, run additional experiments, resubmit to next venue
- If numerical instability persists: Fall back to hybrid approach (BK-Core + standard backprop)
