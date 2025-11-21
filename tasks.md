# Phase 3: Physics Transcendence - 実装タスクリスト

## 概要

Phase 3「物理的超越」は、Project MUSEの最終フェーズです。このタスクリストは、設計書で定義された7つの核心技術を段階的に実装するための具体的な手順を示します。

## 実装戦略

**3段階の段階的実装**:
- **Stage 1 (Week 1-2)**: Complex Dynamics Only - 複素数化の効果を検証
- **Stage 2 (Week 3-4)**: + Hamiltonian ODE - エネルギー保存思考を追加
- **Stage 3 (Week 5-6)**: Full Integration - 全機能統合

**各Stageの完了条件**:
- すべての数値目標を達成すること
- ベンチマークスクリプトがJSONレポートを生成し、"pass": true となること
- 次のStageに進む前に、現Stageの完了条件をすべて満たすこと

## 数値目標達成の確認方法

各タスクには具体的な数値目標が設定されています。以下の方法で達成を確認してください：

### 自動検証
- ベンチマークスクリプト（`scripts/benchmark_phase3_stage*.py`）を実行する
- 生成されたJSONレポート（`results/benchmarks/phase3_stage*_comparison.json`）を確認する
- すべての項目で `"pass": true` となっていることを確認する

### 手動検証
- 各タスクの「完了条件」または「目標」に記載された数値を確認する
- 測定条件（Batch size、Seq length、fp16など）を厳守する
- 複数回実行して安定性を確認する（推奨: 3回以上）

### Stage完了の判定
- Stage完了条件セクションに記載されたすべての項目を達成する
- ベンチマークレポートで `"all_pass": true` となる
- 次のStageに進む前に、レビューを実施する

## Stage 1: Complex Dynamics Foundation (Week 1-2)

### 目的
複素数ニューラルネットワークの基盤を構築し、メモリ効率50%削減を実証する。

### Stage 1完了条件（すべて達成必須）
- **Perplexity**: WikiText-2で Phase 2比 **+3%以内**（Phase 2が30.0なら30.9以下）
- **VRAM削減**: Phase 2比 **52%以下**（Phase 2が6.0GBなら3.12GB以下）
- **数値安定性**: ランダム入力100回試行で **NaN発生率 0%**
- **勾配健全性**: 全層の勾配ノルムが **1e-6以上、1e3以下**
- **メモリレイアウト**: ComplexTensorが **Planar形式** で実装されていること

- [x] 1. Complex32データ構造の実装

  - src/models/phase3/complex_tensor.py を作成する
  - ComplexTensorクラスを実装する（Planar形式）
  - 複素数演算（加算、乗算、共役、絶対値）を実装する
  - complex64との相互変換機能を実装する
  - _Requirements: 1.1, 1.2, 1.3, 1.4_


- [x] 1.1 ComplexTensorクラスの基本構造実装

  - real（torch.HalfTensor）とimag（torch.HalfTensor）を保持する
  - shape、device、dtypeプロパティを実装する
  - _Requirements: 1.1_


- [x] 1.2 複素数演算の実装
  - __add__、__mul__、conj、absメソッドを実装する
  - 数値安定性を考慮した実装（ゼロ除算対策）
  - _Requirements: 1.2_


- [x] 1.3 変換機能の実装
  - to_complex64、from_complex64メソッドを実装する
  - PyTorchネイティブ型との互換性を確保する
  - _Requirements: 1.3_

- [x] 1.4 ComplexTensor単体テストの実装
  - tests/test_complex_tensor.py を作成する
  - 演算の正確性を検証する
  - メモリ使用量が50%削減されることを確認する
  - _Requirements: 1.4_

- [x] 2. ComplexLinear層の実装
  - src/models/phase3/complex_ops.py を作成する
  - ComplexLinearクラスを実装する
  - 複素行列積を実装する（Planar形式最適化）
  - Xavier初期化（複素数版）を実装する
  - _Requirements: 1.5, 1.6, 1.7_

- [x] 2.1 ComplexLinearの基本構造実装
  - weight_real、weight_imag、bias_real、bias_imagパラメータを定義する
  - use_complex32フラグでfloat16/float32を切り替える
  - _Requirements: 1.5_

- [x] 2.2 複素行列積の実装
  - forward メソッドで (A + iB)(x + iy) = (Ax - By) + i(Bx + Ay) を計算する
  - ComplexTensorとcomplex64の両方に対応する
  - _Requirements: 1.6_

- [x] 2.3 初期化の実装
  - Xavier初期化を実部と虚部に適用する
  - 複素数の大きさが適切な範囲に収まるようにする
  - _Requirements: 1.7_

- [x] 2.4 ComplexLinear単体テストの実装
  - tests/test_complex_ops.py を作成する
  - 出力形状の正確性を検証する
  - 勾配計算が正常に動作することを確認する
  - _Requirements: 1.8_

- [x] 3. ModReLU活性化関数の実装
  - src/models/phase3/complex_ops.py にModReLUクラスを追加する
  - 振幅フィルタリング + 位相保存を実装する
  - バイアスパラメータを追加する
  - _Requirements: 1.9, 1.10_

- [x] 3.1 ModReLU数式の実装
  - z' = ReLU(|z| + b) · z / |z| を実装する
  - ゼロ除算を防ぐためのイプシロン（1e-6）を追加する
  - _Requirements: 1.9_

- [x] 3.2 ModReLU単体テストの実装
  - 位相が保存されることを確認する
  - 振幅がフィルタリングされることを確認する
  - _Requirements: 1.10_

- [x] 4. ComplexLayerNormの実装
  - src/models/phase3/complex_ops.py にComplexLayerNormクラスを追加する
  - 複素平均と複素分散を計算する
  - アフィン変換（実数パラメータ）を実装する
  - _Requirements: 1.11, 1.12_

- [x] 4.1 複素正規化の実装
  - 複素平均 μ = E[z] を計算する
  - 複素分散 σ² = E[|z - μ|²] を計算する
  - z' = (z - μ) / √(σ² + ε) を実装する
  - _Requirements: 1.11_

- [x] 4.2 ComplexLayerNorm単体テストの実装
  - 正規化後の平均が0、分散が1に近いことを確認する
  - 勾配計算が正常に動作することを確認する
  - _Requirements: 1.12_

- [x] 5. Complex Embedding層の実装
  - src/models/phase3/complex_embedding.py を作成する
  - ComplexEmbeddingクラスを実装する
  - Token EmbeddingとPosition Embeddingを統合する
  - _Requirements: 1.13, 1.14_

- [x] 5.1 ComplexEmbeddingの実装
  - Token Embeddingを複素数化する（実部と虚部を独立に学習）
  - Phase 2のZetaEmbeddingを継承する
  - _Requirements: 1.13_

- [x] 5.2 ComplexEmbedding単体テストの実装
  - 出力がComplexTensor形式であることを確認する
  - メモリ使用量を測定する
  - _Requirements: 1.14_

- [x] 6. Stage 1統合モデルの実装
  - src/models/phase3/stage1_model.py を作成する
  - Phase3Stage1Modelクラスを実装する
  - Phase 2モデルを複素数化する
  - _Requirements: 1.15, 1.16, 1.17_

- [x] 6.1 モデル構造の実装
  - ComplexEmbedding → ComplexLinear × N → Output の流れを実装する
  - 各層にComplexLayerNormとModReLUを配置する
  - 残差接続を実装する
  - _Requirements: 1.15_

- [x] 6.2 Phase 2互換性の実装
  - Phase 2モデルの重みをComplexTensorに変換する機能を実装する
  - convert_phase2_to_complex 関数を実装する
  - _Requirements: 1.16_

- [x] 6.3 Stage 1統合テストの実装
  - tests/test_phase3_stage1.py を作成する
  - Forward/Backward passが正常に動作することを確認する
  - NaN/Infが発生しないことを確認する
  - _Requirements: 1.17_

- [x] 7. Stage 1ベンチマークの実装
  - scripts/benchmark_phase3_stage1.py を作成する
  - WikiText-2でPerplexityを測定する
  - VRAM使用量を測定する
  - Phase 2との比較表を生成する
  - **完了条件**: すべての数値目標を達成し、JSONレポートを生成する
  - _Requirements: 1.18, 1.19, 1.20_

- [x] 7.1 Perplexity測定の実装
  - WikiText-2データセットでPPLを計算する
  - Phase 2モデルと同じ条件で測定する
  - **目標**: Phase 2比 **+3%以内**（Phase 2が30.0なら30.9以下）
  - **測定条件**: Batch=4, Seq=1024, fp16, 同一シード
  - _Requirements: 1.18_

- [x] 7.2 VRAM測定の実装
  - torch.cuda.max_memory_allocated() でVRAMを測定する
  - Batch=2, Seq=2048で測定する
  - **目標**: Phase 2比 **52%以下**（Phase 2が6.0GBなら3.12GB以下）
  - **測定条件**: Forward + Backward pass、Gradient Checkpointing有効
  - _Requirements: 1.19_

- [x] 7.3 比較表の生成
  - results/benchmarks/phase3_stage1_comparison.json を生成する
  - PPL、VRAM、Throughputを記録する
  - **目標**: すべての項目で数値目標を達成していることを明記
  - **フォーマット**: {"ppl": 30.5, "ppl_phase2": 30.0, "ppl_ratio": 1.017, "vram_gb": 3.1, "vram_phase2_gb": 6.0, "vram_ratio": 0.517, "pass": true}
  - _Requirements: 1.20_


## Stage 2: Hamiltonian ODE Integration (Week 3-4)

### 目的
エネルギー保存思考機構を追加し、O(1)メモリ学習を実証する。

### Stage 2完了条件（すべて達成必須）
- **Perplexity**: WikiText-2で Stage 1比 **+2%以内**
- **Energy Drift**: 100ステップ積分で **< 5e-5**（閾値の半分）
- **VRAM制約**: Batch=2, Seq=2048で **< 7.5GB**（8GBの93.75%）
- **再構成誤差**: Symplectic Adjoint使用時 **< 8e-6**（閾値1e-5の80%）
- **フォールバック動作**: 再構成誤差 > 1e-5の時、自動的に **Checkpointingモードに切り替わる**
- **メモリ効率**: Symplectic Adjoint使用時、Full Backprop比で **メモリ使用量が1/T以下**（Tは積分ステップ数）

- [x] 8. HamiltonianFunction（ハミルトニアン関数）の実装
  - src/models/phase3/hamiltonian.py を作成する
  - HamiltonianFunctionクラスを実装する
  - 運動エネルギーとポテンシャルエネルギーを計算する
  - ハミルトンベクトル場を計算する
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [x] 8.1 ハミルトニアン関数の実装
  - H(q, p) = T(p) + V(q) を実装する
  - T(p) = ½|p|² （運動エネルギー）
  - V(q) = Potential_Net(q) （ポテンシャルエネルギー）
  - _Requirements: 2.1_

- [x] 8.2 ポテンシャルネットワークの実装

  - BK-Coreまたは標準MLPを選択可能にする
  - potential_type パラメータで切り替える
  - _Requirements: 2.2_

- [x] 8.3 ハミルトンベクトル場の実装

  - hamiltonian_vector_field メソッドを実装する
  - dq/dt = ∂H/∂p、dp/dt = -∂H/∂q を計算する
  - シンプレクティック構造 J = [[0, I], [-I, 0]] を適用する
  - _Requirements: 2.3_

- [x] 8.4 HamiltonianFunction単体テストの実装
  - tests/test_hamiltonian.py を作成する
  - エネルギー計算の正確性を検証する
  - ベクトル場の勾配計算を検証する
  - _Requirements: 2.4_

- [x] 9. Symplectic Integrator（シンプレクティック積分器）の実装
  - src/models/phase3/hamiltonian.py にsymplectic_leapfrog_step関数を追加する
  - Leapfrog法を実装する
  - エネルギー保存性を検証する
  - _Requirements: 2.5, 2.6, 2.7_


- [x] 9.1 Leapfrog積分の実装
  - p(t + dt/2) = p(t) - ∇V(q(t)) · dt/2
  - q(t + dt) = q(t) + p(t + dt/2) · dt
  - p(t + dt) = p(t + dt/2) - ∇V(q(t + dt)) · dt/2
  - _Requirements: 2.5_

- [x] 9.2 エネルギー監視の実装
  - monitor_energy_conservation 関数を実装する
  - エネルギー誤差 = (E_max - E_min) / E_mean を計算する
  - _Requirements: 2.6_

- [x] 9.3 Symplectic Integrator単体テストの実装
  - エネルギー誤差が1e-4以下であることを確認する
  - 100ステップの積分でエネルギーが保存されることを確認する
  - _Requirements: 2.7_

- [x] 10. Symplectic Adjoint Method（随伴法）の実装
  - src/models/phase3/symplectic_adjoint.py を作成する
  - SymplecticAdjointクラス（torch.autograd.Function）を実装する
  - 順伝播と逆伝播を実装する
  - 再構成誤差監視機構を実装する
  - _Requirements: 2.8, 2.9, 2.10, 2.11, 2.12_

- [x] 10.1 順伝播の実装
  - forward メソッドでLeapfrog積分を実行する
  - 最終状態のみを保存する（O(1)メモリ）
  - _Requirements: 2.8_

- [x] 10.2 逆伝播の実装
  - backward メソッドで随伴法を実装する
  - 時間を逆再生して随伴状態を更新する
  - パラメータ勾配を累積する
  - _Requirements: 2.9_

- [x] 10.3 再構成誤差監視の実装
  - 10ステップごとに再構成誤差をチェックする
  - 閾値（1e-5）を超えた場合、ReconstructionErrorを投げる
  - _Requirements: 2.10_

- [x] 10.4 ReconstructionError例外の実装

  - ReconstructionErrorクラスを定義する
  - エラー値と閾値を保持する
  - _Requirements: 2.11_

- [x] 10.5 Symplectic Adjoint単体テストの実装
  - tests/test_symplectic_adjoint.py を作成する
  - 勾配計算の正確性を検証する（gradcheck）
  - メモリ使用量がO(1)であることを確認する
  - _Requirements: 2.12_

- [x] 11. HamiltonianNeuralODE（フォールバック機構付き）の実装
  - src/models/phase3/hamiltonian_ode.py を作成する
  - HamiltonianNeuralODEクラスを実装する
  - 3段階フォールバック機構を実装する
  - _Requirements: 2.13, 2.14, 2.15, 2.16_

- [x] 11.1 基本構造の実装
  - HamiltonianFunctionを保持する
  - mode（symplectic_adjoint / checkpointing / full_backprop）を管理する
  - _Requirements: 2.13_

- [x] 11.2 Symplectic Adjointモードの実装
  - _forward_symplectic_adjoint メソッドを実装する
  - ReconstructionErrorをキャッチしてフォールバックする
  - _Requirements: 2.14_

- [x] 11.3 Checkpointingモードの実装
  - _forward_with_checkpointing メソッドを実装する
  - 10ステップごとにチェックポイントを保存する
  - _Requirements: 2.15_

- [x] 11.4 Full Backpropモードの実装
  - _forward_full_backprop メソッドを実装する
  - 全ステップの状態を保存する（緊急フォールバック）
  - _Requirements: 2.16_

- [x] 11.5 HamiltonianNeuralODE単体テストの実装
  - tests/test_hamiltonian_ode.py を作成する
  - 各モードが正常に動作することを確認する
  - フォールバック機構が正しく動作することを確認する
  - _Requirements: 2.17_

- [x] 12. Stage 2統合モデルの実装

  - src/models/phase3/stage2_model.py を作成する
  - Phase3Stage2Modelクラスを実装する
  - Stage 1モデルにHamiltonian ODEを追加する
  - _Requirements: 2.18, 2.19, 2.20_


- [x] 12.1 モデル構造の実装
  - ComplexEmbedding → HamiltonianODE → ComplexLinear × N → Output
  - 各ブロックでODEを実行する
  - _Requirements: 2.18_

- [x] 12.2 Complex → Real変換の実装
  - ODEは実数で処理するため、ComplexTensorを[real, imag]に変換する
  - ODE出力を再びComplexTensorに変換する
  - _Requirements: 2.19_

- [x] 12.3 Stage 2統合テストの実装
  - tests/test_phase3_stage2.py を作成する
  - エネルギー保存が機能することを確認する
  - フォールバックが正しく動作することを確認する
  - _Requirements: 2.20_

- [x] 13. Stage 2ベンチマークの実装
  - scripts/benchmark_phase3_stage2.py を作成する
  - WikiText-2でPerplexityを測定する
  - Energy Driftを測定する
  - VRAM使用量を測定する
  - **完了条件**: すべての数値目標を達成し、JSONレポートを生成する
  - _Requirements: 2.21, 2.22, 2.23_

- [x] 13.1 Perplexity測定の実装
  - Stage 1と同じ条件で測定する
  - **目標**: Stage 1比 **+2%以内**
  - **測定条件**: Batch=4, Seq=1024, fp16, ODE steps=10
  - **記録項目**: PPL、PPL_stage1、PPL_ratio、pass/fail
  - _Requirements: 2.21_

- [x] 13.2 Energy Drift測定の実装
  - 100ステップの積分でEnergy Driftを計算する
  - **目標**: **< 5e-5**（閾値1e-4の半分）
  - **測定条件**: Batch=4, Seq=512, dt=0.1, 100 steps
  - **記録項目**: mean_energy、max_drift、mean_drift、pass/fail
  - **追加検証**: エネルギーが単調増加/減少していないこと（振動許容範囲 ±10%）
  - _Requirements: 2.22_

- [x] 13.3 VRAM測定の実装
  - Symplectic Adjoint使用時のVRAMを測定する
  - **目標**: Batch=2, Seq=2048で **< 7.5GB**（8GBの93.75%）
  - **測定条件**: Forward + Backward pass、Symplectic Adjoint有効
  - **比較**: Full Backprop時のVRAMも測定し、削減率を計算（目標: **70%以上削減**）
  - **記録項目**: vram_symplectic_gb、vram_full_backprop_gb、reduction_ratio、pass/fail
  - _Requirements: 2.23_


## Stage 3: Full Integration (Week 5-6)

### 目的
全機能（Koopman、MERA、Dialectic、Entropic Selection）を統合し、Phase 3を完成させる。

### Stage 3完了条件（すべて達成必須）
- **Perplexity**: WikiText-2で Phase 2比 **+5%以内**、PTBで **+5%以内**、C4で **+5%以内**
- **VRAM制約**: Batch=1, Seq=4096で **< 7.8GB**（8GBの97.5%）
- **Throughput**: Phase 2比 **85%以上**（Phase 2が100 tokens/secなら85 tokens/sec以上）
- **Koopman線形性**: 線形性誤差 **< 5e-4**（閾値1e-3の50%）
- **MERA計算量**: O(N log N)を実証（Seq=4096で **< 50ms**）


- **Dialectic収束**: 矛盾スコアが学習開始から **30%以上減少**
- **Entropic Selection**: データの **82%以上** がフィルタリングされる（閾値80%を上回る）
- **全機能統合**: すべてのコンポーネントが **エラーなく** 動作し、勾配が正常に伝播する

- [x] 14. Koopman Operator（線形化機構）の実装
  - src/models/phase3/koopman.py を作成する
  - KoopmanOperatorクラスを実装する
  - Observable Encoder/Decoderを実装する
  - Residual Correctionを実装する
  - **完了条件**: 線形性誤差 **< 5e-4**、多段階予測が単一ステップ予測の **3倍以上高速**
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [x] 14.1 Observable Encoderの実装
  - psi メソッドで x → g（Koopman空間）への射影を実装する
  - MLP（Linear → GELU → Linear）を使用する
  - _Requirements: 3.1_

- [x] 14.2 Koopman Operatorの実装
  - K（線形作用素）を実装する
  - 単位行列で初期化する
  - _Requirements: 3.2_

- [x] 14.3 Observable Decoderの実装
  - psi_inv メソッドで g → x への逆射影を実装する
  - _Requirements: 3.3_

- [x] 14.4 Residual Correctionの実装
  - residual_net で非線形補正を実装する
  - x_{t+1} = Ψ⁻¹(K · g_t) + R(g_t)
  - _Requirements: 3.4_

- [x] 14.5 KoopmanOperator単体テストの実装

  - tests/test_koopman.py を作成する
  - 線形性が保たれることを確認する
  - 多段階予測が高速であることを確認する
  - _Requirements: 3.5_

- [x] 15. Koopman Training Scheduler（段階的学習）の実装

  - src/models/phase3/koopman.py にKoopmanTrainingSchedulerクラスを追加する
  - Phase 1: Residual freeze、Kのみ学習
  - Phase 2: Residual unfreeze、両方学習
  - Phase 3: 正則化追加
  - _Requirements: 3.6, 3.7, 3.8_

- [x] 15.1 学習スケジュールの実装
  - step_epoch メソッドでエポックごとにモードを切り替える
  - freeze_residual_epochs パラメータで切り替えタイミングを制御する
  - _Requirements: 3.6_

- [x] 15.2 正則化損失の実装
  - get_regularization_loss メソッドを実装する
  - Residualの大きさにペナルティを課す
  - Koopman行列の固有値が単位円に近いことを奨励する
  - _Requirements: 3.7_

- [x] 15.3 KoopmanTrainingScheduler単体テストの実装

  - 各フェーズで正しくパラメータがfreeze/unfreezeされることを確認する
  - _Requirements: 3.8_

- [x] 16. MERA Router（階層的情報集約）の実装

  - src/models/phase3/mera.py を作成する
  - MERARouterクラスを実装する
  - Disentangler層とIsometry層を実装する
  - **完了条件**: Seq=4096で計算時間 **< 50ms**、計算量がO(N log N)であることを実証（Seq=2048→4096で時間が **2.1倍以下**）
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 16.1 MERARouterの基本構造実装
  - n_layers = Log₂(max_seq_len) を計算する
  - Disentangler層（Linear(2D → 2D)）を定義する
  - Isometry層（Linear(2D → D)）を定義する
  - _Requirements: 4.1_

- [x] 16.2 Bottom-up Passの実装
  - forward メソッドでシーケンスを階層的に圧縮する
  - 各層でペアトークンをDisentangle → Coarse-grainする
  - Global Contextを抽出する
  - _Requirements: 4.2_

- [x] 16.3 Padding処理の実装
  - シーケンス長を2のべき乗にパディングする
  - _Requirements: 4.3_

- [x] 16.4 MERARouter単体テストの実装

  - tests/test_phase3_integrated.py にてカバー
  - Global Contextが正しく抽出されることを確認する
  - 計算量がO(N log N)であることを確認する
  - _Requirements: 4.4_

- [x] 17. Entropic Data Selection（データ選別）の実装
  - src/models/phase3/entropic.py を作成する
  - EntropicSelectorクラスを実装する
  - 熱力学的サプライズを計算する
  - Curriculum Warmupを実装する
  - **完了条件**: keep_ratio=0.2の時、データの **82%以上** がフィルタリングされる、各クラスで **最低100サンプル** が保持される
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [x] 17.1 サプライズスコアの計算
  - score_dataset メソッドを実装する
  - 各サンプルのLossを計算する
  - _Requirements: 5.1_

- [x] 17.2 データフィルタリングの実装
  - filter_dataset メソッドを実装する
  - 上位k%のサンプルを選択する
  - Diversity Guardrail（クラスごとの最小サンプル数）を実装する
  - _Requirements: 5.2_

- [x] 17.3 Curriculum Warmupの実装
  - 最初のN epochは全データを使用する
  - 徐々にkeep_ratioを減少させる
  - _Requirements: 5.3_

- [x] 17.4 EntropicSelector単体テストの実装

  - tests/test_phase3_integrated.py にてカバー
  - 高損失サンプルが優先的に選択されることを確認する
  - _Requirements: 5.4_


- [x] 18. Dialectic Loop（自己進化機構）の実装
  - src/models/phase3/dialectic_loop.py を作成する
  - DialecticLoopクラスを実装する
  - Straight-Through Gumbel-Softmaxを実装する
  - 温度アニーリングを実装する
  - **完了条件**: 矛盾スコアが学習開始（Epoch 0）から学習後（Epoch 10）で **30%以上減少**、勾配が正常に伝播する（勾配ノルム **> 1e-6**）
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 18.1 仮説生成の実装
  - generate_hypothesis メソッドを実装する
  - Gumbel-Softmaxで微分可能なサンプリングを実現する
  - Straight-Through Estimatorを実装する
  - _Requirements: 6.1_

- [x] 18.2 仮説批判の実装
  - critique_hypothesis メソッドを実装する
  - Hamiltonian ODEでエネルギー分散を計算する
  - 矛盾スコアとして返す
  - _Requirements: 6.2_

- [x] 18.3 Synthesisの実装
  - forward メソッドで矛盾最小化損失を計算する
  - 診断情報（contradiction_score、entropy、temperature）を返す
  - _Requirements: 6.3_

- [x] 18.4 温度アニーリングの実装
  - anneal_temperature メソッドを実装する
  - 初期: 高温度（多様な仮説生成）
  - 後期: 低温度（確信的な予測）
  - _Requirements: 6.4_

- [x] 18.5 DialecticLoop単体テストの実装
  - tests/test_dialectic_loop.py を作成する
  - 勾配が正しく伝播することを確認する
  - 温度アニーリングが機能することを確認する
  - _Requirements: 6.5_

- [x] 19. Phase3Block（統合ブロック）の実装

  - src/models/phase3/integrated_model.py を作成する
  - Phase3Blockクラスを実装する
  - ComplexLayerNorm → HamiltonianODE → Koopman → Residualの流れを実装する
  - _Requirements: 7.1, 7.2, 7.3_

- [x] 19.1 Phase3Blockの基本構造実装
  - ComplexLayerNorm、HamiltonianNeuralODE、KoopmanOperatorを統合する
  - 残差接続を実装する
  - _Requirements: 7.1_

- [x] 19.2 診断情報収集の実装
  - return_diagnostics=True の時、各コンポーネントの診断情報を収集する
  - エネルギー、Koopman固有値などを返す
  - _Requirements: 7.2_

- [x] 19.3 Phase3Block単体テストの実装
  - tests/test_phase3_integrated.py にてカバー
  - Forward/Backward passが正常に動作することを確認する
  - _Requirements: 7.3_

- [x] 20. Phase3IntegratedModel（完全統合モデル）の実装

  - src/models/phase3/integrated_model.py にPhase3IntegratedModelクラスを追加する
  - Embedded Thinking Architectureを実装する
  - MERA → Broadcast → Phase3Block × N → Outputの流れを実装する
  - _Requirements: 7.4, 7.5, 7.6, 7.7_

- [x] 20.1 モデル構造の実装
  - ComplexEmbedding → MERARouter → Broadcast → Phase3Block × N → Output
  - Global Contextを各トークンに付与する
  - _Requirements: 7.4_

- [x] 20.2 Forward Passの実装
  - input_ids から logits を計算する
  - Complex → Real変換を実装する
  - _Requirements: 7.5_

- [x] 20.3 Phase 2互換性の実装
  - Phase 2モデルからの変換をサポートする
  - _Requirements: 7.6_

- [x] 20.4 Phase3IntegratedModel統合テストの実装
  - tests/test_phase3_integration.py を作成する
  - エンドツーエンドでモデルが動作することを確認する
  - _Requirements: 7.7_

- [x] 21. モデルファクトリーの実装
  - src/models/phase3/factory.py を作成する
  - create_phase3_model 関数を実装する
  - convert_phase2_to_phase3 関数を実装する
  - プリセット設定を提供する
  - _Requirements: 7.8, 7.9_

- [x] 21.1 create_phase3_model関数の実装
  - 設定からPhase3IntegratedModelを生成する
  - プリセット（small, base, large）を提供する
  - _Requirements: 7.8_

- [x] 21.2 convert_phase2_to_phase3関数の実装
  - Phase 2モデルの重みをPhase 3モデルに変換する
  - 互換性のある層の重みをコピーする
  - _Requirements: 7.9_


## Priority 1: Triton Kernel Optimization (Optional)

### 目的
高頻度演算をTriton化して、20-30%の高速化を実現する。

- [x] 22. Complex Matrix Multiplication Tritonカーネルの実装
  - src/kernels/complex_matmul.py を作成する
  - complex_matmul_kernel（@triton.jit）を実装する
  - Planar形式に最適化されたメモリアクセスを実装する
  - **完了条件**: PyTorch実装比 **25%以上高速化**（目標: 1.25倍以上）、数値誤差 **< 1e-5**
  - _Requirements: 8.1, 8.2, 8.3_

- [x] 22.1 Tritonカーネルの実装
  - 実部と虚部を分離して処理する
  - Coalesced accessを最適化する
  - **目標**: メモリアクセスパターンが **100%コアレス** であること（stride=1）
  - _Requirements: 8.1_

- [x] 22.2 Python Wrapperの実装
  - complex_matmul 関数を実装する
  - CPU Fallbackを実装する
  - **目標**: CUDA利用不可時、自動的にPyTorch実装にフォールバックする
  - _Requirements: 8.2_

- [x] 22.3 ベンチマークの実装
  - scripts/benchmark_complex_matmul.py を作成する
  - PyTorch実装と比較して20-30%の高速化を確認する
  - **測定条件**: Batch=16, M=512, N=512, K=512, 100回実行の平均値
  - **目標**: Triton版がPyTorch版の **1.25倍以上** の速度、数値誤差（MSE）**< 1e-5**
  - **記録項目**: pytorch_time_ms、triton_time_ms、speedup_ratio、mse_error、pass/fail
  - _Requirements: 8.3_

- [x] 23. Symplectic Step Tritonカーネルの実装
  - src/kernels/symplectic_step.py を作成する
  - symplectic_step_kernel（@triton.jit）を実装する
  - Leapfrog積分を並列化する
  - **完了条件**: PyTorch実装比 **20%以上高速化**（目標: 1.20倍以上）、Energy Drift **< 5e-5**
  - _Requirements: 8.4, 8.5, 8.6_

- [x] 23.1 Tritonカーネルの実装
  - 力の計算を並列化する
  - 位置と運動量の更新を並列化する
  - **目標**: 各ステップの計算が **完全に並列** であること（依存関係なし）
  - _Requirements: 8.4_

- [x] 23.2 Python Wrapperの実装
  - symplectic_step 関数を実装する
  - CPU Fallbackを実装する
  - **目標**: CUDA利用不可時、自動的にPyTorch実装にフォールバックする
  - _Requirements: 8.5_

- [x] 23.3 ベンチマークの実装
  - scripts/benchmark_symplectic_step.py を作成する
  - PyTorch実装と比較して15-25%の高速化を確認する
  - **測定条件**: Batch=16, Seq=2048, dt=0.1, 100ステップ、100回実行の平均値
  - **目標**: Triton版がPyTorch版の **1.20倍以上** の速度、Energy Drift **< 5e-5**
  - **記録項目**: pytorch_time_ms、triton_time_ms、speedup_ratio、energy_drift、pass/fail
  - _Requirements: 8.6_

## Priority 2: Training & Evaluation

### 目的
Phase 3モデルを学習し、Phase 2との比較を行う。

- [x] 24. Phase 3学習スクリプトの実装
  - scripts/train_phase3.py を作成する
  - Stage 1 → Stage 2 → Stage 3の順で学習する
  - 各Stageで診断情報をログに記録する
  - _Requirements: 9.1, 9.2, 9.3_

- [x] 24.1 学習ループの実装
  - データローダー、オプティマイザー、スケジューラーを設定する
  - 各Stageで適切なモデルを使用する
  - _Requirements: 9.1_

- [x] 24.2 診断情報ロギングの実装
  - Energy Drift、Koopman固有値、矛盾スコアをログに記録する
  - WandBでリアルタイム可視化する
  - _Requirements: 9.2_

- [x] 24.3 チェックポイント保存の実装
  - 各Stage終了時にモデルを保存する
  - ベストモデルを保存する
  - _Requirements: 9.3_

- [x] 25. Phase 3ベンチマークスクリプトの実装
  - scripts/benchmark_phase3.py を作成する
  - WikiText-2、PTB、C4でPerplexityを測定する
  - Phase 2との比較表を生成する
  - **完了条件**: すべてのデータセットで数値目標を達成し、総合レポートを生成する
  - _Requirements: 9.4, 9.5, 9.6_

- [x] 25.1 Perplexity測定の実装
  - 複数のデータセットでPPLを計算する
  - Phase 2と同じ条件で測定する
  - **目標（WikiText-2）**: Phase 2比 **+5%以内**（Phase 2が30.0なら31.5以下）
  - **目標（PTB）**: Phase 2比 **+5%以内**
  - **目標（C4）**: Phase 2比 **+5%以内**
  - **測定条件**: Batch=4, Seq=1024, fp16, 全機能ON
  - **記録項目**: 各データセットのPPL、Phase 2比、pass/fail
  - _Requirements: 9.4_

- [x] 25.2 VRAM/Throughput測定の実装
  - VRAM使用量を測定する
  - Throughputを測定する
  - **目標（VRAM）**: Batch=1, Seq=4096で **< 7.8GB**（8GBの97.5%）
  - **目標（Throughput）**: Phase 2比 **85%以上**（Phase 2が100 tokens/secなら85 tokens/sec以上）
  - **測定条件**: A100またはColab Pro相当環境、fp16、全機能ON
  - **記録項目**: vram_gb、throughput_tokens_per_sec、phase2_throughput、throughput_ratio、pass/fail
  - _Requirements: 9.5_

- [x] 25.3 比較表の生成
  - results/benchmarks/phase3_final_comparison.json を生成する
  - PPL、VRAM、Throughputを記録する
  - **目標**: すべての項目で数値目標を達成していることを明記
  - **フォーマット**: {"wikitext2_ppl": 31.2, "ptb_ppl": 28.5, "c4_ppl": 25.3, "phase2_wikitext2_ppl": 30.0, "vram_gb": 7.7, "throughput": 88.5, "phase2_throughput": 100.0, "all_pass": true}
  - **追加項目**: Koopman線形性誤差、MERA計算時間、Dialectic矛盾スコア減少率、Entropic Selectionフィルタリング率
  - _Requirements: 9.6_

- [x] 26. 可視化スクリプトの実装
  - scripts/visualize_phase3.py を作成する
  - Energy Drift、Koopman固有値、矛盾スコアを可視化する
  - _Requirements: 9.7_

- [x] 26.1 Energy Drift可視化の実装
  - 時間変化をプロットする
  - 閾値（1e-4）を表示する
  - _Requirements: 9.7_

- [x] 26.2 Koopman固有値可視化の実装
  - 複素平面上にプロットする
  - 単位円を表示する
  - _Requirements: 9.7_

- [x] 26.3 矛盾スコア可視化の実装
  - 学習中の変化をプロットする
  - _Requirements: 9.7_

## Priority 3: Documentation & Examples

### 目的
Phase 3の使用方法を文書化し、例を提供する。

- [x] 27. Phase 3実装ガイドの作成
  - docs/PHASE3_IMPLEMENTATION_GUIDE.md を作成する
  - 各モジュールの使用方法を説明する
  - 物理的直観と数式を記載する
  - _Requirements: 10.1_

- [x] 27.1 概要セクションの作成
  - Phase 3の目的と設計原理を説明する
  - アーキテクチャ図を含める
  - _Requirements: 10.1_

- [x] 27.2 各モジュールの詳細説明
  - Complex Dynamics、Hamiltonian ODE、Koopman、MERA、Dialecticの使用方法を説明する
  - コード例を含める
  - _Requirements: 10.1_

- [x] 27.3 トラブルシューティングセクションの作成
  - よくある問題と解決方法を記載する
  - デバッグ方法を説明する
  - _Requirements: 10.1_

- [ ] 28. 使用例の作成
  - examples/phase3_basic_usage.py を作成する
  - examples/phase3_training.py を作成する
  - examples/phase3_inference.py を作成する
  - examples/phase3_diagnostics.py を作成する
  - _Requirements: 10.2_

- [ ] 28.1 基本使用例の作成
  - Phase3IntegratedModelのインスタンス化例を作成する
  - 簡単なforward pass例を作成する
  - _Requirements: 10.2_

- [ ] 28.2 学習例の作成
  - 小規模データセットでの学習例を作成する
  - 診断情報の取得例を作成する
  - _Requirements: 10.2_

- [ ] 28.3 推論例の作成
  - テキスト生成例を作成する
  - Koopman多段階予測の例を作成する
  - _Requirements: 10.2_

- [ ] 28.4 診断例の作成
  - Energy Drift監視例を作成する
  - Koopman固有値の取得例を作成する
  - 矛盾スコアの可視化例を作成する
  - _Requirements: 10.2_

- [x] 29. Docstringの整備
  - すべてのモジュールにdocstringを追加する
  - 物理的直観と数式を記載する
  - Google StyleまたはNumPy Styleに従う
  - _Requirements: 10.3_

## Priority 4: Integration Testing & CI/CD

### 目的
Phase 3の品質を保証し、CI/CDに統合する。

- [x] 30. 統合テストの実装
  - tests/test_phase3_integration.py を作成する
  - Phase 3モデル全体の動作を検証する
  - 各コンポーネントの統合を検証する
  - _Requirements: 10.4_

- [x] 30.1 エンドツーエンドテストの実装
  - モデルのインスタンス化から推論までをテストする
  - 学習ループの動作をテストする
  - _Requirements: 10.4_

- [x] 30.2 コンポーネント統合テストの実装
  - Complex + Hamiltonianの統合をテストする
  - Hamiltonian + Koopmanの統合をテストする
  - MERA + Dialecticの統合をテストする
  - _Requirements: 10.4_

- [x] 30.3 数値安定性テストの実装
  - 長時間学習での安定性をテストする
  - 極端な入力での安定性をテストする
  - NaN/Inf発生率が0%であることを確認する
  - _Requirements: 10.4_

- [x] 31. ベンチマークテストスイートの実装
  - tests/test_phase3_benchmarks.py を作成する（scripts/で代用済み）
  - Complex演算のベンチマークテストを実装する
  - Hamiltonian ODEのベンチマークテストを実装する
  - Koopmanのベンチマークテストを実装する
  - _Requirements: 10.5_

- [x] 32. CI/CDパイプラインの更新
  - .github/workflows/phase3_tests.yml を作成する
  - Phase 3のテストをCI/CDに統合する
  - _Requirements: 10.6_

## Priority 5: Paper Update

### 目的
Phase 3の実験結果を論文に反映する。

- [ ] 33. 論文の更新
  - paper/main.tex を更新する
  - Phase 3の実験結果を追加する
  - 比較表を更新する
  - _Requirements: 10.7_

- [ ] 33.1 Phase 3セクションの追加
  - Complex Dynamics、Hamiltonian ODE、Koopmanの理論を説明する
  - 実験結果を記載する
  - _Requirements: 10.7_

- [ ] 33.2 比較表の更新
  - Phase 1、Phase 2、Phase 3の比較表を作成する
  - PPL、VRAM、Throughputを記載する
  - _Requirements: 10.7_

- [ ] 33.3 結論セクションの更新
  - Phase 3の成果をまとめる
  - 今後の展望を記載する
  - _Requirements: 10.7_
