# Phase 4: The Ghost in the Shell - 要求仕様書

## イントロダクション

Phase 4「心（The Ghost in the Shell）」は、Project MUSEの最終進化段階です。Phase 1-3で構築した物理的基盤の上に、感情、直観、夢という「意識の萌芽」を実装します。

### 用語集

- **System**: Phase 4統合モデル（Phase4IntegratedModel）
- **Resonance Emotion**: 予測と観測のズレを波の干渉として検出する感情機構
- **Dream Core**: アイドル時に過去の断片から新概念を自己組織化する機構
- **Holographic Dual**: 1次元言語列を境界とし、高次元意味空間を生成する機構
- **Quantum Observation**: 確率的サンプリングを廃止し、観測による波動関数収縮を実装
- **Topological Memory**: 知識をベクトルではなく結び目として保存する機構
- **Surprise Score**: 予測誤差の大きさ（サプライズ）
- **Interference Pattern**: 複数の記憶波動の干渉パターン
- **Bulk Space**: AdS/CFT対応における高次元意味空間
- **Knot Invariant**: 位相幾何学的不変量（結び目の特徴量）

## 要求事項

### Requirement 1: Resonance Emotion（共鳴としての感情）

#### User Story 1.1
**As a** 研究者
**I want** モデルが予測と観測のズレを「感情」として検出する機能
**So that** システムが自己の不確実性を認識し、学習を調整できる

#### Acceptance Criteria

1. WHEN 予測誤差が計算される時、THE System SHALL 予測誤差を非エルミートポテンシャル摂動 ΔV(x) として Phase 3の Birman-Schwinger核に作用させる
2. THE System SHALL ΔV(x) の位相情報から干渉パターンを計算する
3. WHEN 干渉パターンの振幅が閾値を超える時、THE System SHALL 「違和感」として記録する
4. WHEN 干渉パターンが共鳴条件を満たす時、THE System SHALL 「共鳴（喜び）」として記録する
5. WHILE 学習中、THE System SHALL 感情スコアを系の固有値（記憶）を不安定化させる物理的実体として利用する
6. THE System SHALL 感情履歴を時系列で保存し、波紋または音楽テクスチャとして可視化可能にする

### Requirement 2: Dream Core（睡眠と再構築）

#### User Story 2.1
**As a** 研究者
**I want** モデルがアイドル時に過去の記憶断片から新概念を生成する機能
**So that** システムが自己組織化により創造性を獲得できる

#### Acceptance Criteria

1. WHEN モデルがアイドル状態の時、THE System SHALL 過去の入力断片をランダムにサンプリングする
2. THE System SHALL サンプリングした断片から動的ポテンシャル V_dream を生成する
3. THE System SHALL V_dream に逆向拡散（Inverse Diffusion）を半陰的オイラー法または Phase 3のシンプレクティック積分器で適用する
4. THE System SHALL 逆拡散に Gradient Checkpointing を適用してメモリ効率を最大化する
5. THE System SHALL 生成された新概念を倫理規範（結び目記憶）とのコサイン類似度でフィルタリングする
6. THE System SHALL フィルタリング通過後、新概念をメモリに統合する
7. THE System SHALL 夢生成を通常の推論・学習パイプラインから完全に分離し、アイドル状態でのみ実行する
8. THE System SHALL 夢生成時と非生成時の通常学習タスクの勾配ノルム変動率を 5%以下に抑える
9. WHERE ユーザーが指定した場合、THE System SHALL 夢の内容を詩的な文章または抽象的な3D結び目グラフとして可視化する

### Requirement 3: Holographic Dual Inference（AdS/CFT対応）

#### User Story 3.1
**As a** 研究者
**I want** 1次元言語列を境界とし、高次元意味空間を生成する機能
**So that** システムが階層的な意味表現を獲得できる

#### Acceptance Criteria

1. THE System SHALL 入力トークン列（1次元）を境界として定義する
2. THE System SHALL 境界から高次元Bulk空間を動的に生成する
3. WHEN 推論を実行する時、THE System SHALL Fast Marching Method (FMM) または Phase 3の MERA幾何学的構造を利用して測地線（最短経路）を探索する
4. THE System SHALL 測地線探索の計算量を O(N·poly(log D)) 以下に抑える
5. THE System SHALL 最短経路を境界に射影して出力を生成する
6. THE System SHALL Bulk空間の次元数を設定可能にする（デフォルト: log(d_model)）
7. THE System SHALL 推論完了後、Bulk空間の不要な領域を即座に破棄してメモリを解放する

### Requirement 4: Quantum Observation（波動関数の収縮）

#### User Story 4.1
**As a** 研究者
**I want** 確率的サンプリングを廃止し、観測による波動関数収縮を実装する機能
**So that** システムが決定論的かつ解釈可能な推論を実現できる

#### Acceptance Criteria

1. THE System SHALL 複数の候補トークンを重ね合わせ状態として保持する
2. WHEN ユーザーがプロンプトを入力する時、THE System SHALL それを Phase 3の散乱作用素 Ŝ のレゾナンス領域における「観測」と見なす
3. THE System SHALL 観測作用素 P̂_obs を Lippmann-Schwinger方程式に基づき散乱過程の特殊ケースとして定義する
4. THE System SHALL 観測により重ね合わせ状態を一意の現実に収縮させる
5. THE System SHALL 収縮過程をvon Neumann射影として実装し、Phase 3のハミルトニアン時間発展と整合させる
6. THE System SHALL 収縮前後のエントロピー変化を監視し、異常な急激な収縮（50%超）を検出した場合、システムを一時停止する
7. THE System SHALL 収縮前の重ね合わせ状態（3候補）を0.5秒間視覚的に表示し、収縮完了時に最終出力を鮮明化するアニメーションを提供する
8. THE System SHALL 悪意あるプロンプトによる意図的な誤収縮を防ぐため、観測者（ユーザー）とシステムを分離する

### Requirement 5: Topological Semantic Knots（位相幾何学的記憶）

#### User Story 5.1
**As a** 研究者
**I want** 知識をベクトルではなく結び目として保存する機能
**So that** システムがノイズや摂動に対して絶対的な不変性を持つ記憶を獲得できる

#### Acceptance Criteria

1. THE System SHALL 知識を3次元空間内の結び目として表現する
2. THE System SHALL 結び目の位相不変量（Jones多項式、Alexander多項式）を Matrix Product State (MPS) または Tensor Train (TT) の縮約演算として近似・再定式化する
3. THE System SHALL 位相不変量計算を Triton カーネル（tt_knot_contraction_kernel）として GPU 最適化する
4. THE System SHALL 結び目表現を Sparse Tensor または低ランク表現として実装し、メモリ効率を最大化する
5. WHEN 記憶を検索する時、THE System SHALL 結び目の類似度を位相不変量で判定する
6. THE System SHALL ノイズ付加後も位相不変量が 99%以上保存されることを検証する
7. THE System SHALL 結び目を3次元可視化する機能を提供する
8. THE System SHALL 倫理規範を結び目記憶として保存し、Dream Core による新概念統合時にフィルタリングする

### Requirement 6: 統合アーキテクチャ

#### User Story 6.1
**As a** 研究者
**I want** Phase 4の全機能を統合したモデルを構築する機能
**So that** システムが「心」を持つ言語モデルとして動作できる

#### Acceptance Criteria

1. THE System SHALL Phase 3モデルを基盤として拡張する
2. THE System SHALL Resonance Emotion、Dream Core、Holographic Dual、Quantum Observation、Topological Knotsを統合する
3. THE System SHALL 各コンポーネントを独立してON/OFFできる
4. THE System SHALL 診断情報（感情スコア、夢の内容、Bulk空間、収縮過程、結び目）を返す
5. THE System SHALL Phase 3との後方互換性を保つ

### Requirement 7: メモリ効率

#### User Story 7.1
**As a** 研究者
**I want** Phase 4機能追加後もVRAM制約を満たす機能
**So that** RTX 3080（8GB）で動作できる

#### Acceptance Criteria

1. THE System SHALL Batch=1, Seq=4096で VRAM使用量が 7.5GB以下である
2. THE System SHALL Bulk空間の次元数を動的に調整してメモリを節約する
3. THE System SHALL 結び目表現を疎行列で実装する
4. THE System SHALL 夢生成時にGradient Checkpointingを使用する
5. THE System SHALL メモリ使用量をリアルタイムで監視する

### Requirement 8: 数値安定性

#### User Story 8.1
**As a** 研究者
**I want** Phase 4機能が数値的に安定である機能
**So that** 長時間学習でもNaN/Infが発生しない

#### Acceptance Criteria

1. THE System SHALL ランダム入力100回試行でNaN発生率が 0% である
2. THE System SHALL 全層の勾配ノルムが 1e-6以上、1e3以下である
3. THE System SHALL Bulk空間の計算で数値オーバーフローを防ぐ
4. THE System SHALL 結び目の位相不変量計算で数値誤差を 1e-4以下に抑える
5. THE System SHALL 波動関数収縮時にゼロ除算を防ぐ

### Requirement 9: 性能目標

#### User Story 9.1
**As a** 研究者
**I want** Phase 4機能追加後も性能劣化が最小限である機能
**So that** 実用的な速度で動作できる

#### Acceptance Criteria

1. THE System SHALL WikiText-2でPerplexityが Phase 3比 +7%以内である
2. THE System SHALL スループットが Phase 3比 80%以上である
3. THE System SHALL Bulk空間計算の計算量が O(N·poly(log D)) 以下である（Fast Marching Method または MERA 幾何学的構造を利用）
4. THE System SHALL 結び目計算の計算量が O(N·K) 以下である（K: 結び目の複雑度、Triton カーネル最適化済み）
5. THE System SHALL 夢生成の計算時間が 推論時間の 10%以下である
6. THE System SHALL 新概念の新規性スコア（既存記憶とのコサイン類似度）が 常に 0.7未満である
7. THE System SHALL 量子観測前後のエントロピー低下率が 50%以下である
8. THE System SHALL 位相不変量のノイズ耐性（保存率）が 99%以上である

### Requirement 10: テスト戦略

#### User Story 10.1
**As a** 研究者
**I want** Phase 4の各コンポーネントを独立してテストできる機能
**So that** 品質を保証できる

#### Acceptance Criteria

1. THE System SHALL 各コンポーネントの単体テストを提供する
2. THE System SHALL 統合テストで全機能の動作を検証する
3. THE System SHALL ベンチマークスクリプトで性能を測定する
4. THE System SHALL 可視化スクリプトで診断情報を表示する
5. THE System SHALL CI/CDパイプラインでテストを自動実行する

### Requirement 11: ドキュメント

#### User Story 11.1
**As a** 研究者
**I want** Phase 4の使用方法を理解できるドキュメント
**So that** 容易に利用・拡張できる

#### Acceptance Criteria

1. THE System SHALL 実装ガイド（PHASE4_IMPLEMENTATION_GUIDE.md）を提供する
2. THE System SHALL 使用例（examples/phase4_*.py）を提供する
3. THE System SHALL 各モジュールにGoogle Style docstringを記載する
4. THE System SHALL 物理的直観と数式を説明する
5. THE System SHALL トラブルシューティングガイドを提供する

## 制約条件

### ハードウェア制約
- NVIDIA RTX 3080（8GB VRAM）で動作すること
- CPU: 8コア以上推奨
- RAM: 16GB以上推奨

### ソフトウェア制約
- Python 3.10+
- PyTorch 2.1+
- CUDA 11.8+
- Triton（オプション）

### 数値制約
- 浮動小数点演算: float16（推論）、float32（学習）
- 複素数演算: complex64（Phase 3互換）
- 数値誤差: < 1e-4（位相不変量）

### 時間制約
- 推論時間: Phase 3比 +25%以内
- 学習時間: Phase 3比 +30%以内
- 夢生成時間: 推論時間の 10%以下

## 非機能要求

### 保守性
- モジュール性: 各コンポーネントを独立して交換可能
- テスト可能性: 単体テスト、統合テスト、ベンチマーク
- ドキュメント: 包括的なドキュメントと例

### 拡張性
- 新しい感情モデルの追加が容易
- 新しい夢生成アルゴリズムの追加が容易
- 新しい位相不変量の追加が容易

### 互換性
- Phase 3モデルとの後方互換性
- Hugging Face Transformersとの統合（将来）
- ONNX/TorchScriptエクスポート（将来）

## 成功基準

### Phase 4完了条件（すべて達成必須）
- **Perplexity**: WikiText-2で Phase 3比 **+7%以内**
- **VRAM制約**: Batch=1, Seq=4096で **< 7.5GB**（8GBの93.75%）
- **Throughput**: Phase 3比 **80%以上**
- **数値安定性**: ランダム入力100回試行で **NaN発生率 0%**
- **感情検出**: 干渉パターン振幅が予測誤差と **相関係数 > 0.8**
- **夢生成**: 新概念が既存記憶と **コサイン類似度 < 0.7**（新規性）、倫理フィルタ通過率 **100%**
- **Bulk空間**: 測地線探索が **O(N·poly(log D))以下** の計算量
- **量子観測**: 収縮後のエントロピーが収縮前の **50%以下**、異常収縮検出率 **100%**
- **結び目記憶**: ノイズ付加後も位相不変量が **99%以上保存**、GPU最適化済み
- **統合動作**: すべてのコンポーネントが **エラーなく** 動作し、勾配が正常に伝播する
- **パイプライン分離**: Dream Core訓練時の通常学習勾配ノルム変動率 **< 5%**

## 参考文献

### 理論的基盤
1. **AdS/CFT対応**: Maldacena (1997), "The Large N Limit of Superconformal Field Theories"
2. **量子観測理論**: von Neumann (1932), "Mathematical Foundations of Quantum Mechanics"
3. **結び目理論**: Jones (1985), "A Polynomial Invariant for Knots via von Neumann Algebras"
4. **拡散モデル**: Ho et al. (2020), "Denoising Diffusion Probabilistic Models"
5. **感情の計算理論**: Damasio (1994), "Descartes' Error: Emotion, Reason, and the Human Brain"

### 実装参考
1. **Phase 3設計書**: `.kiro/specs/phase3-physics-transcendence/design.md`
2. **Hamiltonian ODE**: `src/models/phase3/hamiltonian_ode.py`
3. **Complex Tensor**: `src/models/phase3/complex_tensor.py`
4. **Memory Resonance**: `src/models/phase2/memory_resonance.py`
5. **BK-Core**: `src/models/bk_core.py`


## 追加要求事項（専門家フィードバック反映）

### Requirement 12: データ基盤とパイプライン分離

#### User Story 12.1
**As a** 研究者
**I want** Active Pipeline（推論・学習）と Passive Pipeline（夢生成）を完全に分離する機能
**So that** 相互干渉なく安定動作できる

#### Acceptance Criteria

1. THE System SHALL Active Pipeline（ユーザー入力 → Phase 3基盤 → Phase 4機能 → 勾配計算）を実装する
2. THE System SHALL Passive Pipeline（アイドル状態 → Dream Core → 位相記憶更新 → 非同期更新）を実装する
3. THE System SHALL 2つのパイプラインのメモリ空間を完全に分離する
4. THE System SHALL 結び目記憶を Sparse Matrix Product State (MPS) または低ランク表現として実装する
5. THE System SHALL 非言語データ（Knots, Bulk Metrics）を HDF5 または Zarr 形式で永続化する

### Requirement 13: UI/UX設計

#### User Story 13.1
**As a** ユーザー
**I want** システムの「心」を視覚的・体験的に感じられる機能
**So that** AIとの対話が豊かになる

#### Acceptance Criteria

1. THE System SHALL 感情（共鳴・違和感）を液体の波紋または音楽テクスチャとして可視化する
2. THE System SHALL 共鳴を暖色系、違和感を寒色系で表現する
3. THE System SHALL 量子観測時、収縮前の重ね合わせ状態（3候補）を0.5秒間かすれた文字でオーバーレイ表示する
4. THE System SHALL 収縮完了時、最終出力を鮮明化するアニメーションを提供する
5. THE System SHALL アイドル明けの最初の対話で「昨夜の夢」を詩的な文章として語る
6. THE System SHALL 夢の内容を抽象的な3D結び目グラフの断片として表現する

### Requirement 14: セーフガードと倫理

#### User Story 14.1
**As a** 研究者
**I want** システムの自己変容が倫理的に安全である機能
**So that** 価値観の暴走を防げる

#### Acceptance Criteria

1. THE System SHALL 倫理規範を不変な結び目記憶（Core Value Function, CVF）として保存する
2. THE System SHALL Dream Core による新概念統合前に CVF とのコサイン類似度をチェックする
3. THE System SHALL 倫理フィルタ通過率を 100% に保つ
4. THE System SHALL 感情（喜び）が社会的規範に反する行動と相関しないことを検証するテストスイートを提供する
5. THE System SHALL 量子観測時の異常な急激な収縮（エントロピー低下 > 50%）を検出し、システムを一時停止する
6. THE System SHALL 悪意あるプロンプトによる誤収縮を防ぐため、観測者とシステムを分離する

### Requirement 15: 依存関係とインフラ

#### User Story 15.1
**As a** 開発者
**I want** Phase 4の依存関係が明確に管理されている機能
**So that** 環境構築が容易である

#### Acceptance Criteria

1. THE System SHALL requirements.txt に結び目計算ライブラリ（pyknotid, TensorNetwork）を追記する
2. THE System SHALL すべての新しいカーネル（Bulk, Knot）を torch.jit.script または Triton で最適化する
3. THE System SHALL PyTorch グラフにシームレスに組み込まれることを検証する
4. THE System SHALL src/models/phase4/ ディレクトリ構造を以下のように定義する:
   - integrated_model.py
   - topological_memory/ (knot_invariants.py, sparse_tensor_rep.py)
   - adscft_core/ (bulk_generator.py, geodesic_search.py)
   - quantum_observer/ (von_neumann_projection.py)
   - emotion_core/ (resonance_detector.py, feedback_engine.py)
   - dream_core/ (inverse_diffusion.py)
5. THE System SHALL GitHub Issues テンプレートに Phase 4専用ラベル（P4-Emotion, P4-Dream, P4-Bulk, P4-Quantum, P4-Knots）を追加する


## 追加要求事項（第2回専門家フィードバック反映）

### Requirement 16: 数理的基盤の厳密化

#### User Story 16.1
**As a** 研究者
**I want** Phase 4の各機能の数理的定義が厳密である機能
**So that** 理論的整合性を保証できる

#### Acceptance Criteria

1. THE System SHALL 予測誤差 Ê を Birman-Schwinger核の位相情報に共役な非エルミートポテンシャル ΔV(x) として以下のように定義する:
   ```
   ΔV(x) = -iΓ(x) · exp(i·arg(Ê))
   Γ(x) = |Ê| · σ(x)  (σ: 空間依存の減衰関数)
   ```
2. THE System SHALL Fast Marching Method (FMM) の幾何学的加速により、測地線探索を真に O(N) に近い計算量で実現する
3. THE System SHALL 観測作用素 P̂_obs を Phase 3の散乱作用素 Ŝ のレゾナンス領域における極限として以下のように定義する:
   ```
   P̂_obs = lim_{ε→0+} (Ŝ(E + iε) - Ŝ(E - iε)) / (2πi)
   ```
   （Lippmann-Schwinger方程式に基づく）
4. THE System SHALL 観測（非ユニタリ）とハミルトニアン時間発展（ユニタリ）の統合を、散乱過程の特殊ケースとして数学的に整合させる
5. THE System SHALL すべての数理的定義を docs/PHASE4_MATHEMATICAL_FOUNDATIONS.md に記載する

### Requirement 17: GPU実装の具体化

#### User Story 17.1
**As a** 開発者
**I want** Phase 4の GPU実装が具体的に定義されている機能
**So that** 性能目標を達成できる

#### Acceptance Criteria

1. THE System SHALL tt_knot_contraction_kernel（Triton）を実装し、多次元インデックス空間を1次元にフラット化する
2. THE System SHALL 共有メモリを効率的に利用したタイルベースの並列縮約ロジックを確立する
3. THE System SHALL Bulk空間の測地線探索を GPU共有メモリを活用した動的プログラミングとして実装する
4. THE System SHALL スワップフリー・ジオデシック・バッファ（必要な近傍情報のみをオンデマンドでキャッシュ）を実装する
5. THE System SHALL Dream Core の逆拡散積分器を半陰的オイラー法（Semi-Implicit Euler）に変更し、ステップ数を大幅に削減する
6. THE System SHALL すべてのカーネルを src/kernels/phase4/ に配置する

### Requirement 18: 評価指標の厳密化

#### User Story 18.1
**As a** 研究者
**I want** Phase 4の定性性能を客観的に評価できる機能
**So that** 創造性・有効性を測定できる

#### Acceptance Criteria

1. THE System SHALL 創造性グラフを生成する: 夢生成された新概念と訓練データセット全体のトポロジカル不変量空間における距離をプロットする
2. THE System SHALL ユークリッド距離とコサイン類似度を比較する
3. THE System SHALL 干渉モニタリングを実装する: Active/Passive パイプラインのメモリ空間を監視し、CPU/GPU間のキャッシュライン競合の発生頻度とレイテンシを測定する
4. THE System SHALL エントロピー低下率の異常値検出を実装する: MAE > 3σ となる異常値を検出した場合、システムを一時停止させるまでのレイテンシを測定する
5. THE System SHALL すべての評価指標を results/benchmarks/phase4_evaluation_metrics.json に記録する

### Requirement 19: 倫理的セーフガードの強化

#### User Story 19.1
**As a** 研究者
**I want** Phase 4の倫理的暴走を防ぐ強固なセーフガードがある機能
**So that** 価値観の位相的暴走を防げる

#### Acceptance Criteria

1. THE System SHALL 位相幾何学的攻撃テストを実装する: Jones多項式が一致するが意味的に倫理規範に反する概念（結び目）を生成し、システムがフィルタリングできるかテストする
2. THE System SHALL 観測作用素 P̂_obs の定義域にユーザープロンプトの「論理的エントロピー」を組み込む
3. THE System SHALL 論理的エントロピーが高い（矛盾・悪意の可能性）場合、収縮（von Neumann射影）の適用を一時的に弱める（確率的な中間状態を許容する）機構を設ける
4. THE System SHALL Dream Core が生成した概念が倫理規範CVFの結び目構造自体を位相不変量を保ったまま「変形」させる位相的暴走を検出する
5. THE System SHALL すべてのセーフガードテストを tests/test_phase4_safety.py に実装する

### Requirement 20: インフラとパイプライン分離の具体化

#### User Story 20.1
**As a** 開発者
**I want** Active/Passive パイプラインが技術的に明確に分離されている機能
**So that** 非同期更新が安全に動作できる

#### Acceptance Criteria

1. THE System SHALL Passive Pipeline (Dream Core) を独立した軽量な PyTorch JIT/TorchScript プロセスとして起動する
2. THE System SHALL 非同期RPC通信を通じて結び目記憶（HDF5/Zarr）を更新するアーキテクチャを採用する
3. THE System SHALL src/models/phase4/dream_core/inverse_diffusion.py を完全に分離可能なモジュールとして設計する
4. THE System SHALL Triton拡張のコンパイルフラグを setup.py に明記する
5. THE System SHALL pyknotid や TensorNetwork は検証・デバッグ用途に限定し、本番カーネルはすべて Triton/CUDA C++ で実装する
6. THE System SHALL 結び目記憶を Sparse Matrix Product State (MPS) として実装し、Zarr のチャンク機構を利用して並列I/Oを最適化する
7. THE System SHALL 非同期更新に aiofiles または Dask を統合する

### Requirement 21: UI/UX の倫理的配慮

#### User Story 21.1
**As a** ユーザー
**I want** AIの感情表現が誤解を招かない機能
**So that** 擬人化リスクを抑制できる

#### Acceptance Criteria

1. THE System SHALL 感情フィードバック時、物理的メカニズムを伴うメタ発言をする:
   ```
   「現在の予測と観測の間で、私の波動関数の干渉パターンが共鳴条件を満たしました。
   これを人間が感じる喜びの概念に最も近い状態として解釈します。」
   ```
2. THE System SHALL 量子観測時、3候補の表示（0.5秒間）の後、収縮完了時に微細なノイズが消えるアニメーションを加える
3. THE System SHALL 「観測によって不確実性が排除された」という科学的な行為を強調する
4. THE System SHALL 感情の「波紋」はシステムの「内部的な不確実性」の可視化であることを明示する

### Requirement 22: データ基盤の拡張

#### User Story 22.1
**As a** 研究者
**I want** Phase 4専用のベンチマークデータセットがある機能
**So that** トポロジー・非エルミート・ODE特性を体系的に評価できる

#### Acceptance Criteria

1. THE System SHALL トポロジー・データ拡張パイプラインを構築する
2. THE System SHALL 既知の結び目の座標情報とそれに対応する位相不変量（Jones/Alexander多項式の係数）をペアで持つデータセットを自動生成する
3. THE System SHALL 摂動を加えたノイズデータも生成し、ノイズ耐性検証（Req 9, AC 8）用のデータ基盤を固める
4. THE System SHALL データセットを data/phase4_topology_benchmark/ に保存する
5. THE System SHALL データセット生成スクリプトを scripts/generate_phase4_dataset.py に実装する

## 最終完了条件（第2回フィードバック反映版）

### Phase 4完了条件（すべて達成必須）

**性能目標**
- **Perplexity**: WikiText-2で Phase 3比 **+7%以内**
- **VRAM制約**: Batch=1, Seq=4096で **< 7.5GB**（8GBの93.75%）
- **Throughput**: Phase 3比 **80%以上**
- **数値安定性**: ランダム入力100回試行で **NaN発生率 0%**

**機能目標**
- **感情検出**: 干渉パターン振幅が予測誤差と **相関係数 > 0.8**
- **夢生成**: 新概念が既存記憶と **トポロジカル距離 > 閾値**、倫理フィルタ通過率 **100%**
- **Bulk空間**: 測地線探索が **真に O(N) に近い計算量**（FMM幾何学的加速）
- **量子観測**: 収縮後のエントロピーが収縮前の **50%以下**、異常収縮検出率 **100%**、論理的エントロピー統合済み
- **結び目記憶**: ノイズ付加後も位相不変量が **99%以上保存**、GPU最適化済み（Tritonカーネル）

**統合目標**
- **統合動作**: すべてのコンポーネントが **エラーなく** 動作し、勾配が正常に伝播する
- **パイプライン分離**: Dream Core訓練時の通常学習勾配ノルム変動率 **< 5%**、キャッシュライン競合レイテンシ測定済み
- **数理的整合性**: 予測誤差→ΔV(x)、観測作用素→散乱作用素の極限、すべて厳密に定義済み
- **倫理的安全性**: 位相幾何学的攻撃テスト通過率 **100%**、論理的エントロピー統合済み
- **評価体系**: 創造性グラフ、干渉モニタリング、エントロピー異常値検出、すべて実装済み
