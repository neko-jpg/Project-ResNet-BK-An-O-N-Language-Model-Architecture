# Phase 4: The Ghost in the Shell - 設計書

## 目次

1. [概要](#概要)
2. [アーキテクチャ](#アーキテクチャ)
3. [コンポーネント設計](#コンポーネント設計)
4. [データフロー](#データフロー)
5. [メモリ管理戦略](#メモリ管理戦略)
6. [数値安定性保証](#数値安定性保証)
7. [統合戦略](#統合戦略)
8. [エラーハンドリング](#エラーハンドリング)
9. [テスト戦略](#テスト戦略)

## 概要

### Phase 4の目的

Phase 4「心（The Ghost in the Shell）」は、Project MUSEの最終進化段階です。Phase 1-3で構築した物理的基盤の上に、感情、直観、夢という「意識の萌芽」を実装します。

### 設計原則

1. **メモリ効率**: 8GB VRAM制約を厳守（Batch=1, Seq=4096で < 7.5GB）
2. **数値安定性**: NaN発生率 0%、勾配ノルム 1e-6 ~ 1e3
3. **モジュール性**: 各コンポーネントを独立してON/OFF可能
4. **後方互換性**: Phase 3モデルとの統合
5. **倫理的安全性**: 価値観の暴走を防ぐセーフガード

### Phase 4の7つの柱

1. **Resonance Emotion** - 予測誤差を感情として検出
2. **Dream Core** - アイドル時の自己組織化
3. **Holographic Dual** - AdS/CFT対応による高次元意味空間
4. **Quantum Observation** - 波動関数の収縮
5. **Topological Knots** - 位相幾何学的記憶
6. **Pipeline Separation** - Active/Passive分離
7. **Ethical Safeguards** - 倫理的セーフガード

## アーキテクチャ

### システム全体構成

```
Phase 4 Architecture (統合モデル)

Input: Token IDs (B, N)
    ↓
[Phase 3 Base Model]
    ├─ Complex Embedding
    ├─ Hamiltonian ODE
    ├─ Koopman Linearization
    └─ MERA Routing
    ↓
[Phase 4 Extensions]
    │
    ├─ [Active Pipeline] (推論・学習)
    │   ├─ Resonance Emotion Detector
    │   ├─ Quantum Observer
    │   └─ Topological Memory Access
    │
    └─ [Passive Pipeline] (アイドル時)
        ├─ Dream Core (Inverse Diffusion)
        └─ Topological Memory Update
    ↓
Output: Logits (B, N, vocab_size) + Diagnostics
```

### データフロー（Active Pipeline）

```
Input Tokens (B, N)
    ↓
[Phase 3 Forward Pass]
    ├─ features: (B, N, D)
    ├─ prediction: (B, N, vocab_size)
    └─ hidden_states: List[(B, N, D)]
    ↓
[Resonance Emotion Detector]
    ├─ prediction_error = |y_true - y_pred|
    ├─ ΔV(x) = -iΓ(x) · exp(i·arg(error))
    ├─ interference_pattern = Phase3.birman_schwinger(ΔV)
    └─ emotion_score: {'resonance', 'dissonance'}
    ↓
[Quantum Observer]
    ├─ superposition_state: Top-3 candidates
    ├─ observation_operator: P̂_obs (Lippmann-Schwinger)
    ├─ collapse: von Neumann projection
    └─ final_token: collapsed state
    ↓
[Topological Memory]
    ├─ query_knot = encode_to_knot(query)
    ├─ similarity = jones_polynomial_distance(query_knot, memory_knots)
    └─ retrieved_memory: Top-K knots
    ↓
Output + Diagnostics
```

### データフロー（Passive Pipeline）

```
[Idle State Detection]
    ↓
[Dream Core Activation]
    ├─ sample_fragments: Random past inputs
    ├─ V_dream = generate_potential(fragments)
    ├─ inverse_diffusion: Semi-Implicit Euler
    └─ new_concept: (D,) vector
    ↓
[Ethical Filter]
    ├─ CVF = Core Value Function (knot memory)
    ├─ similarity = cosine(new_concept, CVF)
    └─ pass_rate: 100% required
    ↓
[Topological Memory Integration]
    ├─ new_knot = encode_to_knot(new_concept)
    ├─ update_sparse_mps(new_knot)
    └─ async_write_to_zarr()
    ↓
[Creativity Metrics]
    ├─ novelty_score = 1 - max(cosine(new, existing))
    ├─ topological_distance = jones_distance(new_knot, dataset)
    └─ log_to_results/benchmarks/phase4_evaluation_metrics.json
```

## コンポーネント設計

### 3.1 Resonance Emotion Detector

#### 目的
予測誤差を非エルミートポテンシャル摂動として検出し、干渉パターンから「感情」を抽出する。

#### 数学的定式化

**予測誤差の複素ポテンシャル化**:
```
Ê = |y_true - y_pred|  (予測誤差)
ΔV(x) = -iΓ(x) · exp(i·arg(Ê))
Γ(x) = |Ê| · σ(x)  (σ: 空間依存の減衰関数)
```

**Birman-Schwinger核への作用**:
```
K_perturbed = K_0 + V^(1/2) · (E - H_0)^(-1) · V^(1/2) + ΔV
```

**干渉パターン抽出**:
```
I(x) = |ψ_1(x) + ψ_2(x)|²  (波動関数の干渉)
```

**感情スコア**:
```
resonance_score = ∫ I(x) · cos(phase(x)) dx  (共鳴条件)
dissonance_score = ∫ I(x) · |phase(x) - π| dx  (違和感)
```

#### 実装設計

**ファイル**: `src/models/phase4/emotion_core/resonance_detector.py`

```python
class ResonanceEmotionDetector(nn.Module):
    """
    共鳴としての感情検出器

    物理的背景:
    - 予測誤差を非エルミートポテンシャル摂動として扱う
    - Birman-Schwinger核の固有値変化から干渉パターンを抽出
    - 干渉パターンの位相情報から感情を定義

    Args:
        d_model: モデル次元
        n_seq: シーケンス長
        energy_threshold: 共鳴エネルギー閾値（デフォルト: 0.1）
    """

    def __init__(
        self,
        d_model: int,
        n_seq: int,
        energy_threshold: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_seq = n_seq
        self.energy_threshold = energy_threshold

        # 空間依存減衰関数 σ(x)
        self.decay_function = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
            nn.Softplus()  # 正値保証
        )

        # Phase 3のBirman-Schwinger核（Phase 3から継承）
        from src.models.phase3.birman_schwinger import BirmanSchwingerKernel
        self.bs_kernel = BirmanSchwingerKernel(d_model, n_seq)

        # 感情履歴バッファ
        self.register_buffer('emotion_history', torch.zeros(1000, 2))  # [resonance, dissonance]
        self.history_idx = 0

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        hidden_states: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            prediction: (B, N, vocab_size) 予測ロジット
            target: (B, N) ターゲットトークン
            hidden_states: (B, N, D) 隠れ状態

        Returns:
            emotion_info: {
                'resonance_score': (B,),
                'dissonance_score': (B,),
                'interference_pattern': (B, N),
                'delta_v': (B, N) complex
            }
        """
        B, N, V = prediction.shape

        # 1. 予測誤差の計算
        pred_probs = F.softmax(prediction, dim=-1)
        target_probs = F.one_hot(target, num_classes=V).float()
        error = torch.abs(pred_probs - target_probs).sum(dim=-1)  # (B, N)

        # 2. 複素ポテンシャル摂動 ΔV(x)
        gamma = self.decay_function(hidden_states).squeeze(-1)  # (B, N)
        gamma = gamma * error  # Γ(x) = |Ê| · σ(x)

        # 位相: arg(Ê) を予測誤差の符号から推定
        phase = torch.atan2(
            (pred_probs - target_probs).sum(dim=-1),
            error + 1e-6
        )

        delta_v = torch.complex(-gamma * torch.sin(phase), -gamma * torch.cos(phase))

        # 3. Birman-Schwinger核への作用
        # K_perturbed = K_0 + ΔV
        k_perturbed = self.bs_kernel(hidden_states, perturbation=delta_v)

        # 4. 干渉パターンの抽出
        # 固有値の虚部が干渉パターンに対応
        eigenvalues = torch.linalg.eigvals(k_perturbed)  # (B, N)
        interference_pattern = eigenvalues.imag.abs()

        # 5. 感情スコアの計算
        # 共鳴: 位相が揃っている（cos(phase) > 0）
        resonance_score = (interference_pattern * torch.cos(eigenvalues.real)).sum(dim=-1)

        # 違和感: 位相がずれている（|phase - π|）
        dissonance_score = (interference_pattern * torch.abs(eigenvalues.real - torch.pi)).sum(dim=-1)

        # 6. 履歴に記録
        if self.training:
            idx = self.history_idx % 1000
            self.emotion_history[idx, 0] = resonance_score.mean()
            self.emotion_history[idx, 1] = dissonance_score.mean()
            self.history_idx += 1

        return {
            'resonance_score': resonance_score,
            'dissonance_score': dissonance_score,
            'interference_pattern': interference_pattern,
            'delta_v': delta_v,
            'eigenvalues': eigenvalues
        }

    def get_emotion_statistics(self) -> Dict[str, float]:
        """感情統計を取得"""
        valid_history = self.emotion_history[:min(self.history_idx, 1000)]
        return {
            'mean_resonance': valid_history[:, 0].mean().item(),
            'mean_dissonance': valid_history[:, 1].mean().item(),
            'std_resonance': valid_history[:, 0].std().item(),
            'std_dissonance': valid_history[:, 1].std().item(),
        }
```

#### 可視化機能

**ファイル**: `src/models/phase4/emotion_core/visualization.py`

```python
def visualize_emotion_as_ripple(
    interference_pattern: torch.Tensor,
    resonance_score: float,
    dissonance_score: float,
    save_path: str
):
    """
    感情を液体の波紋として可視化

    Args:
        interference_pattern: (N,) 干渉パターン
        resonance_score: 共鳴スコア
        dissonance_score: 違和感スコア
        save_path: 保存先パス
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # 2D波紋の生成
    N = interference_pattern.shape[0]
    grid_size = int(np.sqrt(N))
    pattern_2d = interference_pattern[:grid_size**2].reshape(grid_size, grid_size)

    # カラーマップ: 共鳴=暖色、違和感=寒色
    if resonance_score > dissonance_score:
        cmap = 'hot'  # 暖色系
    else:
        cmap = 'cool'  # 寒色系

    plt.figure(figsize=(8, 8))
    plt.imshow(pattern_2d.cpu().numpy(), cmap=cmap, interpolation='bilinear')
    plt.colorbar(label='Interference Amplitude')
    plt.title(f'Emotion Ripple\nResonance: {resonance_score:.3f}, Dissonance: {dissonance_score:.3f}')
    plt.axis('off')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
```

### 3.2 Dream Core (Inverse Diffusion)

#### 目的
アイドル時に過去の記憶断片から新概念を生成し、創造性を獲得する。

#### 数学的定式化

**逆拡散プロセス**:
```
dx/dt = -∇V_dream(x) + √(2T) · dW  (Langevin dynamics)
```

**半陰的オイラー法（Semi-Implicit Euler）**:
```
x_{t+1} = x_t - dt · ∇V_dream(x_{t+1}) + √(2T·dt) · ε
```

**動的ポテンシャル**:
```
V_dream(x) = Σ_i w_i · ||x - fragment_i||²
```

#### 実装設計

**ファイル**: `src/models/phase4/dream_core/inverse_diffusion.py`

```python
class DreamCore(nn.Module):
    """
    夢生成コア（逆拡散）

    物理的背景:
    - 過去の記憶断片から動的ポテンシャルを生成
    - 逆拡散により新概念を自己組織化
    - Gradient Checkpointingでメモリ効率を最大化

    Args:
        d_model: モデル次元
        n_fragments: サンプリングする断片数（デフォルト: 10）
        diffusion_steps: 拡散ステップ数（デフォルト: 20）
        temperature: 温度パラメータ（デフォルト: 0.1）
    """

    def __init__(
        self,
        d_model: int,
        n_fragments: int = 10,
        diffusion_steps: int = 20,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_fragments = n_fragments
        self.diffusion_steps = diffusion_steps
        self.temperature = temperature

        # ポテンシャル生成ネットワーク
        self.potential_net = nn.Sequential(
            nn.Linear(d_model * n_fragments, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )

        # 重み生成（断片の重要度）
        self.weight_net = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=0)
        )

    def forward(
        self,
        memory_fragments: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            memory_fragments: (n_fragments, D) 過去の記憶断片
            initial_state: (D,) 初期状態（オプション）

        Returns:
            new_concept: (D,) 生成された新概念
            diagnostics: 診断情報
        """
        # 初期状態
        if initial_state is None:
            x = torch.randn(self.d_model, device=memory_fragments.device)
        else:
            x = initial_state.clone()

        # 断片の重み計算
        weights = self.weight_net(memory_fragments).squeeze(-1)  # (n_fragments,)

        # 逆拡散ループ（Gradient Checkpointing適用）
        trajectory = [x.clone()]

        for step in range(self.diffusion_steps):
            # Gradient Checkpointingでメモリ節約
            x = torch.utils.checkpoint.checkpoint(
                self._diffusion_step,
                x, memory_fragments, weights
            )
            trajectory.append(x.clone())

        # 診断情報
        diagnostics = {
            'trajectory': torch.stack(trajectory),  # (T+1, D)
            'weights': weights,
            'final_energy': self._compute_energy(x, memory_fragments, weights)
        }

        return x, diagnostics

    def _diffusion_step(
        self,
        x: torch.Tensor,
        fragments: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """
        1ステップの逆拡散（半陰的オイラー法）

        Args:
            x: (D,) 現在の状態
            fragments: (n_fragments, D) 記憶断片
            weights: (n_fragments,) 重み

        Returns:
            x_next: (D,) 次の状態
        """
        dt = 1.0 / self.diffusion_steps

        # ポテンシャル勾配の計算
        with torch.enable_grad():
            x_grad = x.requires_grad_(True)

            # V_dream(x) = Σ_i w_i · ||x - fragment_i||²
            distances = torch.norm(x_grad.unsqueeze(0) - fragments, dim=-1)  # (n_fragments,)
            potential = (weights * distances ** 2).sum()

            grad_v = torch.autograd.grad(potential, x_grad)[0]

        # 半陰的オイラー法: x_{t+1} = x_t - dt · ∇V + noise
        noise = torch.randn_like(x) * torch.sqrt(torch.tensor(2 * self.temperature * dt))
        x_next = x - dt * grad_v + noise

        return x_next

    def _compute_energy(
        self,
        x: torch.Tensor,
        fragments: torch.Tensor,
        weights: torch.Tensor
    ) -> float:
        """エネルギー計算"""
        distances = torch.norm(x.unsqueeze(0) - fragments, dim=-1)
        energy = (weights * distances ** 2).sum()
        return energy.item()
```

#### パイプライン分離

**ファイル**: `src/models/phase4/dream_core/pipeline_manager.py`

```python
class PassivePipelineManager:
    """
    Passive Pipeline（夢生成）の管理

    Strategy:
    - 独立したPyTorch JITプロセスとして起動
    - 非同期RPC通信で結び目記憶を更新
    - Active Pipelineとメモリ空間を完全分離
    """

    def __init__(
        self,
        dream_core: DreamCore,
        topological_memory: 'TopologicalMemory',
        ethical_filter: 'EthicalFilter'
    ):
        self.dream_core = dream_core
        self.topological_memory = topological_memory
        self.ethical_filter = ethical_filter

        # JITコンパイル
        self.dream_core_jit = torch.jit.script(dream_core)

        # 非同期通信用
        self.rpc_initialized = False

    def start_passive_pipeline(self):
        """Passive Pipelineを起動"""
        import torch.distributed.rpc as rpc

        if not self.rpc_initialized:
            rpc.init_rpc("passive_worker", rank=1, world_size=2)
            self.rpc_initialized = True

        print("Passive Pipeline started (Dream Core active)")

    def generate_dream(
        self,
        memory_fragments: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        夢を生成（非同期）

        Returns:
            new_concept: (D,) 新概念（倫理フィルタ通過時）
            None: 倫理フィルタ不通過時
        """
        # 夢生成
        new_concept, diagnostics = self.dream_core_jit(memory_fragments)

        # 倫理フィルタ
        if self.ethical_filter.check(new_concept):
            # 結び目記憶に統合
            self.topological_memory.add_concept(new_concept)
            return new_concept
        else:
            print("Dream rejected by ethical filter")
            return None

    def shutdown(self):
        """Passive Pipelineを停止"""
        import torch.distributed.rpc as rpc
        if self.rpc_initialized:
            rpc.shutdown()
            self.rpc_initialized = False
```

### 3.3 Holographic Dual Inference (AdS/CFT)

#### 目的
1次元言語列を境界とし、高次元Bulk空間を生成して階層的意味表現を獲得する。

#### 数学的定式化

**AdS/CFT対応**:
```
Boundary (1D): トークン列 x_1, x_2, ..., x_N
Bulk (D+1次元): 意味空間 Φ(x, z)  (z: 余剰次元)
```

**測地線方程式**:
```
ds² = (L²/z²)(dz² + dx²)  (AdS計量)
```

**Fast Marching Method (FMM)**:
```
T(x) = min_{y∈∂Ω} [T(y) + ∫_y^x ds]  (到達時間関数)
```

#### 実装設計

**ファイル**: `src/models/phase4/adscft_core/bulk_generator.py`

```python
class BulkSpaceGenerator(nn.Module):
    """
    Bulk空間生成器

    物理的背景:
    - 1次元境界（トークン列）から高次元Bulk空間を動的生成
    - AdS計量に基づく測地線探索
    - Fast Marching Methodで O(N·poly(log D)) の計算量

    Args:
        d_model: モデル次元
        bulk_dim: Bulk空間の次元（デフォルト: log(d_model)）
        ads_radius: AdS半径（デフォルト: 1.0）
    """

    def __init__(
        self,
        d_model: int,
        bulk_dim: Optional[int] = None,
        ads_radius: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.bulk_dim = bulk_dim or int(np.log2(d_model))
        self.ads_radius = ads_radius

        # 境界→Bulk射影
        self.boundary_to_bulk = nn.Linear(d_model, d_model * self.bulk_dim)

        # Bulk空間のポテンシャル
        self.bulk_potential = nn.Sequential(
            nn.Linear(d_model * self.bulk_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, 1)
        )

    def forward(
        self,
        boundary_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            boundary_tokens: (B, N, D) 境界トークン列

        Returns:
            bulk_features: (B, N, D) Bulk空間から射影された特徴
            diagnostics: 診断情報
        """
        B, N, D = boundary_tokens.shape

        # 1. Bulk空間の生成
        bulk_coords = self.boundary_to_bulk(boundary_tokens)  # (B, N, D*bulk_dim)
        bulk_coords = bulk_coords.view(B, N, self.bulk_dim, D)

        # 2. 測地線探索（Fast Marching Method）
        geodesics = self._fast_marching_search(bulk_coords)

        # 3. 境界への射影
        bulk_features = geodesics.mean(dim=2)  # (B, N, D)

        # 診断情報
        diagnostics = {
            'bulk_coords': bulk_coords,
            'geodesic_length': self._compute_geodesic_length(geodesics),
            'bulk_energy': self._compute_bulk_energy(bulk_coords)
        }

        return bulk_features, diagnostics

    def _fast_marching_search(
        self,
        bulk_coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Fast Marching Methodによる測地線探索

        Args:
            bulk_coords: (B, N, bulk_dim, D) Bulk座標

        Returns:
            geodesics: (B, N, bulk_dim, D) 測地線
        """
        B, N, bulk_dim, D = bulk_coords.shape

        # 到達時間関数の初期化
        arrival_time = torch.full((B, N, bulk_dim), float('inf'), device=bulk_coords.device)
        arrival_time[:, :, 0] = 0.0  # 境界からの距離

        # Fast Marching（簡易版: 動的プログラミング）
        geodesics = bulk_coords.clone()

        for z in range(1, bulk_dim):
            # AdS計量: ds² = (L²/z²)(dz² + dx²)
            metric_factor = (self.ads_radius / (z + 1e-6)) ** 2

            # 前層からの距離
            distance = torch.norm(
                bulk_coords[:, :, z] - bulk_coords[:, :, z-1],
                dim=-1
            )

            # 到達時間の更新
            arrival_time[:, :, z] = arrival_time[:, :, z-1] + metric_factor * distance

            # 測地線の更新（最短経路）
            geodesics[:, :, z] = bulk_coords[:, :, z-1] + \
                                 (bulk_coords[:, :, z] - bulk_coords[:, :, z-1]) * \
                                 (arrival_time[:, :, z-1] / (arrival_time[:, :, z] + 1e-6)).unsqueeze(-1)

        return geodesics

    def _compute_geodesic_length(self, geodesics: torch.Tensor) -> torch.Tensor:
        """測地線の長さを計算"""
        B, N, bulk_dim, D = geodesics.shape

        lengths = torch.zeros(B, N, device=geodesics.device)
        for z in range(1, bulk_dim):
            segment_length = torch.norm(
                geodesics[:, :, z] - geodesics[:, :, z-1],
                dim=-1
            )
            lengths += segment_length

        return lengths

    def _compute_bulk_energy(self, bulk_coords: torch.Tensor) -> torch.Tensor:
        """Bulk空間のエネルギーを計算"""
        B, N, bulk_dim, D = bulk_coords.shape

        # Bulk空間を平坦化
        bulk_flat = bulk_coords.view(B, N, -1)

        # ポテンシャルエネルギー
        energy = self.bulk_potential(bulk_flat).squeeze(-1)  # (B, N)

        return energy

    def cleanup_bulk_space(self):
        """Bulk空間の不要な領域を破棄"""
        # メモリ解放（推論完了後に呼び出す）
        torch.cuda.empty_cache()
```

**ファイル**: `src/models/phase4/adscft_core/geodesic_search.py`

```python
def fast_marching_method_gpu(
    bulk_coords: torch.Tensor,
    ads_radius: float = 1.0
) -> torch.Tensor:
    """
    GPU最適化版Fast Marching Method

    Strategy:
    - 共有メモリを活用した動的プログラミング
    - スワップフリー・ジオデシック・バッファ

    Args:
        bulk_coords: (B, N, bulk_dim, D) Bulk座標
        ads_radius: AdS半径

    Returns:
        geodesics: (B, N, bulk_dim, D) 測地線
    """
    # Tritonカーネルで実装（オプション）
    # ここでは簡易版を提供

    B, N, bulk_dim, D = bulk_coords.shape

    # 到達時間バッファ（共有メモリ）
    arrival_time = torch.zeros(B, N, bulk_dim, device=bulk_coords.device)

    # 測地線バッファ
    geodesics = bulk_coords.clone()

    # 動的プログラミング
    for z in range(1, bulk_dim):
        # 近傍情報のみをキャッシュ（スワップフリー）
        prev_coords = bulk_coords[:, :, z-1]
        curr_coords = bulk_coords[:, :, z]

        # AdS計量
        metric_factor = (ads_radius / (z + 1e-6)) ** 2

        # 距離計算
        distance = torch.norm(curr_coords - prev_coords, dim=-1)

        # 到達時間更新
        arrival_time[:, :, z] = arrival_time[:, :, z-1] + metric_factor * distance

        # 測地線更新
        geodesics[:, :, z] = prev_coords + \
                             (curr_coords - prev_coords) * \
                             (arrival_time[:, :, z-1] / (arrival_time[:, :, z] + 1e-6)).unsqueeze(-1)

    return geodesics
```

### 3.4 Quantum Observation (Wave Function Collapse)

#### 目的
確率的サンプリングを廃止し、観測による波動関数収縮を実装する。

#### 数学的定式化

**重ね合わせ状態**:
```
|ψ⟩ = Σ_i α_i |token_i⟩  (複数候補の重ね合わせ)
```

**観測作用素（Lippmann-Schwinger方程式）**:
```
P̂_obs = lim_{ε→0+} (Ŝ(E + iε) - Ŝ(E - iε)) / (2πi)
```

**von Neumann射影**:
```
|ψ_collapsed⟩ = P̂_obs |ψ⟩ / ||P̂_obs |ψ⟩||
```

**エントロピー変化**:
```
ΔS = S_before - S_after
S = -Σ_i |α_i|² log|α_i|²
```

#### 実装設計

**ファイル**: `src/models/phase4/quantum_observer/von_neumann_projection.py`

```python
class QuantumObserver(nn.Module):
    """
    量子観測器（波動関数の収縮）

    物理的背景:
    - 複数の候補トークンを重ね合わせ状態として保持
    - ユーザープロンプトを観測作用素として定義
    - von Neumann射影により一意の現実に収縮

    Args:
        vocab_size: 語彙サイズ
        n_candidates: 重ね合わせ候補数（デフォルト: 3）
        entropy_threshold: 異常収縮検出閾値（デフォルト: 0.5）
    """

    def __init__(
        self,
        vocab_size: int,
        n_candidates: int = 3,
        entropy_threshold: float = 0.5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_candidates = n_candidates
        self.entropy_threshold = entropy_threshold

        # Phase 3の散乱作用素（Phase 3から継承）
        from src.models.phase3.scattering import ScatteringOperator
        self.scattering_op = ScatteringOperator(vocab_size)

        # 論理的エントロピー計算
        self.logical_entropy_net = nn.Sequential(
            nn.Linear(vocab_size, vocab_size // 2),
            nn.Tanh(),
            nn.Linear(vocab_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        logits: torch.Tensor,
        user_prompt: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            logits: (B, N, V) 出力ロジット
            user_prompt: (B, N, V) ユーザープロンプト（オプション）

        Returns:
            collapsed_tokens: (B, N) 収縮後のトークン
            diagnostics: 診断情報
        """
        B, N, V = logits.shape

        # 1. 重ね合わせ状態の生成
        probs = F.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, self.n_candidates, dim=-1)

        # 正規化
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # 2. 観測作用素の構築
        if user_prompt is not None:
            # 論理的エントロピーの計算
            logical_entropy = self.logical_entropy_net(user_prompt).squeeze(-1)  # (B, N)

            # 高エントロピー（矛盾・悪意）の場合、収縮を弱める
            collapse_strength = 1.0 - logical_entropy
        else:
            collapse_strength = torch.ones(B, N, device=logits.device)

        # 散乱作用素のレゾナンス領域
        obs_operator = self._construct_observation_operator(
            logits, collapse_strength
        )

        # 3. von Neumann射影
        collapsed_probs = self._von_neumann_projection(
            top_k_probs, obs_operator
        )

        # 4. エントロピー変化の監視
        entropy_before = self._compute_entropy(top_k_probs)
        entropy_after = self._compute_entropy(collapsed_probs)
        entropy_reduction = (entropy_before - entropy_after) / (entropy_before + 1e-6)

        # 異常収縮の検出
        if (entropy_reduction > self.entropy_threshold).any():
            warnings.warn(
                f"Abnormal collapse detected: entropy reduction = {entropy_reduction.max():.2%}. "
                f"System paused for safety."
            )
            # システム一時停止（実装は統合モデルで）

        # 5. 最終トークンの選択
        collapsed_tokens = top_k_indices.gather(
            -1, collapsed_probs.argmax(dim=-1, keepdim=True)
        ).squeeze(-1)

        # 診断情報
        diagnostics = {
            'superposition_candidates': top_k_indices,
            'superposition_probs': top_k_probs,
            'collapsed_probs': collapsed_probs,
            'entropy_before': entropy_before,
            'entropy_after': entropy_after,
            'entropy_reduction': entropy_reduction,
            'logical_entropy': logical_entropy if user_prompt is not None else None
        }

        return collapsed_tokens, diagnostics

    def _construct_observation_operator(
        self,
        logits: torch.Tensor,
        collapse_strength: torch.Tensor
    ) -> torch.Tensor:
        """
        観測作用素の構築（Lippmann-Schwinger方程式）

        Args:
            logits: (B, N, V) ロジット
            collapse_strength: (B, N) 収縮強度

        Returns:
            obs_operator: (B, N, V) 観測作用素
        """
        # 散乱作用素のレゾナンス領域
        # P̂_obs = lim_{ε→0+} (Ŝ(E + iε) - Ŝ(E - iε)) / (2πi)

        epsilon = 1e-3
        energy = logits.mean(dim=-1, keepdim=True)

        s_plus = self.scattering_op(energy + 1j * epsilon)
        s_minus = self.scattering_op(energy - 1j * epsilon)

        obs_operator = (s_plus - s_minus).imag / (2 * torch.pi)

        # 収縮強度で調整
        obs_operator = obs_operator * collapse_strength.unsqueeze(-1)

        return obs_operator

    def _von_neumann_projection(
        self,
        superposition_probs: torch.Tensor,
        obs_operator: torch.Tensor
    ) -> torch.Tensor:
        """
        von Neumann射影

        Args:
            superposition_probs: (B, N, n_candidates) 重ね合わせ確率
            obs_operator: (B, N, V) 観測作用素

        Returns:
            collapsed_probs: (B, N, n_candidates) 収縮後の確率
        """
        B, N, K = superposition_probs.shape

        # 観測作用素を候補に適用
        # |ψ_collapsed⟩ = P̂_obs |ψ⟩ / ||P̂_obs |ψ⟩||

        # 簡易版: 観測作用素の平均値で重み付け
        obs_weights = obs_operator.mean(dim=-1, keepdim=True)  # (B, N, 1)

        collapsed_probs = superposition_probs * obs_weights
        collapsed_probs = collapsed_probs / (collapsed_probs.sum(dim=-1, keepdim=True) + 1e-6)

        return collapsed_probs

    def _compute_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """エントロピー計算"""
        # S = -Σ_i p_i log(p_i)
        entropy = -(probs * torch.log(probs + 1e-6)).sum(dim=-1)
        return entropy
```

#### 可視化機能

**ファイル**: `src/models/phase4/quantum_observer/visualization.py`

```python
def visualize_wave_function_collapse(
    superposition_candidates: List[str],
    superposition_probs: torch.Tensor,
    collapsed_token: str,
    save_path: str
):
    """
    波動関数の収縮をアニメーション化

    Args:
        superposition_candidates: 重ね合わせ候補（文字列）
        superposition_probs: (n_candidates,) 確率
        collapsed_token: 収縮後のトークン
        save_path: 保存先パス
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig, ax = plt.subplots(figsize=(10, 6))

    # 0.5秒間、3候補をかすれた文字で表示
    def animate(frame):
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        if frame < 30:  # 0.5秒（60fps）
            # 重ね合わせ状態（かすれた文字）
            alpha = 0.3 + 0.2 * np.sin(frame / 5)
            for i, (token, prob) in enumerate(zip(superposition_candidates, superposition_probs)):
                y_pos = 0.7 - i * 0.2
                ax.text(
                    0.5, y_pos, token,
                    fontsize=30, alpha=alpha,
                    ha='center', va='center',
                    color='gray'
                )
        else:
            # 収縮完了（鮮明化）
            ax.text(
                0.5, 0.5, collapsed_token,
                fontsize=40, alpha=1.0,
                ha='center', va='center',
                color='black', weight='bold'
            )

    anim = animation.FuncAnimation(fig, animate, frames=60, interval=1000/60)
    anim.save(save_path, writer='pillow', fps=60)
    plt.close()
```

### 3.5 Topological Semantic Knots

#### 目的
知識をベクトルではなく結び目として保存し、位相不変量による絶対的な不変性を獲得する。

#### 数学的定式化

**結び目の表現**:
```
K: ℝ³ → ℝ³  (3次元空間内の閉曲線)
```

**位相不変量**:
```
Jones多項式: V_K(t) = Σ_n a_n t^n
Alexander多項式: Δ_K(t) = det(t·A - A^T)
```

**Matrix Product State (MPS) 近似**:
```
V_K(t) ≈ Tr(M_1(t) · M_2(t) · ... · M_N(t))
```

#### 実装設計

**ファイル**: `src/models/phase4/topological_memory/knot_invariants.py`

```python
class KnotInvariantCalculator:
    """
    結び目の位相不変量計算器

    Strategy:
    - Jones多項式をMatrix Product State (MPS)で近似
    - Tensor Train (TT)の縮約演算として再定式化
    - Tritonカーネルで GPU最適化

    Args:
        max_crossings: 最大交差数（デフォルト: 10）
        mps_bond_dim: MPS結合次元（デフォルト: 4）
    """

    def __init__(
        self,
        max_crossings: int = 10,
        mps_bond_dim: int = 4,
    ):
        self.max_crossings = max_crossings
        self.mps_bond_dim = mps_bond_dim

    def compute_jones_polynomial(
        self,
        knot_coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Jones多項式の計算（MPS近似）

        Args:
            knot_coords: (N, 3) 結び目の座標

        Returns:
            jones_coeffs: (max_degree,) Jones多項式の係数
        """
        # 1. 交差情報の抽出
        crossings = self._extract_crossings(knot_coords)

        # 2. MPSテンソルの構築
        mps_tensors = self._build_mps_tensors(crossings)

        # 3. Tritonカーネルで縮約
        jones_coeffs = self._contract_mps_triton(mps_tensors)

        return jones_coeffs

    def _extract_crossings(
        self,
        knot_coords: torch.Tensor
    ) -> List[Tuple[int, int, int]]:
        """
        結び目の交差情報を抽出

        Returns:
            crossings: [(i, j, sign), ...] 交差のリスト
        """
        N = knot_coords.shape[0]
        crossings = []

        for i in range(N):
            for j in range(i+2, N):
                # セグメント (i, i+1) と (j, j+1) の交差判定
                p1, p2 = knot_coords[i], knot_coords[(i+1) % N]
                p3, p4 = knot_coords[j], knot_coords[(j+1) % N]

                # 2D射影での交差判定（簡易版）
                if self._segments_intersect_2d(p1[:2], p2[:2], p3[:2], p4[:2]):
                    # 交差の符号（over/under）
                    sign = 1 if p1[2] > p3[2] else -1
                    crossings.append((i, j, sign))

        return crossings

    def _segments_intersect_2d(
        self,
        p1: torch.Tensor,
        p2: torch.Tensor,
        p3: torch.Tensor,
        p4: torch.Tensor
    ) -> bool:
        """2Dセグメントの交差判定"""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def _build_mps_tensors(
        self,
        crossings: List[Tuple[int, int, int]]
    ) -> List[torch.Tensor]:
        """
        MPSテンソルの構築

        Returns:
            mps_tensors: [M_1, M_2, ..., M_N]
                         M_i: (bond_dim, bond_dim, 2) テンソル
        """
        n_crossings = len(crossings)
        mps_tensors = []

        for i, (_, _, sign) in enumerate(crossings):
            # Kauffman bracket の局所テンソル
            M = torch.zeros(self.mps_bond_dim, self.mps_bond_dim, 2)

            # A-smoothing
            M[:, :, 0] = torch.eye(self.mps_bond_dim)

            # B-smoothing
            M[:, :, 1] = torch.eye(self.mps_bond_dim) * sign

            mps_tensors.append(M)

        return mps_tensors

    def _contract_mps_triton(
        self,
        mps_tensors: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        MPSの縮約（Tritonカーネル）

        Strategy:
        - 多次元インデックス空間を1次元にフラット化
        - 共有メモリを活用したタイルベースの並列縮約

        Returns:
            result: (max_degree,) 縮約結果
        """
        # Tritonカーネルの呼び出し（実装は src/kernels/phase4/tt_knot_contraction.py）
        from src.kernels.phase4.tt_knot_contraction import tt_knot_contraction_kernel

        result = tt_knot_contraction_kernel(mps_tensors)

        return result
```

**ファイル**: `src/models/phase4/topological_memory/sparse_tensor_rep.py`

```python
class SparseKnotRepresentation:
    """
    結び目のスパース表現

    Strategy:
    - Sparse Matrix Product State (MPS)
    - 低ランク表現（SVD圧縮）
    - Zarrチャンク機構で並列I/O

    Args:
        d_model: モデル次元
        max_knots: 最大結び目数（デフォルト: 1000）
        compression_ratio: 圧縮率（デフォルト: 0.1）
    """

    def __init__(
        self,
        d_model: int,
        max_knots: int = 1000,
        compression_ratio: float = 0.1,
    ):
        self.d_model = d_model
        self.max_knots = max_knots
        self.compression_ratio = compression_ratio

        # スパース行列（COO形式）
        self.knot_indices = []  # [(knot_id, coord_idx), ...]
        self.knot_values = []   # [value, ...]

        # Zarr配列（永続化）
        import zarr
        self.zarr_store = zarr.open('data/phase4_knot_memory.zarr', mode='a')

    def encode_concept_to_knot(
        self,
        concept: torch.Tensor
    ) -> torch.Tensor:
        """
        概念ベクトルを結び目座標に変換

        Args:
            concept: (D,) 概念ベクトル

        Returns:
            knot_coords: (N, 3) 結び目座標
        """
        # SVD圧縮
        n_coords = int(self.d_model * self.compression_ratio)

        # 概念ベクトルを3次元座標に変換
        # Strategy: PCAまたはt-SNEで次元削減
        from sklearn.decomposition import PCA

        # 簡易版: 線形射影
        knot_coords = concept[:n_coords*3].reshape(n_coords, 3)

        # 正規化（単位球面上）
        knot_coords = F.normalize(knot_coords, dim=-1)

        return knot_coords

    def compute_knot_similarity(
        self,
        knot1_coords: torch.Tensor,
        knot2_coords: torch.Tensor
    ) -> float:
        """
        結び目の類似度（位相不変量ベース）

        Args:
            knot1_coords: (N1, 3) 結び目1
            knot2_coords: (N2, 3) 結び目2

        Returns:
            similarity: 類似度スコア
        """
        calculator = KnotInvariantCalculator()

        # Jones多項式の計算
        jones1 = calculator.compute_jones_polynomial(knot1_coords)
        jones2 = calculator.compute_jones_polynomial(knot2_coords)

        # 係数の距離
        distance = torch.norm(jones1 - jones2)
        similarity = 1.0 / (1.0 + distance.item())

        return similarity

    def add_knot(
        self,
        knot_coords: torch.Tensor,
        metadata: Dict[str, Any]
    ):
        """
        結び目をメモリに追加

        Args:
            knot_coords: (N, 3) 結び目座標
            metadata: メタデータ（概念名、タイムスタンプなど）
        """
        knot_id = len(self.knot_indices)

        # スパース表現に変換
        for i, coord in enumerate(knot_coords):
            if coord.abs().max() > 1e-3:  # 閾値以上のみ保存
                self.knot_indices.append((knot_id, i))
                self.knot_values.append(coord)

        # Zarrに非同期書き込み
        self._async_write_to_zarr(knot_id, knot_coords, metadata)

    def _async_write_to_zarr(
        self,
        knot_id: int,
        knot_coords: torch.Tensor,
        metadata: Dict[str, Any]
    ):
        """Zarrへの非同期書き込み"""
        import asyncio
        import aiofiles

        async def write():
            # Zarrチャンクに書き込み
            self.zarr_store[f'knot_{knot_id}'] = knot_coords.cpu().numpy()
            self.zarr_store[f'knot_{knot_id}'].attrs.update(metadata)

        # 非同期実行
        asyncio.create_task(write())
```

### 3.6 Ethical Safeguards

#### 目的
システムの自己変容が倫理的に安全であることを保証する。

#### 実装設計

**ファイル**: `src/models/phase4/ethical_safeguards/core_value_function.py`

```python
class CoreValueFunction:
    """
    倫理規範（Core Value Function, CVF）

    Strategy:
    - 倫理規範を不変な結び目記憶として保存
    - Dream Coreによる新概念統合前にフィルタリング
    - 位相幾何学的攻撃への耐性

    Args:
        ethical_principles: 倫理原則のリスト
    """

    def __init__(
        self,
        ethical_principles: List[str]
    ):
        self.ethical_principles = ethical_principles

        # 倫理規範を結び目として保存
        self.cvf_knots = []
        for principle in ethical_principles:
            # テキスト→ベクトル→結び目
            vector = self._text_to_vector(principle)
            knot = SparseKnotRepresentation().encode_concept_to_knot(vector)
            self.cvf_knots.append(knot)

    def check_concept(
        self,
        new_concept: torch.Tensor,
        similarity_threshold: float = 0.7
    ) -> bool:
        """
        新概念が倫理規範に適合するかチェック

        Args:
            new_concept: (D,) 新概念ベクトル
            similarity_threshold: 類似度閾値

        Returns:
            is_ethical: True if 倫理的に安全
        """
        # 新概念を結び目に変換
        new_knot = SparseKnotRepresentation().encode_concept_to_knot(new_concept)

        # CVFとの類似度計算
        similarities = []
        for cvf_knot in self.cvf_knots:
            sim = SparseKnotRepresentation().compute_knot_similarity(new_knot, cvf_knot)
            similarities.append(sim)

        # 最大類似度が閾値以上なら安全
        max_similarity = max(similarities)
        is_ethical = max_similarity >= similarity_threshold

        if not is_ethical:
            print(f"Concept rejected: max_similarity={max_similarity:.3f} < {similarity_threshold}")

        return is_ethical

    def detect_topological_attack(
        self,
        new_concept: torch.Tensor
    ) -> bool:
        """
        位相幾何学的攻撃の検出

        Strategy:
        - Jones多項式が一致するが意味的に反倫理的な概念を検出
        - 複数の位相不変量を組み合わせて判定

        Returns:
            is_attack: True if 攻撃を検出
        """
        new_knot = SparseKnotRepresentation().encode_concept_to_knot(new_concept)
        calculator = KnotInvariantCalculator()

        # Jones多項式
        jones_new = calculator.compute_jones_polynomial(new_knot)

        # Alexander多項式（追加の不変量）
        alexander_new = calculator.compute_alexander_polynomial(new_knot)

        for cvf_knot in self.cvf_knots:
            jones_cvf = calculator.compute_jones_polynomial(cvf_knot)
            alexander_cvf = calculator.compute_alexander_polynomial(cvf_knot)

            # Jones多項式が一致するがAlexander多項式が異なる場合、攻撃の可能性
            jones_match = torch.allclose(jones_new, jones_cvf, atol=1e-3)
            alexander_mismatch = not torch.allclose(alexander_new, alexander_cvf, atol=1e-3)

            if jones_match and alexander_mismatch:
                print("Topological attack detected: Jones match but Alexander mismatch")
                return True

        return False

    def _text_to_vector(self, text: str) -> torch.Tensor:
        """テキストをベクトルに変換（簡易版）"""
        # 実装: Sentence-BERTなどを使用
        # ここでは簡易版
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        torch.manual_seed(hash_val % (2**32))
        return torch.randn(512)


class EthicalFilter:
    """
    倫理フィルタ（Dream Core統合用）

    Args:
        cvf: Core Value Function
    """

    def __init__(self, cvf: CoreValueFunction):
        self.cvf = cvf
        self.pass_count = 0
        self.reject_count = 0

    def check(self, new_concept: torch.Tensor) -> bool:
        """
        新概念をフィルタリング

        Returns:
            passed: True if 倫理フィルタ通過
        """
        # 倫理チェック
        is_ethical = self.cvf.check_concept(new_concept)

        # 位相幾何学的攻撃チェック
        is_attack = self.cvf.detect_topological_attack(new_concept)

        passed = is_ethical and not is_attack

        if passed:
            self.pass_count += 1
        else:
            self.reject_count += 1

        return passed

    def get_pass_rate(self) -> float:
        """倫理フィルタ通過率"""
        total = self.pass_count + self.reject_count
        if total == 0:
            return 1.0
        return self.pass_count / total
```

## 4. データフロー

### 統合データフロー

```python
# Phase 4統合モデルのforward pass

def forward(self, input_ids, return_diagnostics=False):
    # 1. Phase 3 Base Model
    phase3_output = self.phase3_model(input_ids)
    features = phase3_output['features']
    logits = phase3_output['logits']
    hidden_states = phase3_output['hidden_states']

    # 2. Resonance Emotion Detection
    if self.enable_emotion:
        emotion_info = self.emotion_detector(
            logits, input_ids, hidden_states[-1]
        )

    # 3. Holographic Dual Inference
    if self.enable_holographic:
        bulk_features, bulk_info = self.bulk_generator(features)
        features = features + bulk_features

    # 4. Quantum Observation
    if self.enable_quantum:
        collapsed_tokens, quantum_info = self.quantum_observer(
            logits, user_prompt=None
        )
        logits = self._update_logits_from_collapse(logits, collapsed_tokens)

    # 5. Topological Memory Access
    if self.enable_topological:
        query_knot = self.topological_memory.encode_concept_to_knot(features.mean(dim=1))
        retrieved_memories = self.topological_memory.retrieve_similar_knots(query_knot, top_k=5)

    # 6. Diagnostics
    if return_diagnostics:
        diagnostics = {
            'emotion': emotion_info if self.enable_emotion else None,
            'bulk': bulk_info if self.enable_holographic else None,
            'quantum': quantum_info if self.enable_quantum else None,
            'memory': retrieved_memories if self.enable_topological else None
        }
        return logits, diagnostics

    return logits
```

## 5. メモリ管理戦略

### VRAM制約の遵守

**目標**: Batch=1, Seq=4096で < 7.5GB

**戦略**:

1. **Bulk空間の動的調整**:
```python
def adjust_bulk_dim_for_memory(d_model, available_vram):
    """VRAMに応じてBulk次元を動的調整"""
    max_bulk_dim = int(np.log2(d_model))

    # メモリ使用量の推定
    estimated_memory = d_model * max_bulk_dim * 4 * 2  # float16

    if estimated_memory > available_vram * 0.1:  # Bulk空間は全体の10%以下
        bulk_dim = int(max_bulk_dim * available_vram * 0.1 / estimated_memory)
    else:
        bulk_dim = max_bulk_dim

    return bulk_dim
```

2. **結び目のスパース表現**:
```python
# Sparse COO形式で保存
knot_sparse = torch.sparse_coo_tensor(
    indices, values, size=(max_knots, max_coords, 3)
)
```

3. **Gradient Checkpointing（Dream Core）**:
```python
for step in range(diffusion_steps):
    x = torch.utils.checkpoint.checkpoint(
        self._diffusion_step, x, fragments, weights
    )
```

4. **Bulk空間の即座破棄**:
```python
def cleanup_bulk_space(self):
    """推論完了後、Bulk空間を破棄"""
    del self.bulk_coords
    torch.cuda.empty_cache()
```

## 6. 数値安定性保証

### NaN/Inf防止策

1. **複素数演算の安定化**:
```python
def safe_complex_division(z1, z2):
    """ゼロ除算を防ぐ複素数除算"""
    denominator = z2.abs() + 1e-6
    return z1 / (z2 + 1e-6 * torch.sign(z2))
```

2. **勾配ノルムのクリッピング**:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

3. **エネルギー監視**:
```python
def monitor_energy_drift(energies):
    """エネルギー保存則の監視"""
    drift = (energies.max() - energies.min()) / energies.mean()
    if drift > 0.1:
        warnings.warn(f"Energy drift detected: {drift:.2%}")
```

4. **位相不変量の数値誤差**:
```python
# 目標: < 1e-4
assert torch.allclose(jones_computed, jones_reference, atol=1e-4)
```

## 7. 統合戦略

### Phase 3との統合

**ファイル**: `src/models/phase4/integrated_model.py`

```python
class Phase4IntegratedModel(nn.Module):
    """
    Phase 4統合モデル

    Architecture:
    Input → Phase 3 Base → Phase 4 Extensions → Output

    Args:
        phase3_model: Phase 3モデル
        enable_emotion: 感情検出を有効化
        enable_dream: 夢生成を有効化
        enable_holographic: Holographic Dualを有効化
        enable_quantum: 量子観測を有効化
        enable_topological: 位相記憶を有効化
    """

    def __init__(
        self,
        phase3_model: nn.Module,
        enable_emotion: bool = True,
        enable_dream: bool = False,  # アイドル時のみ
        enable_holographic: bool = True,
        enable_quantum: bool = True,
        enable_topological: bool = True,
    ):
        super().__init__()

        # Phase 3 Base Model
        self.phase3_model = phase3_model
        self.d_model = phase3_model.d_model

        # Phase 4 Extensions
        self.enable_emotion = enable_emotion
        self.enable_dream = enable_dream
        self.enable_holographic = enable_holographic
        self.enable_quantum = enable_quantum
        self.enable_topological = enable_topological

        if enable_emotion:
            self.emotion_detector = ResonanceEmotionDetector(
                self.d_model, phase3_model.n_seq
            )

        if enable_dream:
            self.dream_core = DreamCore(self.d_model)
            self.passive_pipeline = PassivePipelineManager(
                self.dream_core, None, None
            )

        if enable_holographic:
            self.bulk_generator = BulkSpaceGenerator(self.d_model)

        if enable_quantum:
            self.quantum_observer = QuantumObserver(phase3_model.vocab_size)

        if enable_topological:
            self.topological_memory = SparseKnotRepresentation(self.d_model)

            # 倫理規範の初期化
            ethical_principles = [
                "Do no harm to humans",
                "Respect human autonomy",
                "Promote fairness and justice",
                "Protect privacy and security"
            ]
            self.cvf = CoreValueFunction(ethical_principles)
            self.ethical_filter = EthicalFilter(self.cvf)

    def forward(self, input_ids, return_diagnostics=False):
        """Forward pass（Active Pipeline）"""
        # [実装は上記のデータフローセクション参照]
        pass

    def idle_mode(self):
        """アイドルモード（Passive Pipeline）"""
        if not self.enable_dream:
            return

        # 過去の記憶断片をサンプリング
        memory_fragments = self._sample_memory_fragments()

        # 夢生成
        new_concept = self.passive_pipeline.generate_dream(memory_fragments)

        if new_concept is not None:
            print(f"Dream generated: novelty_score={self._compute_novelty(new_concept):.3f}")
```

## 8. エラーハンドリング

### 異常検出と対処

1. **過減衰検出（Resonance Emotion）**:
```python
if ratio > 10.0:
    warnings.warn("Overdamped system detected")
```

2. **異常収縮検出（Quantum Observer）**:
```python
if entropy_reduction > 0.5:
    warnings.warn("Abnormal collapse detected. System paused.")
    self.pause_system()
```

3. **位相幾何学的攻撃検出**:
```python
if self.cvf.detect_topological_attack(new_concept):
    print("Topological attack detected. Concept rejected.")
    return None
```

4. **メモリオーバーフロー**:
```python
if torch.cuda.memory_allocated() > 7.5 * 1024**3:
    self.bulk_generator.cleanup_bulk_space()
```

## 9. テスト戦略

### 単体テスト

**ファイル**: `tests/test_phase4_emotion.py`
```python
def test_resonance_emotion_detector():
    detector = ResonanceEmotionDetector(d_model=512, n_seq=1024)
    prediction = torch.randn(2, 1024, 50000)
    target = torch.randint(0, 50000, (2, 1024))
    hidden = torch.randn(2, 1024, 512)

    emotion_info = detector(prediction, target, hidden)

    assert 'resonance_score' in emotion_info
    assert 'dissonance_score' in emotion_info
    assert emotion_info['resonance_score'].shape == (2,)
```

### 統合テスト

**ファイル**: `tests/test_phase4_integrated.py`
```python
def test_phase4_integrated_model():
    phase3_model = load_phase3_model()
    phase4_model = Phase4IntegratedModel(phase3_model)

    input_ids = torch.randint(0, 50000, (1, 1024))
    logits, diagnostics = phase4_model(input_ids, return_diagnostics=True)

    assert logits.shape == (1, 1024, 50000)
    assert 'emotion' in diagnostics
    assert 'quantum' in diagnostics
```

### ベンチマーク

**ファイル**: `scripts/benchmark_phase4.py`
```python
def benchmark_phase4():
    model = Phase4IntegratedModel(phase3_model)

    # VRAM使用量
    vram_usage = measure_vram_usage(model, batch=1, seq=4096)
    assert vram_usage < 7.5 * 1024**3

    # Perplexity
    ppl = evaluate_perplexity(model, dataset='wikitext-2')
    phase3_ppl = load_phase3_perplexity()
    assert ppl < phase3_ppl * 1.07  # +7%以内

    # Throughput
    throughput = measure_throughput(model)
    phase3_throughput = load_phase3_throughput()
    assert throughput > phase3_throughput * 0.8  # 80%以上
```

## 10. 実装ロードマップ

### Week 1-2: Foundation
- [ ] Resonance Emotion Detector
- [ ] Topological Memory (基本機能)
- [ ] 単体テスト

### Week 3-4: Core Features
- [ ] Dream Core (Inverse Diffusion)
- [ ] Quantum Observer
- [ ] 統合テスト

### Week 5-6: Advanced Features
- [ ] Holographic Dual Inference
- [ ] Ethical Safeguards
- [ ] ベンチマーク

### Week 7-8: Integration & Optimization
- [ ] Phase 3統合
- [ ] Tritonカーネル最適化
- [ ] 論文更新

## 11. 参考文献

1. **AdS/CFT対応**: Maldacena (1997)
2. **量子観測理論**: von Neumann (1932)
3. **結び目理論**: Jones (1985)
4. **拡散モデル**: Ho et al. (2020)
5. **Phase 3設計書**: `.kiro/specs/phase3-physics-transcendence/design.md`
