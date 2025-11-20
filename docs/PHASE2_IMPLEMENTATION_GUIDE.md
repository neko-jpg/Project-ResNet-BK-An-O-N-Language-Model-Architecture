# Phase 2: Breath of Life - 実装ガイド

## 目次

1. [概要](#概要)
2. [アーキテクチャ](#アーキテクチャ)
3. [モジュール詳細](#モジュール詳細)
4. [使用方法](#使用方法)
5. [トラブルシューティング](#トラブルシューティング)
6. [パフォーマンスチューニング](#パフォーマンスチューニング)
7. [FAQ](#faq)

---

## 概要

### Phase 2の目的

Phase 2「生命の息吹 (Breath of Life)」は、Project MUSEのモデルに**動的な時間発展と記憶機構**を導入するフェーズです。Phase 1で構築した静的な物理ベースO(N)アーキテクチャを、生物の神経系のような**動的力学系**へと進化させます。

### 設計原理

Phase 2は以下の4つの核心的な物理概念に基づいて設計されています:

#### 1. **Non-Hermitian Forgetting（非エルミート忘却）**

**物理的背景**: 開放量子系の散逸を模倣した自然な忘却機構

**数学的定式化**:
```
H_eff = H_0 + V - iΓ
```
- `H_0`: 基底ハミルトニアン（Phase 1のBK-Core）
- `V`: 実ポテンシャル（意味的な情報）
- `Γ > 0`: 虚ポテンシャル（減衰率、忘却の速度）

**物理的直観**: 
- 閉じた量子系（エルミート）では情報は永遠に保存される
- 開放量子系（非エルミート）では環境との相互作用により情報が散逸する
- 脳も開放系であり、不要な情報は自然に忘却される

**時間発展**:
```
||ψ(t)||² = exp(-2Γt) ||ψ(0)||²
```
情報の強度は指数関数的に減衰します。

#### 2. **Dissipative Hebbian Dynamics（散逸的ヘッブ力学）**

**生物学的背景**: "Neurons that fire together, wire together"

**数学的定式化**:
```
dW/dt = η(k^T v) - ΓW
```

- `η(k^T v)`: Hebbian項（記憶形成、シナプス強化）
- `-ΓW`: 散逸項（忘却、シナプス減衰）

**離散化**:
```
W_new = exp(-Γ * dt) * W_old + η * (k^T v)
```

**物理的直観**:
- 同時に活性化するニューロン間の結合が強化される（Hebbian学習）
- 使われない結合は自然に減衰する（散逸）
- この2つの力のバランスが記憶の動的平衡を生み出す

**Lyapunov安定性**: エネルギー関数 `E = ||W||²` に対して `dE/dt ≤ 0` を保証

#### 3. **SNR-based Memory Selection（信号対雑音比ベースの記憶選択）**

**生物学的動機**: 脳は重要な記憶だけを長期保持する

**数学的定式化**:
```
SNR_i = |W_i| / σ_noise
```

**選択戦略**:
- `SNR < τ` (閾値): Γを増加 → 急速忘却
- `SNR > τ`: ηを増加 → 学習強化

**物理的直観**:
- 信号（重要な記憶）とノイズ（ランダムな活性化）を区別
- ノイズは自動的に忘却され、信号は保持・強化される
- 生物の記憶選択プロセスの数理モデル

#### 4. **Memory Resonance（記憶共鳴）**

**物理的背景**: 量子系の固有状態分解とゼータ関数の零点分布

**数学的定式化**:
```
W' = U^(-1) W U
```
ここで `U` はリーマンゼータ関数の零点に基づく周波数基底

**物理的直観**:
- 記憶を「周波数空間」で表現
- ゼータ零点は「最も規則的なランダム性」を持つ（GUE統計）
- この基底で対角化すると、記憶の干渉が最小化される
- 類似記憶は同じ周波数で共鳴し、相互に強化される


### Phase 2の革新性

| 特徴 | Phase 1 | Phase 2 |
|------|---------|---------|
| **時間発展** | 静的 | 動的（散逸的Hebbian） |
| **記憶機構** | なし | Fast Weights（短期記憶） |
| **忘却** | なし | Non-Hermitian（自然な散逸） |
| **記憶選択** | なし | SNRベース（重要度判定） |
| **記憶配置** | 標準 | ゼータ零点ベース（干渉最小化） |
| **安定性保証** | 数値的 | Lyapunov理論（数学的保証） |

### 性能目標

| 指標 | 目標値 | 測定条件 |
|------|--------|----------|
| **BK-Core高速化** | 3.0倍以上 | Tritonカーネル vs PyTorch |
| **VRAM使用量** | 8.0GB未満 | Batch=1, Seq=4096, fp16 |
| **PPL劣化** | +5%以内 | Phase 1モデル比 |
| **勾配維持** | 1e-5以上 | Seq=4096の末尾→先頭 |
| **スループット** | 100 tokens/sec以上 | A100/Colab Pro環境 |

---

## アーキテクチャ

### システム全体構成

```
Phase 2 Architecture
├── Priority 0: Kernel Optimization Layer
│   └── BK-Core Triton Kernel (src/kernels/bk_scan.py)
│       ├── Forward Scan (Theta recursion)
│       ├── Backward Scan (Phi recursion)
│       └── Complex Matrix Multiplication
│
├── Core Dynamics Layer (src/models/phase2/)
│   ├── non_hermitian.py
│   │   ├── NonHermitianPotential
│   │   └── DissipativeBKLayer
│   ├── dissipative_hebbian.py
│   │   ├── DissipativeHebbianLayer
│   │   └── LyapunovStabilityMonitor
│   ├── memory_selection.py
│   │   ├── SNRMemoryFilter
│   │   └── MemoryImportanceEstimator
│   └── memory_resonance.py
│       ├── MemoryResonanceLayer
│       └── ZetaBasisTransform
│
├── Initialization Layer (src/models/phase2/)
│   └── zeta_init.py
│       ├── ZetaInitializer
│       ├── ZetaEmbedding
│       └── GUEMatrixGenerator
│
└── Integration Layer (src/models/phase2/)
    ├── integrated_model.py
    │   ├── Phase2IntegratedModel
    │   └── Phase2Block
    └── factory.py
        ├── create_phase2_model()
        └── convert_phase1_to_phase2()
```


### データフロー

```
Input Sequence (B, N, D)
    ↓
[Token Embedding + ZetaEmbedding] ← ゼータ零点ベースの位相エンコーディング
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase2Block (× N layers)                                     │
│                                                               │
│  [Layer Norm]                                                │
│      ↓                                                        │
│  [NonHermitianPotential] → V(x) - iΓ(x) 複素ポテンシャル生成│
│      ↓                              ↓                         │
│      ↓                         [Γ監視] Lyapunov安定性チェック│
│      ↓                              ↓                         │
│  [BK-Core Triton] ← 高速化された三重対角逆行列計算          │
│      ↓                                                        │
│  [Residual Connection]                                       │
│      ↓                                                        │
│  [Layer Norm]                                                │
│      ↓                                                        │
│  [DissipativeHebbianLayer] ← W_new = exp(-Γ*dt)*W_old + η*(k^T v) │
│      ↓                              ↓                         │
│      ↓                         [SNRFilter] 重要記憶の選択    │
│      ↓                              ↓                         │
│  [MemoryResonanceLayer] ← ゼータ基底での対角化と共鳴検出    │
│      ↓                                                        │
│  [Residual Connection]                                       │
│      ↓                                                        │
│  [Layer Norm]                                                │
│      ↓                                                        │
│  [FFN (Feed-Forward Network)]                                │
│      ↓                                                        │
│  [Residual Connection]                                       │
└─────────────────────────────────────────────────────────────┘
    ↓
[Final Layer Norm]
    ↓
[LM Head (Linear)]
    ↓
Output Logits (B, N, vocab_size)
```

### 計算量分析

| コンポーネント | 複雑度 | 備考 |
|--------------|--------|------|
| BK-Core Triton | O(N) | Phase 1と同じ、3倍高速化 |
| NonHermitian Potential | O(N·D) | 線形射影 |
| Dissipative Hebbian | O(N·H·D²) | Fast Weights更新 |
| SNR Filter | O(H·D²) | 統計計算 |
| Memory Resonance | O(H·D³) | 行列対角化（ボトルネック） |
| **Total** | **O(N·H·D² + H·D³)** | D³項は小さいヘッド次元で緩和 |

**注意**: Memory Resonanceの`O(D³)`項は、ヘッド次元`D_h=64`程度に抑えることで実用的な計算量に収まります。


---

## モジュール詳細

### 1. BK-Core Triton Kernel

#### 概要

Phase 1のボトルネックであるBK-Core再帰計算をTritonで並列化し、**3倍以上の高速化**を達成します。

#### ファイル

- `src/kernels/bk_scan.py`

#### 主要コンポーネント

##### 1.1 複素数演算ユーティリティ

```python
@triton.jit
def complex_mul(r1, i1, r2, i2):
    """
    複素数乗算: (r1+i1j)*(r2+i2j) = (r1r2-i1i2) + (r1i2+i1r2)j
    
    物理的直観:
    - 複素数は振幅と位相を持つ波動を表現
    - 乗算は振幅の積と位相の和
    - GPU上では実部と虚部を分離して計算
    """
    real = r1 * r2 - i1 * i2
    imag = r1 * i2 + i1 * r2
    return real, imag
```

##### 1.2 Forward Scan Kernel (Theta)

**数学的定式化**:
```
theta_i = alpha_i * theta_{i-1} + beta_i * theta_{i-2}
```

**行列形式**:
```
[theta_i  ]   [alpha_i  beta_i] [theta_{i-1}]
[theta_{i-1}] = [1        0     ] [theta_{i-2}]
```

**並列化戦略**:
- ブロック内: シリアルスキャン（簡潔性優先）
- ブロック間: 将来的に並列スキャン拡張可能

**使用例**:
```python
from src.kernels.bk_scan import bk_scan_triton

# 入力
a = torch.randn(batch, seq_len, dtype=torch.complex64, device='cuda')
b = torch.randn(seq_len-1, dtype=torch.complex64, device='cuda')
c = torch.randn(seq_len-1, dtype=torch.complex64, device='cuda')
z = torch.tensor(0.1 + 0.1j, dtype=torch.complex64, device='cuda')

# Tritonカーネル実行
diag_elements = bk_scan_triton(a, b, c, z)
# diag_elements: (batch, seq_len) complex64
```

#### 性能ベンチマーク

```bash
python scripts/benchmark_bk_triton.py
```

**期待される結果**:
```
BK-Core Performance Comparison
==============================
PyTorch vmap:  125.3 ms
Triton kernel:  41.7 ms
Speedup:        3.0x ✓

Numerical Error: 8.3e-5 ✓ (< 1e-4)
```


### 2. Non-Hermitian Forgetting

#### 概要

複素ポテンシャル `V - iΓ` を生成し、情報の自然な散逸（忘却）を実現します。

#### ファイル

- `src/models/phase2/non_hermitian.py`

#### 主要クラス

##### 2.1 NonHermitianPotential

**物理的背景**:
```
H_eff = H_0 + V - iΓ
```
- `V`: 実ポテンシャル（意味的な情報エネルギー）
- `Γ > 0`: 虚ポテンシャル（減衰率、忘却の速度）

**時間発展**:
```
||ψ(t)||² = exp(-2Γt) ||ψ(0)||²
```

**使用例**:
```python
from src.models.phase2.non_hermitian import NonHermitianPotential

# インスタンス化
potential = NonHermitianPotential(
    d_model=512,
    n_seq=1024,
    base_decay=0.01,      # 最小減衰率
    adaptive_decay=True,  # 入力依存の減衰
    schatten_p=1.0        # Nuclear Norm
)

# Forward pass
x = torch.randn(4, 1024, 512)  # (B, N, D)
complex_potential = potential(x)  # (B, N) complex64

# 実部と虚部を取得
v_real = complex_potential.real  # 意味的ポテンシャル
gamma = -complex_potential.imag  # 減衰率（正の値）
```

**重要なパラメータ**:

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `base_decay` | 0.01 | 最小減衰率（Γの下限） |
| `adaptive_decay` | True | 入力依存の減衰を使用 |
| `schatten_p` | 1.0 | Schatten Normのp値 |
| `stability_threshold` | 1e-3 | 安定性監視の閾値 |

**Schatten Norm監視**:

```python
# 過減衰検出
# Γ >> |V| の場合、情報が即座に消失
if gamma.mean() / (v_real.abs().mean() + 1e-6) > 10.0:
    warnings.warn("Overdamped system detected")
```

##### 2.2 DissipativeBKLayer

BK-CoreとNonHermitianPotentialを統合したラッパー。

**使用例**:
```python
from src.models.phase2.non_hermitian import DissipativeBKLayer

layer = DissipativeBKLayer(
    d_model=512,
    n_seq=1024,
    use_triton=True  # Tritonカーネルを使用
)

x = torch.randn(4, 1024, 512)
features = layer(x)  # (B, N, 2) [Re(G_ii), Im(G_ii)]
```


### 3. Dissipative Hebbian Dynamics

#### 概要

記憶形成（Hebbian）と散逸（Non-Hermitian）を統合した動的記憶機構。

#### ファイル

- `src/models/phase2/dissipative_hebbian.py`

#### 核心方程式

**連続時間**:
```
dW/dt = η(k^T v) - ΓW
```

**離散化**:
```
W_new = exp(-Γ * dt) * W_old + η * (k^T v)
```

**物理的解釈**:
- `η(k^T v)`: シナプス強化（記憶形成）
- `-ΓW`: シナプス減衰（忘却）
- `exp(-Γ*dt)`: 時間発展演算子

#### 主要クラス

##### 3.1 DissipativeHebbianLayer

**使用例**:
```python
from src.models.phase2.dissipative_hebbian import DissipativeHebbianLayer

layer = DissipativeHebbianLayer(
    d_model=512,
    head_dim=64,
    num_heads=8,
    eta=0.1,   # Hebbian学習率
    dt=1.0     # 時間ステップ
)

# Forward pass
x = torch.randn(4, 1024, 512)  # (B, N, D)
gamma = torch.rand(4, 1024) * 0.1  # (B, N) 減衰率

output, new_state = layer(x, gamma, state=None)
# output: (B, N, D)
# new_state: (B, H, D_h, D_h) Fast Weight行列
```

**重要なパラメータ**:

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `eta` | 0.1 | Hebbian学習率（記憶形成の強度） |
| `dt` | 1.0 | 時間ステップ（離散化の粒度） |
| `num_heads` | 8 | ヘッド数（並列記憶チャネル） |
| `head_dim` | 64 | ヘッド次元（記憶容量） |

**Fast Weightsの更新**:

```python
# 各時刻ステップで
for t in range(seq_len):
    # 散逸項
    decay = torch.exp(-gamma[:, t] * dt)  # (B,)
    
    # Hebbian更新項
    k_t = k[:, t]  # (B, H, D_h)
    v_t = v[:, t]  # (B, H, D_h)
    update = eta * torch.einsum('bhi,bhj->bhij', k_t, v_t)
    
    # 統合更新
    state = decay.view(B, 1, 1, 1) * state + update
    
    # 読み出し
    q_t = q[:, t]  # (B, H, D_h)
    y_t = torch.einsum('bhij,bhj->bhi', state, q_t)
```

##### 3.2 LyapunovStabilityMonitor

**Lyapunov関数**:
```
E = ||W||²_F  (Frobenius norm)
```

**安定性条件**:
```
dE/dt ≤ 0
```

**使用例**:
```python
monitor = LyapunovStabilityMonitor(gamma_adjust_rate=0.01)

# 各ステップで
metrics = monitor.check(state, decay, update)

if not metrics['is_stable']:
    # Γを自動調整
    gamma = gamma + metrics['suggested_gamma_adjust']
```


### 4. Memory Selection (SNR-based)

#### 概要

信号対雑音比（SNR）に基づいて重要な記憶を選択的に保持します。

#### ファイル

- `src/models/phase2/memory_selection.py`

#### 核心概念

**SNR定義**:
```
SNR_i = |W_i| / σ_noise
```

**選択戦略**:
- `SNR < τ` (閾値): その成分のΓを増加 → 急速忘却
- `SNR > τ`: その成分のηを増加 → 学習強化

**生物学的動機**: 脳は重要な記憶だけを長期保持し、ノイズは自動的に忘却する

#### 主要クラス

##### 4.1 SNRMemoryFilter

**使用例**:
```python
from src.models.phase2.memory_selection import SNRMemoryFilter

filter = SNRMemoryFilter(
    threshold=2.0,        # SNR閾値
    gamma_boost=2.0,      # 低SNR成分のΓ増加率
    eta_boost=1.5         # 高SNR成分のη増加率
)

# Fast Weightsのフィルタリング
weights = torch.randn(4, 8, 64, 64)  # (B, H, D_h, D_h)
gamma = torch.rand(4) * 0.1          # (B,)
eta = 0.1

adjusted_gamma, adjusted_eta = filter(weights, gamma, eta)

# 統計情報の取得
stats = filter.get_statistics()
print(f"Mean SNR: {stats['mean_snr']:.2f}")
print(f"Low SNR ratio: {stats['low_snr_ratio']:.2%}")
```

**重要なパラメータ**:

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `threshold` | 2.0 | SNR閾値（重要度判定） |
| `gamma_boost` | 2.0 | 低SNR成分のΓ増加率 |
| `eta_boost` | 1.5 | 高SNR成分のη増加率 |

**ノイズ標準偏差の推定**:
```python
# 全体の重み分布から推定
sigma_noise = torch.std(weights) + 1e-6

# 各成分のSNR計算
snr = torch.abs(weights) / sigma_noise
```

**動的調整**:
```python
# 平均SNR
mean_snr = snr.mean(dim=[1, 2, 3])  # (B,)

# 低SNR → Γ増加（急速忘却）
gamma_adjust = torch.where(
    mean_snr < threshold,
    gamma * gamma_boost,
    gamma
)

# 高SNR → η増加（強化学習）
eta_adjust = eta * (eta_boost if mean_snr.mean() > threshold else 1.0)
```


### 5. Memory Resonance

#### 概要

ゼータ零点基底で記憶を対角化し、共鳴する記憶を検出・強化します。

#### ファイル

- `src/models/phase2/memory_resonance.py`

#### 物理的背景

**量子カオスとゼータ関数**:
- リーマンゼータ関数の零点の虚部は、量子カオス系のエネルギー準位と同じ統計的性質を持つ（GUE統計）
- この「最も規則的なランダム性」を持つ基底で対角化すると、記憶の干渉が最小化される

**数学的定式化**:
```
W' = U^(-1) W U
```
ここで `U` はゼータ零点由来の周波数基底

**基底行列の構成**:
```
U[i, j] = exp(2πi * gamma_j * i / N) / sqrt(N)
```
ここで `gamma_j` はj番目のゼータ零点の虚部

#### 主要クラス

##### 5.1 ZetaBasisTransform

**ゼータ零点の取得**:
```python
from src.models.phase2.memory_resonance import ZetaBasisTransform

zeta_basis = ZetaBasisTransform(max_dim=512)

# 最初のn個の零点を取得
zeros = zeta_basis.get_zeta_zeros(n=64)
# zeros: [14.134725, 21.022040, 25.010858, ...]

# 基底行列を生成（キャッシュ付き）
U = zeta_basis.get_basis_matrix(dim=64, device='cuda')
# U: (64, 64) complex64
```

**精密値とGUE近似**:
- `n <= 10`: 精密な零点値を使用
- `n > 10`: GUE統計に基づく近似生成

**GUE統計による近似**:
```python
# ランダムエルミート行列の固有値を生成
A = torch.randn(k, k, dtype=torch.complex64)
H = (A + A.conj().T) / 2
eigs = torch.linalg.eigvalsh(H.real)

# スケーリングして零点分布に合わせる
spacings = eigs[1:] - eigs[:-1]
spacings = spacings / spacings.mean() * 2.5  # 平均間隔調整
```

##### 5.2 MemoryResonanceLayer

**使用例**:
```python
from src.models.phase2.memory_resonance import MemoryResonanceLayer

layer = MemoryResonanceLayer(
    d_model=512,
    head_dim=64,
    num_heads=8,
    energy_threshold=0.1  # 共鳴エネルギー閾値
)

# Forward pass
weights = torch.randn(4, 8, 64, 64)  # (B, H, D_h, D_h) Fast Weights
x = torch.randn(4, 1024, 512)        # (B, N, D) 入力

filtered_weights, resonance_info = layer(weights, x)

# 共鳴情報
print(f"Resonant modes: {resonance_info['num_resonant']:.1f}")
print(f"Total energy: {resonance_info['total_energy']:.3f}")
```

**対角化と共鳴検出**:
```python
# ゼータ基底への変換
U = zeta_basis.get_basis_matrix(D_h, device=weights.device)
U_inv = torch.linalg.inv(U)

# 対角化: W' = U^(-1) W U
weights_diag = torch.einsum('ij,bhjk,kl->bhil', U_inv, weights, U)

# 対角成分のエネルギー
diag_energy = torch.abs(torch.diagonal(weights_diag, dim1=-2, dim2=-1))

# エネルギー閾値でフィルタリング
mask = diag_energy > energy_threshold

# マスク適用
weights_diag_filtered = weights_diag * mask.unsqueeze(-1)

# 元の基底に戻す
filtered_weights = torch.einsum('ij,bhjk,kl->bhil', U, weights_diag_filtered, U_inv)
```

**物理的直観**:
- 高エネルギー成分 = 重要な記憶（共鳴）
- 低エネルギー成分 = ノイズ（フィルタリング）
- 類似記憶は同じ周波数で共鳴し、相互に強化される


### 6. Zeta Initialization

#### 概要

ゼータ零点分布に基づく初期化で、情報の干渉を最小化する効率的な分散表現を実現します。

#### ファイル

- `src/models/phase2/zeta_init.py`

#### 主要クラス

##### 6.1 ZetaInitializer

**線形層の初期化**:
```python
from src.models.phase2.zeta_init import ZetaInitializer

# 線形層の特異値をゼータ零点分布に基づいて初期化
linear = nn.Linear(512, 512)
ZetaInitializer.initialize_linear_zeta(linear, scale=10.0)
```

**数学的定式化**:
```
W = U Σ V^T
```
ここで特異値 `Σ_i = scale / zero_i`

**物理的直観**:
- ゼータ零点は「最も規則的なランダム性」を持つ
- この分布に基づく初期化により、情報の干渉が最小化される
- 学習の初期段階から効率的な表現を獲得

**Embeddingの初期化**:
```python
# Embeddingをゼータ零点ベースの位相パターンで初期化
embedding = nn.Embedding(50257, 512)
ZetaInitializer.initialize_embedding_zeta(embedding, scale=1.0)
```

##### 6.2 ZetaEmbedding

**ゼータ零点ベースの位置埋め込み**:
```python
from src.models.phase2.zeta_init import ZetaEmbedding

# 標準のSinusoidal Embeddingの代替
pos_embedding = ZetaEmbedding(
    max_len=2048,
    d_model=512,
    trainable=False  # 固定 or 学習可能
)

# Forward pass
positions = torch.arange(1024).unsqueeze(0)  # (1, N)
embeddings = pos_embedding(positions)         # (1, N, D)
```

**数学的定式化**:
```
PE(pos, 2i)   = sin(pos / zero_i)
PE(pos, 2i+1) = cos(pos / zero_i)
```

**標準のSinusoidalとの比較**:

| 特徴 | Sinusoidal | Zeta |
|------|-----------|------|
| **周波数** | 固定（2^i） | ゼータ零点由来 |
| **統計的性質** | 規則的 | GUE統計（量子カオス） |
| **干渉** | あり | 最小化 |
| **長距離依存** | 標準 | 改善 |

**物理的直観**:
- 標準のSinusoidalは規則的な周波数（2の累乗）
- Zetaは「最も規則的なランダム性」を持つ周波数
- 長距離依存関係の表現が改善される


---

## 使用方法

### 基本的な使用例

#### 1. モデルの作成

```python
from src.models.phase2 import create_phase2_model, Phase2Config

# 設定の作成
config = Phase2Config(
    vocab_size=50257,
    d_model=512,
    n_layers=6,
    n_seq=1024,
    num_heads=8,
    head_dim=64,
    use_triton_bk=True,
    base_decay=0.01,
    hebbian_eta=0.1,
    snr_threshold=2.0,
    resonance_enabled=True
)

# モデルの作成
model = create_phase2_model(config)
model = model.cuda()

print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
```

#### 2. Forward Pass

```python
import torch

# 入力データ
input_ids = torch.randint(0, 50257, (4, 1024)).cuda()  # (B, N)

# Forward pass
logits = model(input_ids)  # (B, N, vocab_size)

# 診断情報付き
logits, diagnostics = model(input_ids, return_diagnostics=True)

# 診断情報の確認
print(f"Mean Gamma: {diagnostics['gamma_values'][0].mean():.4f}")
print(f"Mean SNR: {diagnostics['snr_stats'][0]['mean_snr']:.2f}")
print(f"Resonant modes: {diagnostics['resonance_info'][0]['num_resonant']:.1f}")
```

#### 3. 学習ループ

```python
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

# オプティマイザー
optimizer = AdamW(model.parameters(), lr=1e-4)
criterion = CrossEntropyLoss()

# 学習ループ
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        input_ids = batch['input_ids'].cuda()
        labels = batch['labels'].cuda()
        
        # Forward
        logits = model(input_ids)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # 勾配クリッピング（重要）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if step % 100 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
```

#### 4. Phase 1からの変換

```python
from src.models.phase1 import Phase1IntegratedModel, Phase1Config
from src.models.phase2 import convert_phase1_to_phase2, Phase2Config

# Phase 1モデルのロード
phase1_config = Phase1Config.for_hardware(vram_gb=8.0)
phase1_model = Phase1IntegratedModel(phase1_config)
phase1_model.load_state_dict(torch.load('phase1_checkpoint.pt'))

# Phase 2への変換
phase2_config = Phase2Config.from_phase1(phase1_config)
phase2_model = convert_phase1_to_phase2(phase1_model, phase2_config)

# 学習の継続
# phase2_model は phase1_model の重みを継承
```

### プリセット設定

```python
from src.models.phase2 import create_phase2_model

# Small (デバッグ用)
model_small = create_phase2_model('small')
# d_model=256, n_layers=4, num_heads=4

# Base (標準)
model_base = create_phase2_model('base')
# d_model=512, n_layers=6, num_heads=8

# Large (高性能)
model_large = create_phase2_model('large')
# d_model=768, n_layers=12, num_heads=12
```


---

## トラブルシューティング

### よくある問題と解決方法

#### 1. Tritonカーネルのエラー

**症状**:
```
TritonKernelError: Failed to compile kernel
```

**原因**:
- Tritonがインストールされていない
- CUDAバージョンの不一致
- GPUがサポートされていない

**解決方法**:
```python
# 1. Tritonのインストール確認
pip install triton

# 2. フォールバックを有効化
config = Phase2Config(
    use_triton_bk=False  # PyTorch実装を使用
)

# 3. 自動フォールバック（推奨）
# モデルは自動的にPyTorch実装にフォールバックします
```

**デバッグ**:
```python
# Tritonの動作確認
from src.kernels.bk_scan import test_triton_availability

if test_triton_availability():
    print("Triton is available")
else:
    print("Triton is not available, using PyTorch fallback")
```

#### 2. Lyapunov安定性違反

**症状**:
```
UserWarning: Lyapunov stability violated. Increasing gamma by 1.0%
```

**原因**:
- Fast Weightsのエネルギーが増加している（`dE/dt > 0`）
- Hebbian学習率 `η` が大きすぎる
- 減衰率 `Γ` が小さすぎる

**解決方法**:
```python
# 1. Hebbian学習率を減少
config = Phase2Config(
    hebbian_eta=0.05  # デフォルト: 0.1
)

# 2. 基底減衰率を増加
config = Phase2Config(
    base_decay=0.02  # デフォルト: 0.01
)

# 3. 自動調整を有効化（デフォルトで有効）
config = Phase2Config(
    gamma_adjust_rate=0.01  # Γの自動調整率
)
```

**監視**:
```python
# 診断情報で安定性を確認
logits, diagnostics = model(input_ids, return_diagnostics=True)

for layer_idx, stability in enumerate(diagnostics['stability_metrics']):
    if not stability['is_stable']:
        print(f"Layer {layer_idx}: Unstable (dE/dt = {stability['dE_dt']:.3e})")
```

#### 3. 過減衰（Overdamped System）

**症状**:
```
UserWarning: Overdamped system detected: Γ/|V| = 15.2
```

**原因**:
- 減衰率 `Γ` が振動エネルギー `|V|` に比べて大きすぎる
- 情報が即座に消失している

**解決方法**:
```python
# 1. 基底減衰率を減少
config = Phase2Config(
    base_decay=0.005  # デフォルト: 0.01
)

# 2. 適応的減衰を無効化
config = Phase2Config(
    adaptive_decay=False  # 固定減衰率を使用
)

# 3. Schatten Norm監視を調整
config = Phase2Config(
    stability_threshold=1e-2  # デフォルト: 1e-3
)
```

#### 4. VRAM不足

**症状**:
```
RuntimeError: CUDA out of memory
```

**原因**:
- バッチサイズが大きすぎる
- シーケンス長が長すぎる
- Fast Weightsのメモリ消費

**解決方法**:
```python
# 1. バッチサイズを減少
batch_size = 2  # デフォルト: 4

# 2. シーケンス長を減少
config = Phase2Config(
    n_seq=512  # デフォルト: 1024
)

# 3. ヘッド次元を減少
config = Phase2Config(
    head_dim=32,  # デフォルト: 64
    num_heads=4   # デフォルト: 8
)

# 4. Memory Resonanceを無効化
config = Phase2Config(
    resonance_enabled=False  # O(D³)項を削減
)

# 5. 勾配チェックポイントを使用
model.gradient_checkpointing_enable()
```

**メモリ使用量の確認**:
```python
import torch

# Forward pass前
torch.cuda.reset_peak_memory_stats()

logits = model(input_ids)

# メモリ使用量
peak_memory = torch.cuda.max_memory_allocated() / 1024**3
print(f"Peak memory: {peak_memory:.2f} GB")
```


#### 5. 勾配消失/爆発

**症状**:
```
# 勾配消失
Gradient norm: 1.2e-8

# 勾配爆発
Gradient norm: 3.5e+6
```

**原因**:
- 複素勾配の不安定性
- Fast Weightsの暴走
- 学習率の不適切な設定

**解決方法**:
```python
# 1. 勾配クリッピング（必須）
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 2. 複素勾配の安全性機構を確認
from src.models.phase2.gradient_safety import check_gradient_safety

# 学習ループ内で
loss.backward()
check_gradient_safety(model)  # NaN/Infを自動修正
optimizer.step()

# 3. 学習率を調整
optimizer = AdamW(model.parameters(), lr=5e-5)  # デフォルト: 1e-4

# 4. ウォームアップを使用
from torch.optim.lr_scheduler import LinearLR

scheduler = LinearLR(
    optimizer,
    start_factor=0.1,
    total_iters=1000  # 1000ステップでウォームアップ
)
```

**勾配の監視**:
```python
# 各層の勾配ノルムを確認
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if grad_norm < 1e-6 or grad_norm > 1e3:
            print(f"{name}: grad_norm = {grad_norm:.3e}")
```

#### 6. 数値精度の問題

**症状**:
```
# BK-Core Tritonカーネルの誤差が大きい
Numerical error: 5.2e-3 (> 1e-4)
```

**原因**:
- 複素数演算の精度不足
- Tritonカーネルのバグ

**解決方法**:
```python
# 1. 数値精度検証スクリプトを実行
python scripts/verify_triton_correctness.py

# 2. PyTorch実装にフォールバック
config = Phase2Config(
    use_triton_bk=False
)

# 3. float64を使用（デバッグ用）
model = model.double()
input_ids = input_ids.long()
```

#### 7. SNRフィルタが機能しない

**症状**:
```
# すべての記憶が保持される
Low SNR ratio: 0.0%
```

**原因**:
- SNR閾値が低すぎる
- ノイズ推定が不正確

**解決方法**:
```python
# 1. SNR閾値を調整
config = Phase2Config(
    snr_threshold=3.0  # デフォルト: 2.0
)

# 2. SNR統計を確認
logits, diagnostics = model(input_ids, return_diagnostics=True)
snr_stats = diagnostics['snr_stats'][0]
print(f"Mean SNR: {snr_stats['mean_snr']:.2f}")
print(f"Min SNR: {snr_stats['min_snr']:.2f}")
print(f"Max SNR: {snr_stats['max_snr']:.2f}")

# 3. Gamma/Eta boostを調整
config = Phase2Config(
    snr_gamma_boost=3.0,  # デフォルト: 2.0
    snr_eta_boost=2.0     # デフォルト: 1.5
)
```

### デバッグ方法

#### 1. 診断情報の活用

```python
# 詳細な診断情報を取得
logits, diagnostics = model(input_ids, return_diagnostics=True)

# 各層の情報を確認
for layer_idx in range(len(diagnostics['layer_outputs'])):
    print(f"\n=== Layer {layer_idx} ===")
    
    # Gamma値
    gamma = diagnostics['gamma_values'][layer_idx]
    print(f"Gamma: mean={gamma.mean():.4f}, std={gamma.std():.4f}")
    
    # SNR統計
    snr = diagnostics['snr_stats'][layer_idx]
    print(f"SNR: mean={snr['mean_snr']:.2f}, low_ratio={snr['low_snr_ratio']:.2%}")
    
    # 共鳴情報
    resonance = diagnostics['resonance_info'][layer_idx]
    print(f"Resonance: modes={resonance['num_resonant']:.1f}, energy={resonance['total_energy']:.3f}")
    
    # 安定性
    stability = diagnostics['stability_metrics'][layer_idx]
    print(f"Stability: is_stable={stability['is_stable']}, dE/dt={stability['dE_dt']:.3e}")
```

#### 2. 可視化

```python
# 学習曲線と診断情報の可視化
python scripts/visualize_phase2.py --checkpoint checkpoints/phase2_epoch10.pt

# Gamma値の時間変化
python scripts/visualize_phase2.py --mode gamma --checkpoint checkpoints/

# SNR分布
python scripts/visualize_phase2.py --mode snr --checkpoint checkpoints/

# 共鳴エネルギー
python scripts/visualize_phase2.py --mode resonance --checkpoint checkpoints/
```

#### 3. ユニットテストの実行

```python
# すべてのPhase 2テストを実行
pytest tests/test_phase2_*.py -v

# 特定のモジュールのテスト
pytest tests/test_non_hermitian.py -v
pytest tests/test_dissipative_hebbian.py -v
pytest tests/test_memory_resonance.py -v

# 統合テスト
pytest tests/test_phase2_integrated.py -v
```

#### 4. ベンチマーク

```python
# BK-Core Tritonカーネルのベンチマーク
python scripts/benchmark_bk_triton.py

# 長期依存関係テスト
python scripts/test_long_context.py --seq_len 4096

# メモリ使用量の測定
python scripts/test_long_context.py --measure_memory
```


---

## パフォーマンスチューニング

### 1. Tritonカーネルの最適化

```python
# ブロックサイズの調整
config = Phase2Config(
    triton_block_size=256  # デフォルト: 256
    # 試す値: 128, 256, 512
)

# ベンチマークで最適値を探索
for block_size in [128, 256, 512]:
    config.triton_block_size = block_size
    # ベンチマーク実行
```

### 2. Fast Weightsの最適化

```python
# ヘッド数とヘッド次元のバランス
# メモリ: O(H * D_h^2)
# 計算: O(N * H * D_h^2)

# オプション1: 多ヘッド・小次元（並列性重視）
config = Phase2Config(
    num_heads=16,
    head_dim=32
)

# オプション2: 少ヘッド・大次元（容量重視）
config = Phase2Config(
    num_heads=4,
    head_dim=128
)

# 推奨: バランス型
config = Phase2Config(
    num_heads=8,
    head_dim=64
)
```

### 3. Memory Resonanceの高速化

```python
# 共鳴層を選択的に適用
# すべての層ではなく、特定の層のみで使用

class SelectiveResonanceModel(Phase2IntegratedModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 最後の2層のみ共鳴を有効化
        for i, block in enumerate(self.blocks):
            if i < len(self.blocks) - 2:
                block.resonance = None  # 無効化
```

### 4. 混合精度学習

```python
from torch.cuda.amp import autocast, GradScaler

# GradScalerの初期化
scaler = GradScaler()

# 学習ループ
for batch in dataloader:
    optimizer.zero_grad()
    
    # 混合精度でForward
    with autocast():
        logits = model(input_ids)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
    
    # スケールされた勾配でBackward
    scaler.scale(loss).backward()
    
    # 勾配クリッピング
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # オプティマイザーステップ
    scaler.step(optimizer)
    scaler.update()
```

### 5. 勾配チェックポイント

```python
# メモリ削減（速度とのトレードオフ）
model.gradient_checkpointing_enable()

# または、特定の層のみ
for i, block in enumerate(model.blocks):
    if i % 2 == 0:  # 偶数層のみ
        block.gradient_checkpointing = True
```

### 6. データローダーの最適化

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=4,
    num_workers=4,      # 並列データロード
    pin_memory=True,    # GPU転送の高速化
    prefetch_factor=2   # プリフェッチ
)
```

### 7. コンパイル最適化（PyTorch 2.0+）

```python
# torch.compileで最適化
model = torch.compile(model, mode='max-autotune')

# または、特定のモードで
model = torch.compile(model, mode='reduce-overhead')
```

---

## FAQ

### Q1: Phase 1とPhase 2の主な違いは何ですか？

**A**: Phase 1は静的な物理ベースO(N)アーキテクチャで、Phase 2は動的な記憶機構を追加します。

| 特徴 | Phase 1 | Phase 2 |
|------|---------|---------|
| 時間発展 | 静的 | 動的（散逸的Hebbian） |
| 記憶 | なし | Fast Weights（短期記憶） |
| 忘却 | なし | Non-Hermitian（自然な散逸） |
| 計算量 | O(N) | O(N·H·D² + H·D³) |

### Q2: Tritonカーネルは必須ですか？

**A**: いいえ、オプションです。Tritonが利用できない場合、自動的にPyTorch実装にフォールバックします。ただし、Tritonを使用すると3倍以上の高速化が期待できます。

```python
# Triton無効化
config = Phase2Config(use_triton_bk=False)
```

### Q3: VRAM 8GB制約を守るには？

**A**: 以下のパラメータを調整してください：

```python
config = Phase2Config(
    d_model=512,        # モデル次元
    n_seq=1024,         # シーケンス長
    num_heads=8,        # ヘッド数
    head_dim=64,        # ヘッド次元
    resonance_enabled=True  # 必要に応じて無効化
)

# バッチサイズも重要
batch_size = 2  # または 4
```

### Q4: Phase 1モデルをPhase 2に変換できますか？

**A**: はい、`convert_phase1_to_phase2()`関数を使用してください。

```python
from src.models.phase2 import convert_phase1_to_phase2

phase2_model = convert_phase1_to_phase2(phase1_model, phase2_config)
```

Phase 1の重みは保持され、Phase 2固有の層は新規初期化されます。

### Q5: Lyapunov安定性とは何ですか？

**A**: Fast Weightsのエネルギー `E = ||W||²` が時間とともに発散しないことを保証する数学的条件です。

```
dE/dt ≤ 0
```

この条件が満たされない場合、システムは自動的にΓ（減衰率）を増加させて安定化します。

### Q6: SNR閾値はどう設定すべきですか？

**A**: デフォルトの2.0から始めて、診断情報を見ながら調整してください。

```python
# SNR統計を確認
logits, diagnostics = model(input_ids, return_diagnostics=True)
snr_stats = diagnostics['snr_stats'][0]

# 平均SNRが閾値より大きい場合、閾値を上げる
if snr_stats['mean_snr'] > 5.0:
    config.snr_threshold = 3.0
```

### Q7: Memory Resonanceは常に有効にすべきですか？

**A**: タスクによります。長期依存関係が重要なタスクでは有効ですが、計算コスト（O(D³)）がかかります。

```python
# 長期依存タスク（推奨）
config = Phase2Config(resonance_enabled=True)

# 短期タスク or VRAM制約が厳しい場合
config = Phase2Config(resonance_enabled=False)
```

### Q8: ゼータ初期化の効果は？

**A**: 情報の干渉を最小化し、学習の初期段階から効率的な表現を獲得します。特に長期依存関係の学習が改善されます。

```python
# ゼータ初期化を使用（推奨）
config = Phase2Config(use_zeta_init=True)

# 標準初期化
config = Phase2Config(use_zeta_init=False)
```

### Q9: 複素勾配の安全性は自動的に保証されますか？

**A**: はい、モデルは自動的に勾配のクリッピングとNaN/Inf処理を行います。ただし、学習ループでも勾配クリッピングを適用することを推奨します。

```python
# 学習ループ内で
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### Q10: Phase 2の学習は Phase 1より遅いですか？

**A**: Fast Weightsの更新により、約1.5〜2倍の計算時間がかかります。ただし、Tritonカーネルの高速化により、Phase 1比で実質的な速度低下は最小限に抑えられます。

| 構成 | 相対速度 |
|------|---------|
| Phase 1 (PyTorch) | 1.0x |
| Phase 1 (Triton) | 3.0x |
| Phase 2 (Triton) | 2.0x |

---

## まとめ

Phase 2「生命の息吹」は、Project MUSEに動的な時間発展と記憶機構を導入します。物理法則に基づく設計により、安定性と効率性を両立しながら、生物の神経系のような適応的な情報処理を実現します。

**核心的な革新**:
1. **散逸的Hebbian方程式**: 記憶形成と忘却の統一
2. **Lyapunov安定性保証**: 数学的に保証された安定性
3. **SNRベースの選択**: 重要記憶の自動選択
4. **ゼータ共鳴**: フラクタル記憶配置による干渉最小化

**次のステップ**:
1. `examples/phase2_*.py` で使用例を確認
2. `scripts/train_phase2.py` で学習を開始
3. `scripts/visualize_phase2.py` で診断情報を可視化
4. Phase 3「感情の芽生え」への準備

**参考資料**:
- 設計書: `.kiro/specs/phase2-breath-of-life/design.md`
- 要件定義: `.kiro/specs/phase2-breath-of-life/requirements.md`
- タスクリスト: `.kiro/specs/phase2-breath-of-life/tasks.md`
- 実装例: `examples/phase2_*.py`

Phase 2の完成により、MUSEは静的な関数から動的な生命体へと進化します。
