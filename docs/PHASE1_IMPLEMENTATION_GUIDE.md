# Phase 1 Efficiency Engine - Implementation Guide

## 概要 (Overview)

Phase 1: Efficiency Engineは、Project MUSEの物理ベースO(N)言語モデルを家庭用GPU（8-10GB VRAM）で動作可能にする3つの革新的アルゴリズムの実装です。

### 設計目標

1. **メモリ効率**: HTT圧縮と適応ランク機構により90%のVRAM削減
2. **計算効率**: Fused Tritonカーネルにより3倍の高速化とO(N)複雑度
3. **数学的安定性**: Birman-Schwinger作用素の安定性維持 (|det(I - K_ε)| > δ)
4. **品質保持**: FP16ベースラインと比較してPerplexity劣化を5%以下に抑制
5. **Phase 2対応**: 将来の複素数値非エルミート演算への拡張性

### コアコンポーネント

1. **Adaptive Rank Semiseparable Layer (AR-SSM)** - 入力複雑度に基づく動的ランク調整
2. **Holographic Tensor Train Embedding (HTT)** - 位相エンコーディングによる90%パラメータ圧縮
3. **Logarithmic Number System Kernel (LNS)** - 乗算から加算への変換による計算効率化

---

## 1. Adaptive Rank Semiseparable Layer (AR-SSM)

### 数学的基礎

AR-SSM層は、適応ランクを持つ半可分行列構造 H = T + UV^T を実装します：

```
H_eff = T + U_gated · V_gated^T

where:
  T: 三重対角行列（局所相互作用） - O(N) ストレージ
  U, V: 低ランク因子（大域相互作用） - O(N·r) ストレージ
  r: 適応ランク ∈ [r_min, r_max]、位置ごとに動的調整
```

**物理的直観:**
- 単純なトークン（低エントロピー） → 低ランク → 最小限の計算
- 複雑な文脈（高エントロピー/乱流） → 高ランク → 完全な計算能力

### 実装構造

```python
from src.models.phase1 import AdaptiveRankSemiseparableLayer

# 基本的な使用法
ar_ssm = AdaptiveRankSemiseparableLayer(
    d_model=512,        # モデル次元
    max_rank=32,        # 最大ランク容量
    min_rank=4,         # 安定性のための最小ランク
    gate_hidden_dim=128,  # 複雑度ゲートの隠れ層次元
    l1_regularization=0.001,  # ゲートスパース性のためのL1正則化
    use_fused_scan=True,  # Tritonカーネルを使用
)

# 順伝播
x = torch.randn(batch_size, seq_len, d_model)
output = ar_ssm(x)  # shape: (batch_size, seq_len, d_model)
```


### コンポーネント詳細

#### 1. 複雑度ゲートネットワーク

```python
# 内部構造
complexity_gate = nn.Sequential(
    nn.Linear(d_model, gate_hidden_dim),
    nn.ReLU(),
    nn.Linear(gate_hidden_dim, max_rank),
    nn.Sigmoid()  # [0, 1]範囲の出力
)
```

ゲートは各位置の複雑度を推定し、ランク次元ごとにソフトゲート値を出力します。

#### 2. 低ランク射影

```python
# U, V射影
U_proj = nn.Linear(d_model, max_rank)
V_proj = nn.Linear(d_model, max_rank)

# ゲーティング適用
gates = complexity_gate(x)  # (B, L, max_rank)
u = U_proj(x) * gates  # 要素ごとの乗算
v = V_proj(x) * gates
```

#### 3. 局所相互作用（T成分）

```python
# 深さ方向畳み込み
T_conv = nn.Conv1d(
    d_model, d_model,
    kernel_size=3,
    padding=1,
    groups=d_model  # 効率的な深さ方向畳み込み
)
```

#### 4. Fused Associative Scan

```python
from src.kernels import fused_associative_scan

# 標準PyTorch（遅い）
# k_cumsum = torch.cumsum(u_gated, dim=1)

# Phase 1: Fused Tritonカーネル（3倍高速）
k_cumsum = fused_associative_scan(u_gated)
```

### メモリ複雑度解析

```
標準Attention: O(N²) メモリ
既存SemiseparableMatrix: O(N log N) メモリ
AR-SSM with Gating: O(N log N) + O(N·r_eff) where r_eff < r_max

期待される削減:
- 標準Attentionと比較して70%削減（既存）
- 適応ランク削減により追加で20-40%削減
- 合計: 80-85%のメモリ削減
```

### 既存SemiseparableMatrixとの統合

```python
from src.models import SemiseparableMatrix

# 既存: 静的ランク r = ⌈log₂(N)⌉
semisep = SemiseparableMatrix(n_seq=N, rank=static_rank)

# Phase 1: ゲーティング付き適応ランク
ar_ssm = AdaptiveRankSemiseparableLayer(
    d_model=D,
    max_rank=32,
    min_rank=4,
    base_semisep=semisep  # 既存の三重対角構造を再利用
)
```

### 参考文献

- Eidelman & Gohberg (1999): "Fast inversion algorithms for diagonal plus semiseparable matrices"
- Vandebril et al. (2008): "Matrix Computations and Semiseparable Matrices"
- Gu & Dao (2023): "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"

---

## 2. Holographic Tensor Train Embedding (HTT)

### 数学的基礎

HTTは埋め込み行列 E ∈ R^(V×D) をTensor Trainコアに分解します：

```
E[i, :] = Contract(Core1[i₁], Core2[i₂], ..., CoreK[iₖ])

where:
  i = i₁·V₂·...·Vₖ + i₂·V₃·...·Vₖ + ... + iₖ  (インデックス分解)
  V = V₁ × V₂ × ... × Vₖ  (語彙の因数分解)
  D = D₁ × D₂ × ... × Dₖ  (次元の因数分解)

ホログラフィック拡張:
  ランク次元での位相回転: Core1_mod = Core1 · exp(iθ)
  干渉パターンによる意味関係の保存
```


### パラメータ数比較

```
標準Embedding: V × D パラメータ
  例: 50,000 × 1,024 = 51.2M params

Tensor Train (2コア、ランクr=16):
  Core1: V₁ × 1 × r × D₁ = 224 × 1 × 16 × 32 = 114,688
  Core2: V₂ × r × 1 × D₂ = 224 × 16 × 1 × 32 = 114,688
  合計: ~230K params (99.5%圧縮!)

実用的（ランクr=16）:
  圧縮率: 0.004 (99.6%削減)
  品質: PPL劣化 < 2% (実験的)
```

### 実装構造

```python
from src.models.phase1 import HolographicTTEmbedding

# 基本的な使用法
htt_embedding = HolographicTTEmbedding(
    vocab_size=50000,
    d_model=1024,
    rank=16,  # 圧縮-品質トレードオフの設定可能パラメータ
    num_cores=2,  # シンプルさのため2コア、大規模語彙には3+
    phase_encoding=True,  # ホログラフィック位相回転を有効化
)

# 順伝播
input_ids = torch.randint(0, 50000, (batch_size, seq_len))
embeddings = htt_embedding(input_ids)  # shape: (batch_size, seq_len, d_model)
```

### コンポーネント詳細

#### 1. インデックス分解

```python
# トークンID → (idx1, idx2, ..., idxK)
# 平方根分解を使用してバランスの取れたコアを実現
V1 = int(math.sqrt(vocab_size))
V2 = (vocab_size + V1 - 1) // V1

idx1 = token_ids // V2
idx2 = token_ids % V2

# 境界条件の処理
idx1 = torch.clamp(idx1, 0, V1 - 1)
idx2 = torch.clamp(idx2, 0, V2 - 1)
```

#### 2. Tensor Trainコア

```python
# コア初期化
core1 = nn.Parameter(torch.randn(V1, 1, rank, D1) * 0.02)
core2 = nn.Parameter(torch.randn(V2, rank, 1, D2) * 0.02)

# 位相パラメータ（学習可能）
phase_shift = nn.Parameter(torch.randn(rank))
```

#### 3. ホログラフィック収縮

```python
# コアの収集
c1 = core1[idx1]  # (B, L, 1, rank, D1)
c2 = core2[idx2]  # (B, L, rank, 1, D2)

# 位相回転の適用（実数値近似）
phase_mod = torch.cos(phase_shift)  # (rank,)
c1_mod = c1 * phase_mod.view(1, 1, 1, rank, 1)

# Einsum収縮
output = torch.einsum('blird,blrjf->blidjf', c1_mod, c2)
output = output.reshape(batch_size, seq_len, D1 * D2)

# 正確なd_modelサイズにクロップ
output = output[:, :, :d_model]
```

### 既存モデルへの統合

```python
# 前（標準）
self.token_embedding = nn.Embedding(vocab_size, d_model)

# 後（Phase 1 HTT）
self.token_embedding = HolographicTTEmbedding(
    vocab_size=vocab_size,
    d_model=d_model,
    rank=16,
    num_cores=2
)
```

### 参考文献

- Oseledets (2011): "Tensor-Train Decomposition"
- Novikov et al. (2015): "Tensorizing Neural Networks"
- Plate (1995): "Holographic Reduced Representations"

---

## 3. Logarithmic Number System (LNS) Kernel

### 数学的基礎

LNSは乗算を対数領域での加算に変換します：

```
標準: c = a × b  (高コストなFMA演算)
LNS: log(c) = log(a) + log(b)  (安価な加算)

行列乗算:
  C = A @ B = Σₖ A[i,k] × B[k,j]
  
LNS近似（Max-Log）:
  log(C[i,j]) ≈ maxₖ(log(A[i,k]) + log(B[k,j]))
```

**物理的直観:**
- 乗算器(FMA) → 加算器(ADD)への変換
- 消費電力: FMAはADDの約3-5倍
- スループット: ADDはFMAより2倍高速


### 実装構造

```python
from src.kernels import lns_matmul
from src.models.phase1 import LNSLinear

# 低レベルカーネル使用
a = torch.randn(M, K, device='cuda')
b = torch.randn(K, N, device='cuda')
c = lns_matmul(a, b)  # Tritonカーネル

# 高レベル線形層
lns_layer = LNSLinear(
    in_features=512,
    out_features=2048,
    use_lns=True  # 推論時のみLNSを使用
)

# 推論モード
lns_layer.eval()
output = lns_layer(x)  # LNSカーネルを使用

# 訓練モード
lns_layer.train()
output = lns_layer(x)  # 標準matmulにフォールバック
```

### 数値的考慮事項

Max-Log近似はバイアスを導入しますが、以下を維持します：
- **単調性**: arg maxが保存される
- **スパース性**: 支配的な項が強調される
- **安定性**: 指数オーバーフローなし

### 使用ケース

LNSカーネルは**オプション**であり、選択的に使用されます：

1. **推論専用モデル**: 重みを一度対数領域に変換
2. **大規模行列乗算**: FMAがボトルネックとなる場合
3. **実験的対数領域訓練**: 研究目的

### 制限事項

- **訓練**: 勾配の問題により主に推論用
- **精度**: Max-Log近似により数値誤差が発生
- **適用範囲**: 大規模matmul（>1024×1024）でのみ有効

### 参考文献

- Arnold et al. (2011): "Logarithmic Number Systems for Neural Network Inference"
- Coleman et al. (2008): "Arithmetic on the European Logarithmic Microprocessor"

---

## 4. Fused Associative Scan Kernel

### 数学的基礎

結合的スキャン（プレフィックス和）はO(N)シーケンス処理の核心です：

```
入力: x = [x₁, x₂, x₃, ..., xₙ]
出力: y = [x₁, x₁+x₂, x₁+x₂+x₃, ..., Σxᵢ]

逐次的: O(N) 時間、O(1) 空間
並列: O(log N) 深さ、O(N) 作業量

結合性:
  (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)
  where ⊕ can be +, ×, max, etc.
```

**物理的直観:**
- 半可分層の計算は「前のトークンからの情報を線形に累積」
- GPUレジスタ内で完結させることで、メモリ帯域幅を最小化
- グローバルメモリアクセス: O(N) → O(N/log N)

### 実装構造

```python
from src.kernels import fused_associative_scan

# 基本的な使用法
x = torch.randn(batch_size, seq_len, d_model, device='cuda')
cumsum = fused_associative_scan(x, dim=1)  # 次元1に沿ってスキャン

# AR-SSM層での統合
class AdaptiveRankSemiseparableLayer(nn.Module):
    def forward(self, x):
        # ... (複雑度ゲート、U/V射影)
        
        # 標準PyTorch（遅い）
        # k_cumsum = torch.cumsum(u_gated, dim=1)
        
        # Phase 1: Fused Tritonカーネル（3倍高速）
        k_cumsum = fused_associative_scan(u_gated, dim=1)
        
        # ... (残りの順伝播)
```

### パフォーマンス目標

```
シーケンス長 | torch.cumsum | Fused Scan | 高速化
------------|--------------|------------|--------
512         | 0.12 ms      | 0.05 ms    | 2.4x
1024        | 0.25 ms      | 0.08 ms    | 3.1x
2048        | 0.51 ms      | 0.15 ms    | 3.4x
4096        | 1.05 ms      | 0.30 ms    | 3.5x
8192        | 2.15 ms      | 0.62 ms    | 3.5x
```

### アルゴリズム（Blelloch Scan）

1. **Up-sweep phase**: 削減ツリーの構築（O(log N)深さ）
2. **Down-sweep phase**: 部分和の伝播（O(log N)深さ）
3. **合計**: O(N)作業量、O(log N)深さ

### 参考文献

- Blelloch (1990): "Prefix Sums and Their Applications"
- Harris et al. (2007): "Parallel Prefix Sum (Scan) with CUDA"

---

## 5. Birman-Schwinger Stability Monitor

### 数学的基礎

Birman-Schwinger作用素 K_ε(z) は安定性条件を満たす必要があります：

```
K_ε(z) = |V_ε|^(1/2) · R₀(z) · |V_ε|^(1/2)

安定性条件:
  |det(I - K_ε)| > δ > 0  (δ = 安定性マージン)

det(I - K_ε) → 0 のとき:
  - レゾルベント (H - zI)^(-1) が特異になる
  - 勾配が爆発する
  - モデルが発散する

Schatten ノルム境界（理論から）:
  ||K_ε||_S1 ≤ C₁ · ||V||_L1 / ε
  ||K_ε||_S2 ≤ C₂ · ||V||_L2 / √ε
```

**物理的直観:**
- det(I - K_ε) は「系の安定性マージン」
- ゼロに近づくと、物理系が特異点（共鳴）に達する
- 学習中に監視し、発散を事前に防ぐ


### 実装構造

```python
from src.models.phase1 import BKStabilityMonitor

# 初期化
stability_monitor = BKStabilityMonitor(
    stability_threshold=1e-6,  # det条件の閾値
    schatten_s1_bound=100.0,   # S1ノルムの上限
    schatten_s2_bound=50.0,    # S2ノルムの上限
    gradient_norm_threshold=10.0,  # 勾配ノルムの閾値
)

# 訓練ループでの使用
for batch in dataloader:
    # 順伝播
    output, diagnostics = model(batch)
    
    # 安定性チェック
    stability_info = stability_monitor.check_stability(
        G_ii=diagnostics['G_ii'],
        v=diagnostics['potential'],
        epsilon=model.epsilon,
    )
    
    if not stability_info['is_stable']:
        logger.warning(f"安定性違反: {stability_info['warnings']}")
        
        # アクションを実行
        if 'clip_gradients' in stability_info['actions']:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        if 'reduce_lr' in stability_info['actions']:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
    
    # 逆伝播
    loss.backward()
    optimizer.step()
```

### 監視される指標

1. **行列式条件**: |det(I - K_ε)|
2. **Schattenノルム**: ||K||_S1, ||K||_S2
3. **最小固有値**: min(eig(I - K_ε))
4. **勾配ノルム**: ||∇L||_2

### 自動リカバリアクション

- **勾配クリッピング**: 安定性閾値違反時にトリガー
- **学習率削減**: 推奨される場合
- **スペクトルクリッピング**: Schattenノルム違反時
- **すべての安定性イベントのログ記録**

### 参考文献

- Birman & Schwinger (1948): "On the Bound States of a Given Potential"
- Simon (1971): "Quantum Mechanics for Hamiltonians Defined as Quadratic Forms"
- Yafaev (1992): "Mathematical Scattering Theory"

---

## 6. 統合と設定

### Phase 1モデルファクトリ

```python
from src.models.phase1 import create_phase1_model, Phase1Config

# 設定の作成
config = Phase1Config(
    # AR-SSM設定
    ar_ssm_enabled=True,
    ar_ssm_max_rank=32,
    ar_ssm_min_rank=4,
    ar_ssm_use_fused_scan=True,
    
    # HTT Embedding設定
    htt_enabled=True,
    htt_rank=16,
    htt_num_cores=2,
    
    # LNSカーネル設定（オプション）
    lns_enabled=False,  # 実験的、推論専用
    
    # 安定性監視設定
    stability_monitoring_enabled=True,
    stability_threshold=1e-6,
    
    # メモリ最適化設定
    use_gradient_checkpointing=True,
    
    # パフォーマンス目標
    target_vram_gb=8.0,
    target_ppl_degradation=0.05,
)

# モデルの作成
model = create_phase1_model(config)
```

### プリセット設定

```python
from src.models.phase1 import get_preset_config

# 8GB VRAM用
config_8gb = get_preset_config('8gb')

# 10GB VRAM用
config_10gb = get_preset_config('10gb')

# 24GB VRAM用（最大品質）
config_24gb = get_preset_config('24gb')

# 推論専用（LNS有効）
config_inference = get_preset_config('inference')

# 最大効率（最大圧縮）
config_max_efficiency = get_preset_config('max_efficiency')
```

### 既存モデルからの変換

```python
from src.models.phase1 import convert_to_phase1

# 既存モデルの読み込み
baseline_model = torch.load('baseline_model.pt')

# Phase 1への変換
phase1_model = convert_to_phase1(
    baseline_model,
    config=config,
    preserve_weights=True,  # 可能な限り重みを保持
)

# 変換されたモデルの保存
torch.save(phase1_model, 'phase1_model.pt')
```

---

## 7. エラーハンドリングとリカバリ

### カスタム例外

Phase 1は情報豊富なエラーメッセージを提供します：

```python
from src.models.phase1.errors import (
    VRAMExhaustedError,
    NumericalInstabilityError,
    InvalidConfigError,
    HardwareCompatibilityError,
)

try:
    model = create_phase1_model(config)
    output = model(input_ids)
except VRAMExhaustedError as e:
    print(f"VRAM不足: {e.current_mb:.1f}MB / {e.limit_mb:.1f}MB")
    print("提案:")
    for suggestion in e.suggestions:
        print(f"  - {suggestion}")
except NumericalInstabilityError as e:
    print(f"数値不安定性: {e.component}")
    print(f"診断: {e.diagnostics}")
```

### 自動リカバリ

```python
from src.models.phase1 import Phase1ErrorRecovery

recovery = Phase1ErrorRecovery()

try:
    output = model(input_ids)
except VRAMExhaustedError as e:
    # 自動リカバリを試行
    if recovery.handle_vram_exhausted(e, model, config):
        print("リカバリ成功、再試行中...")
        output = model(input_ids)
    else:
        raise
```

---

## 8. パフォーマンス検証

### メモリ使用量の検証

```python
from scripts import validate_phase1_memory

# 8GB VRAM制約の検証
results = validate_phase1_memory(
    config=config,
    batch_size=4,
    seq_len=2048,
    target_vram_gb=8.0,
)

print(f"ピークVRAM: {results['peak_vram_gb']:.2f}GB")
print(f"目標内: {results['within_target']}")
```

### スループットベンチマーク

```python
from scripts import benchmark_phase1_throughput

# スループットの測定
results = benchmark_phase1_throughput(
    config=config,
    seq_lengths=[512, 1024, 2048, 4096],
    num_iterations=100,
)

for result in results:
    print(f"Seq {result['seq_len']}: {result['tokens_per_sec']:.1f} tokens/sec")
```

### Perplexity検証

```python
from scripts import validate_phase1_perplexity

# WikiText-103でのPPL測定
results = validate_phase1_perplexity(
    model=phase1_model,
    baseline_model=baseline_model,
    dataset='wikitext-103',
    max_degradation=0.05,  # 5%
)

print(f"ベースラインPPL: {results['baseline_ppl']:.2f}")
print(f"Phase 1 PPL: {results['phase1_ppl']:.2f}")
print(f"劣化: {results['degradation']:.2%}")
```

---

## 9. ベストプラクティス

### メモリ効率

1. **勾配チェックポイントを有効化**: `use_gradient_checkpointing=True`
2. **適切なバッチサイズを選択**: 8GB VRAMでは batch_size=4, seq_len=2048
3. **混合精度訓練を使用**: `torch.cuda.amp.autocast()`
4. **不要な場合はLNSを無効化**: `lns_enabled=False`

### 数値安定性

1. **安定性監視を有効化**: `stability_monitoring_enabled=True`
2. **勾配クリッピングを使用**: `torch.nn.utils.clip_grad_norm_()`
3. **適切な学習率を選択**: 1e-4から開始、必要に応じて削減
4. **ウォームアップステップを使用**: 最初の1000ステップでランクをウォームアップ

### パフォーマンス最適化

1. **Fused Scanを有効化**: `ar_ssm_use_fused_scan=True`
2. **適切なランクを選択**: max_rank=32は良い出発点
3. **L1正則化を調整**: gate_sparsityを0.001から開始
4. **torch.compileを使用**: PyTorch 2.0+で追加の高速化

---

## 10. トラブルシューティング

### 一般的な問題

#### VRAM不足

**症状**: `VRAMExhaustedError` または CUDA out of memory

**解決策**:
1. バッチサイズを削減
2. シーケンス長を削減
3. `ar_ssm_max_rank`を削減（32 → 16）
4. 勾配チェックポイントを有効化
5. LNSカーネルを無効化（有効な場合）

#### 数値不安定性

**症状**: NaN/Inf損失、`NumericalInstabilityError`

**解決策**:
1. 学習率を削減（0.5倍）
2. 勾配クリッピングを有効化（max_norm=1.0）
3. `stability_threshold`を増加（1e-6 → 1e-5）
4. 入力データにNaN/Infがないか確認

#### 低速なパフォーマンス

**症状**: 期待されるスループットより遅い

**解決策**:
1. `ar_ssm_use_fused_scan=True`を確認
2. CUDAが利用可能か確認
3. Tritonがインストールされているか確認
4. `torch.compile()`を使用
5. 適切なブロックサイズでプロファイル

#### 高いPerplexity劣化

**症状**: PPL劣化 > 5%

**解決策**:
1. `htt_rank`を増加（16 → 32）
2. `ar_ssm_max_rank`を増加（32 → 64）
3. より長い訓練（カリキュラム学習）
4. L1正則化を削減（0.001 → 0.0001）

---

## 11. 次のステップ

### Phase 2への準備

Phase 1コンポーネントは、Phase 2の複素数値演算との互換性を考慮して設計されています：

```python
# 複素数サポートの確認
from src.models.phase1 import check_complex_support

support_info = check_complex_support(model)
print(f"AR-SSM複素数対応: {support_info['ar_ssm']}")
print(f"HTT複素数対応: {support_info['htt']}")
```

### さらなる最適化

1. **カスタムTritonカーネル**: 特定のワークロード用に最適化
2. **量子化**: INT8/FP8量子化でさらなるメモリ削減
3. **スパース性**: 構造化スパース性でさらなる高速化
4. **分散訓練**: 複数GPUでのスケーリング

### コミュニティへの貢献

- バグレポート: GitHub Issues
- 機能リクエスト: GitHub Discussions
- プルリクエスト: CONTRIBUTING.mdを参照
- ベンチマーク結果の共有: results/ディレクトリ

---

## 付録A: アーキテクチャ図

```
┌─────────────────────────────────────────────────────────────────┐
│                     MUSE Language Model                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────┐                                            │
│  │ HTT Embedding    │  ← Phase 1.2: 90% compression             │
│  │ (Tensor Train)   │                                            │
│  └────────┬─────────┘                                            │
│           │ (B, L, D)                                            │
│           ▼                                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              ResNet-BK Block (Existing)                   │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │  MoE-FFN → v_proj → BK-Core (Existing)            │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│           │                                                       │
│           ▼                                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         AR-SSM Layer (Phase 1.1 - NEW)                   │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │  Complexity Gate → Adaptive Rank Gating            │  │   │
│  │  │  T-Conv (Local) + UV^T (Global) → Fused Scan      │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│           │                                                       │
│           ▼                                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         LNS Kernel Layer (Phase 1.3 - Optional)          │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │  Log-Domain MatMul (Triton) → Max-Log Accumulation│  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│           │                                                       │
│           ▼                                                       │
│  ┌──────────────────┐                                            │
│  │ Output Projection│                                            │
│  │ (HTT Compressed) │                                            │
│  └──────────────────┘                                            │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

         Stability Monitoring
         ┌────────────────────────────────┐
         │ BK Stability Monitor           │
         │ - Track |det(I - K_ε)|        │
         │ - Schatten norm bounds         │
         │ - Gradient norm monitoring     │
         └────────────────────────────────┘
```

## 付録B: 主要な数式

### AR-SSM層

```
H_eff = T + U_gated · V_gated^T

gates = σ(W₂ · ReLU(W₁ · x))
U_gated = U · gates
V_gated = V · gates

output = T_conv(x) + output_proj(cumsum(U_gated) · V_gated^T)
```

### HTT Embedding

```
E[i, :] = Σᵣ Core1[i₁, 1, r, :] · exp(iθᵣ) · Core2[i₂, r, 1, :]

i₁ = ⌊i / V₂⌋
i₂ = i mod V₂
```

### LNS Kernel

```
log(C[i,j]) ≈ maxₖ(log(|A[i,k]|) + log(|B[k,j]|))
```

### Birman-Schwinger安定性

```
|det(I - K_ε)| > δ

||K_ε||_S1 ≤ C₁ · ||V||_L1 / ε
||K_ε||_S2 ≤ C₂ · ||V||_L2 / √ε
```

---

**最終更新**: 2024年
**バージョン**: Phase 1.0
**ライセンス**: プロジェクトライセンスを参照
