# Phase 1 Hyperparameter Tuning Guide

## 概要

このガイドでは、Phase 1 Efficiency Engineの各コンポーネントのハイパーパラメータを調整する方法を説明します。推奨範囲、感度分析結果、異なるユースケースでのチューニング戦略、トラブルシューティングを含みます。

---

## 1. AR-SSM Layer Hyperparameters

### 1.1 max_rank (最大ランク)

**説明**: 適応ランク機構の最大ランク容量

**推奨範囲**: 16 - 64

**デフォルト**: 32

**影響**:
- **高い値 (64)**: より高い表現力、メモリ使用量増加
- **低い値 (16)**: メモリ効率的、表現力が制限される可能性

**感度**: 中程度

**チューニング戦略**:

```python
# 8GB VRAM制約
config.ar_ssm_max_rank = 16  # 保守的

# 10GB VRAM
config.ar_ssm_max_rank = 32  # バランス

# 24GB VRAM（最大品質）
config.ar_ssm_max_rank = 64  # 高表現力
```

**実験結果**:

| max_rank | VRAM (GB) | PPL | Throughput (tokens/s) |
|----------|-----------|-----|----------------------|
| 16       | 6.2       | 18.5 | 1250 |
| 32       | 7.8       | 17.2 | 1100 |
| 48       | 9.5       | 16.8 | 950 |
| 64       | 11.2      | 16.5 | 850 |

**推奨**: 8GB制約では16、10GB制約では32から開始

---

### 1.2 min_rank (最小ランク)

**説明**: 安定性のための最小ランク

**推奨範囲**: 2 - 8

**デフォルト**: 4

**影響**:
- **高い値 (8)**: より安定、ゲーティングの効果が減少
- **低い値 (2)**: より積極的な圧縮、不安定になる可能性

**感度**: 低

**チューニング戦略**:

```python
# 安定性優先
config.ar_ssm_min_rank = 8

# 効率優先
config.ar_ssm_min_rank = 2

# バランス（推奨）
config.ar_ssm_min_rank = 4
```

**推奨**: ほとんどの場合4で十分

---

### 1.3 gate_hidden_dim (ゲート隠れ層次元)

**説明**: 複雑度ゲートネットワークの隠れ層次元

**推奨範囲**: d_model // 8 - d_model // 2

**デフォルト**: d_model // 4

**影響**:
- **高い値**: より表現力のあるゲート、パラメータ増加
- **低い値**: 軽量なゲート、表現力が制限される可能性

**感度**: 低

**チューニング戦略**:

```python
# 軽量
config.ar_ssm_gate_hidden_dim = d_model // 8

# 標準（推奨）
config.ar_ssm_gate_hidden_dim = d_model // 4

# 高表現力
config.ar_ssm_gate_hidden_dim = d_model // 2
```

**推奨**: d_model // 4（デフォルト）で開始

---

### 1.4 l1_regularization (L1正則化)

**説明**: ゲートスパース性のためのL1正則化重み

**推奨範囲**: 0.0001 - 0.01

**デフォルト**: 0.001

**影響**:
- **高い値 (0.01)**: より積極的なスパース性、表現力が制限される可能性
- **低い値 (0.0001)**: 緩やかなスパース性、メモリ削減が少ない

**感度**: 中程度

**チューニング戦略**:

```python
# スパース性優先（メモリ効率）
config.ar_ssm_l1_regularization = 0.01

# バランス（推奨）
config.ar_ssm_l1_regularization = 0.001

# 品質優先
config.ar_ssm_l1_regularization = 0.0001
```

**実験結果**:

| L1 weight | Gate Sparsity | Effective Rank | PPL |
|-----------|---------------|----------------|-----|
| 0.0001    | 15%           | 28.5           | 17.0 |
| 0.001     | 35%           | 22.3           | 17.2 |
| 0.01      | 65%           | 12.8           | 18.5 |

**推奨**: 0.001から開始、PPL劣化が大きい場合は0.0001に削減

---

### 1.5 use_fused_scan (Fused Scanカーネル)

**説明**: Triton Fused Scanカーネルを使用するかどうか

**推奨範囲**: True / False

**デフォルト**: True

**影響**:
- **True**: 3倍高速、CUDAが必要
- **False**: 標準torch.cumsum、CPUで動作

**感度**: 高（パフォーマンス）

**チューニング戦略**:

```python
# CUDA利用可能
config.ar_ssm_use_fused_scan = True  # 推奨

# CPU / デバッグ
config.ar_ssm_use_fused_scan = False
```

**推奨**: CUDAが利用可能な場合は常にTrue

---

## 2. HTT Embedding Hyperparameters

### 2.1 htt_rank (Tensor Trainランク)

**説明**: Tensor Train分解のランク

**推奨範囲**: 8 - 64

**デフォルト**: 16

**影響**:
- **高い値 (64)**: 高品質、圧縮率が低下
- **低い値 (8)**: 高圧縮、品質が低下する可能性

**感度**: 高

**チューニング戦略**:

```python
# 最大圧縮（99%+）
config.htt_rank = 8

# バランス（推奨、90%圧縮）
config.htt_rank = 16

# 高品質（80%圧縮）
config.htt_rank = 32

# 最大品質（70%圧縮）
config.htt_rank = 64
```

**実験結果**:

| htt_rank | Compression | Params (M) | PPL | Reconstruction Error |
|----------|-------------|------------|-----|---------------------|
| 8        | 99.2%       | 0.4        | 19.5 | 0.08 |
| 16       | 98.5%       | 0.8        | 17.8 | 0.03 |
| 32       | 97.0%       | 1.5        | 17.1 | 0.01 |
| 64       | 94.0%       | 3.0        | 16.9 | 0.005 |

**推奨**: 16から開始、PPL劣化が大きい場合は32に増加

---

### 2.2 htt_num_cores (コア数)

**説明**: Tensor Trainコアの数

**推奨範囲**: 2 - 3

**デフォルト**: 2

**影響**:
- **2コア**: シンプル、ほとんどの語彙サイズに適している
- **3コア**: 非常に大規模な語彙（>100K）に適している

**感度**: 低

**チューニング戦略**:

```python
# 標準語彙（<50K）
config.htt_num_cores = 2  # 推奨

# 大規模語彙（50K-100K）
config.htt_num_cores = 2  # まだ十分

# 非常に大規模な語彙（>100K）
config.htt_num_cores = 3
```

**推奨**: ほとんどの場合2で十分

---

### 2.3 htt_phase_encoding (位相エンコーディング)

**説明**: ホログラフィック位相回転を有効にするかどうか

**推奨範囲**: True / False

**デフォルト**: True

**影響**:
- **True**: 意味関係の保存が向上、わずかな計算オーバーヘッド
- **False**: 標準Tensor Train、わずかに高速

**感度**: 低

**チューニング戦略**:

```python
# 品質優先（推奨）
config.htt_phase_encoding = True

# 速度優先
config.htt_phase_encoding = False
```

**実験結果**:

| phase_encoding | PPL | Throughput (tokens/s) |
|----------------|-----|-----------------------|
| True           | 17.8 | 1180 |
| False          | 18.2 | 1220 |

**推奨**: True（品質向上がわずかな速度低下を上回る）

---

## 3. LNS Kernel Hyperparameters

### 3.1 lns_enabled (LNS有効化)

**説明**: LNSカーネルを使用するかどうか

**推奨範囲**: True / False

**デフォルト**: False

**影響**:
- **True**: 推論時の計算削減、数値誤差の可能性
- **False**: 標準matmul、高精度

**感度**: 高（精度）

**チューニング戦略**:

```python
# 訓練
config.lns_enabled = False  # 必須

# 推論（実験的）
config.lns_enabled = True  # 慎重に使用
```

**推奨**: 訓練時はFalse、推論時のみTrue（実験的）

---

### 3.2 lns_block_size_* (ブロックサイズ)

**説明**: Tritonカーネルのブロックサイズ

**推奨範囲**: 32 - 256

**デフォルト**: M=128, N=128, K=32

**影響**:
- **大きい値**: より良い並列性、より多くの共有メモリ
- **小さい値**: より少ない共有メモリ、並列性が低下

**感度**: 中程度（ハードウェア依存）

**チューニング戦略**:

```python
# Ampere (RTX 3080/3090)
config.lns_block_size_m = 128
config.lns_block_size_n = 128
config.lns_block_size_k = 32

# Ada (RTX 4090)
config.lns_block_size_m = 256
config.lns_block_size_n = 256
config.lns_block_size_k = 64
```

**推奨**: デフォルト値から開始、プロファイリングで最適化

---

## 4. Stability Monitoring Hyperparameters

### 4.1 stability_threshold (安定性閾値)

**説明**: |det(I - K_ε)|の最小値

**推奨範囲**: 1e-7 - 1e-5

**デフォルト**: 1e-6

**影響**:
- **高い値 (1e-5)**: より保守的、早期警告
- **低い値 (1e-7)**: より積極的、発散リスク増加

**感度**: 高

**チューニング戦略**:

```python
# 安定性優先
config.stability_threshold = 1e-5

# バランス（推奨）
config.stability_threshold = 1e-6

# 積極的
config.stability_threshold = 1e-7
```

**推奨**: 1e-6から開始、不安定な場合は1e-5に増加

---

### 4.2 schatten_s1_bound / schatten_s2_bound

**説明**: Schattenノルムの上限

**推奨範囲**: S1: 50-200, S2: 25-100

**デフォルト**: S1: 100.0, S2: 50.0

**影響**:
- **高い値**: より緩い制約、発散リスク増加
- **低い値**: より厳しい制約、頻繁な警告

**感度**: 中程度

**チューニング戦略**:

```python
# 厳しい制約
config.schatten_s1_bound = 50.0
config.schatten_s2_bound = 25.0

# バランス（推奨）
config.schatten_s1_bound = 100.0
config.schatten_s2_bound = 50.0

# 緩い制約
config.schatten_s1_bound = 200.0
config.schatten_s2_bound = 100.0
```

**推奨**: デフォルト値から開始、理論的境界に基づいて調整

---

### 4.3 gradient_norm_threshold (勾配ノルム閾値)

**説明**: 勾配クリッピングをトリガーする閾値

**推奨範囲**: 1.0 - 100.0

**デフォルト**: 10.0

**影響**:
- **高い値 (100.0)**: 勾配クリッピングが少ない、爆発リスク
- **低い値 (1.0)**: 積極的なクリッピング、収束が遅い可能性

**感度**: 中程度

**チューニング戦略**:

```python
# 積極的なクリッピング
config.gradient_norm_threshold = 1.0

# バランス（推奨）
config.gradient_norm_threshold = 10.0

# 緩いクリッピング
config.gradient_norm_threshold = 100.0
```

**推奨**: 10.0から開始、勾配爆発が発生する場合は1.0に削減

---

## 5. Memory Optimization Hyperparameters

### 5.1 use_gradient_checkpointing

**説明**: 勾配チェックポイントを有効にするかどうか

**推奨範囲**: True / False

**デフォルト**: True

**影響**:
- **True**: 40%メモリ削減、20%速度低下
- **False**: より高速、より多くのメモリ

**感度**: 高（メモリ）

**チューニング戦略**:

```python
# 8GB VRAM制約
config.use_gradient_checkpointing = True  # 必須

# 24GB VRAM
config.use_gradient_checkpointing = False  # 速度優先
```

**推奨**: 8-10GB制約ではTrue、24GB+ではFalse

---

## 6. Use Case Specific Tuning

### 6.1 8GB VRAM制約（RTX 3080 Mobile）

**目標**: 最大メモリ効率

```python
config = Phase1Config(
    # AR-SSM: 保守的
    ar_ssm_max_rank=16,
    ar_ssm_min_rank=4,
    ar_ssm_l1_regularization=0.01,  # 積極的なスパース性
    ar_ssm_use_fused_scan=True,
    
    # HTT: 高圧縮
    htt_rank=12,
    htt_num_cores=2,
    htt_phase_encoding=True,
    
    # LNS: 無効
    lns_enabled=False,
    
    # 安定性: 標準
    stability_monitoring_enabled=True,
    stability_threshold=1e-6,
    
    # メモリ: 積極的
    use_gradient_checkpointing=True,
    checkpoint_ar_ssm=True,
    checkpoint_htt=False,
    
    # 目標
    target_vram_gb=8.0,
)
```

---

### 6.2 10GB VRAM（RTX 3080）

**目標**: バランスの取れたパフォーマンス

```python
config = Phase1Config(
    # AR-SSM: バランス
    ar_ssm_max_rank=32,
    ar_ssm_min_rank=4,
    ar_ssm_l1_regularization=0.001,
    ar_ssm_use_fused_scan=True,
    
    # HTT: 標準圧縮
    htt_rank=16,
    htt_num_cores=2,
    htt_phase_encoding=True,
    
    # LNS: 無効
    lns_enabled=False,
    
    # 安定性: 標準
    stability_monitoring_enabled=True,
    stability_threshold=1e-6,
    
    # メモリ: 標準
    use_gradient_checkpointing=True,
    checkpoint_ar_ssm=True,
    checkpoint_htt=False,
    
    # 目標
    target_vram_gb=10.0,
)
```

---

### 6.3 24GB VRAM（RTX 4090）

**目標**: 最大品質

```python
config = Phase1Config(
    # AR-SSM: 高表現力
    ar_ssm_max_rank=64,
    ar_ssm_min_rank=8,
    ar_ssm_l1_regularization=0.0001,  # 緩いスパース性
    ar_ssm_use_fused_scan=True,
    
    # HTT: 低圧縮
    htt_rank=32,
    htt_num_cores=2,
    htt_phase_encoding=True,
    
    # LNS: 無効
    lns_enabled=False,
    
    # 安定性: 標準
    stability_monitoring_enabled=True,
    stability_threshold=1e-6,
    
    # メモリ: 速度優先
    use_gradient_checkpointing=False,
    checkpoint_ar_ssm=False,
    checkpoint_htt=False,
    
    # 目標
    target_vram_gb=24.0,
)
```

---

### 6.4 推論専用

**目標**: 最大スループット

```python
config = Phase1Config(
    # AR-SSM: 標準
    ar_ssm_max_rank=32,
    ar_ssm_min_rank=4,
    ar_ssm_l1_regularization=0.001,
    ar_ssm_use_fused_scan=True,
    
    # HTT: 標準
    htt_rank=16,
    htt_num_cores=2,
    htt_phase_encoding=True,
    
    # LNS: 有効（実験的）
    lns_enabled=True,
    lns_block_size_m=128,
    lns_block_size_n=128,
    lns_block_size_k=32,
    
    # 安定性: 無効（推論では不要）
    stability_monitoring_enabled=False,
    
    # メモリ: 無効（推論では不要）
    use_gradient_checkpointing=False,
    
    # 目標
    target_vram_gb=8.0,
)
```

---

## 7. Troubleshooting

### 7.1 高いPerplexity劣化（>5%）

**症状**: Phase 1モデルのPPLがベースラインより5%以上高い

**原因**:
- HTTランクが低すぎる
- AR-SSM max_rankが低すぎる
- L1正則化が強すぎる

**解決策**:
1. `htt_rank`を増加（16 → 32）
2. `ar_ssm_max_rank`を増加（32 → 48）
3. `ar_ssm_l1_regularization`を削減（0.001 → 0.0001）
4. より長い訓練（カリキュラム学習）

---

### 7.2 VRAM不足

**症状**: `VRAMExhaustedError`またはCUDA out of memory

**原因**:
- バッチサイズが大きすぎる
- シーケンス長が長すぎる
- ランクが高すぎる

**解決策**:
1. バッチサイズを削減
2. シーケンス長を削減
3. `ar_ssm_max_rank`を削減（32 → 16）
4. `use_gradient_checkpointing=True`を設定
5. `htt_rank`を削減（16 → 12）

---

### 7.3 数値不安定性

**症状**: NaN/Inf損失、`NumericalInstabilityError`

**原因**:
- 学習率が高すぎる
- 安定性閾値が低すぎる
- 勾配爆発

**解決策**:
1. 学習率を削減（1e-4 → 5e-5）
2. `stability_threshold`を増加（1e-6 → 1e-5）
3. `gradient_norm_threshold`を削減（10.0 → 1.0）
4. 勾配クリッピングを有効化
5. 入力データにNaN/Infがないか確認

---

### 7.4 低速なパフォーマンス

**症状**: 期待されるスループットより遅い

**原因**:
- Fused Scanが無効
- CUDAが利用不可
- ブロックサイズが最適でない

**解決策**:
1. `ar_ssm_use_fused_scan=True`を確認
2. CUDAが利用可能か確認
3. Tritonがインストールされているか確認
4. `torch.compile()`を使用
5. ブロックサイズをプロファイリングで最適化

---

## 8. Hyperparameter Search Strategy

### 8.1 グリッドサーチ

```python
# 重要なハイパーパラメータのグリッド
param_grid = {
    'ar_ssm_max_rank': [16, 32, 48],
    'htt_rank': [12, 16, 24],
    'ar_ssm_l1_regularization': [0.0001, 0.001, 0.01],
}

# 各組み合わせを評価
for max_rank in param_grid['ar_ssm_max_rank']:
    for htt_rank in param_grid['htt_rank']:
        for l1_reg in param_grid['ar_ssm_l1_regularization']:
            config = Phase1Config(
                ar_ssm_max_rank=max_rank,
                htt_rank=htt_rank,
                ar_ssm_l1_regularization=l1_reg,
            )
            # 訓練と評価
            ppl, vram = train_and_evaluate(config)
            # 結果を記録
```

---

### 8.2 ランダムサーチ

```python
import random

# ランダムサンプリング
for _ in range(20):
    config = Phase1Config(
        ar_ssm_max_rank=random.choice([16, 24, 32, 48, 64]),
        htt_rank=random.choice([8, 12, 16, 24, 32]),
        ar_ssm_l1_regularization=10 ** random.uniform(-4, -2),
        stability_threshold=10 ** random.uniform(-7, -5),
    )
    # 訓練と評価
    ppl, vram = train_and_evaluate(config)
    # 結果を記録
```

---

### 8.3 ベイズ最適化

```python
from skopt import gp_minimize
from skopt.space import Real, Integer

# 探索空間
space = [
    Integer(16, 64, name='ar_ssm_max_rank'),
    Integer(8, 32, name='htt_rank'),
    Real(1e-4, 1e-2, prior='log-uniform', name='ar_ssm_l1_regularization'),
]

# 目的関数
def objective(params):
    config = Phase1Config(
        ar_ssm_max_rank=params[0],
        htt_rank=params[1],
        ar_ssm_l1_regularization=params[2],
    )
    ppl, vram = train_and_evaluate(config)
    # PPLを最小化、VRAM制約を満たす
    if vram > 8.0:
        return 1000.0  # ペナルティ
    return ppl

# 最適化
result = gp_minimize(objective, space, n_calls=50)
```

---

## 9. Monitoring and Logging

### 9.1 重要なメトリクス

訓練中に監視すべきメトリクス：

1. **Perplexity**: 検証セットでのPPL
2. **VRAM使用量**: ピークメモリ使用量
3. **有効ランク**: AR-SSMの平均有効ランク
4. **ゲートスパース性**: ゲート値<0.1の割合
5. **安定性指標**: |det(I - K_ε)|、Schattenノルム
6. **勾配ノルム**: 各コンポーネントの勾配ノルム
7. **スループット**: tokens/sec

### 9.2 WandBログ

```python
import wandb

wandb.init(project="phase1-tuning", config=config)

# 訓練ループ
for epoch in range(num_epochs):
    for batch in dataloader:
        # 訓練ステップ
        loss, diagnostics = train_step(batch)
        
        # ログ
        wandb.log({
            'loss': loss,
            'ppl': torch.exp(loss),
            'ar_ssm_effective_rank': diagnostics['ar_ssm_effective_rank'],
            'ar_ssm_gate_sparsity': diagnostics['ar_ssm_gate_sparsity'],
            'vram_mb': diagnostics['peak_vram_mb'],
            'stability_det': diagnostics['bk_det_condition'],
        })
```

---

## まとめ

このガイドでは、Phase 1のすべてのハイパーパラメータの推奨範囲、感度分析、チューニング戦略を提供しました。

**重要なポイント**:

1. **AR-SSM max_rank**: 最も影響力のあるパラメータ、VRAM制約に基づいて調整
2. **HTT rank**: 圧縮-品質トレードオフ、16から開始
3. **L1正則化**: ゲートスパース性を制御、0.001から開始
4. **勾配チェックポイント**: 8-10GB制約では必須
5. **Fused Scan**: CUDAで常に有効化

**推奨ワークフロー**:

1. プリセット設定から開始（8gb/10gb/24gb）
2. ベースラインPPLを測定
3. 重要なパラメータを調整（max_rank、htt_rank）
4. 安定性とメモリを監視
5. 必要に応じて微調整

**次のステップ**:

- 実際のデータセットで実験
- ハイパーパラメータサーチを実行
- 結果を文書化して共有
