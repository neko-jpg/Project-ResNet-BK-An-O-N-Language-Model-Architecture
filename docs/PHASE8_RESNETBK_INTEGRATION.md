# Phase8 ResNetBK統合ガイド

## 概要

Phase8は**Phase7（ResNetBK）の拡張**であり、独立したモデルではありません。Phase7のBK-Core（三重対角逆行列計算）を心臓部として、双曲幾何学、トポロジー、論理的含意などの高度な機能を追加したものです。

## アーキテクチャ階層

```
Phase8IntegratedModel
├── Phase7IntegratedModel (継承)
│   ├── ResNetBK (BK-Core)
│   │   └── G_ii計算 (O(N)の源泉)
│   ├── Hybrid Hyperbolic Attention
│   └── AR-SSM
└── Phase8拡張モジュール
    ├── BK-Core Hyperbolic Integration (G_iiゲーティング)
    ├── AR-SSM Hyperbolic Fusion
    ├── Entailment Cones (オプション)
    ├── Persistent Homology (オプション)
    └── Sheaf Attention (オプション)
```

## Phase7からの継承関係

### 1. BK-Core（三重対角逆行列計算）

Phase7のResNetBKが提供する**Green関数対角成分 G_ii** を物理情報として活用します。

```python
# Phase7のBK-Coreから取得
phase7_output = self.phase7_model(x)
g_ii = phase7_output["diagnostics"]["bk_core"]["g_ii"]

# Phase8でG_iiを活用
gate = self.bk_hyperbolic.compute_gate(g_ii)
```

### 2. O(N)複雑度の維持

Phase8はPhase7のO(N)複雑度を**完全に維持**します。

- **BK-Core**: O(N) 三重対角逆行列計算
- **Hyperbolic Attention**: O(N) ブロック単位距離計算
- **AR-SSM**: O(N) 連想スキャン

### 3. 8GB VRAM制約

Phase7と同様に、**RTX 3080 (10GB)** で動作することを保証します。

## Phase8固有の機能

### 1. BK-Core Hyperbolic Integration

G_iiを使った双曲空間のゲーティング機構。

```python
config = Phase8Config(
    use_bk_hyperbolic=True,  # G_iiゲーティングを有効化
    bk_gate_threshold=0.5    # ゲート閾値
)
```

**効果**:
- 物理的共鳴状態に基づく注意重みの調整
- 散乱エネルギーによる情報フロー制御

### 2. AR-SSM Hyperbolic Fusion

AR-SSMと双曲空間の融合。

```python
config = Phase8Config(
    use_ar_ssm_fusion=True,  # AR-SSM融合を有効化
    ar_ssm_rank_threshold=0.8  # ランク閾値
)
```

**効果**:
- 双曲距離を複雑度信号として活用
- 高ランク時の自動曲率増加

### 3. Entailment Cones（オプション）

論理的含意関係の幾何学的検証。

```python
config = Phase8Config(
    use_entailment_cones=True,  # 含意コーンを有効化
    entailment_aperture=0.3     # 開口角
)
```

**効果**:
- 論理的整合性の幾何学的チェック
- AND/OR演算の双曲空間実装

### 4. Persistent Homology（オプション）

トポロジカル解析と循環論理検出。

```python
config = Phase8Config(
    use_persistent_homology=True,  # トポロジー解析を有効化
    homology_threshold=0.1         # Betti数閾値
)
```

**効果**:
- 循環論理の自動検出
- 曲率調整の提案

### 5. Sheaf Attention（オプション）

マルチヘッド間の構造的整合性。

```python
config = Phase8Config(
    use_sheaf_attention=True,  # Sheaf Attentionを有効化
    sheaf_agreement_threshold=0.7  # 合意閾値
)
```

**効果**:
- ヘッド間の情報整合性チェック
- 矛盾情報のフィルタリング

## 使用例

### 基本的な使用

```python
from src.models.phase8.integrated_model import Phase8IntegratedModel
from src.models.phase8.config import Phase8Config

# Phase8設定（Phase7を継承）
config = Phase8Config(
    d_model=512,
    n_heads=8,
    n_layers=6,
    vocab_size=32000,
    max_seq_len=2048,
    # Phase7機能
    use_bk_core=True,
    use_hyperbolic=True,
    use_ar_ssm=True,
    # Phase8機能
    use_bk_hyperbolic=True,
    use_ar_ssm_fusion=True,
    use_entailment_cones=False,  # オプション
    use_persistent_homology=False,  # オプション
    use_sheaf_attention=False  # オプション
)

# モデル構築
model = Phase8IntegratedModel(config).cuda()

# Forward pass
x = torch.randn(2, 1024, 512).cuda()
output = model(x)

# 診断情報
diagnostics = output["diagnostics"]
print(f"G_ii Mean: {diagnostics['bk_core']['g_ii_real_mean']}")
print(f"Gate Mean: {diagnostics['bk_hyperbolic']['gate_mean']}")
```

### Phase7からの移行

```python
# Phase7設定
phase7_config = Phase7Config(
    d_model=512,
    n_heads=8,
    n_layers=6,
    use_bk_core=True,
    use_hyperbolic=True
)

# Phase8設定（Phase7を継承）
phase8_config = Phase8Config(
    **phase7_config.__dict__,  # Phase7設定を継承
    use_bk_hyperbolic=True,    # Phase8機能を追加
    use_ar_ssm_fusion=True
)

# Phase8モデル
model = Phase8IntegratedModel(phase8_config)
```

## 既存モジュールの再利用

Phase8の以下のモジュールは**完璧に実装済み**で、そのまま再利用できます。

| モジュール | ファイル | 状態 |
|-----------|---------|------|
| BK-Core Hyperbolic | `bk_core_hyperbolic.py` | ✅ 完璧 |
| AR-SSM Fusion | `ar_ssm_fusion.py` | ✅ 完璧 |
| Entailment Cones | `entailment_cones.py` | ✅ 完璧 |
| Persistent Homology | `persistent_homology.py` | ✅ 完璧 |
| Sheaf Attention | `sheaf_attention.py` | ✅ 完璧 |
| Quantization | `quantization.py` | ✅ 完璧 |
| Linear Attention | `linear_attention.py` | ✅ 完璧 |
| Precision Manager | `precision_manager.py` | ✅ 完璧 |
| Block Distance | `block_distance.py` | ✅ 完璧 |
| Hyperbolic SSM | `hyperbolic_ssm.py` | ✅ 完璧 |
| KV Cache | `kv_cache.py` | ✅ 完璧 |

## 重要な設計原則

### 1. Phase7の拡張であること

Phase8は**Phase7の置き換えではなく拡張**です。Phase7の全機能を継承し、その上に新機能を追加します。

### 2. BK-Coreの重要性

**BK-Core**（三重対角逆行列計算）がO(N)複雑度の源泉です。Phase8はこれを活用して物理情報を統合します。

### 3. G_iiの活用

**Green関数対角成分 G_ii** は物理的共鳴状態を表します。Phase8はこれをゲーティング信号として活用します。

### 4. オプション機能

Entailment Cones、Persistent Homology、Sheaf Attentionは**オプション機能**です。必要に応じて有効化してください。

### 5. 8GB VRAM制約

Phase7と同様に、**RTX 3080 (10GB)** で動作することを保証します。

## ベンチマーク

### Phase7 vs Phase8比較

```bash
python scripts/benchmark_phase7_vs_phase8_resnetbk.py
```

**期待される結果**:
- スループット: Phase7と同等以上
- メモリ: Phase7と同等以下
- 精度: Phase7と同等

### BK-Core統合効果

```bash
python scripts/benchmark_bk_core_integration_effect.py
```

**測定項目**:
- G_ii統計（平均、標準偏差）
- ゲーティング効果（ゲート値の分布）
- 物理情報活用度（共鳴検出頻度、曲率調整頻度）

### O(N)複雑度検証

```bash
python scripts/verify_phase8_complexity.py
```

**検証内容**:
- シーケンス長を変えて計算時間を測定
- 線形回帰でO(N)スケーリングを確認

## トラブルシューティング

### Q: Phase8がPhase7より遅い

**A**: オプション機能を無効化してください。

```python
config = Phase8Config(
    use_entailment_cones=False,
    use_persistent_homology=False,
    use_sheaf_attention=False
)
```

### Q: メモリ不足エラー

**A**: バッチサイズまたはシーケンス長を減らしてください。

```python
config = Phase8Config(
    max_seq_len=1024,  # 2048 → 1024
    batch_size=1       # 2 → 1
)
```

### Q: G_iiが取得できない

**A**: `use_bk_core=True`を確認してください。

```python
config = Phase8Config(
    use_bk_core=True  # 必須
)
```

## まとめ

Phase8は**Phase7（ResNetBK）の拡張**であり、BK-Coreを心臓部として双曲幾何学、トポロジー、論理的含意などの高度な機能を追加したものです。Phase7の全機能を継承し、O(N)複雑度と8GB VRAM制約を維持しながら、物理情報を活用した新しい機能を提供します。
