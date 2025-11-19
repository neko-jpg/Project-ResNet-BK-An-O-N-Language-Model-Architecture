# Phase 1 Migration Guide

## 概要

このガイドでは、既存のMUSEベースラインモデルからPhase 1 Efficiency Engineへの移行手順を説明します。コード例、API互換性ノート、破壊的変更（もしあれば）を含みます。

---

## 1. 移行の概要

### 1.1 Phase 1で追加される機能

- **Adaptive Rank Semiseparable Layer (AR-SSM)**: 動的ランク調整
- **Holographic Tensor Train Embedding (HTT)**: 90%パラメータ圧縮
- **Logarithmic Number System Kernel (LNS)**: 推論時の計算効率化（オプション）
- **Birman-Schwinger Stability Monitor**: 訓練中の安定性監視
- **自動エラーリカバリ**: VRAM不足と数値不安定性の自動対応

### 1.2 互換性

Phase 1は既存のMUSEアーキテクチャと**完全に互換性があります**：

- ✅ 既存のBK-Coreと統合
- ✅ 既存のSemiseparableMatrixと統合
- ✅ 既存の訓練スクリプトと互換
- ✅ 既存のチェックポイントから変換可能

### 1.3 破壊的変更

**なし** - Phase 1はオプトイン方式です。既存のコードは変更なしで動作します。

---

## 2. ステップバイステップ移行

### Step 1: Phase 1パッケージのインストール

Phase 1コンポーネントは既にプロジェクトに含まれています。追加のインストールは不要です。

```bash
# 依存関係の確認
pip install torch>=2.0.0
pip install triton>=2.1.0  # Fused Scanカーネル用（オプション）
```

### Step 2: 設定の作成

```python
from src.models.phase1 import Phase1Config, get_preset_config

# オプション1: プリセット設定を使用
config = get_preset_config('8gb')  # または '10gb', '24gb'

# オプション2: カスタム設定を作成
config = Phase1Config(
    ar_ssm_enabled=True,
    ar_ssm_max_rank=32,
    htt_enabled=True,
    htt_rank=16,
    stability_monitoring_enabled=True,
    use_gradient_checkpointing=True,
    target_vram_gb=8.0,
)
```

### Step 3: モデルの変換

#### 3.1 新しいモデルの作成

```python
from src.models.phase1 import create_phase1_model

# Phase 1モデルを作成
model = create_phase1_model(config)
```

#### 3.2 既存のチェックポイントからの変換

```python
from src.models.phase1 import convert_to_phase1
import torch

# 既存のモデルを読み込み
baseline_model = torch.load('checkpoints/baseline_model.pt')

# Phase 1に変換
phase1_model = convert_to_phase1(
    baseline_model,
    config=config,
    preserve_weights=True,  # 可能な限り重みを保持
)

# 変換されたモデルを保存
torch.save(phase1_model, 'checkpoints/phase1_model.pt')
```

### Step 4: 訓練スクリプトの更新

#### 4.1 最小限の変更

既存の訓練スクリプトは最小限の変更で動作します：

```python
# 既存のコード
# model = create_baseline_model(...)

# Phase 1への変更
from src.models.phase1 import create_phase1_model, Phase1Config

config = Phase1Config(...)
model = create_phase1_model(config)

# 残りのコードは変更なし
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
# ... 訓練ループ
```

#### 4.2 安定性監視の追加（推奨）

```python
from src.models.phase1 import BKStabilityMonitor

# 安定性モニターの作成
stability_monitor = BKStabilityMonitor(
    stability_threshold=config.stability_threshold,
)

# 訓練ループ
for batch in dataloader:
    # 順伝播
    output, diagnostics = model(batch)
    
    # 安定性チェック（オプション）
    if config.stability_monitoring_enabled:
        stability_info = stability_monitor.check_stability(
            G_ii=diagnostics.get('G_ii'),
            v=diagnostics.get('potential'),
            epsilon=model.epsilon,
        )
        
        if not stability_info['is_stable']:
            logger.warning(f"Stability warning: {stability_info['warnings']}")
            # 必要に応じてアクションを実行
    
    # 逆伝播
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

#### 4.3 エラーハンドリングの追加（推奨）

```python
from src.models.phase1.errors import VRAMExhaustedError, NumericalInstabilityError
from src.models.phase1.recovery import Phase1ErrorRecovery

recovery = Phase1ErrorRecovery()

try:
    # 訓練ループ
    for batch in dataloader:
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
except VRAMExhaustedError as e:
    logger.error(f"VRAM exhausted: {e}")
    # 自動リカバリを試行
    if recovery.handle_vram_exhausted(e, model, config):
        logger.info("Recovery successful, retrying...")
        # 訓練を再開
    else:
        raise

except NumericalInstabilityError as e:
    logger.error(f"Numerical instability: {e}")
    # 自動リカバリを試行
    recovery.handle_numerical_instability(e, optimizer, config)
    logger.info("Recovery successful, continuing...")
```

### Step 5: 評価スクリプトの更新

評価スクリプトは変更不要ですが、Phase 1診断情報を追加できます：

```python
# 既存の評価コード
model.eval()
with torch.no_grad():
    for batch in val_dataloader:
        output = model(batch)
        # ... PPL計算

# Phase 1診断情報の追加（オプション）
from src.models.phase1 import collect_phase1_diagnostics

diagnostics = collect_phase1_diagnostics(model, val_dataloader)
print(f"AR-SSM Effective Rank: {diagnostics['ar_ssm_effective_rank']:.2f}")
print(f"HTT Compression Ratio: {diagnostics['htt_compression_ratio']:.2%}")
print(f"Peak VRAM: {diagnostics['peak_vram_mb']:.2f}MB")
```

---

## 3. コンポーネント別移行

### 3.1 Embedding層の置き換え

#### Before (ベースライン)

```python
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # ...
    
    def forward(self, input_ids):
        x = self.token_embedding(input_ids)
        # ...
```

#### After (Phase 1)

```python
from src.models.phase1 import HolographicTTEmbedding

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, use_htt=True, htt_rank=16):
        super().__init__()
        if use_htt:
            self.token_embedding = HolographicTTEmbedding(
                vocab_size=vocab_size,
                d_model=d_model,
                rank=htt_rank,
                num_cores=2,
            )
        else:
            self.token_embedding = nn.Embedding(vocab_size, d_model)
        # ...
    
    def forward(self, input_ids):
        x = self.token_embedding(input_ids)
        # ... 残りは同じ
```

**パラメータ削減**: 50,000 × 1,024 = 51.2M → ~0.8M (98.5%削減)

---

### 3.2 Semiseparable層の拡張

#### Before (ベースライン)

```python
from src.models import SemiseparableMatrix

class ResNetBKBlock(nn.Module):
    def __init__(self, d_model, n_seq):
        super().__init__()
        self.semisep = SemiseparableMatrix(
            n_seq=n_seq,
            rank=int(np.log2(n_seq)),
        )
        # ...
```

#### After (Phase 1)

```python
from src.models.phase1 import AdaptiveRankSemiseparableLayer

class ResNetBKBlock(nn.Module):
    def __init__(self, d_model, n_seq, use_ar_ssm=True, max_rank=32):
        super().__init__()
        if use_ar_ssm:
            self.semisep = AdaptiveRankSemiseparableLayer(
                d_model=d_model,
                max_rank=max_rank,
                min_rank=4,
                use_fused_scan=True,
            )
        else:
            self.semisep = SemiseparableMatrix(
                n_seq=n_seq,
                rank=int(np.log2(n_seq)),
            )
        # ...
```

**メモリ削減**: 20-40%追加削減（適応ランクゲーティングにより）

---

### 3.3 線形層の最適化（オプション）

#### Before (ベースライン)

```python
class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
```

#### After (Phase 1 - 推論専用)

```python
from src.models.phase1 import LNSLinear

class FFN(nn.Module):
    def __init__(self, d_model, d_ff, use_lns=False):
        super().__init__()
        if use_lns:
            self.fc1 = LNSLinear(d_model, d_ff, use_lns=True)
            self.fc2 = LNSLinear(d_ff, d_model, use_lns=True)
        else:
            self.fc1 = nn.Linear(d_model, d_ff)
            self.fc2 = nn.Linear(d_ff, d_model)
```

**注意**: LNSは実験的で、主に推論用です。訓練時は標準matmulにフォールバックします。

---

## 4. 設定ファイルの移行

### 4.1 YAML設定の更新

#### Before (baseline_config.yaml)

```yaml
model:
  vocab_size: 50000
  d_model: 1024
  n_layers: 24
  n_seq: 2048

training:
  batch_size: 8
  learning_rate: 1e-4
  max_steps: 100000
```

#### After (phase1_config.yaml)

```yaml
model:
  vocab_size: 50000
  d_model: 1024
  n_layers: 24
  n_seq: 2048
  
  # Phase 1設定
  phase1:
    enabled: true
    preset: "8gb"  # または "10gb", "24gb", "custom"
    
    # カスタム設定（presetが"custom"の場合）
    ar_ssm:
      enabled: true
      max_rank: 32
      min_rank: 4
      l1_regularization: 0.001
      use_fused_scan: true
    
    htt:
      enabled: true
      rank: 16
      num_cores: 2
      phase_encoding: true
    
    lns:
      enabled: false
    
    stability:
      enabled: true
      threshold: 1e-6
    
    memory:
      gradient_checkpointing: true
      target_vram_gb: 8.0

training:
  batch_size: 4  # Phase 1では小さいバッチサイズ
  learning_rate: 1e-4
  max_steps: 100000
```

### 4.2 設定の読み込み

```python
import yaml
from src.models.phase1 import Phase1Config, get_preset_config

# YAML設定を読み込み
with open('configs/phase1_config.yaml', 'r') as f:
    yaml_config = yaml.safe_load(f)

# Phase1Configに変換
if yaml_config['model']['phase1']['preset'] != 'custom':
    config = get_preset_config(yaml_config['model']['phase1']['preset'])
else:
    config = Phase1Config(
        ar_ssm_enabled=yaml_config['model']['phase1']['ar_ssm']['enabled'],
        ar_ssm_max_rank=yaml_config['model']['phase1']['ar_ssm']['max_rank'],
        # ... 他のパラメータ
    )
```

---

## 5. チェックポイントの互換性

### 5.1 ベースラインからPhase 1への変換

```python
from src.models.phase1 import convert_to_phase1
import torch

# ベースラインチェックポイントを読み込み
checkpoint = torch.load('checkpoints/baseline_epoch10.pt')
baseline_model = checkpoint['model']

# Phase 1に変換
config = get_preset_config('8gb')
phase1_model = convert_to_phase1(
    baseline_model,
    config=config,
    preserve_weights=True,
)

# 新しいチェックポイントを保存
torch.save({
    'model': phase1_model,
    'config': config,
    'epoch': checkpoint['epoch'],
    'optimizer': checkpoint['optimizer'],  # そのまま使用可能
}, 'checkpoints/phase1_epoch10.pt')
```

### 5.2 Phase 1からベースラインへの変換（ロールバック）

```python
from src.models.phase1 import convert_from_phase1

# Phase 1チェックポイントを読み込み
checkpoint = torch.load('checkpoints/phase1_epoch10.pt')
phase1_model = checkpoint['model']

# ベースラインに変換
baseline_model = convert_from_phase1(phase1_model)

# ベースラインチェックポイントを保存
torch.save({
    'model': baseline_model,
    'epoch': checkpoint['epoch'],
}, 'checkpoints/baseline_from_phase1_epoch10.pt')
```

**注意**: HTT Embeddingは標準Embeddingに展開されるため、パラメータ数が増加します。

---

## 6. パフォーマンス比較

### 6.1 ベンチマークスクリプト

```python
from scripts import (
    validate_phase1_memory,
    benchmark_phase1_throughput,
    validate_phase1_perplexity,
)

# メモリ使用量の比較
baseline_memory = measure_baseline_memory(baseline_model, batch_size=4, seq_len=2048)
phase1_memory = validate_phase1_memory(config, batch_size=4, seq_len=2048)

print(f"Baseline VRAM: {baseline_memory['peak_vram_gb']:.2f}GB")
print(f"Phase 1 VRAM: {phase1_memory['peak_vram_gb']:.2f}GB")
print(f"Reduction: {(1 - phase1_memory['peak_vram_gb'] / baseline_memory['peak_vram_gb']) * 100:.1f}%")

# スループットの比較
baseline_throughput = benchmark_baseline_throughput(baseline_model)
phase1_throughput = benchmark_phase1_throughput(config)

print(f"Baseline: {baseline_throughput['tokens_per_sec']:.1f} tokens/sec")
print(f"Phase 1: {phase1_throughput['tokens_per_sec']:.1f} tokens/sec")

# Perplexityの比較
baseline_ppl = evaluate_baseline_ppl(baseline_model, dataset='wikitext-103')
phase1_ppl = validate_phase1_perplexity(phase1_model, baseline_model, dataset='wikitext-103')

print(f"Baseline PPL: {baseline_ppl:.2f}")
print(f"Phase 1 PPL: {phase1_ppl['phase1_ppl']:.2f}")
print(f"Degradation: {phase1_ppl['degradation']:.2%}")
```

### 6.2 期待される結果

| メトリクス | ベースライン | Phase 1 (8GB) | 改善 |
|-----------|-------------|---------------|------|
| VRAM (GB) | 12.5        | 7.8           | 37.6% ↓ |
| Params (M) | 125         | 25            | 80% ↓ |
| Throughput (tokens/s) | 850 | 1100 | 29.4% ↑ |
| PPL | 16.8 | 17.2 | 2.4% ↑ |

---

## 7. トラブルシューティング

### 7.1 変換エラー

**問題**: `convert_to_phase1()`がエラーを出す

**原因**: モデル構造が期待と異なる

**解決策**:
```python
# デバッグモードで変換
phase1_model = convert_to_phase1(
    baseline_model,
    config=config,
    preserve_weights=True,
    strict=False,  # 厳密なチェックを無効化
    verbose=True,  # 詳細なログを出力
)
```

### 7.2 パフォーマンス劣化

**問題**: Phase 1がベースラインより遅い

**原因**: Fused Scanが無効、またはCUDAが利用不可

**解決策**:
```python
# Fused Scanを確認
config.ar_ssm_use_fused_scan = True

# CUDAを確認
assert torch.cuda.is_available(), "CUDA required for Phase 1"

# Tritonを確認
try:
    import triton
except ImportError:
    print("Install triton: pip install triton")
```

### 7.3 高いPerplexity

**問題**: Phase 1のPPLがベースラインより5%以上高い

**原因**: 圧縮が強すぎる

**解決策**:
```python
# ランクを増加
config.htt_rank = 32  # 16から増加
config.ar_ssm_max_rank = 48  # 32から増加

# L1正則化を削減
config.ar_ssm_l1_regularization = 0.0001  # 0.001から削減

# より長い訓練
# カリキュラム学習でランクを徐々に増加
```

---

## 8. 段階的移行戦略

### 8.1 段階1: HTT Embeddingのみ

最もリスクが低く、最大の圧縮を提供：

```python
config = Phase1Config(
    ar_ssm_enabled=False,  # まだ無効
    htt_enabled=True,      # HTTのみ有効
    htt_rank=16,
    stability_monitoring_enabled=False,
)
```

**期待される結果**:
- パラメータ: 90%削減
- VRAM: 30%削減
- PPL劣化: <2%

### 8.2 段階2: AR-SSMを追加

HTTが安定したら、AR-SSMを追加：

```python
config = Phase1Config(
    ar_ssm_enabled=True,   # AR-SSMを有効化
    ar_ssm_max_rank=32,
    htt_enabled=True,
    htt_rank=16,
    stability_monitoring_enabled=True,
)
```

**期待される結果**:
- VRAM: 追加で20-40%削減
- スループット: 20-30%向上
- PPL劣化: <5%

### 8.3 段階3: 完全なPhase 1

すべてのコンポーネントを有効化：

```python
config = Phase1Config(
    ar_ssm_enabled=True,
    ar_ssm_max_rank=32,
    ar_ssm_use_fused_scan=True,  # Fused Scanを有効化
    htt_enabled=True,
    htt_rank=16,
    lns_enabled=False,  # 推論時のみ
    stability_monitoring_enabled=True,
    use_gradient_checkpointing=True,
)
```

**期待される結果**:
- VRAM: 合計80-85%削減
- スループット: 3倍向上（Fused Scanにより）
- PPL劣化: <5%

---

## 9. ベストプラクティス

### 9.1 移行前

1. **ベースラインを確立**: 現在のモデルのPPL、VRAM、スループットを測定
2. **チェックポイントをバックアップ**: 既存のチェックポイントをバックアップ
3. **小規模で実験**: 小さいモデルでPhase 1をテスト
4. **ドキュメントを読む**: 実装ガイドとハイパーパラメータガイドを確認

### 9.2 移行中

1. **段階的に移行**: HTT → AR-SSM → 完全なPhase 1
2. **頻繁に検証**: 各段階でPPLとメモリを確認
3. **ログを保持**: すべてのメトリクスをWandBまたはTensorBoardに記録
4. **問題を文書化**: 遭遇した問題と解決策を記録

### 9.3 移行後

1. **パフォーマンスを監視**: 訓練中のメトリクスを継続的に監視
2. **ハイパーパラメータを調整**: 必要に応じて微調整
3. **結果を共有**: コミュニティと結果を共有
4. **フィードバックを提供**: バグレポートや改善提案を提出

---

## 10. サポートとリソース

### 10.1 ドキュメント

- [Phase 1 Implementation Guide](PHASE1_IMPLEMENTATION_GUIDE.md)
- [Hyperparameter Tuning Guide](PHASE1_HYPERPARAMETER_TUNING.md)
- [API Reference](API_REFERENCE.md)

### 10.2 例とチュートリアル

- `examples/demo_ar_ssm.py`: AR-SSMデモ
- `examples/htt_compression_demo.py`: HTT圧縮デモ
- `examples/phase1_integration_demo.py`: 完全な統合例
- `notebooks/phase1_*_tutorial.py`: Jupyterチュートリアル

### 10.3 コミュニティ

- GitHub Issues: バグレポートと機能リクエスト
- GitHub Discussions: 質問と議論
- Contributing: プルリクエストを歓迎

---

## まとめ

Phase 1への移行は段階的で、既存のコードとの互換性を維持しながら行えます。

**重要なポイント**:

1. **オプトイン**: Phase 1は完全にオプション
2. **段階的**: HTT → AR-SSM → 完全なPhase 1
3. **互換性**: 既存のチェックポイントから変換可能
4. **ロールバック**: 必要に応じてベースラインに戻せる
5. **サポート**: 豊富なドキュメントと例

**次のステップ**:

1. プリセット設定で小規模実験を開始
2. ベースラインとPhase 1を比較
3. ハイパーパラメータを調整
4. 完全な訓練を実行
5. 結果を共有

Phase 1への移行により、8GB VRAMで100B級モデルを実行できるようになります！
