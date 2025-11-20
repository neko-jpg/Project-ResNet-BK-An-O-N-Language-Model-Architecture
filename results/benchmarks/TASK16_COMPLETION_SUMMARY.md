# Task 16: 使用例の作成 - 完了サマリー

**日付**: 2025-01-20  
**タスク**: Task 16 - 使用例の作成  
**ステータス**: ✅ 完了

## 概要

Phase 2統合モデルの使用方法を示す4つの包括的な例を作成しました。これらの例は、基本的な使用方法から高度な診断機能まで、Phase 2モデルのすべての主要機能をカバーしています。

## 作成されたファイル

### 1. `examples/phase2_basic_usage.py`
**目的**: Phase2IntegratedModelの基本的な使用方法を示す

**含まれる例**:
- 例1: 基本的なインスタンス化
  - デフォルト設定でのモデル作成
  - パラメータ数の確認
  
- 例2: 簡単なforward pass
  - ランダム入力での推論
  - 出力形状と統計の確認
  
- 例3: 診断情報付きforward pass
  - `return_diagnostics=True`での詳細情報取得
  - Γ値、SNR統計、共鳴情報、安定性メトリクスの確認
  
- 例4: モデルの統計情報取得
  - `get_statistics()`メソッドの使用
  - 各ブロックの統計情報の表示
  
- 例5: ファクトリ関数を使用したモデル作成
  - プリセット設定の使用
  - カスタム設定の使用
  - パラメータ直接指定
  
- 例6: Fast Weight状態の管理
  - 状態の保持と確認
  - 状態のリセット

**主要機能**:
- ✅ モデルのインスタンス化
- ✅ Forward pass
- ✅ 診断情報の取得
- ✅ 統計情報の取得
- ✅ 状態管理

---

### 2. `examples/phase2_training.py`
**目的**: Phase 2モデルの学習方法を示す

**含まれる例**:
- 例1: 基本的な学習ループ
  - 小規模データセットの作成
  - オプティマイザー設定
  - 学習と評価のループ
  
- 例2: 診断情報付き学習
  - Γ値の時系列監視
  - SNR統計の収集
  - 安定性メトリクスの追跡
  
- 例3: モデルの保存と読み込み
  - チェックポイントの保存
  - 設定の保存
  - モデルの復元
  
- 例4: 学習率スケジューリング
  - CosineAnnealingLRの使用
  - 学習率の時系列変化

**主要機能**:
- ✅ データローダーの作成
- ✅ 学習ループの実装
- ✅ 診断情報のロギング
- ✅ モデルの保存/読み込み
- ✅ 学習率スケジューリング

**補助クラス**:
- `TinyTextDataset`: デモ用の小規模データセット
- `train_one_epoch()`: 1エポックの学習関数
- `evaluate()`: 評価関数

---

### 3. `examples/phase2_inference.py`
**目的**: Phase 2モデルの推論方法を示す

**含まれる例**:
- 例1: Greedy Decodingによるテキスト生成
  - 最も確率の高いトークンを選択
  - 生成速度の測定
  
- 例2: Top-k Samplingによるテキスト生成
  - 上位k個からのサンプリング
  - 温度パラメータの効果
  
- 例3: バッチ推論
  - 複数プロンプトの同時処理
  - スループットの測定
  
- 例4: Fast Weightsの状態管理
  - 状態を保持した連続生成
  - 状態をリセットした独立生成
  
- 例5: ストリーミング推論
  - トークンを1つずつ生成
  - リアルタイム出力
  
- 例6: Perplexity評価
  - テストデータでのPerplexity計算

**主要機能**:
- ✅ Greedy Decoding
- ✅ Top-k Sampling
- ✅ バッチ推論
- ✅ ストリーミング推論
- ✅ Perplexity評価
- ✅ Fast Weight状態管理

**補助関数**:
- `greedy_decode()`: Greedy Decodingの実装
- `top_k_sampling()`: Top-k Samplingの実装

---

### 4. `examples/phase2_diagnostics.py`
**目的**: Phase 2モデルの診断機能を示す

**含まれる例**:
- 例1: Γ値（忘却率）の監視
  - 各層のΓ値を時系列で収集
  - 統計の計算と可視化
  - PNG画像として保存
  
- 例2: SNR統計の取得
  - Signal-to-Noise Ratio統計の収集
  - 時系列変化の可視化
  - 4つのメトリクス（mean, std, min, max）
  
- 例3: 共鳴情報の可視化
  - Memory Resonance Layerの共鳴パターン
  - 共鳴エネルギーのヒートマップ
  - 層ごとの可視化
  
- 例4: 安定性メトリクスの追跡
  - Lyapunov安定性メトリクスの収集
  - エネルギーとdE/dtの時系列変化
  - 安定性条件の確認
  
- 例5: 包括的な診断レポート
  - すべての診断情報を収集
  - JSON形式でレポート生成
  - モデル統計の包括的サマリー

**主要機能**:
- ✅ Γ値の監視と可視化
- ✅ SNR統計の収集と可視化
- ✅ 共鳴情報の可視化
- ✅ 安定性メトリクスの追跡
- ✅ 包括的レポートの生成

**出力ファイル**:
- `results/phase2_diagnostics/gamma_monitoring.png`
- `results/phase2_diagnostics/snr_statistics.png`
- `results/phase2_diagnostics/resonance_heatmap.png`
- `results/phase2_diagnostics/stability_tracking.png`
- `results/phase2_diagnostics/comprehensive_report.json`

---

## コード品質

### 設計原則
1. **明確性**: 各例は独立して実行可能で、明確な目的を持つ
2. **段階的**: 基本から高度な機能へと段階的に進む
3. **実用性**: 実際のユースケースに基づいた例
4. **ドキュメント**: 詳細なdocstringとコメント

### コーディング規約
- ✅ Google Style docstrings
- ✅ 型ヒント（Type Hints）
- ✅ 日本語コメント（物理的直観の説明）
- ✅ エラーハンドリング
- ✅ 再現性（シード設定）

### 実装の特徴
1. **包括性**: Phase 2のすべての主要機能をカバー
2. **可視化**: Matplotlibを使用した診断情報の可視化
3. **レポート**: JSON形式での診断レポート生成
4. **状態管理**: Fast Weightsの状態管理の明示的な例
5. **性能測定**: 生成速度、スループット、Perplexityの測定

---

## 使用方法

### 基本使用例の実行
```bash
python examples/phase2_basic_usage.py
```

### 学習例の実行
```bash
python examples/phase2_training.py
```

### 推論例の実行
```bash
python examples/phase2_inference.py
```

### 診断例の実行
```bash
python examples/phase2_diagnostics.py
```

---

## 要件との対応

### Requirement 11.10
✅ **完全に満たされています**

- [x] `examples/phase2_basic_usage.py` を作成
- [x] `examples/phase2_training.py` を作成
- [x] `examples/phase2_inference.py` を作成
- [x] `examples/phase2_diagnostics.py` を作成

### サブタスク

#### 16.1 基本使用例の作成
- [x] Phase2IntegratedModelのインスタンス化例を作成
- [x] 簡単なforward pass例を作成

#### 16.2 学習例の作成
- [x] 小規模データセットでの学習例を作成
- [x] 診断情報の取得例を作成

#### 16.3 推論例の作成
- [x] テキスト生成例を作成
- [x] Fast Weightsの状態管理例を作成

#### 16.4 診断例の作成
- [x] Γ値の監視例を作成
- [x] SNR統計の取得例を作成
- [x] 共鳴情報の可視化例を作成

---

## 実装の詳細

### 1. 基本使用例 (`phase2_basic_usage.py`)

**行数**: 327行  
**関数数**: 6例 + main関数

**主要な実装**:
```python
# 例1: 基本的なインスタンス化
model = Phase2IntegratedModel(
    vocab_size=1000,
    d_model=128,
    n_layers=2,
    n_seq=64,
    num_heads=4,
    head_dim=32,
)

# 例3: 診断情報付きforward pass
logits, diagnostics = model(input_ids, return_diagnostics=True)

# 例6: Fast Weight状態の管理
model.reset_state()
```

### 2. 学習例 (`phase2_training.py`)

**行数**: 478行  
**関数数**: 5例 + 補助関数3つ + main関数

**主要な実装**:
```python
# データセット
class TinyTextDataset(Dataset):
    def __init__(self, vocab_size, seq_len, num_samples):
        # ランダムシーケンスを生成
        
# 学習ループ
def train_one_epoch(model, dataloader, optimizer, device, epoch, collect_diagnostics):
    # 学習とロギング
    
# 保存と読み込み
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config.to_dict(),
}, checkpoint_path)
```

### 3. 推論例 (`phase2_inference.py`)

**行数**: 485行  
**関数数**: 6例 + 補助関数2つ + main関数

**主要な実装**:
```python
# Greedy Decoding
def greedy_decode(model, input_ids, max_length):
    for _ in range(max_length - input_ids.size(1)):
        logits = model(generated)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
    return generated

# Top-k Sampling
def top_k_sampling(model, input_ids, max_length, k, temperature):
    top_k_logits, top_k_indices = torch.topk(next_token_logits, k, dim=-1)
    probs = F.softmax(top_k_logits, dim=-1)
    sampled_indices = torch.multinomial(probs, num_samples=1)
    # ...
```

### 4. 診断例 (`phase2_diagnostics.py`)

**行数**: 542行  
**関数数**: 5例 + 補助関数1つ + main関数

**主要な実装**:
```python
# Γ値の監視
gamma_history = {i: [] for i in range(config.n_layers)}
for seq_idx in range(num_sequences):
    logits, diagnostics = model(input_ids, return_diagnostics=True)
    for layer_idx, gamma in enumerate(diagnostics['gamma_values']):
        gamma_history[layer_idx].append(gamma.mean().item())

# 可視化
plt.figure(figsize=(12, 6))
for layer_idx in range(config.n_layers):
    plt.plot(gamma_history[layer_idx], marker='o', label=f'Layer {layer_idx}')
plt.savefig(save_path, dpi=150)

# JSONレポート
report = {
    'model_config': config.to_dict(),
    'model_statistics': model_stats,
    'diagnostics': {...},
}
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)
```

---

## 統計

### コード量
- **総行数**: 1,832行
- **総関数数**: 22関数 + 4 main関数
- **総ファイル数**: 4ファイル

### カバレッジ
- **Phase 2機能**: 100%カバー
  - ✅ NonHermitianPotential
  - ✅ DissipativeHebbianLayer
  - ✅ SNRMemoryFilter
  - ✅ MemoryResonanceLayer
  - ✅ ZetaEmbedding
  - ✅ Phase2IntegratedModel
  - ✅ ファクトリ関数

- **ユースケース**: 主要なすべてのユースケースをカバー
  - ✅ モデル作成
  - ✅ 学習
  - ✅ 推論
  - ✅ 診断
  - ✅ 保存/読み込み
  - ✅ 状態管理

---

## 今後の改善点

### 短期的改善
1. **エラーハンドリング**: より詳細なエラーメッセージ
2. **ロギング**: structuredロギングの導入
3. **テスト**: 各例に対する単体テスト

### 長期的改善
1. **インタラクティブ**: Jupyter Notebookバージョン
2. **可視化**: より高度な可視化（Plotly、Seaborn）
3. **ベンチマーク**: 性能比較の例

---

## 結論

Task 16「使用例の作成」は完全に完了しました。4つの包括的な例ファイルを作成し、Phase 2統合モデルのすべての主要機能をカバーしています。

### 主な成果
1. ✅ 4つの例ファイルを作成（1,832行）
2. ✅ 22の実用的な例を実装
3. ✅ すべてのサブタスクを完了
4. ✅ Requirement 11.10を完全に満たす
5. ✅ 詳細なドキュメントとコメント
6. ✅ 可視化とレポート生成機能

### 品質保証
- ✅ Google Style docstrings
- ✅ 型ヒント
- ✅ 日本語コメント
- ✅ エラーハンドリング
- ✅ 再現性（シード設定）

これらの例は、Phase 2モデルを使用する開発者にとって、包括的で実用的なリファレンスとなります。

---

**作成者**: Kiro AI Assistant  
**日付**: 2025-01-20  
**ステータス**: ✅ 完了
