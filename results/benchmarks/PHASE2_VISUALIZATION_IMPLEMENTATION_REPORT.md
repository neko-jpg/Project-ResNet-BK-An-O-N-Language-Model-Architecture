# Phase 2 可視化スクリプト実装レポート

**実装日**: 2024-XX-XX  
**タスク**: Task 14 - 可視化スクリプトの実装  
**ステータス**: ✅ 完了

---

## 実装概要

Phase 2モデルの学習過程と診断情報を包括的に可視化するスクリプトを実装しました。

### 実装ファイル

1. **メインスクリプト**: `scripts/visualize_phase2.py`
   - 学習曲線、Γ変化、SNR統計、共鳴情報の可視化
   - コマンドライン引数による柔軟な制御
   - 高品質なグラフ生成（300 DPI）

2. **デモスクリプト**: `examples/phase2_visualization_demo.py`
   - サンプルデータ生成
   - 可視化機能のデモンストレーション

3. **クイックリファレンス**: `docs/quick-reference/PHASE2_VISUALIZATION_QUICK_REFERENCE.md`
   - 使用方法の詳細説明
   - トラブルシューティングガイド

---

## 実装詳細

### Task 14.1: 学習曲線可視化 ✅

**実装内容**:
- Loss曲線のプロット
- Perplexity曲線のプロット
- 移動平均の追加（ウィンドウサイズ50）
- 高品質なグラフスタイル

**出力**: `learning_curves.png`

**機能**:
```python
def plot_learning_curves(self):
    """
    学習曲線を可視化
    
    - Loss曲線: 青線（実測値）+ 紫破線（移動平均）
    - Perplexity曲線: オレンジ線（実測値）+ 赤破線（移動平均）
    """
```

**検証結果**:
- ✅ Loss曲線が正しくプロット
- ✅ Perplexity曲線が正しくプロット
- ✅ 移動平均が適切に計算・表示
- ✅ グラフが高品質（300 DPI）

---

### Task 14.2: Γ変化可視化 ✅

**実装内容**:
- 各層のΓ値の時系列プロット
- Γ値のヒートマップ（層×時間）
- 統計情報の出力

**出力**: `gamma_evolution.png`

**機能**:
```python
def plot_gamma_evolution(self):
    """
    Γ（忘却率）の時間変化を可視化
    
    上段: 時系列プロット（各層を異なる色で表示）
    下段: ヒートマップ（YlOrRd カラーマップ）
    """
```

**検証結果**:
- ✅ 各層のΓ値が時系列でプロット
- ✅ ヒートマップが正しく生成
- ✅ 統計情報（平均、標準偏差、最小、最大）が出力
- ✅ デモデータでΓ平均値: 0.022896

**KPI確認**:
- Γの時間変化が可視化され、適応的忘却の動作を確認可能
- 層ごとのΓの違いが明確に表示

---

### Task 14.3: SNR統計可視化 ✅

**実装内容**:
- SNR時系列（平均±標準偏差）
- SNR範囲（Min-Max）
- SNR分布ヒストグラム
- 低SNR比率の時間変化

**出力**: `snr_statistics.png`

**機能**:
```python
def plot_snr_statistics(self):
    """
    SNR統計を可視化
    
    4つのサブプロット:
    1. 平均SNR±標準偏差
    2. SNR範囲（Min-Max）
    3. SNR分布ヒストグラム
    4. 低SNR比率の時間変化
    """
```

**検証結果**:
- ✅ SNR時系列が正しくプロット
- ✅ 閾値（2.0）が赤破線で表示
- ✅ SNR分布ヒストグラムが生成
- ✅ 低SNR比率が追跡可能
- ✅ デモデータで最終SNR平均: 3.1051

**KPI確認**:
- SNR閾値（2.0）との比較が視覚的に明確
- 記憶の質の時間変化が追跡可能

---

### Task 14.4: 共鳴情報可視化 ✅

**実装内容**:
- 共鳴成分数の時系列
- 総共鳴エネルギーの時系列
- 共鳴エネルギーヒートマップ（最終ステップ）
- 共鳴マスクの可視化（最終ステップ）

**出力**: `resonance_info.png`

**機能**:
```python
def plot_resonance_info(self):
    """
    共鳴情報を可視化
    
    6つのサブプロット:
    1. 共鳴成分数の時間変化
    2. 総共鳴エネルギーの時間変化
    3. 共鳴エネルギーヒートマップ（ヘッド×次元）
    4. 共鳴マスク（緑=アクティブ、赤=非アクティブ）
    """
```

**検証結果**:
- ✅ 共鳴成分数が時系列でプロット
- ✅ 総エネルギーが時系列でプロット
- ✅ エネルギーヒートマップが生成（viridis カラーマップ）
- ✅ 共鳴マスクが生成（RdYlGn カラーマップ）
- ✅ デモデータで最終共鳴成分数: 30.73

**KPI確認**:
- 共鳴成分の時間変化が追跡可能
- スパース性（80%フィルタリング）が視覚的に確認可能

---

## コマンドライン引数

### 基本的な使い方

```bash
# すべての可視化を実行
python scripts/visualize_phase2.py --log-dir results/phase2_training

# 出力ディレクトリを指定
python scripts/visualize_phase2.py \
    --log-dir results/phase2_training \
    --output-dir results/my_visualizations

# 特定の可視化のみ実行
python scripts/visualize_phase2.py --log-dir results/phase2_training --only learning_curves
python scripts/visualize_phase2.py --log-dir results/phase2_training --only gamma
python scripts/visualize_phase2.py --log-dir results/phase2_training --only snr
python scripts/visualize_phase2.py --log-dir results/phase2_training --only resonance

# インタラクティブモード
python scripts/visualize_phase2.py --log-dir results/phase2_training --interactive
```

### 引数一覧

| 引数 | 説明 | デフォルト |
|------|------|-----------|
| `--log-dir` | 学習ログディレクトリ | `results/phase2_training` |
| `--output-dir` | 出力ディレクトリ | `results/visualizations` |
| `--only` | 特定の可視化のみ実行 | なし（すべて実行） |
| `--interactive` | インタラクティブモード | False |

---

## ログファイル形式

### 1. `training_log.json`

```json
{
  "loss": [5.0, 4.8, 4.6, ...],
  "perplexity": [148.4, 121.5, 99.5, ...]
}
```

### 2. `diagnostics_log.json`

```json
{
  "gamma_values": [
    {
      "layer_0": [0.012, 0.013, 0.011, 0.012],
      "layer_1": [0.015, 0.016, 0.014, 0.015],
      ...
    },
    ...
  ],
  "snr_stats": [
    {
      "mean_snr": 2.5,
      "std_snr": 0.8,
      "min_snr": 0.5,
      "max_snr": 5.0
    },
    ...
  ],
  "resonance_info": [
    {
      "num_resonant": 25.3,
      "total_energy": 1.2,
      "diag_energy": [[...], [...], ...],
      "resonance_mask": [[...], [...], ...]
    },
    ...
  ]
}
```

---

## デモ実行結果

### コマンド

```bash
python examples/phase2_visualization_demo.py
```

### 出力

```
============================================================
Phase 2 可視化デモ
============================================================

[1] サンプルログを生成中...
✓ 学習ログを生成: results\phase2_demo_logs\training_log.json
✓ 診断ログを生成: results\phase2_demo_logs\diagnostics_log.json

[2] 可視化スクリプトを実行中...
✓ 学習ログをロード: results\phase2_demo_logs\training_log.json
✓ 診断ログをロード: results\phase2_demo_logs\diagnostics_log.json

============================================================
Phase 2 可視化スクリプト
============================================================

[14.1] 学習曲線を可視化中...
  ✓ 学習曲線を保存: results\phase2_demo_visualizations\learning_curves.png

[14.2] Γ変化を可視化中...
  ✓ Γ変化を保存: results\phase2_demo_visualizations\gamma_evolution.png
  - Γ平均値: 0.022896
  - Γ標準偏差: 0.008053
  - Γ最小値: 0.004654
  - Γ最大値: 0.045451

[14.3] SNR統計を可視化中...
  ✓ SNR統計を保存: results\phase2_demo_visualizations\snr_statistics.png
  - 最終SNR平均: 3.1051
  - 最終SNR標準偏差: 0.7001
  - 全体SNR平均: 1.9916

[14.4] 共鳴情報を可視化中...
  ✓ 共鳴情報を保存: results\phase2_demo_visualizations\resonance_info.png
  - 最終共鳴成分数: 30.73
  - 最終総エネルギー: 1.800284
  - 平均共鳴成分数: 19.94

============================================================
✓ すべての可視化が完了しました
  出力ディレクトリ: results\phase2_demo_visualizations
============================================================
```

### 生成されたファイル

1. `results/phase2_demo_visualizations/learning_curves.png`
2. `results/phase2_demo_visualizations/gamma_evolution.png`
3. `results/phase2_demo_visualizations/snr_statistics.png`
4. `results/phase2_demo_visualizations/resonance_info.png`

---

## 技術的詳細

### 依存関係

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
```

**requirements.txt**:
- `matplotlib==3.8.2`
- `seaborn==0.13.0`
- `numpy>=1.24.0`

### グラフスタイル

```python
# Seabornスタイル
sns.set_style("whitegrid")

# Matplotlibパラメータ
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
```

### カラーパレット

- **学習曲線**: `#2E86AB` (青), `#A23B72` (紫), `#F18F01` (オレンジ), `#C73E1D` (赤)
- **Γヒートマップ**: `YlOrRd` (黄色→オレンジ→赤)
- **共鳴エネルギー**: `viridis` (紫→緑→黄)
- **共鳴マスク**: `RdYlGn` (赤→黄→緑)

---

## 統合テスト

### テストケース1: サンプルデータでの動作確認

**コマンド**:
```bash
python examples/phase2_visualization_demo.py
```

**結果**: ✅ 成功
- すべての可視化が正常に生成
- 統計情報が正しく出力
- グラフが高品質（300 DPI）

### テストケース2: 個別可視化の実行

**コマンド**:
```bash
python scripts/visualize_phase2.py --log-dir results/phase2_demo_logs --only learning_curves
python scripts/visualize_phase2.py --log-dir results/phase2_demo_logs --only gamma
python scripts/visualize_phase2.py --log-dir results/phase2_demo_logs --only snr
python scripts/visualize_phase2.py --log-dir results/phase2_demo_logs --only resonance
```

**結果**: ✅ 成功
- 各可視化が個別に実行可能
- 出力ファイルが正しく生成

### テストケース3: エラーハンドリング

**シナリオ**: ログファイルが存在しない

**結果**: ✅ 適切に処理
```
警告: 学習ログが見つかりません: results/nonexistent/training_log.json
```

---

## KPI達成状況

### Requirement 7.6: 学習曲線と忘却率の可視化

| KPI | 目標 | 達成状況 |
|-----|------|---------|
| Loss曲線の可視化 | 実装 | ✅ 完了 |
| Perplexity曲線の可視化 | 実装 | ✅ 完了 |
| Γ変化の時系列プロット | 実装 | ✅ 完了 |
| Γヒートマップ | 実装 | ✅ 完了 |
| SNR統計の可視化 | 実装 | ✅ 完了 |

### Requirement 10.7: 共鳴情報の可視化

| KPI | 目標 | 達成状況 |
|-----|------|---------|
| 共鳴エネルギーヒートマップ | 実装 | ✅ 完了 |
| 共鳴マスクの可視化 | 実装 | ✅ 完了 |
| 共鳴成分数の追跡 | 実装 | ✅ 完了 |
| 総エネルギーの追跡 | 実装 | ✅ 完了 |

---

## 今後の拡張可能性

### 1. リアルタイム可視化

WandBやTensorBoardとの統合：

```python
import wandb

# 学習ループ内で
wandb.log({
    'loss': loss.item(),
    'gamma_mean': gamma.mean().item(),
    'snr_mean': snr_stats['mean_snr'],
    'num_resonant': resonance_info['num_resonant']
})
```

### 2. インタラクティブダッシュボード

Plotly Dashを使用：

```python
import plotly.graph_objects as go
from dash import Dash, dcc, html

app = Dash(__name__)
app.layout = html.Div([
    dcc.Graph(id='learning-curves'),
    dcc.Graph(id='gamma-evolution'),
    # ...
])
```

### 3. 比較可視化

複数の実験結果を比較：

```python
def plot_comparison(log_dirs: List[Path], labels: List[str]):
    """複数の実験結果を比較"""
    for log_dir, label in zip(log_dirs, labels):
        visualizer = Phase2Visualizer(log_dir, output_dir)
        # プロット
```

### 4. アニメーション

学習過程のアニメーション生成：

```python
from matplotlib.animation import FuncAnimation

def animate_gamma_evolution(gamma_data):
    """Γ変化のアニメーション"""
    # フレームごとにヒートマップを更新
```

---

## ドキュメント

### 作成したドキュメント

1. **クイックリファレンス**: `docs/quick-reference/PHASE2_VISUALIZATION_QUICK_REFERENCE.md`
   - 使用方法の詳細
   - トラブルシューティング
   - 高度な使い方

2. **実装レポート**: `results/benchmarks/PHASE2_VISUALIZATION_IMPLEMENTATION_REPORT.md`（本ドキュメント）
   - 実装詳細
   - テスト結果
   - KPI達成状況

### コード内ドキュメント

すべての関数にdocstringを記載：

```python
def plot_learning_curves(self):
    """
    学習曲線を可視化
    
    - Loss曲線
    - Perplexity曲線
    
    Requirements: 7.6
    """
```

---

## まとめ

### 実装完了項目

- ✅ Task 14.1: 学習曲線可視化の実装
- ✅ Task 14.2: Γ変化可視化の実装
- ✅ Task 14.3: SNR統計可視化の実装
- ✅ Task 14.4: 共鳴情報可視化の実装

### 主要な成果

1. **包括的な可視化**: Phase 2の全診断情報を可視化
2. **高品質なグラフ**: 300 DPIの出版品質
3. **柔軟な制御**: コマンドライン引数による細かい制御
4. **デモスクリプト**: すぐに試せるサンプル実装
5. **詳細なドキュメント**: クイックリファレンスとレポート

### 技術的ハイライト

- Seabornによる美しいグラフスタイル
- 移動平均による滑らかな曲線
- ヒートマップによる多次元データの可視化
- エラーハンドリングによる堅牢性

### 次のステップ

Task 14完了により、Phase 2の可視化インフラが整いました。次は：

1. **Task 15**: Phase 2実装ガイドの作成
2. **Task 16**: 使用例の作成
3. **Task 17**: Docstringの整備

---

## 参考資料

- **設計書**: `.kiro/specs/phase2-breath-of-life/design.md`
- **要件定義**: `.kiro/specs/phase2-breath-of-life/requirements.md`
- **タスクリスト**: `.kiro/specs/phase2-breath-of-life/tasks.md`
- **学習スクリプト**: `scripts/train_phase2.py`

---

**実装者**: Kiro AI Assistant  
**レビュー**: 未実施  
**承認**: 未実施
