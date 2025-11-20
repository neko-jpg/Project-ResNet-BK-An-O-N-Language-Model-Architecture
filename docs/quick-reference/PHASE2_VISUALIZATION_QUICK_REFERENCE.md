# Phase 2 可視化クイックリファレンス

Phase 2モデルの学習過程と診断情報を可視化するためのクイックリファレンスです。

## 概要

`scripts/visualize_phase2.py`は、Phase 2モデルの学習ログから以下の可視化を生成します：

1. **学習曲線**: Loss/Perplexityの時間変化
2. **Γ変化**: 各層の忘却率の時間変化とヒートマップ
3. **SNR統計**: 信号対雑音比の分布と時間変化
4. **共鳴情報**: 記憶共鳴のエネルギーとマスク

## 基本的な使い方

### 1. すべての可視化を実行

```bash
python scripts/visualize_phase2.py --log-dir results/phase2_training
```

出力: `results/visualizations/` に以下のファイルが生成されます
- `learning_curves.png`: 学習曲線
- `gamma_evolution.png`: Γ変化
- `snr_statistics.png`: SNR統計
- `resonance_info.png`: 共鳴情報

### 2. 出力ディレクトリを指定

```bash
python scripts/visualize_phase2.py \
    --log-dir results/phase2_training \
    --output-dir results/my_visualizations
```

### 3. 特定の可視化のみ実行

```bash
# 学習曲線のみ
python scripts/visualize_phase2.py --log-dir results/phase2_training --only learning_curves

# Γ変化のみ
python scripts/visualize_phase2.py --log-dir results/phase2_training --only gamma

# SNR統計のみ
python scripts/visualize_phase2.py --log-dir results/phase2_training --only snr

# 共鳴情報のみ
python scripts/visualize_phase2.py --log-dir results/phase2_training --only resonance
```

### 4. インタラクティブモード

```bash
python scripts/visualize_phase2.py --log-dir results/phase2_training --interactive
```

グラフがウィンドウで表示されます（保存もされます）。

## デモの実行

サンプルデータを使ったデモ：

```bash
python examples/phase2_visualization_demo.py
```

これにより、サンプルログが生成され、可視化が実行されます。

## 必要なログファイル

可視化スクリプトは以下のログファイルを期待します：

### 1. `training_log.json`

学習の基本メトリクス：

```json
{
  "loss": [5.0, 4.8, 4.6, ...],
  "perplexity": [148.4, 121.5, 99.5, ...]
}
```

### 2. `diagnostics_log.json`

Phase 2固有の診断情報：

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

## 学習スクリプトとの統合

`scripts/train_phase2.py`は自動的にこれらのログを生成します：

```python
# train_phase2.py内で
import json

# 学習ループ
training_log = {'loss': [], 'perplexity': []}
diagnostics_log = {'gamma_values': [], 'snr_stats': [], 'resonance_info': []}

for epoch in range(num_epochs):
    for batch in dataloader:
        # 学習ステップ
        loss = train_step(model, batch)
        
        # ログに記録
        training_log['loss'].append(loss.item())
        training_log['perplexity'].append(np.exp(loss.item()))
        
        # 診断情報を取得
        if step % log_interval == 0:
            _, diagnostics = model(batch, return_diagnostics=True)
            diagnostics_log['gamma_values'].append(diagnostics['gamma'])
            diagnostics_log['snr_stats'].append(diagnostics['snr_stats'])
            diagnostics_log['resonance_info'].append(diagnostics['resonance_info'])

# ログを保存
with open('results/phase2_training/training_log.json', 'w') as f:
    json.dump(training_log, f)

with open('results/phase2_training/diagnostics_log.json', 'w') as f:
    json.dump(diagnostics_log, f)
```

## 可視化の詳細

### 1. 学習曲線 (`learning_curves.png`)

**上段**: Training Loss
- 青線: 実際のLoss値
- 紫破線: 移動平均（ウィンドウサイズ50）

**下段**: Perplexity
- オレンジ線: 実際のPerplexity値
- 赤破線: 移動平均

**解釈**:
- Lossが減少していれば学習が進行中
- Perplexityが低いほど良いモデル
- Phase 1モデルと比較してPerplexity劣化が+5%以内であることを確認

### 2. Γ変化 (`gamma_evolution.png`)

**上段**: 時系列プロット
- 各層のΓ値の時間変化
- 層ごとに異なる色

**下段**: ヒートマップ
- 縦軸: 層
- 横軸: 学習ステップ
- 色: Γ値（黄色→赤で高い値）

**解釈**:
- Γが時間とともに変化していれば、適応的忘却が機能中
- 層ごとにΓが異なれば、階層的な記憶管理が実現
- Γが極端に大きい（>0.1）場合、過減衰の可能性

**KPI**:
- 学習初期と学習後でΓの平均値が0.1以上変動していること

### 3. SNR統計 (`snr_statistics.png`)

**左上**: 平均SNRの時間変化
- 青線: 平均SNR
- 青塗りつぶし: ±1標準偏差
- 赤破線: 閾値（2.0）

**右上**: SNR範囲（Min-Max）
- オレンジ塗りつぶし: SNR範囲
- 赤線: 平均SNR

**左下**: SNR分布ヒストグラム
- 全ステップのSNR値の分布
- 赤破線: 閾値（2.0）

**右下**: 低SNR比率の時間変化
- SNR < 2.0 の割合

**解釈**:
- SNRが閾値（2.0）を超えていれば、重要な記憶が保持されている
- SNRが時間とともに増加していれば、記憶の質が向上中
- 低SNR比率が減少していれば、ノイズが除去されている

**KPI**:
- SNR < 2.0 の信号に対して、Hebbian更新量が1/10以下に抑制されること

### 4. 共鳴情報 (`resonance_info.png`)

**左上**: 共鳴成分数の時間変化
- 青線: アクティブな共鳴モード数

**右上**: 総共鳴エネルギーの時間変化
- オレンジ線: 総エネルギー

**中段**: 共鳴エネルギーヒートマップ（最終ステップ）
- 縦軸: ヘッド
- 横軸: 次元
- 色: エネルギー値

**下段**: 共鳴マスク（最終ステップ）
- 縦軸: ヘッド
- 横軸: 次元
- 色: 緑=アクティブ、赤=非アクティブ

**解釈**:
- 共鳴成分数が適度（10-30）であれば、効率的な記憶配置
- エネルギーが特定の次元に集中していれば、重要記憶が共鳴中
- マスクで80%以上がフィルタリングされていれば、スパース性が達成

**KPI**:
- 記憶の80%以上がフィルタリングされ、上位20%のみが保持されること
- 共鳴層の計算時間が層全体の20%以下であること

## トラブルシューティング

### ログファイルが見つからない

```
警告: 学習ログが見つかりません: results/phase2_training/training_log.json
```

**解決策**:
1. `--log-dir`パスが正しいか確認
2. `train_phase2.py`が正常に実行されたか確認
3. ログファイルが生成されているか確認

### データ構造が不正

```
KeyError: 'gamma_values'
```

**解決策**:
1. `diagnostics_log.json`の構造を確認
2. `train_phase2.py`が最新版か確認
3. サンプルデータでテスト: `python examples/phase2_visualization_demo.py`

### グラフが空

```
警告: Γデータがないため、Γ変化をスキップします
```

**解決策**:
1. 診断情報が正しく記録されているか確認
2. `return_diagnostics=True`で学習しているか確認
3. ログ間隔（`log_interval`）が適切か確認

## 高度な使い方

### カスタム可視化の追加

`Phase2Visualizer`クラスを拡張：

```python
from scripts.visualize_phase2 import Phase2Visualizer

class MyVisualizer(Phase2Visualizer):
    def plot_custom_metric(self):
        """カスタムメトリクスの可視化"""
        # 独自の可視化ロジック
        pass

visualizer = MyVisualizer(log_dir, output_dir)
visualizer.plot_custom_metric()
```

### プログラムからの呼び出し

```python
from pathlib import Path
from scripts.visualize_phase2 import Phase2Visualizer

# 可視化実行
visualizer = Phase2Visualizer(
    log_dir=Path('results/phase2_training'),
    output_dir=Path('results/visualizations')
)

# 個別に実行
visualizer.plot_learning_curves()
visualizer.plot_gamma_evolution()
visualizer.plot_snr_statistics()
visualizer.plot_resonance_info()
```

## 参考資料

- **実装ガイド**: `docs/PHASE2_IMPLEMENTATION_GUIDE.md`
- **学習スクリプト**: `scripts/train_phase2.py`
- **デモスクリプト**: `examples/phase2_visualization_demo.py`
- **設計書**: `.kiro/specs/phase2-breath-of-life/design.md`

## 関連タスク

- Task 12: 学習スクリプトの実装（ログ生成）
- Task 14.1: 学習曲線可視化の実装 ✓
- Task 14.2: Γ変化可視化の実装 ✓
- Task 14.3: SNR統計可視化の実装 ✓
- Task 14.4: 共鳴情報可視化の実装 ✓

## 更新履歴

- 2024-XX-XX: 初版作成
- Task 14完了時点でのドキュメント
