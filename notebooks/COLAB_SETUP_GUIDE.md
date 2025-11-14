# Google Colab実行ガイド

## Step 2 Phase 1をColabで実行する方法

### 1. Colabでノートブックを開く

1. [Google Colab](https://colab.research.google.com/)にアクセス
2. 「ファイル」→「ノートブックを開く」→「GitHub」タブ
3. リポジトリURL: `https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture`
4. `notebooks/step2_phase1_colab.ipynb`を選択

### 2. GPU設定

1. 「ランタイム」→「ランタイムのタイプを変更」
2. 「ハードウェアアクセラレータ」を「GPU」に設定（T4推奨）
3. 「保存」をクリック

### 3. 最初のセルに以下を追加

```python
# Google Colab Setup
import os
import sys

# Clone repository (初回のみ)
REPO_NAME = 'Project-ResNet-BK-An-O-N-Language-Model-Architecture'
if not os.path.exists(REPO_NAME):
    !git clone https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture.git {REPO_NAME}
    
# Change to project directory
os.chdir(REPO_NAME)

# Install dependencies
!pip install -q torch torchvision datasets transformers matplotlib numpy

# Verify GPU
import torch
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### 4. 実行

すべてのセルを順番に実行：
- 「ランタイム」→「すべてのセルを実行」

### 5. 期待される結果

実行時間（T4 GPU）:
- Analytic MoE validation: ~10秒
- Mixed precision benchmark: ~30秒
- Batched gradient profiling: ~1分
- GRAD_BLEND grid search: ~5-10分
- 3-epoch training: ~10-15分

**合計: 約20-30分**

### 6. 結果の確認

実行後、以下が生成されます：
- `results/step2_phase1_grad_blend_quick/` - Grid search結果
- `checkpoints/step2_phase1_model.pt` - 学習済みモデル
- `*.png` - 可視化グラフ

### 7. 結果のダウンロード

```python
# Zip results
!zip -r step2_phase1_results.zip results/ checkpoints/ *.png

# Download
from google.colab import files
files.download('step2_phase1_results.zip')
```

## トラブルシューティング

### メモリ不足エラー

バッチサイズを減らす：
```python
# データローダーのバッチサイズを変更
train_loader, val_loader, vocab_size = get_wikitext2_dataloaders(
    batch_size=16,  # 32から16に変更
    seq_len=128,
    num_workers=2
)
```

### データセットのダウンロードエラー

手動でキャッシュをクリア：
```python
!rm -rf ~/.cache/huggingface/datasets
```

### GPU メモリ不足

モデルサイズを小さくする：
```python
config = ResNetBKConfig(
    vocab_size=vocab_size,
    d_model=32,  # 64から32に変更
    n_layers=2,  # 4から2に変更
    n_seq=64,    # 128から64に変更
    num_experts=2,  # 4から2に変更
    top_k=1
)
```

## クイックテスト版（5分で完了）

時間を節約したい場合：

```python
# Quick test configuration
optimizer = GradBlendOptimizer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    alpha_values=[0.0, 0.5, 1.0],  # 3つだけテスト
    epochs_per_trial=1,  # 1エポックのみ
    device=device,
    save_dir='results/step2_phase1_quick'
)
```

## 完全版の実行（本番用）

完全なgrid searchを実行する場合：

```python
# Full configuration
optimizer = GradBlendOptimizer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    alpha_values=[i * 0.1 for i in range(11)],  # 0.0, 0.1, ..., 1.0
    epochs_per_trial=5,  # 5エポック
    device=device,
    save_dir='results/step2_phase1_full'
)
```

実行時間: 約2-3時間（T4 GPU）

## 次のステップ

Step 2 Phase 1が完了したら：
1. 結果を確認（perplexity、convergence speed）
2. 最適なGRAD_BLEND値を記録
3. Task 3（Koopman Operator Learning）に進む

## 参考リンク

- [Google Colab公式ドキュメント](https://colab.research.google.com/notebooks/intro.ipynb)
- [PyTorch on Colab](https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/pytorch_tutorial.ipynb)
