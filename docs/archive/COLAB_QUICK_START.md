# Google Colab クイックスタートガイド

## 🚀 5分で始める

### 1. Colabでノートブックを開く

このリンクをクリック：
👉 [Step 2 Phase 1 Colab Notebook](https://colab.research.google.com/github/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/blob/main/notebooks/step2_phase1_colab.ipynb)

### 2. GPU設定

1. 「ランタイム」→「ランタイムのタイプを変更」
2. 「ハードウェアアクセラレータ」→「**T4 GPU**」を選択
3. 「保存」をクリック

### 3. 実行

「ランタイム」→「すべてのセルを実行」をクリック

**実行時間:** 約20-30分

### 4. 完了！

実行が完了すると、`step2_phase1_results.zip`が自動的にダウンロードされます。

---

## 📊 期待される結果

実行後、以下が確認できます：

- ✅ Mixed precision speedup: **1.5-2.0×**
- ✅ Batched gradient speedup: **2.0-2.5×**
- ✅ 最適なGRAD_BLEND値: **0.3-0.7**
- ✅ Validation perplexity: **減少傾向**
- ✅ 数値安定性: **NaN/Infなし**

---

## 🔧 トラブルシューティング

### メモリ不足エラーが出た場合

セル3（GRAD_BLEND Grid Search）で以下のように変更：

```python
# バッチサイズを減らす
train_loader, val_loader, vocab_size = get_wikitext2_dataloaders(
    batch_size=16,  # 32 → 16に変更
    seq_len=128,
    num_workers=2
)

# モデルサイズを小さくする
config = ResNetBKConfig(
    vocab_size=vocab_size,
    d_model=32,  # 64 → 32に変更
    n_layers=2,  # 4 → 2に変更
    n_seq=64,    # 128 → 64に変更
    num_experts=2,  # 4 → 2に変更
    top_k=1
)
```

### もっと速くテストしたい場合（5分で完了）

セル3で以下のように変更：

```python
optimizer = GradBlendOptimizer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    alpha_values=[0.0, 0.5, 1.0],  # 3つだけテスト
    epochs_per_trial=1,  # 1エポックのみ
    device=device,
    save_dir='results/step2_phase1_colab'
)
```

セル4で：

```python
# 3エポック → 1エポックに変更
for epoch in range(1):  # 3 → 1に変更
```

---

## 📁 ダウンロードされるファイル

`step2_phase1_results.zip`には以下が含まれます：

```
step2_phase1_results.zip
├── results/
│   └── step2_phase1_colab/
│       ├── grad_blend_results.json      # Grid search結果
│       ├── grad_blend_analysis.png      # 可視化グラフ
│       └── summary.json                 # サマリー
├── checkpoints/
│   └── step2_phase1_colab.pt           # 学習済みモデル
└── training_curves.png                  # 学習曲線
```

---

## 📚 詳細ドキュメント

より詳しい情報は以下を参照：

- [完全なセットアップガイド](notebooks/COLAB_SETUP_GUIDE.md)
- [実装サマリー](STEP2_PHASE1_IMPLEMENTATION.md)
- [プロジェクト構造](PROJECT_STRUCTURE.md)

---

## 🎯 次のステップ

Step 2 Phase 1が完了したら：

1. **結果を確認**
   - `grad_blend_results.json`で最適なα値を確認
   - `training_curves.png`で学習曲線を確認

2. **次のタスクに進む**
   - Task 3: Koopman Operator Learning
   - Task 4: Physics-Informed Learning

3. **ローカルで実行**（オプション）
   ```bash
   git pull origin main
   python -m pytest tests/test_step2_phase1.py -v
   ```

---

## 💡 ヒント

- **無料のT4 GPU**で十分実行できます
- **Colab Pro**を使うとさらに高速（A100 GPU）
- 実行中にブラウザを閉じても大丈夫（バックグラウンドで実行）
- セッションは最大12時間まで（無料版）

---

## ❓ 質問・問題

問題が発生した場合：

1. [Issues](https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/issues)で報告
2. エラーメッセージをコピーして共有
3. 実行環境（GPU種類、メモリ）を記載

---

**Happy Training! 🚀**
