# Phase 8 訓練コマンド一覧

## 🚀 クイックスタート

### 1. 環境確認
```bash
make verify-phase7  # Phase 8も同じコマンドでOK
```

### 2. テスト実行（1分で完了）
```bash
make train-phase8-small
```

### 3. 本番訓練
```bash
# データセット設定
make recipe

# 訓練開始
make train-phase8
```

## 📋 全コマンド

### テスト系
```bash
# 小規模テスト（d=256, L=4, 1エポック）
make train-phase8-small

# ダミーデータテスト（最大設定で動作確認）
make train-phase8-test
```

### 訓練系
```bash
# 標準設定（d=512, L=12, ~150M params）
make train-phase8

# 最大設定（d=4096, L=32, ~3B params, 8GB VRAM）
make train-phase8-max

# 最大設定 + SSM（実験的、メモリ増加）
make train-phase8-max-ssm
```

### 再開
```bash
# チェックポイントから再開
make train-phase8-resume CHECKPOINT=checkpoints/phase8/epoch_5.pt
```

### ベンチマーク
```bash
# Phase 7とPhase 8の比較
make bench-phase8-vs-phase7
```

## 🎯 推奨設定

### RTX 3080 (8GB)
```bash
# 最大設定で訓練
make train-phase8-max
```
- Parameters: 3.08B
- VRAM: 5.81 GB
- Batch Size: 1
- Gradient Accumulation: 16

### RTX 3060 (12GB)
```bash
# バッチサイズを増やす
make train-phase8-max BATCH_SIZE=2
```

### RTX 4090 (24GB)
```bash
# さらに大きなモデル
make train-phase8 D_MODEL=6144 N_LAYERS=48 BATCH_SIZE=4
```

## ⚙️ パラメータ調整

### モデルサイズ
```bash
# d_modelを変更
make train-phase8 D_MODEL=768

# レイヤー数を変更
make train-phase8 N_LAYERS=16

# 両方変更
make train-phase8 D_MODEL=1024 N_LAYERS=24
```

### 訓練設定
```bash
# バッチサイズ
make train-phase8 BATCH_SIZE=4

# エポック数
make train-phase8 EPOCHS=20

# シーケンス長
make train-phase8 N_SEQ=1024
```

### 複数パラメータ
```bash
make train-phase8 D_MODEL=768 N_LAYERS=16 BATCH_SIZE=4 EPOCHS=20
```

## 📊 期待される結果

### 標準設定（512次元、12層）
- Parameters: ~150M
- VRAM: ~2-3 GB
- 訓練速度: ~1000 tokens/sec
- 収束: 5-10 epochs

### 最大設定（4096次元、32層）
- Parameters: ~3.08B
- VRAM: ~5.81 GB
- 訓練速度: ~200 tokens/sec
- 収束: 10-20 epochs

## 🔧 トラブルシューティング

### OOM (メモリ不足)
```bash
# バッチサイズを1に
make train-phase8-max BATCH_SIZE=1

# シーケンス長を減らす
make train-phase8-max N_SEQ=256

# モデルサイズを減らす
make train-phase8 D_MODEL=2048 N_LAYERS=24
```

### 訓練が遅い
```bash
# Tritonカーネル確認
make verify-triton

# 小さいモデルでテスト
make train-phase8-small
```

### エラーが出る
```bash
# 環境診断
make doctor

# Phase 8モジュール確認
python -c "from src.models.phase8.linear_attention import TangentSpaceLinearAttention; print('OK')"
```

## 📈 Phase 7との比較

| 項目 | Phase 7 | Phase 8 | コマンド |
|------|---------|---------|---------|
| 訓練 | `make train-phase7` | `make train-phase8` | - |
| 最大設定 | `make train-phase7-max` | `make train-phase8-max` | - |
| テスト | `make train-phase7-small` | `make train-phase8-small` | - |
| 比較 | - | - | `make bench-phase8-vs-phase7` |

### 主な違い
- **Phase 7**: O(N²)アテンション、安定性重視
- **Phase 8**: O(N)アテンション、速度重視

### どちらを使う？
- **初めて**: Phase 7（安定）
- **速度重視**: Phase 8
- **研究目的**: Phase 8

## 💡 ヒント

### データセット設定
```bash
# 最初に必ず実行
make recipe
```

### チェックポイント管理
```bash
# 自動保存先
checkpoints/phase8/epoch_*.pt

# 最終モデル
checkpoints/phase8/final_model.pt
```

### ログ確認
```bash
# 訓練サマリー
cat checkpoints/phase8/training_summary.json

# WandB（設定した場合）
# https://wandb.ai/your-project
```

## 🎓 次のステップ

### 1. 評価
```bash
# Perplexity測定
python scripts/evaluate_phase8.py

# 長文脈テスト
python scripts/test_long_context.py --phase 8
```

### 2. 推論
```bash
# チャット
make chat-ai CHECKPOINT=checkpoints/phase8/final_model.pt
```

### 3. 論文執筆
訓練結果を `paper/main.tex` に追記

## 📚 参考資料

- [Phase 8 Quick Start](PHASE8_QUICK_START.md)
- [Phase 7 vs Phase 8比較](../results/benchmarks/PHASE7_VS_PHASE8_FINAL_SUMMARY_JP.md)
- [設計書](.kiro/specs/phase8-hyperbolic-transcendence/design.md)
