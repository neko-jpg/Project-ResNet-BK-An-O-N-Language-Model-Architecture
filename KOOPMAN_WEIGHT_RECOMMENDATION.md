# Koopman Weight 推奨設定

## 📊 現在の問題

### 観察された結果
```
Epoch 4: Train Loss: 6.2687 (LM: 6.2687, Koopman: 0.0000)
Epoch 5: Train Loss: 6.3411 (LM: 6.2275, Koopman: 2.0563)  Weight: 0.0714
Epoch 6: Train Loss: 6.6788 (LM: 6.1917, Koopman: 3.9526)  Weight: 0.1429
Epoch 7: Train Loss: 7.1401 (LM: 6.2062, Koopman: 4.9417)  Weight: 0.2143
Epoch 8: Train Loss: 7.5829 (LM: 6.2114, Koopman: 4.8001)  Weight: 0.2857
Epoch 9: Train Loss: 7.7511 (LM: 6.1874, Koopman: 5.1835)  Weight: 0.3571
Epoch 10: Train Loss: 8.0606 (LM: 6.2183, Koopman: 4.6448) Weight: 0.4286
```

### 問題点
1. **Koopman lossが高すぎる** (2〜5の範囲、生の値は10〜17)
2. **Total lossが増加** (6.27 → 8.06)
3. **LM lossは安定** (6.19〜6.27) ← これは良い！
4. **Koopman weightが高すぎる** (最大0.5は過剰)

## ✅ 推奨設定

### Option 1: 保守的アプローチ（推奨）
```python
# Notebookのセル4を修正
KOOPMAN_WEIGHT_MAX = 0.05  # 0.5 → 0.05 に変更
KOOPMAN_START_EPOCH = 4    # 3 → 4 に変更（より長いwarmup）
fallback_threshold=8.0     # 10.0 → 8.0 に変更（より早く減衰）
```

**期待される効果**:
- Koopman lossの影響を1/10に削減
- Total loss: 6.2 + 0.05 * 5 = 6.45 程度（許容範囲）
- より安定した学習

### Option 2: 段階的アプローチ
```python
KOOPMAN_WEIGHT_MAX = 0.1   # 中間的な値
KOOPMAN_START_EPOCH = 5    # さらに長いwarmup
schedule_type='exponential' # 指数的に増加（より緩やか）
fallback_threshold=7.0     # より厳しい閾値
```

### Option 3: 最小限アプローチ
```python
KOOPMAN_WEIGHT_MAX = 0.02  # 非常に小さい値
KOOPMAN_START_EPOCH = 5
fallback_threshold=6.0     # 非常に厳しい閾値
```

## 🧮 計算根拠

### 現在の状況
```
LM Loss: ~6.2
Koopman Loss (平均): ~4.5 (生の値: ~15)
Weight: 0.4286
Total Loss: 6.2 + 0.4286 * 4.5 = 8.13 ✗ 高すぎる
```

### 推奨設定（Option 1）
```
LM Loss: ~6.2
Koopman Loss (平均): ~4.5
Weight: 0.05
Total Loss: 6.2 + 0.05 * 4.5 = 6.425 ✓ 許容範囲
```

### 理想的なバランス
```
Koopman lossの寄与 < LM lossの10%
6.2 * 0.1 = 0.62
0.62 / 4.5 = 0.138 → Weight上限は0.1〜0.15が適切
```

## 📝 修正手順

### Notebookで修正する場合

**セル4（Initialize Trainer）を以下に変更**:
```python
# Training configuration
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
KOOPMAN_START_EPOCH = 4  # 3 → 4 に変更
KOOPMAN_WEIGHT_MAX = 0.05  # 0.5 → 0.05 に変更 ★重要★

# Optimizer and criterion
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Create hybrid Koopman trainer
trainer = HybridKoopmanTrainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    koopman_weight_min=0.0,
    koopman_weight_max=KOOPMAN_WEIGHT_MAX,
    koopman_start_epoch=KOOPMAN_START_EPOCH,
    total_epochs=NUM_EPOCHS,
    schedule_type='linear',
    enable_koopman_updates=True,
    fallback_threshold=8.0,  # 10.0 → 8.0 に変更
    device=device
)

print(f"\nTraining Configuration:")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Koopman start epoch: {KOOPMAN_START_EPOCH}")
print(f"  Koopman weight max: {KOOPMAN_WEIGHT_MAX}")
print(f"  Fallback threshold: 8.0")
print(f"  Batch size: {BATCH_SIZE}")
```

### 期待される結果

```
Epoch 4/10:
  Train Loss: 6.2687 (LM: 6.2687, Koopman: 0.0000)
  Koopman Weight: 0.0000
  Koopman Enabled: False

Epoch 5/10:
  Train Loss: 6.2456 (LM: 6.2275, Koopman: 2.5123)
  Koopman Weight: 0.0071  # 0.05 / 7 epochs
  Koopman Enabled: True

Epoch 10/10:
  Train Loss: 6.4123 (LM: 6.1987, Koopman: 4.2567)
  Koopman Weight: 0.0500  # 最大値
  Val PPL: ~480 (ベースラインの30%以内)
```

## 🎯 成功の指標

### ✅ 良い兆候
- Total loss < 6.5
- Val perplexity < 620 (baseline 477 * 1.3)
- LM lossが安定または減少
- Koopman lossが徐々に減少
- 警告が1 epochにつき1回以下

### ⚠️ 調整が必要な兆候
- Total loss > 7.0
- Val perplexityが増加し続ける
- 警告が頻繁に出る
- Koopman lossが増加

## 🔬 さらなる最適化

### Koopman lossが高い根本原因

1. **表現空間が未安定**
   - Epoch 4では言語モデルがまだ収束していない
   - 線形マッピング（Koopman operator）では予測困難

2. **Koopman dimが大きすぎる可能性**
   - 現在: 256次元
   - 推奨: 128次元に削減を検討

3. **Lifting関数の改善**
   - 現在: Tanh活性化
   - 検討: LayerNormやBatchNormの追加

### 長期的な改善案

```python
# より安定したKoopman学習
KOOPMAN_DIM = 128  # 256 → 128
KOOPMAN_START_EPOCH = 5  # より長いwarmup
KOOPMAN_WEIGHT_MAX = 0.03  # より保守的
NUM_EPOCHS = 15  # より長い学習
```

---

**推奨**: まずOption 1を試して、結果を見てから調整してください！
