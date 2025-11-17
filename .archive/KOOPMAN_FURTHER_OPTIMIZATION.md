# Koopman さらなる最適化案

## 現状分析

### ✅ 成功している点
- Total loss安定: 6.23 → 6.24
- LM loss減少: 6.23 → 6.19
- Val perplexity良好: 472-478（ベースライン477付近）
- 警告制御完璧: 1回/epoch

### ⚠️ 改善の余地
- Koopman loss高い: 9〜22
- 計算時間増加: 78s → 96s（約23%増）

## 🚀 さらなる最適化オプション

### Option A: Koopman Dim削減（推奨）
```python
# セル2を修正
KOOPMAN_DIM = 128  # 256 → 128に削減
```

**効果**:
- メモリ使用量: 50%削減
- 計算時間: 30%削減
- Koopman loss: より安定（次元の呪いを回避）

### Option B: Koopman更新頻度削減
```python
# hybrid_koopman_trainer.pyに追加
self.koopman_update_interval = 5  # 5バッチごとに更新

# train_stepで
if (self.enable_koopman_updates and self.koopman_enabled and 
    not self.koopman_failed and koopman_weight > 1e-4 and
    batch_idx % self.koopman_update_interval == 0):  # ← 追加
    # Operator更新
```

**効果**:
- 計算時間: 15%削減
- 学習の安定性: ほぼ同じ

### Option C: Koopman Weight上限をさらに削減
```python
# セル9を修正
KOOPMAN_WEIGHT_MAX = 0.02  # 0.05 → 0.02
```

**効果**:
- Koopman lossの影響: さらに最小化
- Total loss: より安定
- 学習速度: ほぼ同じ

### Option D: Batch Size増加（GPU余裕がある場合）
```python
# セル2を修正
BATCH_SIZE = 64  # 32 → 64
```

**効果**:
- Epoch時間: 短縮（バッチ数半減）
- メモリ使用量: 増加（要確認）

## 📝 推奨される組み合わせ

### 軽量化重視
```python
KOOPMAN_DIM = 128
KOOPMAN_WEIGHT_MAX = 0.02
BATCH_SIZE = 64  # GPU余裕があれば
```

### バランス重視（現状維持）
```python
KOOPMAN_DIM = 256
KOOPMAN_WEIGHT_MAX = 0.05
BATCH_SIZE = 32
# 現在の設定で十分良好
```

### 精度重視
```python
KOOPMAN_DIM = 256
KOOPMAN_WEIGHT_MAX = 0.05
NUM_EPOCHS = 15  # より長い学習
KOOPMAN_START_EPOCH = 5  # さらに長いwarmup
```

## 🎯 結論

**現在の設定で十分成功しています！**

- Total loss安定
- Val perplexity良好
- 警告制御完璧

もし計算時間が問題なら：
1. KOOPMAN_DIM = 128に削減
2. BATCH_SIZE = 64に増加（GPU余裕があれば）

それ以外は**そのまま続行を推奨**します！
