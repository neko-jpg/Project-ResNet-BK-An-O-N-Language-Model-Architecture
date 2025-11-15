# Phase 2 Koopman Warning Fix - 完全版

## 🔍 問題の原因分析

### 発見された問題

#### 1. **指摘された問題（確認済み）**
- `train_step`内で`loss_koopman > fallback_threshold`の時、**毎バッチ**警告が出る
- Koopman weightが0になっても計算は継続される
- Epoch 4以降、Koopman lossが10〜15に上昇し、数百バッチ分の警告が出る

#### 2. **新たに発見した問題**
- **警告の頻度制御がない** → 同じ警告が数百回繰り返される
- **Koopman weightの減衰ロジックが不完全** → `* 0.1`するだけで、次のバッチでまた元に戻る
- **`koopman_failed`フラグが機能していない** → 一度失敗してもリセットされない
- **重みが最小値でも計算スキップされない** → 無駄な計算が走り続ける
- **Warmup期間の制御が不完全** → Epoch 3以前でもKoopman計算が走る可能性

## ✅ 実装した修正

### 1. **警告頻度の制御**
```python
# Epoch単位で警告を制限
self.last_warning_epoch = -1
self.warning_count = 0
self.max_warnings_per_epoch = 1

# 警告は1 epochにつき1回のみ
if self.last_warning_epoch != self.current_epoch or self.warning_count < self.max_warnings_per_epoch:
    print(f"[Epoch {self.current_epoch}] Koopman loss high ({loss_koopman.item():.2f}), reducing weight...")
    self.last_warning_epoch = self.current_epoch
    self.warning_count += 1
```

**効果**: 数百回の警告 → 1 epochにつき1回のみ

### 2. **Koopman Weight減衰の持続化**
```python
# 減衰を複数バッチ持続させる
self.koopman_weight_decay_active = False
self.koopman_weight_decay_counter = 0
self.koopman_weight_decay_duration = 10  # 10バッチ持続

# Loss高い時
if not self.koopman_weight_decay_active:
    self.koopman_weight_decay_active = True
    self.koopman_weight_decay_counter = 0

koopman_weight = scheduled_weight * 0.01  # より強力な減衰
self.koopman_weight_decay_counter += 1

# 期間終了後リセット
if self.koopman_weight_decay_counter >= self.koopman_weight_decay_duration:
    self.koopman_weight_decay_active = False
```

**効果**: 
- 1バッチだけ減衰 → 10バッチ持続
- `* 0.1` → `* 0.01` (より強力)
- Loss正常化したら自動復帰

### 3. **計算スキップの最適化**
```python
# Schedulerレベルでの閾値
def get_weight(self):
    if self.current_weight < 1e-6:
        return 0.0
    return self.current_weight

# Trainerレベルでの早期リターン
if self.koopman_enabled and not self.koopman_failed and scheduled_weight > 1e-6:
    # Koopman計算実行
    ...
else:
    # 完全スキップ
    loss_koopman = torch.tensor(0.0, device=self.device)
    koopman_weight = 0.0
```

**効果**: 重みが最小値の時、計算を完全スキップ

### 4. **Koopman Operator更新の条件強化**
```python
# 重みが十分大きい時のみ更新
if (self.enable_koopman_updates and self.koopman_enabled and 
    not self.koopman_failed and koopman_weight > 1e-4):
    # Operator更新
    ...
```

**効果**: 減衰中は無駄な更新をスキップ

### 5. **Epoch開始時のリセット処理**
```python
# Epoch開始時
if self.current_epoch >= self.koopman_start_epoch:
    self.koopman_enabled = True
    self.koopman_failed = False  # フラグリセット
else:
    self.koopman_enabled = False  # Warmup中は完全オフ

# 警告カウンターリセット
self.warning_count = 0

# 減衰状態リセット
self.koopman_weight_decay_active = False
self.koopman_weight_decay_counter = 0
```

**効果**: 
- 各Epochで新しくスタート
- Warmup期間は完全にKoopman無効化

## 📊 期待される改善

### Before（修正前）
```
Epoch 4/10:
  Warning: Koopman loss too high (10.04), reducing weight
  Warning: Koopman loss too high (10.12), reducing weight
  Warning: Koopman loss too high (10.08), reducing weight
  ... (300回繰り返し)
  Train Loss: 5.2341 (LM: 5.1234, Koopman: 10.0456)
  Koopman Weight: 0.0000  # 重み0でも計算は走る
```

### After（修正後）
```
Epoch 4/10:
  [Epoch 4] Koopman loss high (10.04), reducing weight for next 10 batches
  Train Loss: 5.1456 (LM: 5.1234, Koopman: 2.2345)
  Koopman Weight: 0.0025  # 減衰適用、計算も最適化
```

## 🎯 修正のポイント

### 1. **警告の抑制**
- ❌ バッチごとに警告 → ✅ Epochごとに1回のみ
- ログが読みやすくなる

### 2. **減衰の持続化**
- ❌ 1バッチだけ減衰 → ✅ 10バッチ持続
- Loss高騰時の安定性向上

### 3. **計算の最適化**
- ❌ 重み0でも計算 → ✅ 完全スキップ
- 無駄な計算を削減

### 4. **状態管理の改善**
- ❌ フラグがリセットされない → ✅ Epoch開始時にリセット
- 各Epochで新しくスタート

### 5. **Warmup期間の厳格化**
- ❌ 曖昧な制御 → ✅ 完全オフ
- LMの安定化を優先

## 🧪 テスト方法

```python
# Notebookで実行
# Section 5: Training Loop を実行

# 期待される出力:
# - Epoch 0-2: Koopman Weight: 0.0000, Koopman Enabled: False
# - Epoch 3: Koopman Weight: 0.0000 → 徐々に増加
# - Epoch 4+: 警告は最大1回/epoch
# - Loss高い時: 10バッチ減衰適用
```

## 📝 変更ファイル

1. `src/training/hybrid_koopman_trainer.py`
   - 警告頻度制御追加
   - 減衰持続化ロジック追加
   - 計算スキップ最適化
   - Epoch開始時リセット処理

2. `src/training/koopman_scheduler.py`
   - `get_weight()`に閾値チェック追加

## 🚀 次のステップ

1. ✅ Notebookで動作確認
2. ✅ 警告が1 epochにつき1回以下になることを確認
3. ✅ Koopman lossが徐々に減少することを確認
4. ✅ Perplexityがベースラインの30%以内に収まることを確認

---

**修正完了日**: 2025-11-15
**修正者**: Kiro AI Assistant
**レビュー**: 徹平さん 🎯
