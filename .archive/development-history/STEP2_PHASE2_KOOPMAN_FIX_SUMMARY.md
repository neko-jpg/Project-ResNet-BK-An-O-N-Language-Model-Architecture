# Step 2 Phase 2: Koopman Learning - 修正完了

## 📋 修正概要

前回のテスト結果から判明した問題を修正しました。

### 問題点

1. **Koopman予測が機能していない**
   - 標準forward: PPL 461
   - Koopman forward: PPL 8776 (+1802%!)
   - → Koopman演算子が正しく学習できていない

2. **Koopman演算子の更新が不十分**
   - 各層で0.05-0.06%の変化のみ
   - → 期待値: 5-10%の変化

3. **パープレキシティが高い**
   - 最終PPL: 461
   - → 期待値: 50-100

## ✅ 実施した修正

### 1. Koopman演算子の更新率を上げる

**ファイル**: `src/models/koopman_layer.py` (行 ~200)

```python
# 修正前
alpha = 0.1  # Learning rate for Koopman operator

# 修正後
alpha = 0.3  # Learning rate for Koopman operator (increased for faster adaptation)
```

**効果**: Koopman演算子の更新速度が3倍に → より速く最適な演算子を学習

### 2. バッファサイズを増やす

**ファイル**: `src/models/koopman_layer.py` (行 ~70)

```python
# 修正前
self.register_buffer('Z_current', torch.zeros(koopman_dim, 100))
self.register_buffer('Z_next', torch.zeros(koopman_dim, 100))

# 修正後
buffer_size = 500
self.register_buffer('Z_current', torch.zeros(koopman_dim, buffer_size))
self.register_buffer('Z_next', torch.zeros(koopman_dim, buffer_size))
```

**効果**: DMD推定の精度が向上 → より正確なKoopman演算子

### 3. Koopman損失の重みを上げる

**ファイル**: `notebooks/step2_phase2_koopman.ipynb`

```python
# 修正前
KOOPMAN_WEIGHT_MAX = 0.1

# 修正後
KOOPMAN_WEIGHT_MAX = 0.5
```

**効果**: Koopman損失の影響が5倍に → Koopman学習が強化される

### 4. エポック数を増やす

**ファイル**: `notebooks/step2_phase2_koopman.ipynb`

```python
# 修正前
NUM_EPOCHS = 5

# 修正後
NUM_EPOCHS = 10
```

**効果**: Koopman学習に十分な時間を確保 → より良い収束

## 📊 期待される改善

### 修正前の結果
```
Koopman演算子変化: 0.05-0.06%
Koopman予測PPL: 8776 (標準の19倍)
最終PPL: 461
訓練エポック: 5
```

### 修正後の期待値
```
Koopman演算子変化: 5-10% (100倍改善)
Koopman予測PPL: 500-600 (標準と同程度)
最終PPL: 100-200 (2-4倍改善)
訓練エポック: 10
```

## 🔍 検証ポイント

修正後のテストで以下を確認してください：

### 1. Koopman演算子の更新
```
✓ Mean absolute change > 0.5 (各層)
✓ Relative change > 5%
✓ Final operator norm が変化している
```

### 2. Koopman予測の精度
```
✓ Koopman forward PPL < Standard PPL × 2
✓ Difference < +200%
✓ Koopman loss が減少している
```

### 3. 全体的な性能
```
✓ Final PPL < 200
✓ Loss が単調減少
✓ 訓練が安定している (NaN/Infなし)
```

## 🚀 次のステップ

### Google Colabでテスト

1. **ノートブックを開く**
   ```
   notebooks/step2_phase2_koopman.ipynb
   ```

2. **実行**
   - Runtime → Restart and run all
   - または、セルを順に実行

3. **結果を確認**
   - Koopman演算子の変化
   - Koopman予測の精度
   - 最終パープレキシティ

### 成功基準

以下の条件を満たせば成功：

- ✅ Koopman演算子変化 > 5%
- ✅ Koopman予測PPL < 標準PPL × 2
- ✅ 最終PPL < 200
- ✅ 訓練が安定

### 失敗時の対応

もし結果が改善しない場合：

**オプション1: さらに強化**
```python
alpha = 0.5  # 0.3 → 0.5
KOOPMAN_WEIGHT_MAX = 0.8  # 0.5 → 0.8
NUM_EPOCHS = 15  # 10 → 15
```

**オプション2: 学習率調整**
```python
LEARNING_RATE = 5e-4  # 1e-3 → 5e-4 (より慎重に)
```

**オプション3: バッファサイズ増加**
```python
buffer_size = 1000  # 500 → 1000
```

## 📝 修正ファイル一覧

### 自動修正済み
- ✅ `src/models/koopman_layer.py` - alpha, buffer_size
- ✅ `notebooks/step2_phase2_koopman.ipynb` - KOOPMAN_WEIGHT_MAX, NUM_EPOCHS

### 補助ファイル
- 📄 `KOOPMAN_FIX_INSTRUCTIONS.md` - 詳細な修正手順
- 📄 `fix_koopman_notebook.py` - ノートブック修正スクリプト
- 📄 `STEP2_PHASE2_KOOPMAN_FIX_SUMMARY.md` - このファイル

## 🔧 トラブルシューティング

### 問題: メモリ不足
```python
# 対策
buffer_size = 300  # 500 → 300
BATCH_SIZE = 16  # 32 → 16
```

### 問題: 訓練が不安定
```python
# 対策
KOOPMAN_WEIGHT_MAX = 0.3  # 0.5 → 0.3
LEARNING_RATE = 5e-4  # 1e-3 → 5e-4
```

### 問題: Koopman演算子がまだ更新されない
```python
# 対策
alpha = 0.5  # 0.3 → 0.5
buffer_size = 1000  # 500 → 1000
```

## 📈 期待される訓練曲線

### 損失
```
Epoch 1-3: 勾配学習のみ (warmup)
  Loss: 7.0 → 6.3
  Koopman loss: 0.0

Epoch 4-10: ハイブリッド学習
  Loss: 6.3 → 5.5
  Koopman loss: 0.5 → 0.001 (減少)
```

### Koopman演算子
```
Epoch 1-3: 変化なし (warmup)
  Change: 0%

Epoch 4-10: 学習中
  Change: 徐々に増加 → 5-10%
```

### パープレキシティ
```
Epoch 1: 580
Epoch 3: 490
Epoch 5: 350
Epoch 10: 150-200 (目標)
```

## ✅ 完了チェックリスト

修正完了後、以下を確認：

- [x] `src/models/koopman_layer.py` の alpha を 0.3 に変更
- [x] `src/models/koopman_layer.py` の buffer_size を 500 に変更
- [x] `notebooks/step2_phase2_koopman.ipynb` の KOOPMAN_WEIGHT_MAX を 0.5 に変更
- [x] `notebooks/step2_phase2_koopman.ipynb` の NUM_EPOCHS を 10 に変更
- [ ] Google Colab でテスト実行
- [ ] 結果が期待値を満たすか確認
- [ ] Phase 3 へ進む準備完了

## 🎯 まとめ

**修正内容**:
- Koopman更新率: 0.1 → 0.3 (3倍)
- バッファサイズ: 100 → 500 (5倍)
- Koopman重み: 0.1 → 0.5 (5倍)
- エポック数: 5 → 10 (2倍)

**期待される効果**:
- Koopman演算子の変化: 0.05% → 5-10% (100倍改善)
- Koopman予測精度: 19倍悪化 → 同程度
- 最終PPL: 461 → 150-200 (2-3倍改善)

これらの修正により、Koopman学習が正しく機能し、Phase 3へ進む準備が整います。
