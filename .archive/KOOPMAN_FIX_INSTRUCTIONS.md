# Koopman Learning Phase 2 - 修正手順

## 問題の診断

前回のテスト結果から以下の問題が判明：

1. **Koopman予測が機能していない**: PPL 461 → 8776 (19倍悪化)
2. **Koopman演算子の更新が不十分**: 0.05-0.06%の変化のみ
3. **パープレキシティが高い**: 461 (期待値: 50-100)

## 実施した修正

### 1. Koopman演算子の更新率を上げる
**ファイル**: `src/models/koopman_layer.py`

```python
# 修正前
alpha = 0.1  # Learning rate for Koopman operator

# 修正後
alpha = 0.3  # Learning rate for Koopman operator (increased for faster adaptation)
```

**理由**: 0.1では更新が遅すぎて、Koopman演算子が十分に学習できない

### 2. バッファサイズを増やす
**ファイル**: `src/models/koopman_layer.py`

```python
# 修正前
self.register_buffer('Z_current', torch.zeros(koopman_dim, 100))
self.register_buffer('Z_next', torch.zeros(koopman_dim, 100))

# 修正後
buffer_size = 500
self.register_buffer('Z_current', torch.zeros(koopman_dim, buffer_size))
self.register_buffer('Z_next', torch.zeros(koopman_dim, buffer_size))
```

**理由**: バッファサイズが小さいと、DMD (Dynamic Mode Decomposition) の推定精度が低い

### 3. Koopman損失の重みを上げる (ノートブック修正が必要)
**ファイル**: `notebooks/step2_phase2_koopman.ipynb`

**修正箇所**: セル内の設定

```python
# 修正前
KOOPMAN_WEIGHT_MAX = 0.1  # Maximum Koopman loss weight

# 修正後
KOOPMAN_WEIGHT_MAX = 0.5  # Maximum Koopman loss weight (increased for stronger signal)
```

**理由**: 0.1では損失の影響が小さすぎて、Koopman学習が進まない

### 4. エポック数を増やす (推奨)

```python
# 修正前
NUM_EPOCHS = 5
KOOPMAN_START_EPOCH = 3

# 修正後
NUM_EPOCHS = 10
KOOPMAN_START_EPOCH = 3
```

**理由**: Koopman学習には時間がかかるため、より多くのエポックが必要

## Google Colabでの修正手順

### ステップ1: ノートブックを開く
`notebooks/step2_phase2_koopman.ipynb` を Google Colab で開く

### ステップ2: 設定セルを修正
「Training configuration」セクションで以下を変更：

```python
# Training configuration
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10  # 5 → 10 に変更
KOOPMAN_START_EPOCH = 3
KOOPMAN_WEIGHT_MAX = 0.5  # 0.1 → 0.5 に変更
```

### ステップ3: 再実行
1. Runtime → Restart and run all
2. または、修正したセルから順に実行

## 期待される改善結果

### 修正前
- Koopman演算子変化: 0.05-0.06%
- Koopman予測PPL: 8776 (標準の19倍)
- 最終PPL: 461

### 修正後の期待値
- Koopman演算子変化: 5-10% (100倍改善)
- Koopman予測PPL: 500-600 (標準と同程度)
- 最終PPL: 100-200 (2-4倍改善)

## 検証ポイント

修正後のテストで以下を確認：

1. ✅ **Koopman演算子が更新されている**
   - 各層で5-10%の変化があるか
   - `Mean absolute change` が 0.5 以上か

2. ✅ **Koopman予測が機能している**
   - Koopman forward の PPL が標準の2倍以内か
   - 差が +200% 以下か

3. ✅ **損失が減少している**
   - Koopman loss が減少しているか
   - 最終的に 0.001 以下になるか

4. ✅ **パープレキシティが改善**
   - 最終 PPL が 200 以下か
   - ベースラインより改善しているか

## トラブルシューティング

### 問題1: Koopman演算子がまだ更新されない
**対策**: 
- `alpha` を 0.5 まで上げる
- バッファサイズを 1000 に増やす

### 問題2: 訓練が不安定になる
**対策**:
- `KOOPMAN_WEIGHT_MAX` を 0.3 に下げる
- 学習率を 5e-4 に下げる

### 問題3: メモリ不足
**対策**:
- バッファサイズを 300 に減らす
- バッチサイズを 16 に減らす

## 次のステップ

修正後のテストが成功したら：

1. ✅ Phase 2 完了を確認
2. → Phase 3 (Physics-Informed Learning) へ進む
3. → 両方を組み合わせたハイブリッド学習をテスト

## コード修正の確認

修正が正しく適用されているか確認：

```python
# Python で確認
import sys
sys.path.append('src')
from models.koopman_layer import KoopmanResNetBKLayer

# レイヤーを作成
layer = KoopmanResNetBKLayer(d_model=64, n_seq=128, koopman_dim=256)

# バッファサイズを確認
print(f"Buffer size: {layer.Z_current.shape[1]}")  # 500 であるべき

# ソースコードで alpha を確認
import inspect
source = inspect.getsource(layer.update_koopman_operator)
print("alpha = 0.3" in source)  # True であるべき
```

## まとめ

主な修正:
- ✅ Koopman更新率: 0.1 → 0.3
- ✅ バッファサイズ: 100 → 500
- 🔄 Koopman重み: 0.1 → 0.5 (ノートブックで手動修正)
- 🔄 エポック数: 5 → 10 (推奨、ノートブックで手動修正)

これらの修正により、Koopman学習が正しく機能するはずです。
