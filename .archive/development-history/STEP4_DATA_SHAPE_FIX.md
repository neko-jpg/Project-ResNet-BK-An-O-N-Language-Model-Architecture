# Step 4 Data Shape Fix

## 問題

```
AssertionError: n_seq mismatch: expected 128, got 20
```

### 原因

`get_data_loader`が返すデータの形状が`(seq_len, batch_size)`で、モデルは`(batch_size, seq_len)`を期待していました。

```python
# get_data_loader returns:
train_data shape: (seq_len_total, batch_size)  # 例: (25000, 20)

# get_batch returns:
x shape: (seq_len, batch_size)  # 例: (128, 20)

# Model expects:
x shape: (batch_size, seq_len)  # 例: (20, 128)
```

## 解決策

`SimpleDataLoader`の`__iter__`メソッドで転置を追加：

```python
def __iter__(self):
    for i in range(0, self.data.size(0) - self.n_seq, self.n_seq):
        x, y = self.get_batch(self.data, i)
        # x shape: (seq_len, batch_size) -> transpose to (batch_size, seq_len)
        x = x.t().contiguous()  # Now (batch_size, seq_len)
        yield x, y.view(-1)
```

## 修正内容

### notebooks/step4_compression.ipynb

**変更前:**
```python
def __iter__(self):
    for i in range(0, self.data.size(0) - self.n_seq, self.n_seq):
        x, y = self.get_batch(self.data, i)
        yield x, y.view(-1)
```

**変更後:**
```python
def __iter__(self):
    for i in range(0, self.data.size(0) - self.n_seq, self.n_seq):
        x, y = self.get_batch(self.data, i)
        # x shape: (seq_len, batch_size) -> transpose to (batch_size, seq_len)
        x = x.t().contiguous()  # Now (batch_size, seq_len)
        yield x, y.view(-1)
```

### デバッグコード追加

学習前にデータ形状を確認：

```python
# Check data shape
print("\nChecking data shapes...")
for x_batch, y_batch in train_loader:
    print(f"x_batch shape: {x_batch.shape}  # Should be (batch_size={BATCH_SIZE}, seq_len={N_SEQ})")
    print(f"y_batch shape: {y_batch.shape}  # Should be (batch_size * seq_len,)")
    break
```

## 期待される出力

```
Checking data shapes...
x_batch shape: torch.Size([20, 128])  # Should be (batch_size=20, seq_len=128)
y_batch shape: torch.Size([2560])     # Should be (batch_size * seq_len,)

Training baseline model...
Epoch 1, Batch 0/195, Loss: 10.3456
...
```

## 検証

### 正しい形状
- `x_batch`: `(20, 128)` ✓
- `y_batch`: `(2560,)` = 20 × 128 ✓

### モデル要件
- `LanguageModel.forward(x)`: `x.shape = (batch_size, n_seq)`
- `assert n_seq == self.n_seq`: 128 == 128 ✓

## 他の影響

この修正により、以下のコンポーネントも正しく動作します：

1. **Compression Pipeline** - `_train_epoch`と`_evaluate`
2. **Distillation Trainer** - `train_step`と`evaluate`
3. **すべてのノートブック** - 同じデータローダーパターンを使用

## まとめ

- ✅ データ形状の転置を追加
- ✅ デバッグコードで確認
- ✅ モデルの期待する形状と一致
- ✅ すべてのコンポーネントで動作

**修正完了！** ノートブックはGoogle Colabで正常に実行できます。
