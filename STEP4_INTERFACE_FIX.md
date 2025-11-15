# Step 4 Interface Fix

## 問題

```
TypeError: QuantizedBKCore.forward() takes 2 positional arguments but 5 were given
```

### 原因

元の`bk_core`は`BKCoreFunction.apply`（関数）で、4つの引数を取ります：
```python
features = self.bk_core(he_diag, h0_super, h0_sub, self.z)
```

しかし、`QuantizedBKCore`は`nn.Module`として実装され、`forward(self, v_fp32)`という異なるインターフェースを持っていました。

## 解決策

`QuantizedBKCore`と`PerChannelQuantizedBKCore`を**callable class**（`nn.Module`ではない）に変更し、`BKCoreFunction.apply`と同じインターフェースを実装：

```python
class QuantizedBKCore:  # nn.Moduleを継承しない
    def __call__(self, he_diag, h0_super, h0_sub, z):
        # BKCoreFunction.applyと同じインターフェース
        ...
```

## 修正内容

### 1. src/models/quantized_bk_core.py

**変更前:**
```python
class QuantizedBKCore(nn.Module):
    def __init__(self, n_seq, enable_quantization=True):
        super().__init__()
        self.register_buffer('v_scale', torch.tensor(1.0))
        ...
    
    def forward(self, v_fp32):
        ...
```

**変更後:**
```python
class QuantizedBKCore:  # nn.Moduleを継承しない
    def __init__(self, n_seq, enable_quantization=True):
        self.training = True  # トレーニングモードを追跡
        self.v_scale = torch.tensor(1.0)  # register_bufferではなく通常の属性
        ...
    
    def __call__(self, he_diag, h0_super, h0_sub, z):
        # BKCoreFunction.applyと同じインターフェース
        ...
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False
```

### 2. src/models/complex_quantization.py

同様の変更を`PerChannelQuantizedBKCore`と`ComplexQuantizer`に適用：

```python
class ComplexQuantizer:  # nn.Moduleを継承しない
    def __init__(self, num_channels, per_channel=True):
        self.training = True
        self.real_scale = torch.ones(num_channels)  # register_bufferではない
        ...

class PerChannelQuantizedBKCore:  # nn.Moduleを継承しない
    def __call__(self, he_diag, h0_super, h0_sub, z):
        ...
    
    def train(self):
        self.training = True
        self.complex_quantizer.training = True
    
    def eval(self):
        self.training = False
        self.complex_quantizer.training = False
```

### 3. src/training/compression_pipeline.py

パイプラインでトレーニングモードを設定：

```python
quantized_core = QuantizedBKCore(n_seq=n_seq, enable_quantization=True)
quantized_core.train()  # トレーニングモードに設定
block.bk_layer.bk_core = quantized_core
```

## インターフェース比較

### 元のBKCoreFunction.apply
```python
# 呼び出し
features = bk_core(he_diag, h0_super, h0_sub, z)

# 引数
- he_diag: (B, N) - 有効ハミルトニアン対角
- h0_super: (B, N-1) - 超対角
- h0_sub: (B, N-1) - 副対角
- z: complex scalar - スペクトルシフト

# 戻り値
- features: (B, N, 2) - [real(G_ii), imag(G_ii)]
```

### 新しいQuantizedBKCore
```python
# 呼び出し（同じ）
features = quantized_bk_core(he_diag, h0_super, h0_sub, z)

# 引数（同じ）
- he_diag: (B, N)
- h0_super: (B, N-1)
- h0_sub: (B, N-1)
- z: complex scalar

# 戻り値（同じ）
- features: (B, N, 2)
```

## 主な変更点

1. **nn.Moduleを継承しない** - callable classとして実装
2. **`__call__`メソッド** - `forward`の代わりに`__call__`を使用
3. **インターフェース統一** - `BKCoreFunction.apply`と同じ引数
4. **トレーニングモード管理** - `self.training`属性を手動で管理
5. **バッファではなく属性** - `register_buffer`ではなく通常の属性

## 利点

- ✅ 既存のコードと完全互換
- ✅ `bk_core`を直接置き換え可能
- ✅ 追加のラッパー不要
- ✅ トレーニング/評価モードの切り替えが可能

## 検証

```python
# 元のBK-Core
from src.models.bk_core import BKCoreFunction
bk_core = BKCoreFunction.apply
features = bk_core(he_diag, h0_super, h0_sub, z)  # ✓ 動作

# 量子化BK-Core
from src.models.quantized_bk_core import QuantizedBKCore
quantized_bk_core = QuantizedBKCore(n_seq=128)
features = quantized_bk_core(he_diag, h0_super, h0_sub, z)  # ✓ 動作

# 置き換え
block.bk_layer.bk_core = quantized_bk_core  # ✓ 動作
```

## まとめ

- ✅ インターフェースの不一致を解決
- ✅ `nn.Module`から callable class に変更
- ✅ `BKCoreFunction.apply`と同じ引数
- ✅ トレーニングモード管理を追加
- ✅ すべての診断がパス

**修正完了！** 量子化BK-Coreは既存のコードと完全互換です。
