# Phase 2 Long Context Test - Quick Reference

**Script**: `scripts/test_long_context.py`  
**Purpose**: Phase 2モデルの長期依存関係処理能力を検証

## Quick Start

### 基本的な使用方法

```bash
# デフォルト設定でテスト実行
python scripts/test_long_context.py

# 小規模モデルでテスト
python scripts/test_long_context.py --preset small

# 中規模モデルでテスト
python scripts/test_long_context.py --preset base
```

### 結果をJSONに保存

```bash
python scripts/test_long_context.py --output results/long_context_test.json
```

## Common Use Cases

### 1. VRAM使用量のみをテスト

```bash
# 複数のシーケンス長でVRAM使用量を測定
python scripts/test_long_context.py --seq-lengths 1024 2048 4096
```

### 2. 勾配消失のみをテスト

```bash
# 特定のシーケンス長で勾配フローを検証
python scripts/test_long_context.py --max-seq-len 4096 --min-gradient-norm 1e-5
```

### 3. カスタムモデル設定

```bash
# カスタムモデル次元とレイヤー数
python scripts/test_long_context.py --d-model 512 --n-layers 6
```

### 4. FP32でテスト

```bash
# FP16の代わりにFP32を使用
python scripts/test_long_context.py --fp32
```

### 5. より長いシーケンスでテスト

```bash
# 8192トークンまでテスト
python scripts/test_long_context.py --seq-lengths 1024 2048 4096 8192 --max-seq-len 8192
```

## CLI Options

### モデル設定

| Option | Default | Description |
|--------|---------|-------------|
| `--preset` | `small` | モデルプリセット (small/base/large) |
| `--vocab-size` | `50257` | 語彙サイズ |
| `--d-model` | preset依存 | モデル次元 |
| `--n-layers` | preset依存 | レイヤー数 |

### テスト設定

| Option | Default | Description |
|--------|---------|-------------|
| `--batch-size` | `1` | バッチサイズ |
| `--seq-lengths` | `1024 2048 4096` | テストするシーケンス長 |
| `--max-seq-len` | `4096` | 勾配テスト用の最大シーケンス長 |
| `--min-gradient-norm` | `1e-5` | 最小勾配ノルム閾値 |

### デバイス設定

| Option | Default | Description |
|--------|---------|-------------|
| `--device` | `cuda` (利用可能な場合) | 使用するデバイス |
| `--fp32` | False | FP32を使用（デフォルトはFP16） |

### 出力設定

| Option | Default | Description |
|--------|---------|-------------|
| `--output` | None | 結果を保存するJSONファイルパス |

## Output Format

### コンソール出力

```
================================================================================
Phase 2 Long Context Test Suite
================================================================================

--- Testing Seq=1024 ---
  VRAM usage: 2048.00 MB (2.0000 GB)
  Forward time: 0.1234 sec
  Output shape: torch.Size([1, 1024, 50257])
  Status: ✓ PASSED

--- Testing Seq=4096 ---
  VRAM usage: 7168.00 MB (7.0000 GB)
  Forward time: 0.4567 sec
  Output shape: torch.Size([1, 4096, 50257])
  Status: ✓ PASSED
  ✓ SUCCESS: VRAM usage 7.0000 GB < 8.0 GB

================================================================================
Gradient Flow Verification
================================================================================
Loss: 10.234567
Embedding gradient norm: 1.234567e-03
First layer gradient norm: 4.567890e-04
Last layer gradient norm: 7.890123e-03

--- Layer-wise Gradient Norms ---
  Layer 0: 4.567890e-04
  Layer 1: 5.123456e-04
  Layer 2: 6.789012e-04

--- Gradient Flow Status ---
✓ SUCCESS: First layer gradient norm 4.567890e-04 >= 1.000000e-05
✓ No NaN or Inf in gradients

================================================================================
Final Summary
================================================================================
VRAM Test (Seq=4096): ✓ PASSED
  VRAM usage: 7.0000 GB
Gradient Test (Seq=4096): ✓ PASSED
  Gradient norm: 4.567890e-04
NaN/Inf Check: ✓ PASSED

Overall Status: ✓ ALL TESTS PASSED
================================================================================
```

### JSON出力

```json
{
  "vram_test": {
    "1024": {
      "vram_bytes": 2147483648,
      "vram_mb": 2048.0,
      "vram_gb": 2.0,
      "forward_time_sec": 0.1234,
      "passed": true,
      "output_shape": [1, 1024, 50257]
    },
    "4096": {
      "vram_bytes": 7516192768,
      "vram_mb": 7168.0,
      "vram_gb": 7.0,
      "forward_time_sec": 0.4567,
      "passed": true,
      "output_shape": [1, 4096, 50257]
    }
  },
  "gradient_test": {
    "seq_len": 4096,
    "loss": 10.234567,
    "embedding_grad_norm": 0.00123,
    "first_layer_grad_norm": 0.000456,
    "last_layer_grad_norm": 0.00789,
    "layer_gradient_norms": [
      {"layer": 0, "gradient_norm": 0.000456},
      {"layer": 1, "gradient_norm": 0.000512},
      {"layer": 2, "gradient_norm": 0.000678}
    ],
    "min_gradient_norm": 0.000123,
    "max_gradient_norm": 0.00789,
    "mean_gradient_norm": 0.00234,
    "has_nan_or_inf": false,
    "passed": true
  },
  "all_passed": true,
  "summary": {
    "vram_4096_gb": 7.0,
    "vram_4096_passed": true,
    "gradient_norm": 0.000456,
    "gradient_passed": true,
    "has_nan_or_inf": false
  }
}
```

## Success Criteria

### VRAM Test

- ✅ **Seq=4096でVRAM使用量が8.0GB未満**
- 参考値として他のシーケンス長も測定

### Gradient Test

- ✅ **先頭層の勾配ノルムが1e-5以上**
- ✅ **NaN/Infが発生しない**

## Troubleshooting

### CUDA Out of Memory

```bash
# バッチサイズを減らす（既にデフォルトは1）
python scripts/test_long_context.py --batch-size 1

# より短いシーケンス長でテスト
python scripts/test_long_context.py --seq-lengths 512 1024 2048

# FP16を使用（デフォルト）
python scripts/test_long_context.py  # FP16がデフォルト
```

### Gradient Vanishing

```bash
# 勾配ノルム閾値を調整
python scripts/test_long_context.py --min-gradient-norm 1e-6

# より小さいモデルでテスト
python scripts/test_long_context.py --preset small
```

### Import Errors

```bash
# 必要なパッケージをインストール
pip install torch numpy

# Phase 2モジュールが正しくインストールされているか確認
python -c "from src.models.phase2 import Phase2IntegratedModel; print('OK')"
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
- name: Run Long Context Test
  run: |
    python scripts/test_long_context.py \
      --preset small \
      --seq-lengths 1024 2048 \
      --output results/long_context_test.json
```

### Pre-commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "Running long context test..."
python scripts/test_long_context.py --preset small --seq-lengths 1024 2048
if [ $? -ne 0 ]; then
    echo "Long context test failed!"
    exit 1
fi
```

## Performance Tips

### 1. Use FP16 for Memory Efficiency

```bash
# FP16はデフォルトで有効
python scripts/test_long_context.py
```

### 2. Test on Smaller Models First

```bash
# 小規模モデルで素早くテスト
python scripts/test_long_context.py --preset small
```

### 3. Incremental Sequence Length Testing

```bash
# 段階的にシーケンス長を増やす
python scripts/test_long_context.py --seq-lengths 512 1024 2048 4096
```

## Related Documentation

- [Phase 2 Implementation Guide](../PHASE2_IMPLEMENTATION_GUIDE.md)
- [Long Context Test Implementation Report](../../results/benchmarks/LONG_CONTEXT_TEST_IMPLEMENTATION_REPORT.md)
- [Phase 2 Training Quick Reference](./PHASE2_TRAINING_QUICK_REFERENCE.md)

## Support

問題が発生した場合は、以下を確認してください:

1. PyTorchとCUDAのバージョン
2. 利用可能なVRAM容量
3. Phase 2モジュールのインストール状態

---

**Last Updated**: 2025-01-20  
**Version**: 1.0.0
