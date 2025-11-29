#!/usr/bin/env python3
"""quantized_hyperbolic_triton.pyのテストスクリプト"""

import sys
import torch
import importlib

print("Testing quantized_hyperbolic_triton.py...")

# キャッシュをクリア
if 'src.kernels.quantized_hyperbolic_triton' in sys.modules:
    del sys.modules['src.kernels.quantized_hyperbolic_triton']

try:
    import src.kernels.quantized_hyperbolic_triton as qht
    importlib.reload(qht)
    
    TRITON_AVAILABLE = qht.TRITON_AVAILABLE
    INT8Quantizer = qht.INT8Quantizer
    INT4Quantizer = qht.INT4Quantizer
    QuantizedHyperbolicAttention = qht.QuantizedHyperbolicAttention
    create_quantized_attention = qht.create_quantized_attention
    
    print(f"✓ Import successful")
    print(f"  TRITON_AVAILABLE: {TRITON_AVAILABLE}")
except Exception as e:
    import traceback
    print(f"✗ Import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# INT8量子化テスト
print("\nTesting INT8Quantizer...")
try:
    x = torch.randn(2, 4, 8, 64)
    x_q, scale, zp = INT8Quantizer.quantize(x)
    x_deq = INT8Quantizer.dequantize(x_q, scale, zp)
    error = (x - x_deq).abs().mean().item()
    print(f"✓ INT8 quantization: error={error:.6f}")
except Exception as e:
    print(f"✗ INT8 quantization failed: {e}")

# INT4量子化テスト
print("\nTesting INT4Quantizer...")
try:
    x = torch.randn(2, 64)
    packed, scale, zp = INT4Quantizer.quantize(x)
    x_deq = INT4Quantizer.dequantize(packed, scale, zp)
    error = (x - x_deq).abs().mean().item()
    print(f"✓ INT4 quantization: error={error:.6f}")
    print(f"  Original shape: {x.shape}, Packed shape: {packed.shape}")
except Exception as e:
    print(f"✗ INT4 quantization failed: {e}")

# QuantizedHyperbolicAttentionテスト
print("\nTesting QuantizedHyperbolicAttention...")
try:
    attn = create_quantized_attention(d_model=256, num_heads=4, bits=8)
    x = torch.randn(2, 16, 256)
    out = attn(x)
    print(f"✓ QuantizedHyperbolicAttention: input={x.shape}, output={out.shape}")
except Exception as e:
    print(f"✗ QuantizedHyperbolicAttention failed: {e}")

print("\n" + "="*50)
print("All tests completed!")
