#!/usr/bin/env python3
"""Phase 8 Quick Test Script"""

import json
import torch
from datetime import datetime

# GPU情報取得
gpu_info = {}
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    gpu_info = {
        'name': props.name,
        'total_memory_gb': props.total_memory / (1024**3),
        'compute_capability': f'{props.major}.{props.minor}',
    }
    print(f"GPU: {gpu_info['name']}, VRAM: {gpu_info['total_memory_gb']:.1f} GB")

# Phase 8モジュールテスト
modules_status = {}

print("\nTesting Phase 8 modules...")

try:
    from src.models.phase8 import TangentSpaceLinearAttention, LinearAttentionConfig
    config = LinearAttentionConfig(d_model=256, num_heads=4)
    model = TangentSpaceLinearAttention(config).cuda()
    x = torch.randn(2, 512, 256, device='cuda')
    with torch.no_grad():
        out = model(x)
    modules_status['TangentSpaceLinearAttention'] = 'OK'
    print("  TangentSpaceLinearAttention: OK")
except Exception as e:
    modules_status['TangentSpaceLinearAttention'] = str(e)
    print(f"  TangentSpaceLinearAttention: FAIL - {e}")

try:
    from src.models.phase8 import HyperbolicSSM, HyperbolicSSMConfig
    config = HyperbolicSSMConfig(d_model=256)
    model = HyperbolicSSM(config).cuda()
    x = torch.randn(2, 512, 256, device='cuda')
    with torch.no_grad():
        out = model(x)
    modules_status['HyperbolicSSM'] = 'OK'
    print("  HyperbolicSSM: OK")
except Exception as e:
    modules_status['HyperbolicSSM'] = str(e)
    print(f"  HyperbolicSSM: FAIL - {e}")

try:
    from src.models.phase8 import BlockWiseDistanceComputation, BlockDistanceConfig
    config = BlockDistanceConfig(d_model=256, num_heads=4)
    model = BlockWiseDistanceComputation(config).cuda()
    x = torch.randn(2, 512, 256, device='cuda')
    with torch.no_grad():
        out = model(x)
    modules_status['BlockWiseDistanceComputation'] = 'OK'
    print("  BlockWiseDistanceComputation: OK")
except Exception as e:
    modules_status['BlockWiseDistanceComputation'] = str(e)
    print(f"  BlockWiseDistanceComputation: FAIL - {e}")

try:
    from src.models.phase8 import EntailmentCones, EntailmentConeConfig
    config = EntailmentConeConfig(d_model=256)
    model = EntailmentCones(config).cuda()
    x = torch.randn(2, 512, 256, device='cuda')
    v = torch.randn(2, 512, 256, device='cuda')
    with torch.no_grad():
        out = model(x, v)
    modules_status['EntailmentCones'] = 'OK'
    print("  EntailmentCones: OK")
except Exception as e:
    modules_status['EntailmentCones'] = str(e)
    print(f"  EntailmentCones: FAIL - {e}")

try:
    from src.models.phase8 import SheafAttentionModule, SheafAttentionConfig
    config = SheafAttentionConfig(d_model=256, num_heads=4)
    model = SheafAttentionModule(config).cuda()
    x = torch.randn(2, 512, 256, device='cuda')
    with torch.no_grad():
        out = model(x)
    modules_status['SheafAttentionModule'] = 'OK'
    print("  SheafAttentionModule: OK")
except Exception as e:
    modules_status['SheafAttentionModule'] = str(e)
    print(f"  SheafAttentionModule: FAIL - {e}")

try:
    from src.models.phase8 import LogarithmicQuantizer, QuantizationConfig
    config = QuantizationConfig(bits=8)
    model = LogarithmicQuantizer(config).cuda()
    x = torch.randn(2, 512, 256, device='cuda')
    with torch.no_grad():
        out = model(x)
    modules_status['LogarithmicQuantizer'] = 'OK'
    print("  LogarithmicQuantizer: OK")
except Exception as e:
    modules_status['LogarithmicQuantizer'] = str(e)
    print(f"  LogarithmicQuantizer: FAIL - {e}")

# 結果出力
ok_count = sum(1 for v in modules_status.values() if v == 'OK')
total_count = len(modules_status)

results = {
    'timestamp': datetime.now().isoformat(),
    'gpu_info': gpu_info,
    'modules_status': modules_status,
    'summary': {
        'passed': ok_count,
        'total': total_count,
        'success_rate': ok_count / total_count if total_count > 0 else 0,
    },
    'overall_status': 'PASS' if ok_count == total_count else 'PARTIAL',
}

print(f"\nSummary: {ok_count}/{total_count} modules working")
print(f"Overall status: {results['overall_status']}")

# ファイル保存
with open('results/benchmarks/phase8_rtx3080_final.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nResults saved to results/benchmarks/phase8_rtx3080_final.json")
