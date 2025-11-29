#!/usr/bin/env python3
"""Phase 8モジュールのインポートテスト"""

import sys
print(f"Python: {sys.version}")

try:
    from src.models.phase8 import TangentSpaceLinearAttention
    print("TangentSpaceLinearAttention: OK")
except Exception as e:
    print(f"TangentSpaceLinearAttention: FAIL - {e}")

try:
    from src.models.phase8 import HyperbolicSSM
    print("HyperbolicSSM: OK")
except Exception as e:
    print(f"HyperbolicSSM: FAIL - {e}")

try:
    from src.models.phase8 import BlockWiseDistanceComputation
    print("BlockWiseDistanceComputation: OK")
except Exception as e:
    print(f"BlockWiseDistanceComputation: FAIL - {e}")

try:
    from src.models.phase8 import ARSSMHyperbolicFusion
    print("ARSSMHyperbolicFusion: OK")
except Exception as e:
    print(f"ARSSMHyperbolicFusion: FAIL - {e}")

try:
    from src.models.phase8 import EntailmentCones
    print("EntailmentCones: OK")
except Exception as e:
    print(f"EntailmentCones: FAIL - {e}")

try:
    from src.models.phase8 import SheafAttentionModule
    print("SheafAttentionModule: OK")
except Exception as e:
    print(f"SheafAttentionModule: FAIL - {e}")

print("\nAll imports tested.")
