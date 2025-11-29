#!/usr/bin/env python
"""Test Hyperbolic SSM import."""
import sys
sys.path.insert(0, ".")

try:
    # Direct import from file
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "hyperbolic_ssm", 
        "src/models/phase8/hyperbolic_ssm.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    print("Direct import successful!")
    print(f"HyperbolicSSMConfig: {module.HyperbolicSSMConfig}")
    print(f"HyperbolicSSM: {module.HyperbolicSSM}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
