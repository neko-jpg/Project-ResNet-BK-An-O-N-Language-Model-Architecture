
import sys
import os
sys.path.insert(0, os.getcwd())

print("Step 1: Importing torch...")
import torch
print("Step 1: OK")

print("Step 2: Importing Phase8IntegratedModel...")
try:
    from src.models.phase8.integrated_model import Phase8IntegratedModel, Phase8Config
    print("Step 2: OK")
except Exception as e:
    print(f"Step 2 FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Step 3: Creating Config...")
try:
    config = Phase8Config(d_model=128, n_layers=2)
    print("Step 3: OK")
except Exception as e:
    print(f"Step 3 FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Step 4: Initializing Model...")
try:
    model = Phase8IntegratedModel(config)
    print("Step 4: OK")
except Exception as e:
    print(f"Step 4 FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("All steps passed.")
