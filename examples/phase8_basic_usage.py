import torch
from src.models.phase8.integrated_model import Phase8IntegratedModel
from src.models.phase8.config import Phase8Config

def main():
    print("=== Phase 8 Basic Usage Demo ===")

    # 1. Configuration
    config = Phase8Config(
        enable_adaptive_computation=True,
        enable_topological_norm=True,
        enable_entailment_cones=True
    )
    print("Configuration loaded.")

    # 2. Model Initialization
    d_model = 64
    n_layers = 4
    model = Phase8IntegratedModel(d_model, n_layers, config=config)
    print(f"Model initialized with d_model={d_model}, n_layers={n_layers}")

    # 3. Forward Pass
    seq_len = 16
    x = torch.randn(1, seq_len, d_model)

    print(f"Input shape: {x.shape}")
    out, diagnostics = model(x)
    print(f"Output shape: {out.shape}")

    # 4. Diagnostics
    print("\nDiagnostics:")
    for k, v in diagnostics.items():
        print(f"  {k}: {v}")

    print("\nDemo completed successfully.")

if __name__ == "__main__":
    main()
