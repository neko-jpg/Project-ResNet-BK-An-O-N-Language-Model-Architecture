import torch
import torch.nn as nn
from src.models.birman_schwinger_core import BirmanSchwingerCore
from src.models.semiseparable_matrix import SemiseparableMatrix

def verify_spectral_health():
    print("--- Verifying Spectral Health (Schatten Norms & BitNet) ---")

    # Configuration
    batch_size = 2
    n_seq = 64
    d_model = 32
    use_bitnet = True

    print(f"Config: Batch={batch_size}, N={n_seq}, BitNet={use_bitnet}")

    # 1. Instantiate Core with BitNet
    core = BirmanSchwingerCore(
        n_seq=n_seq,
        epsilon=1.0,
        use_bitnet=use_bitnet,
        use_semiseparable=True,
        schatten_threshold=100.0,
    )

    # 2. Initialize Potential V (Simulate Prime-Bump or Random)
    # Important: V should not be zero or too small.
    V = torch.randn(batch_size, n_seq) * 2.0

    # 3. Forward Pass
    print("\n[Running Forward Pass]...")
    z = 1.0j
    features, diagnostics = core(V, z=z)

    # 4. Inspect Diagnostics
    print("\n--- Diagnostics ---")
    print(f"Schatten S1: {diagnostics.get('schatten_s1', 'N/A')}")
    print(f"Schatten S2: {diagnostics.get('schatten_s2', 'N/A')}")
    print(f"Condition Num: {diagnostics.get('condition_number', 'N/A')}")
    print(f"Semiseparable Active: {diagnostics.get('semiseparable_active', 'N/A')}")
    print(f"Reason: {diagnostics.get('semiseparable_reason', 'N/A')}")

    # 5. Check Output Health
    if torch.allclose(features, torch.zeros_like(features)):
        print("\n[ERROR] Output features are all ZERO!")
    else:
        print(f"\n[OK] Output features magnitude: {features.abs().mean().item():.4f}")

    # 6. Check S2 Norm
    s2 = diagnostics.get('schatten_s2', 0.0)
    if s2 < 1e-6:
        print("[CRITICAL] Schatten S2 is effectively ZERO. Spectral death confirmed.")
    else:
        print("[OK] Schatten S2 is non-zero.")

if __name__ == "__main__":
    try:
        verify_spectral_health()
    except Exception as e:
        print(f"\n[EXCEPTION] {e}")
        import traceback
        traceback.print_exc()
