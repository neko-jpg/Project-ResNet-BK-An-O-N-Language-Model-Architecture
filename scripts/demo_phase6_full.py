import torch
import torch.nn as nn
from src.models.resnet_bk import LanguageModel
from src.models.phase5.integrated_model import Phase5IntegratedModel
from src.models.phase6.physics.precision_field import AdaptivePrecisionField

def demo_phase6_full():
    print("\n--- Phase 6 Full System Demo ---")

    d_model = 32
    n_seq = 16
    vocab_size = 100

    # 1. Setup Model
    base_model = LanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=2,
        n_seq=n_seq,
        use_birman_schwinger=True,
        use_symplectic=True,
        symplectic_mode='verlet'
    )

    model = Phase5IntegratedModel(
        base_model=base_model,
        d_model=d_model,
        vocab_size=vocab_size,
        max_seq_len=n_seq
    )

    # 2. Mock Input with High Complexity (High Temp)
    x = torch.randint(0, vocab_size, (2, n_seq))

    print("[Step 1] Running Forward Pass (Initial State)...")
    logits, diag = model(x)

    print(f"  Pacing Status: {diag['pacing']['status']}")
    print(f"  Energy Level: {diag['pacing']['energy']:.2f}")
    print(f"  Temperature: {diag['pacing']['temperature']:.4f}")

    # 3. Simulate Exhaustion
    print("\n[Step 2] Simulating Intense Workload (Fatigue Accumulation)...")
    # Force high pain signal repeatedly
    for _ in range(5):
        model.pacing.fatigue_model.update(temp=10.0, pain=5.0)

    # Run again
    logits, diag = model(x)
    print(f"  Pacing Status: {diag['pacing']['status']}")
    print(f"  Rec. Integrator: {diag['pacing']['symplectic_mode']}")

    if diag['pacing']['status'] in ['tired', 'exhausted']:
        print("[OK] Model successfully transitioned to Fatigue mode.")
    else:
        print("[FAIL] Model did not get tired.")

    # 4. Test Adaptive Precision Field
    print("\n[Step 3] Testing Adaptive Precision Field...")
    field = AdaptivePrecisionField(n_seq=n_seq)
    # Mock condition profile with a spike at index 5
    profile = torch.zeros(1, n_seq)
    profile[0, 5] = 100.0 # Spike

    mask = field.compute_precision_mask(profile)
    print(f"  Input Spike at Index 5.")
    print(f"  Precision Mask: {mask[0].tolist()}")

    if mask[0, 5] == 1.0 and mask[0, 4] == 1.0: # Check smoothing
        print("[OK] Precision Field correctly smoothed the spike.")
    else:
        print("[FAIL] Precision Field smoothing failed.")

if __name__ == "__main__":
    try:
        demo_phase6_full()
    except Exception as e:
        print(e)
        import traceback
        traceback.print_exc()
