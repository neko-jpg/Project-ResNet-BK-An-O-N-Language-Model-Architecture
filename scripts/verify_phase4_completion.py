
import torch
import sys
import os

# Add root to path
sys.path.append(os.getcwd())

from src.utils.mock_phase3 import MockPhase3Model
from src.models.phase4.integrated_model import Phase4IntegratedModel

def verify_integrated_model():
    print("=== Verifying Phase 4 Integrated Model ===")

    # 1. Initialize
    print("[1] Initializing Model...")
    p3 = MockPhase3Model(d_model=64)
    model = Phase4IntegratedModel(
        phase3_model=p3,
        enable_emotion=True,
        enable_dream=True,
        enable_holographic=True,
        enable_quantum=True,
        enable_topological=True,
        enable_ethics=True,
        enable_meta=True,
        enable_boundary=True
    )
    print("    Model initialized.")

    # 2. Run Forward
    print("[2] Running Forward Pass...")
    input_ids = torch.randint(0, 50257, (1, 32))
    out = model(input_ids)

    diag = out['diagnostics']

    # 3. Check Components
    checks = {
        'Emotion': 'emotion' in diag,
        'Quantum': 'quantum' in diag,
        'Bulk': 'bulk' in diag,
        'Meta': 'meta_commentary' in diag,
        'Boundary': 'boundary_context' in diag,
    }

    all_pass = True
    for name, passed in checks.items():
        status = "OK" if passed else "FAIL"
        print(f"    - {name}: {status}")
        if not passed: all_pass = False

    # Check values
    if checks['Emotion']:
        res = diag['emotion']['resonance_score'].mean().item()
        print(f"    Emotion Resonance: {res:.4f}")

    if checks['Quantum']:
        ent = diag['quantum']['entropy_reduction'].mean().item()
        print(f"    Quantum Entropy Delta: {ent:.4f}")

    if checks['Meta']:
        print(f"    Meta Commentary: \"{diag['meta_commentary']}\"")

    # 4. Check Sleep Mode
    print("[3] Checking Sleep Mode...")
    msg = model.enter_idle_mode(0.01)
    print(f"    Enter: {msg}")
    model.exit_idle_mode()
    print("    Exit: OK")

    if all_pass:
        print("\n=== SUCCESS: All Phase 4 components active and connected ===")
    else:
        print("\n=== FAILURE: Some components missing ===")
        sys.exit(1)

if __name__ == "__main__":
    verify_integrated_model()
