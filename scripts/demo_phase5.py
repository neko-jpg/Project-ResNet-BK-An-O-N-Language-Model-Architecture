import torch
import torch.nn as nn
from src.models.resnet_bk import LanguageModel
from src.models.phase5.integrated_model import Phase5IntegratedModel

def demo_phase5_pipeline():
    print("--- Phase 5.0/5.5 System Check ---")

    # 1. Setup Configuration
    d_model = 64
    vocab_size = 1000
    seq_len = 16
    batch_size = 2

    # 2. Instantiate Base Model (Physics Core)
    print("Initializing Physics Core (ResNet-BK)...")
    base_model = LanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=2,
        n_seq=seq_len,
        use_birman_schwinger=False # Simple mode for demo
    )

    # 3. Instantiate Integrated Model (Ghost)
    print("Initializing Phase 5 Integrated Model (Ghost)...")
    model = Phase5IntegratedModel(
        base_model=base_model,
        d_model=d_model,
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        beam_width=2
    )

    # 4. Mock Inputs
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # 5. Run Standard Forward (Monad Loop)
    print("\n[Step 1] Running Standard Forward (Monad Loop)...")

    # Inject a thought before running
    model.monad.log_thought("I am analyzing the topological structure of this sentence.")

    logits, diag = model(input_ids)

    print(f"Logits Shape: {logits.shape}")
    print("Diagnostics:")
    for k, v in diag.items():
        print(f"  {k}: {v}")

    # Verify Reflector updated params
    print("\n[Step 2] Verifying Reflector Updates...")
    params = diag['physics_params']
    if params['gamma'] != 0.0 or params['bump_scale'] != 0.02:
        print(f"  SUCCESS: Physics parameters updated by Reflector. Gamma: {params['gamma']:.4f}")
    else:
        print("  WARNING: Physics parameters static (Reflector might have output near-zero deltas).")

    # 6. Run Sheaf Ethics
    print("\n[Step 3] Checking Sheaf Ethics Energy...")
    energy = diag['ethics']['sheaf_energy']
    print(f"  Sheaf Energy: {energy:.4f}")
    if energy >= 0:
        print("  SUCCESS: Energy calculated.")

    # 7. Mock Quantum Process (Symbolic)
    print("\n[Step 4] Testing Quantum Process Matrix Logic...")
    from src.models.phase5.quantum.superposition_state import SuperpositionState

    # Create seed state
    seed = SuperpositionState(
        token_ids=[1, 2],
        hidden_state=torch.randn(1, d_model),
        cumulative_log_prob=-0.5,
        cumulative_energy=1.0
    )

    model.quantum.initialize_superposition(seed)

    # Mock expansion inputs
    next_logits = torch.randn(1, vocab_size) # 1 beam
    next_h = torch.randn(1, d_model)
    energies = torch.tensor([0.5])

    print("  Expanding superposition...")
    candidates = model.quantum.expand_superposition(next_logits, next_h, energies)
    print(f"  Active Beams: {len(candidates)}")
    best = model.quantum.collapse()
    print(f"  Collapsed to best action: {best.action:.4f}")

    print("\n--- Phase 5 Verification Complete ---")

if __name__ == "__main__":
    demo_phase5_pipeline()
