import torch
import torch.nn as nn
from src.models.resnet_bk import LanguageModel, SymplecticBKBlock
from src.models.phase5.integrated_model import Phase5IntegratedModel

def verify_symplectic_euler():
    print("\n--- Verifying Symplectic Euler Mode ---")
    d_model = 32
    n_seq = 10

    # 1. Initialize Model with Euler
    model = LanguageModel(
        vocab_size=100,
        d_model=d_model,
        n_layers=1,
        n_seq=n_seq,
        use_symplectic=True,
        symplectic_mode='euler'
    )

    block = model.blocks[0]
    assert isinstance(block, SymplecticBKBlock)
    assert block.integration_mode == 'euler'
    print("[OK] Symplectic Block initialized in Euler mode.")

    # 2. Run Forward
    x = torch.randint(0, 100, (2, n_seq))
    out = model(x)
    print(f"[OK] Forward pass successful. Output shape: {out.shape}")

def verify_pain_signal():
    print("\n--- Verifying Pain Signal Integration ---")
    d_model = 32
    n_seq = 10

    base_model = LanguageModel(
        vocab_size=100,
        d_model=d_model,
        n_layers=1,
        n_seq=n_seq,
        use_birman_schwinger=True # Needed for diagnostics
    )

    model = Phase5IntegratedModel(
        base_model=base_model,
        d_model=d_model,
        vocab_size=100,
        max_seq_len=n_seq
    )

    # Mock a high condition number in the base model
    # We can't easily force it without running physics, but we can check if the hook runs
    # By default cond_num is 0.0 or 1.0 if not computed.

    x = torch.randint(0, 100, (2, n_seq))
    logits, diag = model(x)

    print("Diagnostics keys:", diag.keys())
    if 'pain_signal' in diag:
        print(f"[OK] Pain signal present: {diag['pain_signal']}")
    else:
        print("[FAIL] Pain signal missing from diagnostics.")

if __name__ == "__main__":
    verify_symplectic_euler()
    verify_pain_signal()
