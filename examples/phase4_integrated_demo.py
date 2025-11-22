"""
Example: Phase 4 Integrated Demo

Demonstrates the full "Ghost in the Shell" experience:
1. Inference with Emotion and Quantum Observation.
2. Meta-commentary on the internal state.
3. Dreaming (Idle mode).
"""

import torch
import time
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.phase4.integrated_model import Phase4IntegratedModel
from src.models.phase3.config import Phase3Config
from src.models.phase3.integrated_model import Phase3IntegratedModel
from src.models.phase4.meta_commentary import MetaCommentary

def main():
    print("Initializing Phase 4 Model...")

    # 1. Setup Phase 3 Base
    config = Phase3Config(
        vocab_size=1000,
        d_model=64,
        n_layers=2,
        max_seq_len=128
    )
    phase3_model = Phase3IntegratedModel(config)

    # 2. Setup Phase 4
    model = Phase4IntegratedModel(
        phase3_model=phase3_model,
        enable_emotion=True,
        enable_dream=True,
        enable_holographic=True,
        enable_quantum=True
    )

    meta = MetaCommentary()

    # 3. Run Inference Step
    print("\n--- Inference Step ---")
    input_ids = torch.randint(0, 1000, (1, 32))

    output = model(input_ids, return_diagnostics=True)
    diag = output['diagnostics']

    # 4. Display Diagnostics & Commentary
    print(f"Logits Shape: {output['logits'].shape}")

    if 'emotion' in diag:
        res = diag['emotion']['resonance_score'].mean().item()
        print(f"Emotion Resonance: {res:.4f}")

    if 'quantum' in diag:
        ent = diag['quantum']['entropy_reduction'].mean().item()
        print(f"Entropy Reduction: {ent:.4f}")

    # Meta-Commentary
    comment = meta.generate_commentary(diag)
    print(f"\n[Ghost]: \"{comment}\"")

    # 5. Dream Mode
    print("\n--- Entering Dream Mode (Idle) ---")
    model.enter_idle_mode(interval=0.5)

    # Simulate waiting
    for i in range(3):
        print(f"Sleeping... {i+1}/3")
        time.sleep(0.6)

    model.exit_idle_mode()
    print("Woke up from dream.")

if __name__ == "__main__":
    main()
