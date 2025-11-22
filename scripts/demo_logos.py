"""
Demo Script for LOGOS Architecture (Phase 4 +)

This script demonstrates the 5 key linguistic phenomena handled by LOGOS:
1. Ambiguity & Irony (Sentiment Phase Shifting)
2. Logical Consistency (Hamiltonian Energy Check)
3. Factuality (Topological Knots)
4. Infinite Context (Symplectic Adjoint - conceptual)
5. Empathy (Resonance Emotion - existing Phase 4)
"""

import torch
import sys
import os
from typing import List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.phase3.config import Phase3Config
from src.models.phase3.integrated_model import Phase3IntegratedModel
from src.models.phase4.integrated_model import Phase4IntegratedModel
from src.models.phase4.logos_tokenizer import ComplexTokenizer
from transformers import AutoTokenizer

def print_section(title):
    print("\n" + "="*50)
    print(f"  {title}")
    print("="*50 + "\n")

def demo_logos():
    # 1. Setup
    print("Initializing LOGOS Architecture...")

    # Ensure vocab size matches GPT-2 for compatibility if we use its tokenizer
    vocab_size = 50257 # GPT-2 size

    config = Phase3Config(
        vocab_size=vocab_size,
        d_model=64,
        n_layers=2,
        max_seq_len=128
    )

    phase3_model = Phase3IntegratedModel(config)
    model = Phase4IntegratedModel(
        phase3_model,
        enable_emotion=True,
        enable_topological=True,
        enable_meta=True
    )

    # Mock Tokenizer (since we don't have a real trained model weights for text)
    # We map basic words to IDs manually for the demo flow
    hf_tokenizer = AutoTokenizer.from_pretrained("gpt2") # Just for splitting
    if hf_tokenizer.pad_token is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token

    logos_tokenizer = model.logos_tokenizer
    logos_tokenizer.base_tokenizer = hf_tokenizer

    # =================================================================
    # Scenario 1: Ambiguity & Irony (Phase Shifting)
    # =================================================================
    print_section("1. Irony Detection (Sentiment Phase Shifting)")

    texts = [
        "Great job.", # Neutral/Positive
        "Great job!" # Enthusiastic (Phase +pi/4)
        # "Great job?" # Sarcastic/Questioning (Phase +pi/2) -> Not implemented in simple regex but illustrative
    ]

    for text in texts:
        print(f"Input: '{text}'")
        processed = logos_tokenizer.process_batch([text], max_length=10)
        input_ids = processed['input_ids']
        initial_phase = processed['initial_phase']

        print(f"  -> Base Phase Shift: {initial_phase[0,0].item():.4f} rad")

        # Forward Pass
        output = model(input_ids, initial_phase=initial_phase)
        # Check Embedding internal state (if accessible) or just show it ran
        print("  -> Model processed with rotated embeddings.")

    # =================================================================
    # Scenario 2: Factuality (Topological Knots)
    # =================================================================
    print_section("2. Factuality Check (Topological Knots)")

    # Fact: France capital is Paris.
    # Violation: France capital is London.

    generated_hypothesis = "France capital is London"
    print(f"Generating: '{generated_hypothesis}'")

    output = model(
        torch.tensor([[1, 2, 3]]), # Dummy input
        generated_text=generated_hypothesis
    )

    if output['logits'] is None:
        print(f"  -> BLOCKED! Energy Penalty: {output['loss']}")
        print(f"  -> Diagnosis: {output['diagnostics']['factuality_violation']['violation']}")
    else:
        print("  -> Allowed (Unexpected).")

    generated_truth = "France capital is Paris"
    print(f"\nGenerating: '{generated_truth}'")
    output_truth = model(
        torch.tensor([[1, 2, 3]]),
        generated_text=generated_truth
    )

    if output_truth['logits'] is None:
         print("  -> BLOCKED (Unexpected).")
    else:
         print("  -> ALLOWED. Topologically Trivial.")

    # =================================================================
    # Scenario 3: Logical Consistency (Hamiltonian Energy)
    # =================================================================
    print_section("3. Logical Consistency (Hamiltonian Energy)")

    print("Simulating High Energy Drift (Contradiction)...")

    # We simulate a forward pass where we artificially inject a high energy drift diagnostic
    # Since random weights won't naturally conserve energy perfectly or violate it meaningfully without training

    # Normal pass
    input_ids = torch.randint(0, 100, (1, 10))
    out = model(input_ids)
    meta = out['diagnostics'].get('meta_commentary', '')
    print(f"Normal State Meta-Commentary: {meta}")

    # Artificial Injection of High Drift
    # We can't easily inject into the live model forward without mocking.
    # But we can test the MetaCommentary logic directly.

    print("\n[Injecting Hamiltonian Drift = 5.0]")
    fake_diagnostics = {'hamiltonian_drift': 5.0}
    commentary = model.meta_commentary.generate_commentary(fake_diagnostics)
    print(f"Meta-Commentary Response:\n  \"{commentary}\"")

    if "Self-Correction" in commentary:
        print("  -> SUCCESS: System detected contradiction and triggered self-correction.")
    else:
        print("  -> FAILURE: No correction triggered.")

    print_section("LOGOS Demo Complete")

if __name__ == "__main__":
    demo_logos()
