"""
Example: Phase 4 Dream Demo

Focuses on the Dream Core and Topological Memory.
Demonstrates how memory fragments are consolidated into new concepts.
"""

import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.phase4.dream_core.inverse_diffusion import DreamCore
from src.models.phase4.topological_memory.sparse_tensor_rep import SparseKnotRepresentation

def main():
    print("Initializing Dream Core...")
    d_model = 64
    dream_core = DreamCore(d_model)
    memory = SparseKnotRepresentation(d_model)

    # Create fake memories
    print("Creating memory fragments...")
    fragments = torch.randn(10, d_model)

    # Store them in topological memory (simulation)
    for i in range(10):
        memory.encode_concept_to_knot(fragments[i])

    # Generate a dream
    print("\nGenerating Dream (Inverse Diffusion)...")
    new_concept, diag = dream_core(fragments)

    print(f"New Concept Energy: {diag['final_energy'].item():.4f}")
    print(f"Diffusion Trajectory Length: {len(diag['trajectory'])}")

    # Check if new concept is topologically valid
    print("\nEncoding new concept to knot...")
    knot = memory.encode_concept_to_knot(new_concept)
    print(f"Knot Shape: {knot.shape}")

    print("\nDemo Complete.")

if __name__ == "__main__":
    main()
