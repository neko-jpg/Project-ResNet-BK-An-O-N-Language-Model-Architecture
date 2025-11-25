# experiments/validate_riemannian_adam.py

import torch
import torch.nn as nn
import sys
import os

# Add the project root to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.phase6.geometry.hyperbolic import HyperbolicInitializer
from src.optimizers.riemannian_adam import RiemannianAdam

# ##############################################################################
# # Validation Script
# ##############################################################################

def validate_optimizer():
    """
    Validates that the RiemannianAdam optimizer keeps hyperbolic parameters
    within the Poincaré ball.
    """
    print("--- Starting RiemannianAdam Validation ---")

    # 1. Configuration
    d_model = 2  # Use 2D for easy visualization if needed
    n_vocab = 10
    learning_rate = 1e-2
    num_steps = 100

    # 2. Model Definition
    # A simple model with just a hyperbolic embedding layer
    model = nn.Embedding(n_vocab, d_model)

    # 3. Hyperbolic Initialization
    # This will initialize weights inside the Poincaré ball and set `is_hyperbolic=True`
    initializer = HyperbolicInitializer(d_model=d_model)
    initializer.initialize_embeddings(model)

    print(f"Parameter 'is_hyperbolic' attribute is set: {hasattr(model.weight, 'is_hyperbolic') and model.weight.is_hyperbolic}")
    initial_norm = model.weight.data.norm(dim=-1).max()
    print(f"Initial maximum norm: {initial_norm.item():.4f}")
    assert initial_norm < 1.0, "Initial norms must be < 1."

    # 4. Optimizer Setup
    optimizer = RiemannianAdam(model.parameters(), lr=learning_rate)

    # 5. Training Loop
    print(f"Running optimization for {num_steps} steps...")
    for i in range(num_steps):
        # The goal is to minimize the norm of the embeddings,
        # which means pushing them towards the origin.
        # The gradient of ||w||^2 is 2w.
        loss = model.weight.pow(2).sum()

        # Zero gradients, compute loss, and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation check at each step
        current_max_norm = model.weight.data.norm(dim=-1).max()
        if (i + 1) % 10 == 0:
            print(f"  Step {i+1:3d}/{num_steps} -> Loss: {loss.item():.4f}, Max Norm: {current_max_norm.item():.4f}")

        # The crucial assertion: parameters must remain inside the Poincaré ball
        assert current_max_norm < 1.0, f"Parameter norm exceeded 1 at step {i+1}! Max Norm: {current_max_norm.item()}"

    final_max_norm = model.weight.data.norm(dim=-1).max()
    print("\n--- Validation Summary ---")
    print(f"Initial Max Norm: {initial_norm.item():.4f}")
    print(f"Final Max Norm  : {final_max_norm.item():.4f}")

    if final_max_norm < initial_norm:
        print("✅ Validation Successful: Optimizer correctly minimized the norm while staying within the Poincaré ball.")
    else:
        print("⚠️ Validation Warning: Optimizer did not appear to minimize the norm effectively.")

if __name__ == '__main__':
    validate_optimizer()
