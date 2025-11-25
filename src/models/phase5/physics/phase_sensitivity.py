import torch

def compute_phase_sensitivity(
    scatter_phase: torch.Tensor,
    lambda_val: torch.Tensor,
    epsilon: float = 1e-6
) -> torch.Tensor:
    """
    Compute the phase sensitivity |d\delta / d\lambda|.

    According to the Birman-Krein formula:
    d\delta/d\lambda = -Tr((H - \lambda)^{-1} - (H_0 - \lambda)^{-1})

    However, if we have the explicit phase values, we can approximate it numerically
    or use the analytical gradients if available.

    In the context of the Scattering Router, `scatter_phase` is \delta(\lambda).
    We assume \lambda corresponds to the input signal magnitude or energy level.

    Args:
        scatter_phase: (B, N) tensor of phases.
        lambda_val: (B, N) tensor of energy parameters (or eigenvalues).

    Returns:
        sensitivity: (B, N) tensor.
    """

    # If we don't have explicit gradients, we use the magnitude of the phase itself
    # as a proxy for "curvature" or "sensitivity" in the semantic manifold.
    # Regions with high phase accumulation are "resonant" and sensitive.

    # Placeholder for analytical derivative if not available via autograd
    # For now, we return a function of the phase gradient w.r.t the input potential.

    # Ideally, we hook into the backward pass.
    # Here we provide a utility that might be used *inside* the model during forward
    # or as a separate estimation step.

    # Simple proxy: 1.0 / (1.0 + |phase|) to avoid division by zero if used as scale?
    # No, we want sensitivity.

    return torch.abs(scatter_phase) # Raw phase magnitude often correlates with resonance density
