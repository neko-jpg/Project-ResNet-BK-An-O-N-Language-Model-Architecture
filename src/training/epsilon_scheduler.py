"""
Epsilon Scheduler for Renormalization Group Flow

Implements a scheduler to anneal the epsilon parameter (related to
hyperbolic curvature) during training, following a cosine decay schedule.

This is inspired by the concept of Renormalization Group Flow, where the
model first learns coarse-grained features (large epsilon) and gradually
focuses on finer details (small epsilon).
"""
import math

class EpsilonScheduler:
    """
    Calculates the value of epsilon based on the current training step
    using a cosine decay schedule.

    epsilon(t) = epsilon_min + 0.5 * (epsilon_max - epsilon_min) * (1 + cos(pi * t / t_max))
    """
    def __init__(self, t_max: int, epsilon_max: float = 0.5, epsilon_min: float = 0.01):
        if t_max <= 0:
            raise ValueError("t_max must be positive")
        self.t_max = t_max
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min

    def get_epsilon(self, t: int) -> float:
        """
        Calculates the value of epsilon for a given step t.

        Args:
            t: The current training step.

        Returns:
            The value of epsilon for the current step.
        """
        if t < 0:
            t = 0
        if t > self.t_max:
            t = self.t_max

        # Cosine decay formula
        epsilon = self.epsilon_min + 0.5 * (self.epsilon_max - self.epsilon_min) * \
                  (1 + math.cos(math.pi * t / self.t_max))

        return epsilon

    def update_model_curvature(self, model, t: int):
        """
        Updates the curvature of hyperbolic layers in a model.
        This function assumes the model has modules with a 'log_c' parameter.

        The relationship is assumed to be c = 1 / epsilon.
        Therefore, log_c = -log(epsilon).

        Args:
            model: The model to update.
            t: The current training step.
        """
        epsilon = self.get_epsilon(t)

        # We need to handle epsilon -> 0, which would make log_c -> inf
        # Clamp epsilon to a small positive value.
        epsilon = max(epsilon, 1e-8)

        log_c_value = -math.log(epsilon)

        for module in model.modules():
            if hasattr(module, 'log_c'):
                module.log_c.data.fill_(log_c_value)
