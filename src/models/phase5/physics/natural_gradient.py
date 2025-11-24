import torch

class NaturalGradientHook:
    """
    Implements the Natural Gradient scaling based on Phase Sensitivity.

    scale = 1.0 / |d\delta / d\lambda|

    This acts as a hook on the backward pass of the potential V.
    """

    @staticmethod
    def hook_fn(grad: torch.Tensor, sensitivity: torch.Tensor, epsilon: float = 1e-4) -> torch.Tensor:
        """
        Scale the gradient.

        Args:
            grad: The gradient of the loss w.r.t Potential V.
            sensitivity: The pre-calculated phase sensitivity |d\delta/d\lambda|.

        Returns:
            Scaled gradient.
        """
        # Calculate scale factor
        # Higher sensitivity -> Lower learning rate (High curvature)
        # Lower sensitivity -> Higher learning rate (Flat region)

        # scale = 1 / (sensitivity + epsilon)

        # We need to ensure sensitivity matches grad shape or broadcasts
        # Assuming grad is (B, N) and sensitivity is (B, N) or (B,)

        scale = 1.0 / (sensitivity + epsilon)

        # Clamp scale to prevent explosions in flat regions
        scale = torch.clamp(scale, 0.1, 10.0)

        return grad * scale

    @classmethod
    def register(cls, potential_tensor: torch.Tensor, sensitivity: torch.Tensor):
        """
        Register the hook.

        Note: PyTorch hooks usually only take 'grad'. To pass 'sensitivity',
        we use a closure or partial.
        """
        # We need to capture the current value of sensitivity.
        # Since sensitivity changes per step, this registration needs to happen
        # every forward pass or the hook needs to look up the current sensitivity.

        # Ideally, we define the hook inline in the model forward/backward.
        pass
