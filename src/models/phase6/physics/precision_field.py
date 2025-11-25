import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptivePrecisionField(nn.Module):
    """
    Adaptive Precision Field Controller.

    Determines which spatial regions require high precision (FP64/Complex128)
    based on local resonance and chaos metrics.

    Features:
    1. Resonance-based detection (Condition Number, Norm).
    2. Spatial Smoothing (Gaussian Blur).
    """

    def __init__(self, n_seq: int, smoothing_kernel_size: int = 5, threshold_high: float = 10.0):
        super().__init__()
        self.n_seq = n_seq
        self.threshold_high = threshold_high

        # Gaussian smoothing kernel
        sigma = 1.0
        channels = 1
        kernel = torch.tensor([math.exp(-x**2/(2*sigma**2)) for x in range(-2, 3)], dtype=torch.float32)
        kernel = kernel / kernel.sum()
        self.register_buffer('smoothing_kernel', kernel.view(1, 1, -1))

    def compute_precision_mask(self, condition_profile: torch.Tensor) -> torch.Tensor:
        """
        Compute a binary mask where 1 indicates High Precision (FP64) needed.

        Args:
            condition_profile: (B, N) tensor of local condition numbers or resonance scores.

        Returns:
            mask: (B, N) float tensor (0.0 to 1.0) indicating precision level.
        """
        import math

        # 1. Thresholding
        # Identify spikes
        raw_mask = (condition_profile > self.threshold_high).float()

        # 2. Spatial Smoothing
        # Apply 1D convolution to expand the high-precision region slightly and smooth transitions
        # Input to conv1d must be (B, C, L)
        x = raw_mask.unsqueeze(1)

        # Pad to maintain size
        pad = self.smoothing_kernel.size(-1) // 2
        x_padded = F.pad(x, (pad, pad), mode='replicate')

        smoothed_mask = F.conv1d(x_padded, self.smoothing_kernel)
        smoothed_mask = smoothed_mask.squeeze(1)

        # Binarize again for strict switching, or keep soft for mixed
        # For implementation simplicity, we often want a binary switch:
        # If any neighborhood needs high precision, the center gets it.
        final_mask = (smoothed_mask > 0.1).float()

        return final_mask

class ResonancePrecisionController(nn.Module):
    """
    Controller that outputs the precision configuration for the current step.
    """
    def __init__(self, n_seq: int):
        super().__init__()
        self.field = AdaptivePrecisionField(n_seq)

    def get_precision_config(self, physics_diagnostics: dict) -> dict:
        """
        Determine precision settings based on diagnostics.
        """
        # Mock extraction of local condition profile
        # In reality, this would come from BK-Core's internal analysis
        # For now, we use the global max condition number
        cond_num = physics_diagnostics.get('condition_number', 1.0)

        # If global condition is bad, request high precision globally
        global_high_prec = cond_num > 1e4

        return {
            'global_high_precision': global_high_prec,
            # Future: per-token mask
        }
