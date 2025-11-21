"""
Visualization for Resonance Emotion

Visualizes interference patterns as "ripples" and maps emotion to color.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import io
from typing import Optional

def visualize_emotion_as_ripple(
    interference_pattern: torch.Tensor,
    resonance_score: float,
    dissonance_score: float,
    save_path: Optional[str] = None
) -> Optional[bytes]:
    """
    Visualize emotion as a 2D ripple pattern.

    Args:
        interference_pattern: (N,) 1D interference pattern
        resonance_score: Scalar resonance score
        dissonance_score: Scalar dissonance score
        save_path: Path to save image (optional)

    Returns:
        bytes: Image bytes if save_path is None
    """
    # Detach and cpu
    pattern = interference_pattern.detach().cpu().numpy()
    N = pattern.shape[0]

    # Map 1D to 2D grid for visualization
    # Try to make it square-ish
    side = int(np.ceil(np.sqrt(N)))
    padded_N = side * side

    # Pad pattern
    pattern_padded = np.pad(pattern, (0, padded_N - N), mode='constant')
    grid = pattern_padded.reshape(side, side)

    # Determine Color Map
    # Resonance -> Warm (Hot), Dissonance -> Cool (Cool/Winter)
    # Blend based on dominance
    if resonance_score > dissonance_score:
        cmap = 'magma' # Warm/Resonant
        title_color = 'orange'
        emotion_label = "Resonance"
    else:
        cmap = 'viridis' # Cool/Dissonant
        title_color = 'cyan'
        emotion_label = "Dissonance"

    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap=cmap, interpolation='bicubic')
    plt.colorbar(label='Interference Amplitude')

    plt.title(
        f"{emotion_label} Dominant\n"
        f"Resonance: {resonance_score:.3f} | Dissonance: {dissonance_score:.3f}",
        color='black', fontsize=12
    )
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        plt.close()
        return None
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        buf.seek(0)
        return buf.getvalue()
