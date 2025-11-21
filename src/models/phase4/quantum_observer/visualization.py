import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from typing import List

def visualize_wave_function_collapse(
    superposition_candidates: List[str],
    superposition_probs: torch.Tensor,
    collapsed_token: str,
    save_path: str
):
    """
    Animate wave function collapse.

    Args:
        superposition_candidates: list of 3 token strings
        superposition_probs: (3,) probabilities
        collapsed_token: final token string
        save_path: .gif or .mp4 path
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    def animate(frame):
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Duration: 60 frames (1 sec at 60fps)
        # 0-30 frames: Superposition (blur/overlay)
        # 30-60 frames: Collapse (sharpening)

        if frame < 30:
            # Superposition phase
            t = frame / 30.0
            # Oscillate opacity
            alpha_base = 0.5 + 0.3 * np.sin(t * 10)

            for i, (token, prob) in enumerate(zip(superposition_candidates, superposition_probs)):
                # Position slightly jittered
                x_jit = 0.5 + np.random.randn() * 0.05 * (1-t)
                y_pos = 0.5 + (i - 1) * 0.2 * (1-t) # Converge to center

                # Handle tensor scalar
                p = prob.item() if torch.is_tensor(prob) else prob

                font_size = 20 + p * 20
                ax.text(x_jit, y_pos, token,
                        ha='center', va='center', fontsize=font_size,
                        alpha=alpha_base * p, color='blue')

            ax.set_title("Superposition State |ψ⟩", fontsize=14)

        else:
            # Collapse phase
            t = (frame - 30) / 30.0
            # Sharpen final token
            size = 30 + t * 20
            alpha = min(1.0, 0.2 + t)

            ax.text(0.5, 0.5, collapsed_token,
                    ha='center', va='center', fontsize=size,
                    alpha=alpha, color='black', weight='bold')

            ax.set_title("Collapsed State |0⟩", fontsize=14)

    anim = animation.FuncAnimation(fig, animate, frames=60, interval=1000/60)
    anim.save(save_path, writer='pillow', fps=60)
    plt.close()
