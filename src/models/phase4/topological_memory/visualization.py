"""
Visualization Tools for Topological Memory

Functions to visualize 3D knots and their invariants.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_knot_3d(
    knot_coords: torch.Tensor,
    save_path: str,
    title: str = "Topological Knot",
    invariants: dict = None
):
    """
    Visualize a knot in 3D.

    Args:
        knot_coords: (N, 3) coordinates
        save_path: Path to save the image
        title: Plot title
        invariants: Dictionary of invariants to display
    """
    coords = knot_coords.detach().cpu().numpy()

    # Close the loop
    coords = np.vstack([coords, coords[0]])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot line
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], lw=2, c='blue')

    # Add tube effect (scatter points with varying size/alpha)
    # Simplified: just scatter points
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=10, c=np.linspace(0, 1, len(coords)), cmap='viridis')

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if invariants:
        text_str = "\n".join([f"{k}: {v}" for k, v in invariants.items()])
        plt.figtext(0.02, 0.02, text_str, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
