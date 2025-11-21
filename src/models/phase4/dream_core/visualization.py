import torch
import matplotlib.pyplot as plt
from typing import Optional
import numpy as np

def visualize_dream_as_text(
    concept_vector: torch.Tensor,
    diagnostics: Optional[dict] = None
) -> str:
    """
    Convert a dream concept into poetic text (heuristic).

    Args:
        concept_vector: (d,) tensor
        diagnostics: optional dict from DreamCore

    Returns:
        poem: str
    """
    # simple heuristics based on vector stats
    mean_val = concept_vector.mean().item()
    std_val = concept_vector.std().item()
    norm_val = concept_vector.norm().item()

    # Mood determination
    moods = ["calm", "turbulent", "radiant", "shadowy", "harmonic", "chaotic"]
    idx = int((abs(mean_val) * 100)) % len(moods)
    mood = moods[idx]

    # Structure determination
    structures = ["crystal", "river", "flame", "cloud", "root", "star"]
    idx2 = int((std_val * 100)) % len(structures)
    structure = structures[idx2]

    poem = f"I dreamt of a {mood} {structure}.\n"
    poem += f"Its essence vibrated with intensity {norm_val:.2f}.\n"

    if diagnostics and 'final_energy' in diagnostics:
        energy = diagnostics['final_energy']
        poem += f"It settled at an energy level of {energy:.2f},\n"
        poem += "born from the fragments of the past."

    return poem

def visualize_dream_as_knots(
    concept_vector: torch.Tensor,
    save_path: str
):
    """
    Visualize the dream concept as a 3D knot/curve.

    Args:
        concept_vector: (d,) tensor
        save_path: path to save image
    """
    # Project vector to 3D curve
    # We use the vector to modulate a base curve (e.g. torus knot)

    # p, q for torus knot (p, q integers typically)
    # We map vector values to p, q

    v_sum = concept_vector.sum().item()
    p = 2 + int(abs(v_sum * 10)) % 5
    q = 3 + int(abs(concept_vector.std().item() * 20)) % 7

    t = np.linspace(0, 2*np.pi * 5, 1000)

    # Torus knot:
    # x = (R + cos(q*t)) * cos(p*t)
    # y = (R + cos(q*t)) * sin(p*t)
    # z = sin(q*t)

    R = 2.0
    x = (R + np.cos(q*t)) * np.cos(p*t)
    y = (R + np.cos(q*t)) * np.sin(p*t)
    z = np.sin(q*t)

    # Add some "noise" from the concept vector to make it unique
    # Fold concept vector to match length 1000
    noise_amp = 0.2
    noise_pattern = concept_vector.detach().cpu().numpy()
    if len(noise_pattern) < 1000:
        noise_pattern = np.tile(noise_pattern, 1000 // len(noise_pattern) + 1)
    noise_pattern = noise_pattern[:1000]

    z += noise_amp * noise_pattern

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, lw=2, alpha=0.8)
    ax.set_title(f"Dream Knot (p={p}, q={q})")
    ax.axis('off')

    plt.savefig(save_path)
    plt.close()
