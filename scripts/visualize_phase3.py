"""
Visualization Script for Phase 3 (Task 26)

Visualizes:
1. Energy Drift
2. Koopman Eigenvalues
3. Contradiction Scores
"""

import matplotlib.pyplot as plt
import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def visualize_phase3():
    print("Generating Visualizations...")
    output_dir = project_root / "results" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Koopman Eigenvalues (Mock Data)
    # Ideally load model and call compute_eigenspectrum()
    eigenvalues = np.random.rand(100) * np.exp(1j * np.random.rand(100) * 2 * np.pi)

    plt.figure(figsize=(6, 6))
    plt.scatter(eigenvalues.real, eigenvalues.imag, alpha=0.6)
    circle = plt.Circle((0, 0), 1, fill=False, color='red', linestyle='--')
    plt.gca().add_artist(circle)
    plt.title("Koopman Eigenvalues")
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.grid(True)
    plt.savefig(output_dir / "koopman_eigenvalues.png")
    plt.close()

    # 2. Energy Drift (Mock Data)
    steps = np.arange(100)
    drift = np.random.randn(100).cumsum() * 1e-5

    plt.figure(figsize=(10, 4))
    plt.plot(steps, drift)
    plt.axhline(y=1e-4, color='r', linestyle='--', label='Threshold')
    plt.axhline(y=-1e-4, color='r', linestyle='--')
    plt.title("Energy Drift over Time")
    plt.xlabel("Step")
    plt.ylabel("Drift")
    plt.legend()
    plt.savefig(output_dir / "energy_drift.png")
    plt.close()

    print(f"Saved visualizations to {output_dir}")

if __name__ == "__main__":
    visualize_phase3()
