import math

import torch

from src.utils.physics_checks import analyze_gamma_stability, compute_unitarity_error


def test_unitarity_error_small_for_unitary():
    torch.manual_seed(0)
    z = torch.randn(2, 4, 4, dtype=torch.complex64)
    q, _ = torch.linalg.qr(z)
    err = compute_unitarity_error(q)
    assert err < 1e-4


def test_gamma_stability_detects_overdamping():
    gamma = torch.full((16,), 0.5)
    stats = analyze_gamma_stability(gamma, time_horizon=12.0, vanish_threshold=1e-4)
    assert stats["vanished_frac"] > 0.9


def test_energy_conservation_simple_oscillator():
    # シンプルな調和振動子を半陰的オイラーで計算し、エネルギー保存を確認
    dt = 1e-3
    steps = 1000
    mass = 1.0
    k = 1.0
    x = torch.tensor(1.0)
    v = torch.tensor(0.0)

    def energy(x_val: torch.Tensor, v_val: torch.Tensor) -> float:
        return 0.5 * k * float(x_val**2) + 0.5 * mass * float(v_val**2)

    initial_energy = energy(x, v)
    for _ in range(steps):
        # Symplectic Euler (position then velocity)
        v = v - dt * (k / mass) * x
        x = x + dt * v
    final_energy = energy(x, v)
    rel_err = abs(final_energy - initial_energy) / initial_energy
    assert rel_err < 1e-3
