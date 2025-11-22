"""
非エルミート安定性＆ユニタリティ閾値の事前検証スクリプト。

- NonHermitianPotentialのΓをランダム入力で生成し、長時間での情報消失/発散率を推定
- 散乱行列Sのユニタリティ破れをノイズ掃引で測定し、実装用の閾値を推薦
- 結果をJSONとコンソールに出力（ベンチマーク要件）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch

from src.models.phase2.non_hermitian import NonHermitianPotential
from src.utils.physics_checks import (
    analyze_gamma_stability,
    compute_unitarity_error,
    ensure_results_dir,
    sweep_unitarity_threshold,
)


def sample_gamma(
    d_model: int,
    n_seq: int,
    batch_size: int,
    base_decay: float,
    adaptive: bool,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate Γ samples from NonHermitianPotential with random inputs.
    """
    torch.manual_seed(42)
    potential = NonHermitianPotential(
        d_model=d_model,
        n_seq=n_seq,
        base_decay=base_decay,
        adaptive_decay=adaptive,
    ).to(device)
    x = torch.randn(batch_size, n_seq, d_model, device=device)
    V = potential(x)
    gamma = -V.imag  # positive decay rate
    return gamma.detach().cpu()


def simulate_gamma_boundaries(
    device: torch.device,
    base_decays=(1e-4, 5e-4, 1e-3, 5e-3, 1e-2),
) -> Dict:
    """
    Sweep base_decay to find regimes that avoid overdamping or divergence.
    """
    results = {}
    for bd in base_decays:
        gamma = sample_gamma(
            d_model=64,
            n_seq=128,
            batch_size=32,
            base_decay=bd,
            adaptive=True,
            device=device,
        )
        stats = analyze_gamma_stability(gamma, time_horizon=12.0)
        results[str(bd)] = stats
    return results


def simulate_unitarity(device: torch.device) -> Dict:
    """
    Sweep unitarity violation and return recommended threshold + stats.
    """
    noise_levels = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    threshold, stats = sweep_unitarity_threshold(noise_levels, trials=64)
    # Provide a sanity check example
    z = torch.randn(1, 4, 4, device=device) + 1j * torch.randn(1, 4, 4, device=device)
    q, _ = torch.linalg.qr(z)
    example_error = compute_unitarity_error(q.cpu())
    return {
        "recommended_threshold": threshold,
        "noise_sweep": stats,
        "unitary_example_error": example_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/physics_core/stability_unitarity.json"),
        help="Path to save JSON results.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Using device: {device}")

    gamma_results = simulate_gamma_boundaries(device)
    unitarity_results = simulate_unitarity(device)

    summary = {
        "device": str(device),
        "gamma_analysis": gamma_results,
        "unitarity_analysis": unitarity_results,
    }

    ensure_results_dir(args.output)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"[done] Saved results to {args.output}")

    # Console digest
    best_bd = min(gamma_results.items(), key=lambda kv: kv[1]["vanished_frac"])
    print(f"[gamma] best base_decay={best_bd[0]} vanished={best_bd[1]['vanished_frac']:.3f}")
    print(f"[unitarity] recommended anomaly threshold={unitarity_results['recommended_threshold']:.2e}")


if __name__ == "__main__":
    main()
