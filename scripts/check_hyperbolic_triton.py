"""
Quick smoke test for HyperbolicMultiHeadAttention with optional Triton path.

This is intended to be called via `make triton-attn` to give a reproducible
forward/backward check (with causal mask by default) and emit a JSON summary.

カーネルバージョン:
- fast: 最速版（近似双曲距離）
- v2: 最適化版（事前計算 + autotune）
- v1: 従来版
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import torch

from src.models.phase7.hyperbolic_attention import HyperbolicMultiHeadAttention


def _to_device(requested: str) -> torch.device:
    """Resolve device name to available torch device."""
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cpu")
    return torch.device(requested)


def run_smoke(
    batch: int,
    seq_len: int,
    d_model: int,
    heads: int,
    use_triton: bool,
    use_mask: bool,
    device_str: str,
    json_path: str | None,
    kernel_version: str = 'fast',
) -> Dict[str, Any]:
    """
    Run a small forward/backward pass and return diagnostic dict.

    Args:
        batch: Batch size for the synthetic input.
        seq_len: Sequence length.
        d_model: Model dimension.
        heads: Number of attention heads.
        use_triton: Whether to request Triton kernel usage.
        use_mask: Whether to apply a causal mask.
        device_str: Target device string ("cuda" or "cpu").
        json_path: Optional output path for JSON summary.
        kernel_version: Triton kernel version ('fast', 'v2', 'v1').

    Returns:
        Dictionary with timings, gradient norms, and status flags.
    """
    device = _to_device(device_str)
    torch.manual_seed(0)

    x = torch.randn(batch, seq_len, d_model, device=device, requires_grad=True)
    mask = None
    if use_mask:
        mask = (
            torch.tril(torch.ones(seq_len, seq_len, device=device))
            .view(1, 1, seq_len, seq_len)
        )

    attn = HyperbolicMultiHeadAttention(
        d_model=d_model,
        num_heads=heads,
        use_triton_kernel=use_triton,
        kernel_version=kernel_version,
    ).to(device)

    # Detect if Triton path is available
    triton_available = getattr(attn, "triton_kernel_function", None) is not None

    # Warmup runs (compile kernels, initialize CUDA)
    for _ in range(3):
        with torch.no_grad():
            _ = attn(x, mask=mask, return_diagnostics=False)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Timed forward pass
    t0 = time.time()
    out, diagnostics = attn(x, mask=mask, return_diagnostics=True)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    fwd_time = time.time() - t0

    # Timed backward pass
    t1 = time.time()
    loss = out.sum()
    loss.backward()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    bwd_time = time.time() - t1

    grad_norm = x.grad.norm().item()

    result = {
        "batch": batch,
        "seq_len": seq_len,
        "d_model": d_model,
        "heads": heads,
        "device": str(device),
        "use_triton_requested": use_triton,
        "triton_available": triton_available,
        "kernel_version": kernel_version,
        "used_mask": use_mask,
        "forward_time_sec": fwd_time,
        "backward_time_sec": bwd_time,
        "grad_norm": grad_norm,
        "status": "ok" if (triton_available or not use_triton) else "triton_missing",
    }

    if json_path:
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))

    print(
        f"[triton-attn] status={result['status']} device={result['device']} "
        f"kernel={kernel_version} use_triton={use_triton} mask={use_mask} "
        f"fwd={fwd_time:.4f}s bwd={bwd_time:.4f}s grad_norm={grad_norm:.4f}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=2, help="Batch size.")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length.")
    parser.add_argument("--d-model", type=int, default=64, help="Model dimension.")
    parser.add_argument("--heads", type=int, default=4, help="Number of heads.")
    parser.add_argument(
        "--use-triton",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request Triton kernel usage.",
    )
    parser.add_argument(
        "--use-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply causal mask.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Target device ("cuda" or "cpu").',
    )
    parser.add_argument(
        "--json",
        type=str,
        default="results/triton_attention_check.json",
        help="Path to save JSON summary (set empty to skip).",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="fast",
        choices=["fast", "v2", "v1"],
        help="Triton kernel version: fast (fastest), v2 (optimized), v1 (original).",
    )
    args = parser.parse_args()

    json_path = args.json if args.json else None
    run_smoke(
        batch=args.batch,
        seq_len=args.seq_len,
        d_model=args.d_model,
        heads=args.heads,
        use_triton=args.use_triton,
        use_mask=args.use_mask,
        device_str=args.device,
        json_path=json_path,
        kernel_version=args.kernel,
    )


if __name__ == "__main__":
    main()
