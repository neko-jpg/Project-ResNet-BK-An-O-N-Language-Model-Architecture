"""
Long-context inference benchmark for ResNet-BK vs Transformer.

This script measures:
  - tokens/sec (forward only)
  - wall-clock latency
  - peak CUDA memory (if available)
  - max sequence length that runs without OOM

Designed for Colab: small model config, random token inputs.
"""

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import torch
# Use torch.autocast so device_type is accepted on Colab's Torch builds.
from torch import autocast

# Ensure repo root is on sys.path when executed from notebooks/
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Ensure results dir exists
os.makedirs("benchmarks/results", exist_ok=True)

from src.models.configurable_resnet_bk import ConfigurableResNetBK, ResNetBKConfig
from src.models.transformer_baseline import TransformerConfig, TransformerLM


@dataclass
class BenchmarkConfigLC:
    seq_lengths: List[int]
    batch_size: int = 2
    vocab_size: int = 20000
    d_model: int = 256
    n_layers: int = 6
    num_heads: int = 8
    ffn_dim: int = 1024
    num_experts: int = 4
    top_k: int = 2
    device: str = "auto"  # "auto", "cuda", or "cpu"
    save_path: str = "benchmarks/results/long_context_benchmark.json"


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Using CPU.")
        name = "cpu"
    return torch.device(name)


def build_models(cfg: BenchmarkConfigLC, seq_len: Optional[int] = None):
    # Build models sized for the target sequence length (ResNet-BK requires exact n_seq).
    target_seq = seq_len if seq_len is not None else max(cfg.seq_lengths)
    resnet_cfg = ResNetBKConfig(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_seq=target_seq,
        num_experts=cfg.num_experts,
        top_k=cfg.top_k,
        dropout_p=0.1,
        use_analytic_gradient=True,
        grad_blend=0.5,
    )
    resnet = ConfigurableResNetBK(resnet_cfg)

    transformer_cfg = TransformerConfig(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.num_heads,
        ffn_dim=cfg.ffn_dim,
        max_seq_len=target_seq,
    )
    transformer = TransformerLM(transformer_cfg)
    return resnet, transformer


def measure_model(model: torch.nn.Module, seq_len: int, batch_size: int, device: torch.device, vocab_size: int, use_autocast: bool = False):
    model.eval()
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    torch.cuda.empty_cache() if device.type == "cuda" else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.time()
    with torch.no_grad():
        if use_autocast and device.type == "cuda":
            with autocast("cuda", dtype=torch.float16):
                _ = model(x)
        else:
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    dur = time.time() - start
    tokens = batch_size * seq_len
    tokens_per_s = tokens / max(dur, 1e-6)
    peak_mb = None
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / 1e6
    return {"seq_len": seq_len, "batch_size": batch_size, "time_sec": dur, "tokens_per_sec": tokens_per_s, "peak_mb": peak_mb}


def run_long_context(cfg: BenchmarkConfigLC):
    device = resolve_device(cfg.device)
    results = {"config": asdict(cfg), "resnet_bk": [], "transformer": []}

    for seq in cfg.seq_lengths:
        print(f"=== Seq len {seq} ===")
        resnet, transformer = build_models(cfg, seq_len=seq)
        resnet = resnet.to(device)
        transformer = transformer.to(device)

        # ResNet-BK
        try:
            r = measure_model(resnet, seq, cfg.batch_size, device, cfg.vocab_size, use_autocast=False)
            results["resnet_bk"].append({"status": "ok", **r})
            print(f"ResNet-BK ok: {r['tokens_per_sec']:.0f} tok/s, peak={r['peak_mb']}")
        except RuntimeError as e:
            results["resnet_bk"].append({"status": "oom", "seq_len": seq, "error": str(e)})
            print(f"ResNet-BK OOM at seq {seq}")

        # Transformer
        try:
            t = measure_model(transformer, seq, cfg.batch_size, device, cfg.vocab_size, use_autocast=(device.type == "cuda"))
            results["transformer"].append({"status": "ok", **t})
            print(f"Transformer ok: {t['tokens_per_sec']:.0f} tok/s, peak={t['peak_mb']}")
        except RuntimeError as e:
            results["transformer"].append({"status": "oom", "seq_len": seq, "error": str(e)})
            print(f"Transformer OOM at seq {seq}")

    Path(cfg.save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {cfg.save_path}")


def build_argparser():
    p = argparse.ArgumentParser(description="Long-context inference benchmark (ResNet-BK vs Transformer)")
    p.add_argument("--seq_lengths", type=str, default="2048,4096,8192,16384,32768", help="Comma-separated sequence lengths")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--vocab_size", type=int, default=20000)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--ffn_dim", type=int, default=1024)
    p.add_argument("--num_experts", type=int, default=4)
    p.add_argument("--top_k", type=int, default=2)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--save_path", type=str, default="benchmarks/results/long_context_benchmark.json")
    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()
    seq_list = [int(x) for x in args.seq_lengths.split(",") if x]
    cfg = BenchmarkConfigLC(
        seq_lengths=seq_list,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        num_experts=args.num_experts,
        top_k=args.top_k,
        device=args.device,
        save_path=args.save_path,
    )
    run_long_context(cfg)


if __name__ == "__main__":
    main()
