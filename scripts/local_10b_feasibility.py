"""
10B feasibility (analysis only) for ResNet-BK.

Estimates parameter counts and memory, and constructs a meta-device model
to verify shapes without allocating real memory. This does NOT train a 10B
model; used for documentation/planning.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from src.models.resnet_bk import LanguageModel as ResNetBK


def param_count_resnetbk(vocab_size: int, d_model: int, n_layers: int, n_seq: int) -> int:
    model = ResNetBK(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_seq=n_seq,
        num_experts=4,
        top_k=1,
        dropout_p=0.1,
        use_scattering_router=False,
        use_birman_schwinger=False,
    )
    return sum(p.numel() for p in model.parameters())


def estimate_activation_memory(batch_size: int, seq_length: int, d_model: int, bytes_per_elem: int = 2) -> int:
    tokens = batch_size * seq_length
    return tokens * d_model * bytes_per_elem


def estimate_total_memory(params: int, bytes_per_param: int = 2, activation_bytes: int = 0) -> int:
    return params * bytes_per_param + activation_bytes


def main():
    vocab_size = 50_000
    configs = [
        {"name": "baseline_small", "d_model": 256, "n_layers": 6, "n_seq": 2048},
        {"name": "mid", "d_model": 512, "n_layers": 16, "n_seq": 8192},
        {"name": "target_10b", "d_model": 2048, "n_layers": 48, "n_seq": 32768},
    ]
    estimates = []
    for cfg in configs:
        params = param_count_resnetbk(vocab_size, cfg["d_model"], cfg["n_layers"], cfg["n_seq"])
        act_mem = estimate_activation_memory(batch_size=1, seq_length=cfg["n_seq"], d_model=cfg["d_model"], bytes_per_elem=2)
        total_mem = estimate_total_memory(params, bytes_per_param=2, activation_bytes=act_mem)
        estimates.append(
            {
                "config": cfg,
                "params": params,
                "activation_bytes": act_mem,
                "total_bytes_fp16": total_mem,
            }
        )
    print(json.dumps(estimates, indent=2))

    # Meta-device shape check
    with torch.device("meta"):
        cfg = SimpleNamespace(
            vocab_size=50000,
            d_model=2048,
            n_layers=48,
            n_seq=32768,
            num_experts=4,
            top_k=1,
            dropout_p=0.1,
            use_scattering_router=False,
            use_birman_schwinger=False,
        )
        _ = ResNetBK(
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_seq=cfg.n_seq,
            num_experts=cfg.num_experts,
            top_k=cfg.top_k,
            dropout_p=cfg.dropout_p,
            use_scattering_router=cfg.use_scattering_router,
            use_birman_schwinger=cfg.use_birman_schwinger,
        )
        print("Meta model constructed (no real memory).")

    requirements = {
        "mixed_precision": "fp16 or bf16 mandatory",
        "activation_checkpointing": True,
        "gradient_accumulation": True,
        "offload": "ZeRO/FSDP-style sharding not available on stock Colab",
        "multi_gpu": "Single T4/RTX is insufficient for true 10B; requires multi-GPU or heavy offload",
        "long_context": "1M tokens require chunking/state-saving; not feasible on a single GPU naively",
    }
    print(json.dumps(requirements, indent=2))


if __name__ == "__main__":
    main()
