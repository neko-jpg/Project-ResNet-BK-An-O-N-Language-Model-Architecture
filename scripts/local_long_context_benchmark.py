"""
Local head-to-head benchmark: ResNet-BK vs Mamba on long contexts.

Intended for local GPU runs (e.g., RTX 3080). Mirrors the Colab notebook but
as a CLI script so you can run and log results offline without notebook
overhead.

Features:
- Identical hyperparameters for both models.
- Multiple sequence lengths and seeds in one run.
- Optional theory features (scattering router + Birman-Schwinger) toggle.
- Saves JSON results under results/benchmarks/ and optional PNG loss curves.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.mamba_baseline import MambaLM, create_mamba_from_resnetbk_config
from src.models.resnet_bk import LanguageModel as ResNetBK

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_lm_dataset(dataset_name: str, dataset_config: str, tokenizer, seq_length: int):
    raw = load_dataset(dataset_name, dataset_config)

    def tok_fn(examples):
        return tokenizer(examples["text"], add_special_tokens=False)

    tokenized = raw["train"].map(tok_fn, batched=True, remove_columns=["text"])
    seq_plus_one = seq_length + 1

    def group_texts(examples):
        concatenated = list(itertools.chain.from_iterable(examples["input_ids"]))
        total_length = len(concatenated) // seq_plus_one * seq_plus_one
        concatenated = concatenated[:total_length]
        result = [
            concatenated[i : i + seq_plus_one] for i in range(0, total_length, seq_plus_one)
        ]
        return {"input_ids": result}

    # tokenized is Dataset (train split already selected)
    grouped = tokenized.map(group_texts, batched=True, remove_columns=tokenized.column_names)
    grouped.set_format(type="torch", columns=["input_ids"])
    return grouped


def make_dataloader(dataset, batch_size: int, seed: int):
    g = torch.Generator().manual_seed(seed)

    def collate(batch):
        inputs = torch.stack([b["input_ids"][:-1] for b in batch])
        targets = torch.stack([b["input_ids"][1:] for b in batch])
        return inputs, targets

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=g,
        collate_fn=collate,
    )


def build_models(seq_length: int, vocab_size: int, model_cfg: Dict, use_theory: bool):
    resnet_model = ResNetBK(
        vocab_size=vocab_size,
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_seq=seq_length,
        num_experts=model_cfg["num_experts"],
        top_k=model_cfg["top_k"],
        dropout_p=model_cfg["dropout"],
        use_scattering_router=use_theory,
        use_birman_schwinger=use_theory,
    )

    resnet_cfg = argparse.Namespace(
        vocab_size=vocab_size,
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_seq=seq_length,
        dropout=model_cfg["dropout"],
        tie_weights=True,
    )
    mamba_model = MambaLM(create_mamba_from_resnetbk_config(resnet_cfg))
    return resnet_model, mamba_model


def train_one(
    model_name: str,
    model,
    dataloader,
    max_steps: int,
    lr: float,
    min_lr: float,
    weight_decay: float,
    log_every: int,
    grad_clip: float | None,
    use_amp: bool,
):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps, eta_min=min_lr)
    
    # Use new AMP API to avoid deprecation warning
    device_type = 'cuda' if DEVICE == 'cuda' else 'cpu'
    scaler = torch.amp.GradScaler(device_type, enabled=use_amp and DEVICE == "cuda")

    losses: List[float] = []
    wall_start = time.time()
    diverged = False
    
    for step, (inputs, targets) in enumerate(dataloader):
        if step >= max_steps:
            break
        
        try:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            opt.zero_grad(set_to_none=True)
            
            # Use new AMP API
            with torch.amp.autocast(device_type, enabled=use_amp and DEVICE == "cuda"):
                # MambaLM returns (logits, loss); ResNetBK returns logits tensor.
                outputs = model(inputs)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            if not torch.isfinite(loss):
                print(f"{model_name} divergence at step {step+1} loss={loss.item():.4f}")
                diverged = True
                break
            
            scaler.scale(loss).backward()
            
            if grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            
            losses.append(loss.item())
            
            if (step + 1) % log_every == 0:
                print(
                    f"{model_name} step {step+1}/{max_steps} loss={loss.item():.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )
            
            # Clear cache periodically to avoid memory fragmentation
            if (step + 1) % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            if "CUDA" in str(e) or "out of memory" in str(e):
                print(f"{model_name} CUDA error at step {step+1}: {e}")
                print("Attempting to recover...")
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                diverged = True
                break
            else:
                raise
    
    return {
        "model": model_name,
        "losses": losses,
        "steps": len(losses),
        "wall_clock_sec": time.time() - wall_start,
        "diverged": diverged,
    }


def plot_losses(result_dict: Dict, seq_length: int, run_tag: str, out_dir: Path):
    plt.figure(figsize=(10, 5))
    for entry in result_dict[seq_length]:
        seed = entry["seed"]
        for name, color in [("resnet_bk", "blue"), ("mamba", "red")]:
            losses = entry[name]["losses"]
            steps = range(1, len(losses) + 1)
            plt.plot(steps, losses, label=f"{name}-seed{seed}-{run_tag}", color=color, alpha=0.5)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"Seq {seq_length} ({run_tag})")
    plt.legend()
    plt.grid(alpha=0.3)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"loss_{seq_length}_{run_tag}.png"
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    print("Saved", fname)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Local long-context benchmark: ResNet-BK vs Mamba")
    parser.add_argument(
        "--seq-lengths", nargs="+", type=int, default=[8192, 32768, 131072], help="Sequence lengths"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44], help="Seeds")
    parser.add_argument("--max-steps", nargs="+", type=int, default=[200, 120, 60], help="Steps per seq_len")
    parser.add_argument("--batch-size", type=int, default=2, help="Base batch size (halved for >=32k)")
    parser.add_argument("--use-theory", action="store_true", help="Enable scattering router + Birman-Schwinger")
    parser.add_argument("--save-plots", action="store_true", help="Save loss plots")
    parser.add_argument("--dataset-name", default="wikitext", help="HF dataset name")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1", help="HF dataset config")
    parser.add_argument("--tokenizer", default="gpt2", help="HF tokenizer name")
    parser.add_argument("--out-dir", default="results/benchmarks", help="Output directory for results")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision (use FP32)")
    args = parser.parse_args()

    run_tag = "theory_on" if args.use_theory else "vanilla"
    print(f"Device: {DEVICE}, run_tag: {run_tag}")

    # align max_steps with seq_lengths
    if len(args.max_steps) == 1:
        steps_map = {sl: args.max_steps[0] for sl in args.seq_lengths}
    elif len(args.max_steps) == len(args.seq_lengths):
        steps_map = dict(zip(args.seq_lengths, args.max_steps))
    else:
        raise ValueError("max-steps must be length 1 or match seq-lengths length")

    tok = get_tokenizer(args.tokenizer)
    model_cfg = {
        "d_model": 256,
        "n_layers": 6,
        "num_experts": 4,
        "top_k": 1,
        "dropout": 0.1,
    }
    train_cfg = {
        "learning_rate": 3e-4,
        "min_lr": 1e-5,
        "weight_decay": 0.01,
        "log_every": 20,
        "grad_clip": 1.0,
        "use_amp": not args.no_amp,  # Disable AMP if --no-amp flag is set
        "batch_size": args.batch_size,
    }

    all_results: Dict[int, List[Dict]] = {}
    for seq_len in args.seq_lengths:
        dataset = load_lm_dataset(args.dataset_name, args.dataset_config, tok, seq_len)
        max_steps = steps_map[seq_len]
        seed_results = []
        for seed in args.seeds:
            set_seed(seed)
            bs = train_cfg["batch_size"]
            if seq_len >= 32768:
                bs = max(1, bs // 2)
            dataloader = make_dataloader(dataset, batch_size=bs, seed=seed)
            resnet_model, mamba_model = build_models(seq_len, tok.vocab_size, model_cfg, args.use_theory)
            print(f"\n=== seq_len {seq_len} | seed {seed} | tag {run_tag} | batch {bs} ===")
            resnet_result = train_one(
                "resnet_bk",
                resnet_model,
                dataloader,
                max_steps=max_steps,
                lr=train_cfg["learning_rate"],
                min_lr=train_cfg["min_lr"],
                weight_decay=train_cfg["weight_decay"],
                log_every=train_cfg["log_every"],
                grad_clip=train_cfg["grad_clip"],
                use_amp=train_cfg["use_amp"],
            )
            # Free ResNet model before running Mamba to avoid GPU memory pressure.
            del resnet_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats()
            
            # Recreate dataloader for Mamba to ensure fresh data iteration
            dataloader = make_dataloader(dataset, batch_size=bs, seed=seed)
            
            mamba_result = train_one(
                "mamba",
                mamba_model,
                dataloader,
                max_steps=max_steps,
                lr=train_cfg["learning_rate"],
                min_lr=train_cfg["min_lr"],
                weight_decay=train_cfg["weight_decay"],
                log_every=train_cfg["log_every"],
                grad_clip=train_cfg["grad_clip"],
                use_amp=train_cfg["use_amp"],
            )
            resnet_result.update({"seed": seed, "seq_length": seq_len, "batch_size": bs, "use_theory": args.use_theory})
            mamba_result.update({"seed": seed, "seq_length": seq_len, "batch_size": bs, "use_theory": args.use_theory})
            seed_results.append({"seed": seed, "resnet_bk": resnet_result, "mamba": mamba_result})
        all_results[seq_len] = seed_results

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"local_long_context_{run_tag}.json"
    out_json.write_text(json.dumps(all_results, indent=2))
    print("Saved", out_json)

    if args.save_plots:
        for seq_len in args.seq_lengths:
            plot_losses(all_results, seq_len, run_tag, out_dir)


if __name__ == "__main__":
    main()
