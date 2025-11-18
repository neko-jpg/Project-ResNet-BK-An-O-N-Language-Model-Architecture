"""
Local efficiency benchmark: ResNet-BK (with/without GradientCache) vs Mamba.

Toy-scale FLOPs estimate using torch profiler, intended for local GPU (e.g., RTX 3080).
Saves JSON under results/benchmarks/.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from src.models.mamba_baseline import MambaLM, create_mamba_from_resnetbk_config
from torch.utils.checkpoint import checkpoint
from src.models.resnet_bk import LanguageModel as ResNetBK

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def make_loader(seq_length: int, batch_size: int, seed: int, dataset_name: str, dataset_config: str, tokenizer):
    raw = load_dataset(dataset_name, dataset_config)

    def tok_fn(examples):
        return tokenizer(examples["text"], add_special_tokens=False)

    # Use only a small subset of the data to avoid timeout during preprocessing
    tokenized = raw["train"].select(range(1000)).map(tok_fn, batched=True, remove_columns=["text"])
    seq_plus_one = seq_length + 1

    def group(examples):
        concat = list(itertools.chain.from_iterable(examples["input_ids"]))
        total = len(concat) // seq_plus_one * seq_plus_one
        concat = concat[:total]
        return {"input_ids": [concat[i : i + seq_plus_one] for i in range(0, total, seq_plus_one)]}

    grouped = tokenized.map(group, batched=True, remove_columns=tokenized.column_names)
    grouped.set_format(type="torch", columns=["input_ids"])
    g = torch.Generator().manual_seed(seed)

    def collate(batch):
        inputs = torch.stack([b["input_ids"][:-1] for b in batch])
        targets = torch.stack([b["input_ids"][1:] for b in batch])
        return inputs, targets

    return DataLoader(grouped, batch_size=batch_size, shuffle=True, drop_last=True, generator=g, collate_fn=collate)


def build_models(seq_length: int, vocab_size: int, d_model: int, n_layers: int, dropout: float):
    bk_base = ResNetBK(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_seq=seq_length,
        num_experts=4,
        top_k=1,
        dropout_p=dropout,
        use_scattering_router=False,
        use_birman_schwinger=False,
    )
    bk_act = ResNetBK(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_seq=seq_length,
        num_experts=4,
        top_k=1,
        dropout_p=dropout,
        use_scattering_router=False,
        use_birman_schwinger=False,
    )
    res_cfg = argparse.Namespace(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_seq=seq_length, dropout=dropout, tie_weights=True
    )
    mamba = MambaLM(create_mamba_from_resnetbk_config(res_cfg))
    return bk_base, bk_act, mamba


def count_parameters(model):
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_and_profile(model, loader, use_act: bool, steps: int, lr: float, model_name: str = "Model"):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    model.train()

    start_time = time.time()
    progress_bar = tqdm(total=steps, desc=f"Training {model_name}")

    for idx, (inp, tgt) in enumerate(loader):
        if idx >= steps:
            break
        inp = inp.to(DEVICE)
        tgt = tgt.to(DEVICE)
        opt.zero_grad()

        def forward_pass(x):
            return model(x)

        if use_act:
            # Use torch's built-in activation checkpointing
            output = checkpoint(forward_pass, inp, use_reentrant=False)
        else:
            output = forward_pass(inp)

        # Handle tuple output from Mamba model
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))
        loss.backward()
        opt.step()
        losses.append(loss.item())

        progress_bar.update(1)
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    progress_bar.close()
    end_time = time.time()

    training_time = end_time - start_time
    num_params = count_parameters(model)

    return {
        "losses": losses,
        "training_time_sec": training_time,
        "num_params": num_params,
        "final_loss": losses[-1] if losses else None
    }


def main():
    parser = argparse.ArgumentParser(description="Local efficiency benchmark: ResNet-BK vs Mamba")
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--train-steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-name", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--out-dir", default="results/benchmarks")
    args = parser.parse_args()

    set_seed(args.seed)
    tok = get_tokenizer(args.tokenizer)
    loader = make_loader(
        args.seq_length, args.batch_size, args.seed, args.dataset_name, args.dataset_config, tok
    )
    bk_base, bk_act, mamba = build_models(
        args.seq_length, tok.vocab_size, d_model=args.d_model, n_layers=args.n_layers, dropout=0.1
    )

    res_base = train_and_profile(bk_base, loader, use_act=False, steps=args.train_steps, lr=args.lr, model_name="ResNet-BK")
    res_act = train_and_profile(bk_act, loader, use_act=True, steps=args.train_steps, lr=args.lr, model_name="ResNet-BK (GC)")
    res_mamba = train_and_profile(mamba, loader, use_act=False, steps=args.train_steps, lr=args.lr, model_name="Mamba")

    results = {"resnet_bk": res_base, "resnet_bk_act": res_act, "mamba": res_mamba}

    # --- Enhanced Console Output ---
    print("\n--- Benchmark Results ---")
    print(f"Configuration: sequence_length={args.seq_length}, d_model={args.d_model}, n_layers={args.n_layers}, steps={args.train_steps}")
    print("-" * 70)
    print(f"{'Model':<20} | {'Parameters (M)':<15} | {'Training Time (s)':<20} | {'Final Loss':<15}")
    print("-" * 70)

    models = ["resnet_bk", "resnet_bk_act", "mamba"]
    display_names = ["ResNet-BK", "ResNet-BK (GC)", "Mamba"]

    for model_key, display_name in zip(models, display_names):
        params_m = results[model_key]['num_params'] / 1e6
        train_time = results[model_key]['training_time_sec']
        final_loss = results[model_key]['final_loss'] if results[model_key]['final_loss'] else 'N/A'
        print(f"{display_name:<20} | {params_m:<15.2f} | {train_time:<20.2f} | {final_loss:<15.4f}")

    print("-" * 70)

    # --- Unique Filename and Saving ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"efficiency_seq{args.seq_length}_d{args.d_model}_l{args.n_layers}_{timestamp}.json"
    out_path = out_dir / filename

    out_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
