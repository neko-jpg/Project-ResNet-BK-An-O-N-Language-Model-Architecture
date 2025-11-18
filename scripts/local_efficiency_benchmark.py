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
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.models.mamba_baseline import MambaLM, create_mamba_from_resnetbk_config
from src.models.resnet_bk import LanguageModel as ResNetBK
from src.training.gradient_caching import GradientCache

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

    tokenized = raw["train"].map(tok_fn, batched=True, remove_columns=["text"])
    seq_plus_one = seq_length + 1

    def group(examples):
        concat = list(itertools.chain.from_iterable(examples["input_ids"]))
        total = len(concat) // seq_plus_one * seq_plus_one
        concat = concat[:total]
        return {"input_ids": [concat[i : i + seq_plus_one] for i in range(0, total, seq_plus_one)]}

    grouped = tokenized.map(group, batched=True, remove_columns=tokenized["train"].column_names)
    grouped.set_format(type="torch", columns=["input_ids"])
    g = torch.Generator().manual_seed(seed)

    def collate(batch):
        inputs = torch.stack([b["input_ids"][:-1] for b in batch])
        targets = torch.stack([b["input_ids"][1:] for b in batch])
        return inputs, targets

    return DataLoader(grouped["train"], batch_size=batch_size, shuffle=True, drop_last=True, generator=g, collate_fn=collate)


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


def train_and_profile(model, loader, use_act: bool, steps: int, lr: float):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    flop_samples = []
    model.train()
    for idx, (inp, tgt) in enumerate(loader):
        if idx >= steps:
            break
        inp = inp.to(DEVICE)
        tgt = tgt.to(DEVICE)
        opt.zero_grad()

        def forward_pass(x):
            return model(x)

        if use_act:
            cache = GradientCache(chunk_size=max(1, inp.shape[1] // 2))
            logits = cache(forward_pass, inp)
        else:
            logits = forward_pass(inp)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))
        loss.backward()
        opt.step()
        losses.append(loss.item())

        with torch.autograd.profiler.profile(enabled=True, use_cuda=DEVICE == "cuda") as prof:
            _ = forward_pass(inp)
        flops = sum(e.flops or 0 for e in prof.function_events)
        flop_samples.append(flops)
    avg_flops = float(sum(flop_samples) / max(1, len(flop_samples)))
    return {"losses": losses, "avg_flops": avg_flops}


def main():
    parser = argparse.ArgumentParser(description="Local efficiency benchmark: ResNet-BK vs Mamba")
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--train-steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2)
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
        args.seq_length, tok.vocab_size, d_model=256, n_layers=6, dropout=0.1
    )

    res_base = train_and_profile(bk_base, loader, use_act=False, steps=args.train_steps, lr=args.lr)
    res_act = train_and_profile(bk_act, loader, use_act=True, steps=args.train_steps, lr=args.lr)
    res_mamba = train_and_profile(mamba, loader, use_act=False, steps=args.train_steps, lr=args.lr)

    results = {"resnet_bk": res_base, "resnet_bk_act": res_act, "mamba": res_mamba}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "local_efficiency_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print("Saved", out_path)


if __name__ == "__main__":
    main()
