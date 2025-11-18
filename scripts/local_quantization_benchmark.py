"""
Local quantization benchmark: ResNet-BK vs Mamba (FP32 / INT8 / fake-INT4).

Intended for local GPU (e.g., RTX 3080). Mirrors the Colab quantization
notebook with a CLI. Small model + WikiText-2 for fast turnaround.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.models.mamba_baseline import MambaLM, create_mamba_from_resnetbk_config
from src.models.resnet_bk import LanguageModel as ResNetBK

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_data(dataset_name: str, dataset_config: str, tokenizer, seq_length: int):
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
    return grouped["train"], raw["validation"]


def make_loader(dataset, batch_size: int, seed: int):
    g = torch.Generator().manual_seed(seed)

    def collate(batch):
        inputs = torch.stack([b["input_ids"][:-1] for b in batch])
        targets = torch.stack([b["input_ids"][1:] for b in batch])
        return inputs, targets

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, generator=g, collate_fn=collate)


def build_models(seq_length: int, vocab_size: int, model_cfg: Dict) -> Tuple[ResNetBK, MambaLM]:
    bk = ResNetBK(
        vocab_size=vocab_size,
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_seq=seq_length,
        num_experts=model_cfg["num_experts"],
        top_k=model_cfg["top_k"],
        dropout_p=model_cfg["dropout"],
        use_scattering_router=False,
        use_birman_schwinger=False,
    )
    res_cfg = argparse.Namespace(
        vocab_size=vocab_size,
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_seq=seq_length,
        dropout=model_cfg["dropout"],
        tie_weights=True,
    )
    mb = MambaLM(create_mamba_from_resnetbk_config(res_cfg))
    return bk, mb


def train_small(model, loader, steps: int, lr: float, weight_decay: float):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    losses = []
    model.train()
    for step, (inp, tgt) in enumerate(loader):
        if step >= steps:
            break
        inp = inp.to(DEVICE)
        tgt = tgt.to(DEVICE)
        opt.zero_grad()
        logits = model(inp)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


@torch.no_grad()
def eval_ppl(model, loader, max_tokens: int = 200000):
    model = model.to(DEVICE)
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for inp, tgt in loader:
        inp = inp.to(DEVICE)
        tgt = tgt.to(DEVICE)
        logits = model(inp)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += tgt.numel()
        if total_tokens > max_tokens:
            break
    return float(torch.exp(torch.tensor(total_loss / total_tokens)))


def quantize_int8_linear(model):
    import torch.ao.quantization as tq

    return tq.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def quantize_int4_linear(model):
    import copy

    qmodel = copy.deepcopy(model).cpu()
    for _, mod in qmodel.named_modules():
        if isinstance(mod, torch.nn.Linear):
            w = mod.weight.data
            scale = w.abs().max() / 7.0 + 1e-8
            q = torch.clamp(torch.round(w / scale), -8, 7)
            mod.weight.data = (q * scale).to(mod.weight.dtype)
    return qmodel


def main():
    parser = argparse.ArgumentParser(description="Local quantization benchmark: ResNet-BK vs Mamba")
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--train-steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-name", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--out-dir", default="results/benchmarks")
    args = parser.parse_args()

    set_seed(args.seed)
    tok = get_tokenizer(args.tokenizer)
    train_data, val_raw = load_data(args.dataset_name, args.dataset_config, tok, args.seq_length)
    train_loader = make_loader(train_data, args.batch_size, args.seed)
    val_tokenized = val_raw.map(lambda ex: tok(ex["text"], add_special_tokens=False), batched=True, remove_columns=["text"])
    val_tokenized = val_tokenized.map(
        lambda ex: {"input_ids": [ids[: args.seq_length + 1] for ids in ex["input_ids"] if len(ids) >= args.seq_length + 1]},
        batched=True,
    )
    val_tokenized = val_tokenized.filter(lambda ex: len(ex["input_ids"]) > 0)
    val_tokenized.set_format(type="torch", columns=["input_ids"])
    val_loader = make_loader(val_tokenized["validation"], batch_size=1, seed=0)

    model_cfg = {"d_model": 256, "n_layers": 4, "num_experts": 2, "top_k": 1, "dropout": 0.1}
    bk, mb = build_models(args.seq_length, tok.vocab_size, model_cfg)

    print("Training ResNet-BK...")
    train_small(bk, train_loader, steps=args.train_steps, lr=args.lr, weight_decay=args.weight_decay)
    print("Training Mamba...")
    train_small(mb, train_loader, steps=args.train_steps, lr=args.lr, weight_decay=args.weight_decay)

    print("Evaluating FP32...")
    ppl_bk_fp32 = eval_ppl(bk, val_loader)
    ppl_mb_fp32 = eval_ppl(mb, val_loader)

    print("Quantizing INT8...")
    ppl_bk_int8 = eval_ppl(quantize_int8_linear(bk), val_loader)
    ppl_mb_int8 = eval_ppl(quantize_int8_linear(mb), val_loader)

    print("Quantizing fake INT4 (Linear weights)...")
    ppl_bk_int4 = eval_ppl(quantize_int4_linear(bk), val_loader)
    ppl_mb_int4 = eval_ppl(quantize_int4_linear(mb), val_loader)

    results = {
        "seq_length": args.seq_length,
        "train_steps": args.train_steps,
        "ppl": {
            "resnet_bk": {"fp32": ppl_bk_fp32, "int8": ppl_bk_int8, "int4_fake": ppl_bk_int4},
            "mamba": {"fp32": ppl_mb_fp32, "int8": ppl_mb_int8, "int4_fake": ppl_mb_int4},
        },
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "local_quant_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print("Saved", out_path)


if __name__ == "__main__":
    main()
