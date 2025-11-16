"""
Colab-ready comparison script: ResNet-BK vs Transformer (small-scale).

Usage (Colab cell):
    !pip install -q datasets
    %cd /content/Project-ResNet-BK-An-O-N-Language-Model-Architecture
    !python notebooks/transformer_vs_resnetbk_colab.py --dataset wikitext2 --total_steps 200

Defaults are kept small so it runs quickly on Colab T4. Adjust flags for longer runs.
"""

import argparse
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset

# Ensure project root is on sys.path when executed as a script in Colab
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.models.configurable_resnet_bk import ConfigurableResNetBK, ResNetBKConfig
from src.models.transformer_baseline import TransformerConfig, TransformerLM


@dataclass
class ColabConfig:
    dataset: str = "wikitext2"  # {"wikitext2", "wikitext103"}
    seq_len: int = 256
    batch_size: int = 8
    vocab_size: int = 20000
    total_steps: int = 500
    eval_interval: int = 50
    eval_max_batches: Optional[int] = None  # limit eval steps for fairness
    log_interval: int = 20
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 20
    grad_clip: float = 1.0
    data_limit: Optional[int] = 500_000  # tokens for speed
    val_limit: Optional[int] = 100_000
    save_path: str = "benchmarks/results/colab_resnetbk_vs_transformer.json"
    device: str = "auto"  # "auto", "cuda", or "cpu"

    # Shared model dimensions
    d_model: int = 256
    n_layers: int = 6
    num_heads: int = 8
    ffn_dim: int = 1024
    num_experts: int = 4
    top_k: int = 2


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_vocab(texts: List[str], vocab_size: int) -> Tuple[Dict[str, int], Dict[int, str]]:
    counter = Counter()
    for line in texts:
        tokens = line.strip().split()
        if tokens:
            counter.update(tokens)

    special_tokens = ["<unk>"]
    stoi: Dict[str, int] = {}
    itos: Dict[int, str] = {}

    for token in special_tokens:
        idx = len(stoi)
        stoi[token] = idx
        itos[idx] = token

    for token, _ in counter.most_common(vocab_size - len(special_tokens)):
        if token not in stoi:
            idx = len(stoi)
            stoi[token] = idx
            itos[idx] = token

    return stoi, itos


def encode_texts(texts: List[str], stoi: Dict[str, int], limit_tokens: Optional[int] = None) -> torch.Tensor:
    unk_id = stoi["<unk>"]
    ids: List[int] = []
    total_tokens = 0
    for line in texts:
        for token in line.strip().split():
            ids.append(stoi.get(token, unk_id))
            total_tokens += 1
            if limit_tokens is not None and total_tokens >= limit_tokens:
                return torch.tensor(ids[:limit_tokens], dtype=torch.long)
    return torch.tensor(ids, dtype=torch.long)


def batchify(data: torch.Tensor, batch_size: int) -> torch.Tensor:
    seq_len = data.size(0) // batch_size
    data = data.narrow(0, 0, seq_len * batch_size)
    data = data.view(batch_size, seq_len).t().contiguous()
    return data


def load_language_dataset(
    name: str,
    seq_len: int,
    batch_size: int,
    vocab_size: int,
    data_limit: Optional[int],
    val_limit: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    if name == "wikitext2":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        train_texts = dataset["train"]["text"]
        val_texts = dataset["validation"]["text"]
    elif name == "wikitext103":
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
        train_texts = dataset["train"]["text"]
        val_texts = dataset["validation"]["text"]
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    stoi, itos = build_vocab(train_texts, vocab_size)
    train_ids = encode_texts(train_texts, stoi, data_limit)
    val_ids = encode_texts(val_texts, stoi, val_limit)

    train_data = batchify(train_ids, batch_size)
    val_data = batchify(val_ids, batch_size)

    # Truncate to equal length for fair eval
    min_len = min(train_data.size(0), val_data.size(0))
    train_data = train_data[:min_len]
    val_data = val_data[:min_len]

    vocab = {"stoi": stoi, "itos": itos, "size": len(stoi)}
    print(
        f"Loaded dataset={name} | train tokens={train_ids.numel():,} | "
        f"val tokens={val_ids.numel():,} | vocab size={vocab['size']:,}"
    )
    print(f"Batched train shape={train_data.shape}, val shape={val_data.shape}")
    return train_data, val_data, vocab


def get_batch(source: torch.Tensor, seq_len: int, batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    i = batch_idx * seq_len
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len]
    data = data.t().contiguous()
    target = target.t().contiguous()
    return data, target


def cycle_indices(data_source: torch.Tensor, seq_len: int):
    total_segments = (data_source.size(0) - 1) // seq_len
    while True:
        for idx in range(total_segments):
            yield idx


def evaluate(
    model: torch.nn.Module,
    data_source: torch.Tensor,
    seq_len: int,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        total_segments = (data_source.size(0) - 1) // seq_len
        limit = total_segments if max_batches is None else min(total_segments, max_batches)
        for seg_idx in range(limit):
            i = seg_idx * seq_len
            data, targets = get_batch(data_source, seq_len, i // seq_len)
            data = data.to(device)
            targets = targets.to(device)
            logits = model(data)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            losses.append(loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


def train_model(label: str, model: torch.nn.Module, train_data: torch.Tensor, val_data: torch.Tensor, cfg: ColabConfig) -> Dict:
    device_str = cfg.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    if device_str == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but torch was built without CUDA support. Falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    def lr_lambda(step: int):
        if step < cfg.warmup_steps:
            return float(step + 1) / float(max(1, cfg.warmup_steps))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    seq_indices = cycle_indices(train_data, cfg.seq_len)
    history = []
    start_time = time.time()
    last_log = start_time
    tokens_per_step = cfg.seq_len * cfg.batch_size
    throughput_meter = []

    oom = False
    loss = None
    try:
        for step in range(1, cfg.total_steps + 1):
            batch_idx = next(seq_indices)
            data, targets = get_batch(train_data, cfg.seq_len, batch_idx)
            data = data.to(device)
            targets = targets.to(device)

            logits = model(data)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            if step % cfg.log_interval == 0:
                now = time.time()
                elapsed = now - last_log
                tokens = cfg.log_interval * tokens_per_step
                throughput = tokens / max(elapsed, 1e-6)
                throughput_meter.append(throughput)
                last_log = now
                print(f"[{label}] step {step:04d} loss={loss.item():.4f} tokens/s={throughput:,.0f}")

            if step % cfg.eval_interval == 0:
                val_loss = evaluate(model, val_data, cfg.seq_len, device, cfg.eval_max_batches)
                ppl = math.exp(val_loss)
                history.append(
                    {
                        "step": step,
                        "train_loss": loss.item(),
                        "val_loss": val_loss,
                        "val_ppl": ppl,
                        "elapsed_sec": time.time() - start_time,
                    }
                )
                print(f"[{label}] Eval step {step}: val_loss={val_loss:.4f} | ppl={ppl:.2f}")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            oom = True
            print(f"[{label}] OOM encountered during training. Stopping early.")
        else:
            raise

    total_time = time.time() - start_time
    avg_throughput = sum(throughput_meter) / max(len(throughput_meter), 1)
    final_val_loss = evaluate(model, val_data, cfg.seq_len, device, cfg.eval_max_batches) if not oom else float("inf")

    summary = {
        "label": label,
        "final_val_loss": final_val_loss,
        "final_val_ppl": math.exp(final_val_loss) if final_val_loss != float("inf") else float("inf"),
        "total_time_sec": total_time,
        "tokens_processed": cfg.total_steps * tokens_per_step,
        "avg_tokens_per_sec": avg_throughput,
        "device": device_str,
        "oom": oom,
    }
    return {"history": history, "summary": summary}


def run_colab(cfg: ColabConfig):
    set_seed()
    train_data, val_data, vocab = load_language_dataset(
        cfg.dataset,
        cfg.seq_len,
        cfg.batch_size,
        cfg.vocab_size,
        cfg.data_limit,
        cfg.val_limit,
    )

    resnet_config = ResNetBKConfig(
        vocab_size=vocab["size"],
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_seq=cfg.seq_len,
        num_experts=cfg.num_experts,
        top_k=cfg.top_k,
        dropout_p=0.1,
        use_analytic_gradient=True,
        grad_blend=0.5,
    )
    resnet_model = ConfigurableResNetBK(resnet_config)

    transformer_cfg = TransformerConfig(
        vocab_size=vocab["size"],
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.num_heads,
        ffn_dim=cfg.ffn_dim,
        max_seq_len=cfg.seq_len,
    )
    transformer_model = TransformerLM(transformer_cfg)

    print("=== Training ResNet-BK ===")
    resnet_results = train_model("resnet_bk", resnet_model, train_data, val_data, cfg)

    print("=== Training Transformer Baseline ===")
    transformer_results = train_model("transformer", transformer_model, train_data, val_data, cfg)

    # Ensure save directory exists
    Path(cfg.save_path).parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "config": asdict(cfg),
        "vocab_size": vocab["size"],
        "results": {"resnet_bk": resnet_results, "transformer": transformer_results},
    }
    with open(cfg.save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved comparison to {cfg.save_path}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Colab comparison: ResNet-BK vs Transformer (small-scale)")
    parser.add_argument("--dataset", type=str, default="wikitext2", choices=["wikitext2", "wikitext103"])
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--total_steps", type=int, default=500)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--eval_max_batches", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--data_limit", type=int, default=500_000)
    parser.add_argument("--val_limit", type=int, default=100_000)
    parser.add_argument("--save_path", type=str, default="benchmarks/results/colab_resnetbk_vs_transformer.json")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--ffn_dim", type=int, default=1024)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=2)
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    cfg = ColabConfig(
        dataset=args.dataset,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        total_steps=args.total_steps,
        eval_interval=args.eval_interval,
        eval_max_batches=args.eval_max_batches,
        log_interval=args.log_interval,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        data_limit=args.data_limit,
        val_limit=args.val_limit,
        save_path=args.save_path,
        device=args.device,
        d_model=args.d_model,
        n_layers=args.n_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        num_experts=args.num_experts,
        top_k=args.top_k,
    )
    run_colab(cfg)


if __name__ == "__main__":
    main()
