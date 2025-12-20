#!/usr/bin/env python3
"""
Learning Efficiency Benchmark - Japanese
=========================================
Measures ResNet-BK's data efficiency compared to Transformer baseline.

Metrics:
- PPL drop per 1M tokens
- Tokens to reach target PPL
- Efficiency ratio (ResNet-BK / Transformer)

Usage:
    python scripts/benchmark_learning_efficiency.py --max-tokens 10000000
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).parent.parent))

# Use tiktoken (GPT-4 tokenizer) - open and works well for Japanese
try:
    import tiktoken
    TOKENIZER = tiktoken.get_encoding("cl100k_base")
    USE_TIKTOKEN = True
except ImportError:
    from transformers import AutoTokenizer
    TOKENIZER = None
    USE_TIKTOKEN = False


@dataclass
class EfficiencyMetrics:
    """Metrics for learning efficiency."""
    model_name: str
    tokens_seen: List[int] = field(default_factory=list)
    ppl_history: List[float] = field(default_factory=list)
    loss_history: List[float] = field(default_factory=list)
    time_per_1k_tokens: List[float] = field(default_factory=list)
    
    # Efficiency calculations
    ppl_per_1m_tokens: float = 0.0  # PPL improvement per 1M tokens
    tokens_to_ppl_1000: Optional[int] = None
    tokens_to_ppl_500: Optional[int] = None
    tokens_to_ppl_100: Optional[int] = None
    tokens_to_ppl_50: Optional[int] = None
    final_ppl: float = float('inf')
    total_tokens: int = 0
    total_time: float = 0.0
    
    def to_dict(self):
        return asdict(self)
    
    def save_json(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


def load_japanese_data(
    tokenizer,
    n_seq: int = 512,
    max_tokens: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load Japanese data for benchmark.
    
    Returns:
        train_data: tokenized training data
        valid_data: tokenized validation data
    """
    data_paths = [
        Path("data/japanese/pretrain_combined/data.jsonl"),
        Path("data/japanese/wikipedia_ja/data.jsonl"),
        Path("data/japanese/cc100_ja/data.jsonl"),
    ]
    
    # Find first available data file
    data_path = None
    for p in data_paths:
        if p.exists():
            data_path = p
            break
    
    if data_path is None:
        raise RuntimeError("No local data found. Run 'make prepare-japanese-data' first.")
    
    print(f"📂 Loading data from {data_path}")
    texts = []
    total_chars = 0
    # Estimate: ~4 chars per token for Japanese
    max_chars = (max_tokens * 4) if max_tokens else float('inf')
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            text = item.get("text", "")
            texts.append(text)
            total_chars += len(text)
            if total_chars > max_chars:
                break
    
    all_text = "\n".join(texts)
    print(f"📝 Loaded {len(texts):,} documents, {total_chars:,} characters")
    
    # Tokenize
    print("🔤 Tokenizing...")
    if USE_TIKTOKEN:
        tokens = tokenizer.encode(all_text)
    else:
        tokens = tokenizer.encode(all_text, add_special_tokens=False)
    
    if max_tokens:
        tokens = tokens[:max_tokens]
    
    print(f"📊 Total tokens: {len(tokens):,}")
    
    # Split into train/valid (90/10)
    split_idx = int(len(tokens) * 0.9)
    train_tokens = tokens[:split_idx]
    valid_tokens = tokens[split_idx:]
    
    def batchify(data, bsz):
        """Divide data into batches."""
        nbatch = len(data) // bsz
        data = data[:nbatch * bsz]
        return torch.tensor(data).view(bsz, -1)
    
    return batchify(train_tokens, 1), batchify(valid_tokens, 1)


def create_resnet_bk_model(vocab_size: int, d_model: int, n_layers: int, n_seq: int, device: str):
    """Create ResNet-BK model for benchmark."""
    from src.models.resnet_bk import LanguageModel
    from src.models.config import ResNetBKConfig
    
    config = ResNetBKConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_seq=n_seq,
        num_experts=4,
        top_k=1,
        dropout_p=0.1,
        use_birman_schwinger=True,  # Enable BK physics
        prime_bump_init=True,       # Enable prime bump initialization
        use_hybrid_attention=False, # Keep simple
        use_symplectic=False,       # Keep simple
        use_bitnet=False,           # Keep simple
    )
    
    model = LanguageModel(config).to(device)
    return model


def create_transformer_model(vocab_size: int, d_model: int, n_layers: int, n_seq: int, device: str):
    """Create Transformer baseline model."""
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
    
    class TransformerLM(nn.Module):
        def __init__(self, vocab_size, d_model, n_layers, n_seq, nhead=4):
            super().__init__()
            self.d_model = d_model
            self.n_seq = n_seq
            
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_embedding = nn.Embedding(n_seq, d_model)
            
            encoder_layer = TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True
            )
            self.transformer = TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.lm_head = nn.Linear(d_model, vocab_size)
            
            # Count parameters
            self.num_params = sum(p.numel() for p in self.parameters())
        
        def forward(self, x):
            B, L = x.shape
            positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
            
            x = self.embedding(x) + self.pos_embedding(positions)
            
            # Causal mask
            mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
            
            x = self.transformer(x, src_mask=mask, is_causal=True)
            return self.lm_head(x)
    
    model = TransformerLM(vocab_size, d_model, n_layers, n_seq).to(device)
    return model


def train_and_measure(
    model: nn.Module,
    model_name: str,
    train_data: torch.Tensor,
    valid_data: torch.Tensor,
    n_seq: int,
    device: str,
    checkpoint_interval: int = 100000,
    lr: float = 1e-3,
    grad_clip: float = 1.0,
) -> EfficiencyMetrics:
    """
    Train model and measure learning efficiency.
    """
    metrics = EfficiencyMetrics(model_name=model_name)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"📐 {model_name}: {num_params:,} parameters")
    
    model.train()
    
    total_tokens = 0
    total_loss = 0.0
    num_batches = 0
    start_time = time.time()
    checkpoint_start = time.time()
    
    # Training loop
    train_data = train_data.to(device)
    n_batches = (train_data.size(1) - 1) // n_seq
    
    for batch_idx in range(n_batches):
        # Get batch
        i = batch_idx * n_seq
        data = train_data[:, i:i+n_seq]
        target = train_data[:, i+1:i+n_seq+1]
        
        if data.size(1) != n_seq or target.size(1) != n_seq:
            continue
        
        # Forward
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        total_tokens += n_seq
        
        # Checkpoint
        if total_tokens % checkpoint_interval == 0 and total_tokens > 0:
            avg_loss = total_loss / num_batches
            ppl = torch.exp(torch.tensor(avg_loss)).item()
            elapsed = time.time() - checkpoint_start
            
            metrics.tokens_seen.append(total_tokens)
            metrics.ppl_history.append(ppl)
            metrics.loss_history.append(avg_loss)
            metrics.time_per_1k_tokens.append(elapsed / (checkpoint_interval / 1000))
            
            print(f"  [{model_name}] Tokens: {total_tokens:,} | Loss: {avg_loss:.4f} | PPL: {ppl:.2f}")
            
            # Check milestones
            if metrics.tokens_to_ppl_1000 is None and ppl < 1000:
                metrics.tokens_to_ppl_1000 = total_tokens
            if metrics.tokens_to_ppl_500 is None and ppl < 500:
                metrics.tokens_to_ppl_500 = total_tokens
            if metrics.tokens_to_ppl_100 is None and ppl < 100:
                metrics.tokens_to_ppl_100 = total_tokens
            if metrics.tokens_to_ppl_50 is None and ppl < 50:
                metrics.tokens_to_ppl_50 = total_tokens
            
            checkpoint_start = time.time()
    
    # Final metrics
    if num_batches > 0:
        final_loss = total_loss / num_batches
        metrics.final_ppl = torch.exp(torch.tensor(final_loss)).item()
    
    metrics.total_tokens = total_tokens
    metrics.total_time = time.time() - start_time
    
    # Calculate PPL improvement per 1M tokens
    if len(metrics.ppl_history) >= 2:
        initial_ppl = metrics.ppl_history[0]
        final_ppl = metrics.ppl_history[-1]
        tokens_m = total_tokens / 1_000_000
        metrics.ppl_per_1m_tokens = (initial_ppl - final_ppl) / tokens_m
    
    return metrics


def run_benchmark(args):
    """Run the full benchmark."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load tokenizer
    print("\n📝 Loading tokenizer...")
    if USE_TIKTOKEN:
        tokenizer = TOKENIZER
        vocab_size = tokenizer.n_vocab
        print(f"   Using tiktoken cl100k_base")
    else:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")  # fallback
        vocab_size = len(tokenizer)
    
    # Round up to power of 2 for efficiency
    # tiktoken cl100k_base has ~100k vocab, use 131072 (2^17) for padding
    vocab_size = 131072  # Fixed for tiktoken compatibility
    
    print(f"   Vocab size: {vocab_size:,}")
    
    # Load data
    print("\n📚 Loading Japanese data...")
    train_data, valid_data = load_japanese_data(
        tokenizer, 
        n_seq=args.n_seq,
        max_tokens=args.max_tokens
    )
    
    results = {}
    
    # Benchmark ResNet-BK
    print("\n" + "="*60)
    print("🧠 Benchmarking ResNet-BK")
    print("="*60)
    
    resnet_model = create_resnet_bk_model(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_seq=args.n_seq,
        device=device
    )
    
    resnet_metrics = train_and_measure(
        model=resnet_model,
        model_name="ResNet-BK",
        train_data=train_data,
        valid_data=valid_data,
        n_seq=args.n_seq,
        device=device,
        checkpoint_interval=args.checkpoint_interval,
        lr=args.lr,
        grad_clip=args.grad_clip,
    )
    results["resnet_bk"] = resnet_metrics
    
    # Free memory
    del resnet_model
    torch.cuda.empty_cache()
    
    # Benchmark Transformer
    print("\n" + "="*60)
    print("🔄 Benchmarking Transformer Baseline")
    print("="*60)
    
    transformer_model = create_transformer_model(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_seq=args.n_seq,
        device=device
    )
    
    transformer_metrics = train_and_measure(
        model=transformer_model,
        model_name="Transformer",
        train_data=train_data,
        valid_data=valid_data,
        n_seq=args.n_seq,
        device=device,
        checkpoint_interval=args.checkpoint_interval,
        lr=args.lr,
        grad_clip=args.grad_clip,
    )
    results["transformer"] = transformer_metrics
    
    del transformer_model
    torch.cuda.empty_cache()
    
    # Print comparison
    print("\n" + "="*60)
    print("📊 RESULTS COMPARISON")
    print("="*60)
    
    print(f"\n{'Metric':<30} {'ResNet-BK':>15} {'Transformer':>15} {'Efficiency':>12}")
    print("-"*72)
    
    r = resnet_metrics
    t = transformer_metrics
    
    print(f"{'Final PPL':<30} {r.final_ppl:>15.2f} {t.final_ppl:>15.2f} {t.final_ppl/r.final_ppl:>11.2f}x")
    print(f"{'PPL drop per 1M tokens':<30} {r.ppl_per_1m_tokens:>15.2f} {t.ppl_per_1m_tokens:>15.2f} {r.ppl_per_1m_tokens/max(t.ppl_per_1m_tokens,1):>11.2f}x")
    print(f"{'Total time (s)':<30} {r.total_time:>15.1f} {t.total_time:>15.1f}")
    
    if r.tokens_to_ppl_1000 and t.tokens_to_ppl_1000:
        eff = t.tokens_to_ppl_1000 / r.tokens_to_ppl_1000
        print(f"{'Tokens to PPL 1000':<30} {r.tokens_to_ppl_1000:>15,} {t.tokens_to_ppl_1000:>15,} {eff:>11.2f}x")
    
    if r.tokens_to_ppl_100 and t.tokens_to_ppl_100:
        eff = t.tokens_to_ppl_100 / r.tokens_to_ppl_100
        print(f"{'Tokens to PPL 100':<30} {r.tokens_to_ppl_100:>15,} {t.tokens_to_ppl_100:>15,} {eff:>11.2f}x")
    
    # Estimate requirements for 10B model
    print("\n" + "="*60)
    print("📈 ESTIMATED REQUIREMENTS FOR 10B MODEL")
    print("="*60)
    
    # Scaling law assumption: larger models need more data proportionally
    # but ResNet-BK efficiency should scale similarly
    efficiency_ratio = t.final_ppl / r.final_ppl if r.final_ppl > 0 else 1.0
    
    print(f"\nResNet-BK is {efficiency_ratio:.2f}x more data-efficient than Transformer")
    
    if r.ppl_per_1m_tokens > 0:
        # Estimate tokens needed for various PPL targets
        current_ppl = r.ppl_history[0] if r.ppl_history else 1000000
        
        for target_ppl in [1000, 100, 50, 25, 10]:
            if current_ppl > target_ppl:
                ppl_drop_needed = current_ppl - target_ppl
                tokens_needed = (ppl_drop_needed / r.ppl_per_1m_tokens) * 1_000_000
                tokens_b = tokens_needed / 1_000_000_000
                print(f"  PPL {target_ppl:<5}: ~{tokens_b:.1f}B tokens needed")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    resnet_metrics.save_json(output_dir / "resnet_bk_efficiency.json")
    transformer_metrics.save_json(output_dir / "transformer_efficiency.json")
    
    # Combined summary
    summary = {
        "benchmark_config": {
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "n_seq": args.n_seq,
            "max_tokens": args.max_tokens,
            "checkpoint_interval": args.checkpoint_interval,
        },
        "efficiency_ratio": efficiency_ratio,
        "resnet_bk_final_ppl": r.final_ppl,
        "transformer_final_ppl": t.final_ppl,
        "resnet_bk_ppl_per_1m": r.ppl_per_1m_tokens,
        "transformer_ppl_per_1m": t.ppl_per_1m_tokens,
    }
    
    with open(output_dir / "efficiency_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Results saved to {output_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark learning efficiency")
    parser.add_argument("--max-tokens", type=int, default=5_000_000, help="Max tokens to train on")
    parser.add_argument("--checkpoint-interval", type=int, default=100_000, help="Checkpoint every N tokens")
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension")
    parser.add_argument("--n-layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--n-seq", type=int, default=256, help="Sequence length")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--output-dir", type=str, default="results/learning_efficiency", help="Output directory")
    
    args = parser.parse_args()
    
    print("="*60)
    print("🧪 Learning Efficiency Benchmark")
    print("="*60)
    print(f"Max tokens: {args.max_tokens:,}")
    print(f"Model: d_model={args.d_model}, n_layers={args.n_layers}")
    print(f"Sequence length: {args.n_seq}")
    print(f"Checkpoint interval: {args.checkpoint_interval:,} tokens")
    print("="*60)
    
    run_benchmark(args)


if __name__ == "__main__":
    main()
