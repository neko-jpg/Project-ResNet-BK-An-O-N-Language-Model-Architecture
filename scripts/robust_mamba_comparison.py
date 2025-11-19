#!/usr/bin/env python3
"""
Robust Mamba vs ResNet-BK Comparison Benchmark

Designed to prove ResNet-BK superiority over Mamba with:
1. Long-context stability (論文 Table 1)
2. Lower perplexity at equal compute
3. Graceful degradation under memory pressure

Features:
- Aggressive memory management
- FP32 fallback for stability
- Checkpoint saving after each run
- Detailed error logging
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

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
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_gpu_memory_info() -> Dict[str, float]:
    """Get current GPU memory usage."""
    if not torch.cuda.is_available():
        return {}
    
    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    
    return {
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "max_allocated_gb": max_allocated,
    }


def clear_gpu_memory():
    """Aggressively clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def get_tokenizer(name: str):
    """Load tokenizer with padding token."""
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def prepare_dataset(dataset_name: str, dataset_config: str, tokenizer, seq_length: int, max_samples: int = 1000):
    """Prepare dataset with limited samples for faster iteration."""
    print(f"Loading dataset {dataset_name}/{dataset_config}...")
    raw = load_dataset(dataset_name, dataset_config, split="train")
    
    # Limit samples for faster benchmarking
    if len(raw) > max_samples:
        raw = raw.select(range(max_samples))
    
    def tokenize_fn(examples):
        return tokenizer(examples["text"], add_special_tokens=False)
    
    tokenized = raw.map(tokenize_fn, batched=True, remove_columns=["text"])
    
    # Group into sequences
    seq_plus_one = seq_length + 1
    
    def group_texts(examples):
        concatenated = []
        for ids in examples["input_ids"]:
            concatenated.extend(ids)
        
        total_length = len(concatenated) // seq_plus_one * seq_plus_one
        concatenated = concatenated[:total_length]
        
        result = [
            concatenated[i:i + seq_plus_one] 
            for i in range(0, total_length, seq_plus_one)
        ]
        return {"input_ids": result}
    
    grouped = tokenized.map(group_texts, batched=True, remove_columns=tokenized.column_names)
    grouped.set_format(type="torch", columns=["input_ids"])
    
    print(f"Dataset prepared: {len(grouped)} sequences of length {seq_length}")
    return grouped


def make_dataloader(dataset, batch_size: int, seed: int):
    """Create dataloader with proper collation."""
    g = torch.Generator().manual_seed(seed)
    
    def collate_fn(batch):
        inputs = torch.stack([b["input_ids"][:-1] for b in batch])
        targets = torch.stack([b["input_ids"][1:] for b in batch])
        return inputs, targets
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=g,
        collate_fn=collate_fn,
    )


def build_resnet_bk(seq_length: int, vocab_size: int, config: Dict, use_theory: bool = True):
    """Build ResNet-BK model with theory features."""
    return ResNetBK(
        vocab_size=vocab_size,
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_seq=seq_length,
        num_experts=config["num_experts"],
        top_k=config["top_k"],
        dropout_p=config["dropout"],
        use_scattering_router=use_theory,
        use_birman_schwinger=use_theory,
    )


def build_mamba(seq_length: int, vocab_size: int, config: Dict):
    """Build Mamba baseline model."""
    resnet_cfg = argparse.Namespace(
        vocab_size=vocab_size,
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_seq=seq_length,
        dropout=config["dropout"],
        tie_weights=True,
    )
    return MambaLM(create_mamba_from_resnetbk_config(resnet_cfg))


def train_model(
    model_name: str,
    model,
    dataloader,
    max_steps: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    log_every: int = 10,
    use_fp32: bool = True,
) -> Dict:
    """
    Train model and return results.
    
    Returns dict with:
    - losses: list of loss values
    - final_loss: final loss value
    - steps_completed: number of steps completed
    - diverged: whether training diverged
    - wall_time: training time in seconds
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    model = model.to(DEVICE)
    
    # Use FP32 for stability
    if use_fp32:
        model = model.float()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_steps,
        eta_min=lr * 0.1,
    )
    
    losses = []
    diverged = False
    start_time = time.time()
    
    try:
        for step, (inputs, targets) in enumerate(dataloader):
            if step >= max_steps:
                break
            
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass
            outputs = model(inputs)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
            # Compute loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
            
            # Check for divergence
            if not torch.isfinite(loss):
                print(f"⚠️  {model_name} DIVERGED at step {step+1}: loss={loss.item()}")
                diverged = True
                break
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            scheduler.step()
            
            losses.append(loss.item())
            
            # Logging
            if (step + 1) % log_every == 0:
                mem_info = get_gpu_memory_info()
                mem_str = f"GPU: {mem_info.get('allocated_gb', 0):.2f}GB" if mem_info else ""
                print(
                    f"  Step {step+1:3d}/{max_steps} | "
                    f"Loss: {loss.item():.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                    f"{mem_str}"
                )
            
            # Periodic memory cleanup
            if (step + 1) % 20 == 0:
                clear_gpu_memory()
    
    except RuntimeError as e:
        print(f"❌ {model_name} ERROR at step {step+1}: {e}")
        traceback.print_exc()
        diverged = True
    
    wall_time = time.time() - start_time
    
    # Calculate final metrics
    final_loss = losses[-1] if losses else float('inf')
    avg_loss = np.mean(losses) if losses else float('inf')
    
    result = {
        "model": model_name,
        "losses": losses,
        "final_loss": final_loss,
        "avg_loss": avg_loss,
        "steps_completed": len(losses),
        "diverged": diverged,
        "wall_time_sec": wall_time,
        "memory_peak_gb": get_gpu_memory_info().get("max_allocated_gb", 0),
    }
    
    print(f"\n{model_name} Results:")
    print(f"  Steps: {len(losses)}/{max_steps}")
    print(f"  Final Loss: {final_loss:.4f}")
    print(f"  Avg Loss: {avg_loss:.4f}")
    print(f"  Diverged: {diverged}")
    print(f"  Time: {wall_time:.1f}s")
    
    return result


def run_comparison(
    seq_length: int,
    seed: int,
    tokenizer,
    dataset,
    model_config: Dict,
    train_config: Dict,
    use_theory: bool = True,
) -> Dict:
    """Run single comparison between ResNet-BK and Mamba."""
    
    print(f"\n{'#'*60}")
    print(f"# Sequence Length: {seq_length} | Seed: {seed}")
    print(f"{'#'*60}")
    
    set_seed(seed)
    
    batch_size = train_config["batch_size"]
    dataloader = make_dataloader(dataset, batch_size, seed)
    
    results = {
        "seq_length": seq_length,
        "seed": seed,
        "batch_size": batch_size,
        "use_theory": use_theory,
    }
    
    # Train ResNet-BK
    print("\n[1/2] Building ResNet-BK...")
    resnet_model = build_resnet_bk(seq_length, tokenizer.vocab_size, model_config, use_theory)
    
    resnet_result = train_model(
        "ResNet-BK",
        resnet_model,
        dataloader,
        max_steps=train_config["max_steps"],
        lr=train_config["lr"],
        weight_decay=train_config["weight_decay"],
        grad_clip=train_config["grad_clip"],
        log_every=train_config["log_every"],
        use_fp32=train_config["use_fp32"],
    )
    
    results["resnet_bk"] = resnet_result
    
    # Clean up before Mamba
    del resnet_model
    clear_gpu_memory()
    
    # Train Mamba
    print("\n[2/2] Building Mamba...")
    dataloader = make_dataloader(dataset, batch_size, seed)  # Recreate dataloader
    mamba_model = build_mamba(seq_length, tokenizer.vocab_size, model_config)
    
    mamba_result = train_model(
        "Mamba",
        mamba_model,
        dataloader,
        max_steps=train_config["max_steps"],
        lr=train_config["lr"],
        weight_decay=train_config["weight_decay"],
        grad_clip=train_config["grad_clip"],
        log_every=train_config["log_every"],
        use_fp32=train_config["use_fp32"],
    )
    
    results["mamba"] = mamba_result
    
    # Clean up
    del mamba_model
    clear_gpu_memory()
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    resnet_better = resnet_result["final_loss"] < mamba_result["final_loss"]
    resnet_stable = not resnet_result["diverged"]
    mamba_stable = not mamba_result["diverged"]
    
    print(f"ResNet-BK: Loss={resnet_result['final_loss']:.4f} | Stable={resnet_stable}")
    print(f"Mamba:     Loss={mamba_result['final_loss']:.4f} | Stable={mamba_stable}")
    
    if resnet_stable and not mamba_stable:
        print("✅ ResNet-BK WINS: Stable while Mamba diverged!")
    elif resnet_better and resnet_stable:
        improvement = (mamba_result['final_loss'] - resnet_result['final_loss']) / mamba_result['final_loss'] * 100
        print(f"✅ ResNet-BK WINS: {improvement:.1f}% lower loss!")
    elif not resnet_stable and mamba_stable:
        print("❌ Mamba wins: ResNet-BK diverged")
    else:
        print("⚖️  Mixed results")
    
    results["winner"] = "resnet_bk" if (resnet_better and resnet_stable) or (resnet_stable and not mamba_stable) else "mamba"
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Robust Mamba vs ResNet-BK Comparison")
    parser.add_argument("--seq-lengths", nargs="+", type=int, default=[4096, 8192], help="Sequence lengths to test")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Random seeds")
    parser.add_argument("--max-steps", type=int, default=100, help="Training steps per run")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--no-theory", action="store_true", help="Disable theory features (scattering router, etc)")
    parser.add_argument("--dataset", default="wikitext", help="Dataset name")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1", help="Dataset config")
    parser.add_argument("--tokenizer", default="gpt2", help="Tokenizer name")
    parser.add_argument("--out-dir", default="results/mamba_comparison", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=1000, help="Max dataset samples")
    args = parser.parse_args()
    
    print("="*60)
    print("ROBUST MAMBA vs RESNET-BK COMPARISON")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Sequence lengths: {args.seq_lengths}")
    print(f"Seeds: {args.seeds}")
    print(f"Max steps: {args.max_steps}")
    print(f"Theory features: {not args.no_theory}")
    
    # Load tokenizer
    tokenizer = get_tokenizer(args.tokenizer)
    
    # Model config
    model_config = {
        "d_model": 256,
        "n_layers": 6,
        "num_experts": 4,
        "top_k": 1,
        "dropout": 0.1,
    }
    
    # Training config
    train_config = {
        "max_steps": args.max_steps,
        "lr": args.lr,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "log_every": 10,
        "batch_size": args.batch_size,
        "use_fp32": True,  # Always use FP32 for stability
    }
    
    # Run comparisons
    all_results = []
    
    for seq_length in args.seq_lengths:
        # Prepare dataset for this sequence length
        dataset = prepare_dataset(
            args.dataset,
            args.dataset_config,
            tokenizer,
            seq_length,
            max_samples=args.max_samples,
        )
        
        for seed in args.seeds:
            try:
                result = run_comparison(
                    seq_length=seq_length,
                    seed=seed,
                    tokenizer=tokenizer,
                    dataset=dataset,
                    model_config=model_config,
                    train_config=train_config,
                    use_theory=not args.no_theory,
                )
                all_results.append(result)
                
                # Save checkpoint after each run
                out_dir = Path(args.out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_file = out_dir / f"checkpoint_seq{seq_length}_seed{seed}.json"
                with open(checkpoint_file, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"\n💾 Checkpoint saved: {checkpoint_file}")
                
            except Exception as e:
                print(f"\n❌ FAILED: seq_length={seq_length}, seed={seed}")
                print(f"Error: {e}")
                traceback.print_exc()
    
    # Save final results
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    theory_tag = "theory" if not args.no_theory else "vanilla"
    results_file = out_dir / f"comparison_results_{theory_tag}.json"
    
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ ALL RESULTS SAVED: {results_file}")
    print(f"{'='*60}")
    
    # Print summary
    print("\nFINAL SUMMARY:")
    resnet_wins = sum(1 for r in all_results if r.get("winner") == "resnet_bk")
    mamba_wins = sum(1 for r in all_results if r.get("winner") == "mamba")
    print(f"  ResNet-BK wins: {resnet_wins}/{len(all_results)}")
    print(f"  Mamba wins: {mamba_wins}/{len(all_results)}")
    
    if resnet_wins > mamba_wins:
        print("\n🎉 ResNet-BK DOMINATES! 🎉")
    elif mamba_wins > resnet_wins:
        print("\n⚠️  Mamba performed better")
    else:
        print("\n⚖️  Tie")


if __name__ == "__main__":
    main()
