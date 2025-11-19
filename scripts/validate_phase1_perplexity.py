"""
Phase 1 Perplexity Validation Script

WikiText-103またはC4サブセットでperplexityを測定し、
Phase 1モデルとFP16ベースラインを比較します。
PPL劣化が5%未満であることを検証します。

Requirements: 9.1, 9.2, 9.5
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.phase1.config import Phase1Config
from src.models.phase1.factory import create_phase1_model
from src.models.resnet_bk import LanguageModel as BaselineModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class PerplexityResult:
    """Perplexity測定結果"""
    
    model_name: str
    dataset_name: str
    dataset_config: str
    
    # Perplexity metrics
    perplexity: float
    bits_per_byte: float
    avg_loss: float
    
    # Configuration
    config: Dict
    seq_length: int
    batch_size: int
    num_samples: int
    
    # Comparison metrics (if baseline available)
    baseline_perplexity: Optional[float] = None
    ppl_degradation_percent: Optional[float] = None
    passes_5_percent_threshold: Optional[bool] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def print_summary(self):
        """サマリーを出力"""
        print("\n" + "=" * 70)
        print(f"PERPLEXITY VALIDATION: {self.model_name}")
        print("=" * 70)
        print(f"Dataset: {self.dataset_name} ({self.dataset_config})")
        print(f"Samples: {self.num_samples}")
        print(f"Sequence Length: {self.seq_length}")
        print(f"Batch Size: {self.batch_size}")
        print()
        print(f"Results:")
        print(f"  Perplexity:     {self.perplexity:>10.4f}")
        print(f"  Bits per Byte:  {self.bits_per_byte:>10.4f}")
        print(f"  Avg Loss:       {self.avg_loss:>10.4f}")
        
        if self.baseline_perplexity is not None:
            print()
            print(f"Comparison to Baseline:")
            print(f"  Baseline PPL:   {self.baseline_perplexity:>10.4f}")
            print(f"  Degradation:    {self.ppl_degradation_percent:>10.2f}%")
            
            if self.passes_5_percent_threshold:
                print(f"  Status:         ✅ PASS (< 5% degradation)")
            else:
                print(f"  Status:         ❌ FAIL (>= 5% degradation)")
        
        print("=" * 70)


def prepare_dataset(
    dataset_name: str,
    dataset_config: str,
    tokenizer,
    seq_length: int,
    split: str = "validation",
    max_samples: Optional[int] = None,
):
    """
    データセットを準備
    
    Args:
        dataset_name: データセット名
        dataset_config: データセット設定
        tokenizer: トークナイザー
        seq_length: シーケンス長
        split: データセット分割
        max_samples: 最大サンプル数
    
    Returns:
        準備されたデータセット
    """
    print(f"📚 Loading dataset: {dataset_name} ({dataset_config})")
    
    # Load dataset
    try:
        raw_dataset = load_dataset(dataset_name, dataset_config, split=split)
    except Exception as e:
        print(f"⚠️  Failed to load {split} split: {e}")
        print(f"   Trying 'test' split instead...")
        raw_dataset = load_dataset(dataset_name, dataset_config, split="test")
    
    # Limit samples if requested
    if max_samples is not None and len(raw_dataset) > max_samples:
        raw_dataset = raw_dataset.select(range(max_samples))
    
    print(f"   Loaded {len(raw_dataset)} samples")
    
    # Tokenize
    def tokenize_function(examples):
        # Only return input_ids, no attention_mask
        result = tokenizer(examples["text"], add_special_tokens=False)
        return {"input_ids": result["input_ids"]}
    
    print(f"   Tokenizing...")
    tokenized = raw_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing",
    )
    
    # Group into sequences
    def group_texts(examples):
        # Concatenate all texts
        concatenated = []
        for input_ids in examples["input_ids"]:
            concatenated.extend(input_ids)
        
        # Split into chunks of seq_length
        total_length = (len(concatenated) // seq_length) * seq_length
        result = {
            "input_ids": [
                concatenated[i : i + seq_length]
                for i in range(0, total_length, seq_length)
            ]
        }
        return result
    
    print(f"   Grouping into sequences of length {seq_length}...")
    grouped = tokenized.map(
        group_texts,
        batched=True,
        desc="Grouping",
    )
    
    grouped.set_format(type="torch", columns=["input_ids"])
    
    print(f"   Final dataset size: {len(grouped)} sequences")
    
    return grouped


def evaluate_perplexity(
    model: torch.nn.Module,
    dataloader: DataLoader,
    model_name: str,
) -> tuple[float, float]:
    """
    モデルのperplexityを評価
    
    Args:
        model: 評価対象のモデル
        dataloader: データローダー
        model_name: モデル名
    
    Returns:
        (perplexity, avg_loss): perplexityと平均損失
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    print(f"🔍 Evaluating {model_name}...")
    
    # Clear CUDA cache before evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            try:
                input_ids = batch["input_ids"].to(DEVICE)
                
                # Create targets (shifted by 1)
                targets = input_ids[:, 1:].contiguous()
                inputs = input_ids[:, :-1].contiguous()
                
                # Forward pass
                outputs = model(inputs)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                
                # Calculate loss
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    reduction="sum",
                )
                
                total_loss += loss.item()
                total_tokens += targets.numel()
                
                # Periodic cleanup to avoid memory fragmentation
                if (batch_idx + 1) % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "CUDA" in str(e) or "out of memory" in str(e):
                    print(f"\n⚠️  CUDA error at batch {batch_idx}: {e}")
                    print("   Attempting to recover...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    # Skip this batch and continue
                    continue
                else:
                    raise
    
    # Calculate perplexity
    if total_tokens == 0:
        print("⚠️  No tokens processed. Returning infinite perplexity.")
        return float('inf'), float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity, avg_loss


def validate_phase1_perplexity(
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
    seq_length: int = 512,
    batch_size: int = 8,
    max_samples: Optional[int] = 1000,
    vocab_size: int = 50257,
    d_model: int = 512,
    n_layers: int = 8,
    compare_baseline: bool = True,
    phase1_configs: Optional[List[Phase1Config]] = None,
) -> Dict[str, PerplexityResult]:
    """
    Phase 1モデルのperplexityを検証
    
    Args:
        dataset_name: データセット名
        dataset_config: データセット設定
        seq_length: シーケンス長
        batch_size: バッチサイズ
        max_samples: 最大サンプル数
        vocab_size: 語彙サイズ
        d_model: モデル次元
        n_layers: レイヤー数
        compare_baseline: ベースラインと比較するか
        phase1_configs: Phase 1設定のリスト
    
    Returns:
        Dict[str, PerplexityResult]: モデル名をキーとするperplexity結果
    """
    print(f"📊 Phase 1 Perplexity Validation")
    print(f"   Dataset: {dataset_name} ({dataset_config})")
    print(f"   Sequence Length: {seq_length}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Max Samples: {max_samples}")
    print()
    
    # Prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare dataset
    dataset = prepare_dataset(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        tokenizer=tokenizer,
        seq_length=seq_length + 1,  # +1 for target shift
        split="validation",
        max_samples=max_samples,
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    
    results = {}
    baseline_ppl = None
    
    # Evaluate baseline
    if compare_baseline:
        print("\n" + "=" * 70)
        print("BASELINE MODEL")
        print("=" * 70)
        
        baseline_config = {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_seq": seq_length,
            "num_experts": 4,
            "top_k": 1,
            "dropout_p": 0.0,  # No dropout for evaluation
            "use_scattering_router": False,  # Disable for stability
            "use_birman_schwinger": False,  # Disable for stability
        }
        
        try:
            baseline_model = BaselineModel(**baseline_config).to(DEVICE)
            baseline_model.eval()  # Ensure eval mode
            
            # Warmup to avoid initial CUDA issues
            print("   Warming up model...")
            with torch.no_grad():
                dummy_input = torch.randint(0, vocab_size, (2, seq_length), device=DEVICE)
                _ = baseline_model(dummy_input)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            
            ppl, avg_loss = evaluate_perplexity(
                model=baseline_model,
                dataloader=dataloader,
                model_name="Baseline (ResNet-BK)",
            )
        except Exception as e:
            print(f"⚠️  Baseline evaluation failed: {e}")
            print("   Skipping baseline comparison...")
            compare_baseline = False
            baseline_ppl = None
            del baseline_model
            torch.cuda.empty_cache()
            
            # Continue without baseline
            if not compare_baseline:
                results = {}
        else:
            
            baseline_ppl = ppl
            
            result = PerplexityResult(
                model_name="Baseline (ResNet-BK)",
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                perplexity=ppl,
                bits_per_byte=avg_loss / 0.69314718056,  # log(2)
                avg_loss=avg_loss,
                config=baseline_config,
                seq_length=seq_length,
                batch_size=batch_size,
                num_samples=len(dataset),
            )
            
            results["baseline"] = result
            result.print_summary()
            
            del baseline_model
            torch.cuda.empty_cache()
    
    # Evaluate Phase 1 configurations
    if phase1_configs is None:
        phase1_configs = [
            Phase1Config(
                ar_ssm_enabled=True,
                ar_ssm_max_rank=32,
                htt_enabled=True,
                htt_rank=16,
                lns_enabled=False,
            )
        ]
    
    for i, phase1_config in enumerate(phase1_configs):
        config_name = f"phase1_config_{i}"
        
        print("\n" + "=" * 70)
        print(f"PHASE 1 MODEL (Config {i})")
        print("=" * 70)
        print(f"  AR-SSM: {phase1_config.ar_ssm_enabled} (rank={phase1_config.ar_ssm_max_rank})")
        print(f"  HTT: {phase1_config.htt_enabled} (rank={phase1_config.htt_rank})")
        print(f"  LNS: {phase1_config.lns_enabled}")
        print()
        
        phase1_config_dict = {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_seq": seq_length,
            "phase1_config": phase1_config,
        }
        
        phase1_model = create_phase1_model(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_seq=seq_length,
            config=phase1_config,
        ).to(DEVICE)
        
        ppl, avg_loss = evaluate_perplexity(
            model=phase1_model,
            dataloader=dataloader,
            model_name=f"Phase 1 (Config {i})",
        )
        
        result = PerplexityResult(
            model_name=f"Phase 1 (Config {i})",
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            perplexity=ppl,
            bits_per_byte=avg_loss / 0.69314718056,
            avg_loss=avg_loss,
            config=phase1_config_dict,
            seq_length=seq_length,
            batch_size=batch_size,
            num_samples=len(dataset),
        )
        
        # Calculate degradation if baseline available
        if baseline_ppl is not None:
            result.baseline_perplexity = baseline_ppl
            result.ppl_degradation_percent = ((ppl - baseline_ppl) / baseline_ppl) * 100
            result.passes_5_percent_threshold = result.ppl_degradation_percent < 5.0
        
        results[config_name] = result
        result.print_summary()
        
        del phase1_model
        torch.cuda.empty_cache()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 Perplexity Validation Script"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="wikitext",
        help="Dataset name (wikitext or c4)",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="wikitext-103-raw-v1",
        help="Dataset configuration",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=512,
        help="Sequence length",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples (None for all)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=50257,
        help="Vocabulary size",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=512,
        help="Model dimension",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=8,
        help="Number of layers",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip baseline comparison",
    )
    parser.add_argument(
        "--test-htt-ranks",
        nargs="+",
        type=int,
        default=[16],
        help="HTT ranks to test",
    )
    parser.add_argument(
        "--test-ar-ssm-ranks",
        nargs="+",
        type=int,
        default=[32],
        help="AR-SSM max ranks to test",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/benchmarks",
        help="Output directory",
    )
    args = parser.parse_args()
    
    # Create Phase 1 configurations to test
    phase1_configs = []
    for htt_rank in args.test_htt_ranks:
        for ar_ssm_rank in args.test_ar_ssm_ranks:
            config = Phase1Config(
                ar_ssm_enabled=True,
                ar_ssm_max_rank=ar_ssm_rank,
                ar_ssm_min_rank=4,
                htt_enabled=True,
                htt_rank=htt_rank,
                htt_num_cores=2,
                lns_enabled=False,
                use_gradient_checkpointing=False,  # Not needed for eval
            )
            phase1_configs.append(config)
    
    # Run validation
    results = validate_phase1_perplexity(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        compare_baseline=not args.no_baseline,
        phase1_configs=phase1_configs,
    )
    
    # Save results
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results_dict = {
        name: result.to_dict() for name, result in results.items()
    }
    
    out_path = out_dir / "phase1_perplexity_validation.json"
    with open(out_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n💾 Results saved to: {out_path}")
    
    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 70)
    
    baseline_ppl = None
    if "baseline" in results:
        baseline_ppl = results["baseline"].perplexity
        print(f"Baseline Perplexity: {baseline_ppl:.4f}")
        print()
    
    all_pass = True
    for name, result in results.items():
        if name == "baseline":
            continue
        
        print(f"{result.model_name}:")
        print(f"  Perplexity: {result.perplexity:.4f}")
        
        if result.baseline_perplexity is not None:
            print(f"  Degradation: {result.ppl_degradation_percent:.2f}%")
            
            if result.passes_5_percent_threshold:
                print(f"  Status: ✅ PASS")
            else:
                print(f"  Status: ❌ FAIL")
                all_pass = False
        print()
    
    if all_pass and baseline_ppl is not None:
        print("✅ All Phase 1 configurations pass the 5% PPL degradation threshold!")
    elif baseline_ppl is not None:
        print("❌ Some Phase 1 configurations exceed the 5% PPL degradation threshold.")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
