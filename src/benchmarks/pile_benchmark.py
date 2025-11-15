"""
The Pile Comprehensive Benchmark
Train ResNet-BK on The Pile (1B token subset) and evaluate domain-specific performance.

This script implements task 9.6:
- Train on 1B token subset from The Pile
- Evaluate domain-specific performance across Pile's 22 domains
- Compare to WikiText-2, WikiText-103, Penn Treebank, and C4 results
- Track FLOPs, wall-clock time, memory usage
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import math
import json
from pathlib import Path
from dataclasses import dataclass, asdict, replace
from typing import Dict, List, Optional
from collections import Counter
from datasets import load_dataset

# Optional imports for plotting
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from src.models.configurable_resnet_bk import ConfigurableResNetBK, FULL_CONFIG
from src.utils import TrainingMetrics, MetricsLogger
from src.benchmarks.flops_counter import FLOPsCounter


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""
    model_name: str
    d_model: int
    n_layers: int
    n_seq: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    grad_clip: float
    device: str
    seed: int
    
    # Data configuration
    data_limit: int = 1_000_000_000  # 1B tokens (default for Pile benchmark)
    
    # Optimization flags
    use_analytic_gradient: bool = True
    use_koopman: bool = False
    use_physics_informed: bool = False
    use_quantization: bool = False
    use_pruning: bool = False
    use_mixed_precision: bool = False
    use_act: bool = False
    use_multi_scale: bool = False
    use_sparse_bk: bool = False
    use_early_exit: bool = False
    use_curriculum: bool = False
    use_active_learning: bool = False


@dataclass
class BenchmarkResults:
    """Results from a benchmark run."""
    model_name: str
    dataset_name: str
    config: Dict
    
    # Training metrics
    final_loss: float
    final_perplexity: float
    best_perplexity: float
    training_time: float  # seconds
    
    # Dataset info
    total_tokens: int
    vocab_size: int
    
    # Domain-specific perplexities (The Pile has 22 domains)
    domain_perplexities: Dict[str, float]
    
    # FLOPs metrics
    forward_flops: int
    backward_flops: int
    optimizer_flops: int
    total_flops_per_step: int
    total_training_flops: int
    
    # Memory metrics
    peak_memory_mb: float
    model_size_mb: float
    
    # Per-epoch metrics
    epoch_losses: List[float]
    epoch_perplexities: List[float]
    epoch_times: List[float]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def save_json(self, filepath: str):
        """Save results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Results saved to {filepath}")


def load_pile_data(batch_size, n_seq, data_limit=1_000_000_000, vocab_size_limit=32000):
    """
    Load The Pile dataset.
    
    The Pile is a 825 GiB diverse dataset containing 22 domains:
    - Pile-CC, PubMed Central, Books3, OpenWebText2, ArXiv, Github, FreeLaw,
      StackExchange, USPTO, PubMed Abstracts, Gutenberg, OpenSubtitles,
      Wikipedia, DM Mathematics, Ubuntu IRC, BookCorpus2, EuroParl, HackerNews,
      YoutubeSubtitles, PhilPapers, NIH ExPorter, Enron Emails
    
    Args:
        batch_size: batch size
        n_seq: sequence length
        data_limit: maximum number of tokens (default: 1B)
        vocab_size_limit: maximum vocabulary size
    
    Returns:
        train_data: (seq_len_total, batch_size) LongTensor
        vocab: dict with stoi, itos, vocab_size
        get_batch: function (source, i) -> (data, target)
    """
    print("Loading The Pile dataset...")
    print(f"Target: {data_limit:,} tokens (1 billion)")
    print("The Pile contains 22 diverse domains for comprehensive evaluation")
    
    try:
        # Load The Pile dataset (using 'all' configuration for all domains)
        # Note: The Pile is very large, so we use streaming mode
        dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
        print("Successfully loaded The Pile dataset (streaming mode)")
    except Exception as e:
        print(f"Failed to load The Pile: {e}")
        print("Attempting alternative: EleutherAI/pile subset...")
        try:
            # Try alternative source
            dataset = load_dataset("EleutherAI/pile", split="train", streaming=True)
            print("Successfully loaded The Pile from alternative source")
        except Exception as e2:
            print(f"Failed to load from alternative source: {e2}")
            print("Please check network connection or dataset availability")
            print("You may need to: pip install datasets")
            return None, None, None
    
    # Build vocabulary from first portion of data
    print("Building vocabulary...")
    counter = Counter()
    vocab_sample_size = min(20000, data_limit // 50000)  # Sample for vocab
    
    for i, example in enumerate(dataset):
        if i >= vocab_sample_size:
            break
        text = example['text']
        tokens = text.strip().split()
        if tokens:
            counter.update(tokens)
        
        if (i + 1) % 2000 == 0:
            print(f"  Processed {i+1}/{vocab_sample_size} examples for vocabulary...")
    
    special_tokens = ["<unk>", "<pad>"]
    stoi = {}
    itos = []
    
    for sp in special_tokens:
        stoi[sp] = len(itos)
        itos.append(sp)
    
    # Limit vocabulary size
    for tok, freq in counter.most_common(vocab_size_limit - len(special_tokens)):
        if tok not in stoi:
            stoi[tok] = len(itos)
            itos.append(tok)
    
    vocab_size = len(itos)
    unk_id = stoi["<unk>"]
    
    print(f"Vocabulary size: {vocab_size:,}")
    
    # Encode texts up to data_limit tokens
    print(f"Encoding texts (target: {data_limit:,} tokens)...")
    print("This may take a while for 1B tokens...")
    
    def encode_texts_streaming(dataset, max_tokens):
        ids = []
        total_tokens = 0
        num_examples = 0
        
        for example in dataset:
            text = example['text']
            for tok in text.strip().split():
                ids.append(stoi.get(tok, unk_id))
                total_tokens += 1
                
                if total_tokens >= max_tokens:
                    break
            
            num_examples += 1
            if num_examples % 5000 == 0:
                print(f"  Encoded {total_tokens:,} tokens from {num_examples:,} examples...")
            
            if total_tokens >= max_tokens:
                break
        
        return torch.tensor(ids, dtype=torch.long)
    
    train_ids = encode_texts_streaming(dataset, data_limit)
    
    actual_tokens = train_ids.numel()
    print(f"Total tokens encoded: {actual_tokens:,}")
    
    # Batchify
    def batchify(data, bsz):
        seq_len = data.size(0) // bsz
        data = data.narrow(0, 0, seq_len * bsz)
        data = data.view(bsz, seq_len).t().contiguous()  # (seq_len, batch_size)
        return data
    
    train_data = batchify(train_ids, batch_size)
    
    print(f"Batched data shape: {train_data.shape}")
    
    def get_batch(source, i):
        seq_len = min(n_seq, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].reshape(-1)
        return data, target
    
    vocab = {
        "stoi": stoi,
        "itos": itos,
        "vocab_size": vocab_size,
    }
    
    return train_data, vocab, get_batch



class PileBenchmark:
    """
    Comprehensive benchmark for The Pile dataset.
    """
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """
        Initialize benchmark.
        
        Args:
            output_dir: directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: Dict[str, BenchmarkResults] = {}
        
        # The Pile's 22 domains
        self.pile_domains = [
            'Pile-CC', 'PubMed Central', 'Books3', 'OpenWebText2', 'ArXiv',
            'Github', 'FreeLaw', 'StackExchange', 'USPTO', 'PubMed Abstracts',
            'Gutenberg', 'OpenSubtitles', 'Wikipedia', 'DM Mathematics',
            'Ubuntu IRC', 'BookCorpus2', 'EuroParl', 'HackerNews',
            'YoutubeSubtitles', 'PhilPapers', 'NIH ExPorter', 'Enron Emails'
        ]
    
    def run_benchmark(self, config: BenchmarkConfig) -> BenchmarkResults:
        """
        Run a single benchmark with given configuration.
        
        Args:
            config: benchmark configuration
        
        Returns:
            BenchmarkResults object
        """
        print("=" * 80)
        print(f"Running Benchmark: {config.model_name}")
        print("=" * 80)
        
        # Set random seed
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
        
        # Set device
        device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")
        
        # Load data
        train_data, vocab, get_batch = load_pile_data(
            batch_size=config.batch_size,
            n_seq=config.n_seq,
            data_limit=config.data_limit
        )
        
        if train_data is None:
            raise ValueError("Failed to load The Pile dataset")
        
        total_tokens = train_data.numel()
        print(f"Total training tokens: {total_tokens:,}")
        
        # Create model
        print("\nCreating model...")
        model = self._create_model(config, vocab['vocab_size'])
        model = model.to(device)
        
        # Print model summary
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,}")
        
        # Count FLOPs
        print("\nCounting FLOPs...")
        flops_counter = FLOPsCounter(model, config.batch_size, config.n_seq)
        flops_per_step = flops_counter.count_total_flops()
        flops_counter.print_summary()
        
        # Create optimizer and scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        
        criterion = nn.CrossEntropyLoss()
        
        num_steps_per_epoch = train_data.size(0) // config.n_seq
        num_total_steps = num_steps_per_epoch * config.epochs
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_total_steps,
            eta_min=config.lr / 10
        )
        
        print(f"\nTotal training steps: {num_total_steps:,}")
        print(f"Steps per epoch: {num_steps_per_epoch:,}")
        
        # Training loop
        print("\nStarting training...")
        model.train()
        
        epoch_losses = []
        epoch_perplexities = []
        epoch_times = []
        best_perplexity = float('inf')
        
        # Track memory
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        
        training_start = time.time()
        
        for epoch in range(1, config.epochs + 1):
            epoch_start = time.time()
            total_loss = 0.0
            num_batches = 0
            
            # Progress tracking
            print_interval = max(1, num_steps_per_epoch // 20)
            
            for i in range(0, train_data.size(0) - 1, config.n_seq):
                x_batch, y_batch = get_batch(train_data, i)
                x_batch = x_batch.t().contiguous()
                
                if x_batch.size(1) != config.n_seq:
                    continue
                
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                
                if config.use_mixed_precision and device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        logits = model(x_batch)
                        loss = criterion(logits.view(-1, logits.size(-1)), y_batch)
                else:
                    logits = model(x_batch)
                    loss = criterion(logits.view(-1, logits.size(-1)), y_batch)
                
                # Skip if loss is NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN/Inf loss at epoch {epoch}, batch {num_batches}, skipping")
                    continue
                
                # Backward pass
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.grad_clip
                )
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Print progress
                if num_batches % print_interval == 0:
                    current_loss = total_loss / num_batches
                    current_ppl = math.exp(min(current_loss, 20))
                    elapsed = time.time() - epoch_start
                    tokens_per_sec = (num_batches * config.batch_size * config.n_seq) / elapsed
                    print(f"  Epoch {epoch}, Batch {num_batches}/{num_steps_per_epoch}: "
                          f"Loss={current_loss:.4f}, PPL={current_ppl:.2f}, "
                          f"Time={elapsed:.1f}s, Tokens/s={tokens_per_sec:.0f}")
            
            # Epoch summary
            epoch_time = time.time() - epoch_start
            avg_loss = total_loss / max(1, num_batches)
            perplexity = math.exp(min(avg_loss, 20))
            
            epoch_losses.append(avg_loss)
            epoch_perplexities.append(perplexity)
            epoch_times.append(epoch_time)
            
            if perplexity < best_perplexity:
                best_perplexity = perplexity
            
            print(f"\nEpoch {epoch}/{config.epochs} Complete: "
                  f"Loss={avg_loss:.4f}, PPL={perplexity:.2f}, "
                  f"Time={epoch_time:.1f}s ({epoch_time/60:.1f}min), "
                  f"Best PPL={best_perplexity:.2f}\n")
        
        training_time = time.time() - training_start
        
        # Get memory stats
        if device.type == 'cuda':
            peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        else:
            peak_memory_mb = 0.0
        
        # Calculate model size
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        
        # Calculate total training FLOPs
        total_training_flops = flops_per_step.total * num_total_steps
        
        # Measure domain-specific perplexities
        print("\nMeasuring domain-specific perplexities...")
        domain_perplexities = self._measure_domain_perplexities(model, device, vocab, config)
        
        # Create results
        results = BenchmarkResults(
            model_name=config.model_name,
            dataset_name='pile',
            config=asdict(config),
            final_loss=epoch_losses[-1],
            final_perplexity=epoch_perplexities[-1],
            best_perplexity=best_perplexity,
            training_time=training_time,
            total_tokens=total_tokens,
            vocab_size=vocab['vocab_size'],
            domain_perplexities=domain_perplexities,
            forward_flops=flops_per_step.forward,
            backward_flops=flops_per_step.backward,
            optimizer_flops=flops_per_step.optimizer,
            total_flops_per_step=flops_per_step.total,
            total_training_flops=total_training_flops,
            peak_memory_mb=peak_memory_mb,
            model_size_mb=model_size_mb,
            epoch_losses=epoch_losses,
            epoch_perplexities=epoch_perplexities,
            epoch_times=epoch_times,
        )
        
        # Save results
        results_file = self.output_dir / f"{config.model_name}_pile_results.json"
        results.save_json(str(results_file))
        
        self.results[config.model_name] = results
        
        print("\n" + "=" * 80)
        print(f"Benchmark Complete: {config.model_name}")
        print(f"Dataset: The Pile (22 diverse domains)")
        print(f"Total Tokens: {total_tokens:,}")
        print(f"Final Perplexity: {results.final_perplexity:.2f}")
        print(f"Best Perplexity: {results.best_perplexity:.2f}")
        print(f"Training Time: {results.training_time:.1f}s ({results.training_time/60:.1f}min)")
        print(f"Total FLOPs: {results.total_training_flops/1e12:.2f} TFLOPs")
        print("\nDomain-Specific Perplexities:")
        for domain, ppl in sorted(domain_perplexities.items(), key=lambda x: x[1]):
            print(f"  {domain}: {ppl:.2f}")
        print("=" * 80 + "\n")
        
        return results

    
    def _measure_domain_perplexities(self, model, device, vocab, config):
        """
        Measure perplexity across different domains in The Pile.
        
        The Pile contains 22 distinct domains. We'll sample from each domain
        to measure domain-specific performance.
        
        Args:
            model: trained model
            device: torch device
            vocab: vocabulary dict
            config: benchmark configuration
        
        Returns:
            dict mapping domain names to perplexities
        """
        model.eval()
        domain_perplexities = {}
        
        # For The Pile, we'll measure perplexity on validation set samples from each domain
        try:
            # Load validation set
            val_dataset = load_dataset("monology/pile-uncopyrighted", split="validation", streaming=True)
            
            stoi = vocab['stoi']
            unk_id = stoi["<unk>"]
            criterion = nn.CrossEntropyLoss()
            
            # Sample domains (we'll evaluate on a subset due to computational constraints)
            # Focus on most representative domains
            eval_domains = [
                'Pile-CC',           # Common Crawl (web text)
                'PubMed Central',    # Scientific papers
                'Books3',            # Books
                'OpenWebText2',      # Reddit links
                'ArXiv',             # Academic papers
                'Github',            # Code
                'StackExchange',     # Q&A
                'Wikipedia',         # Encyclopedia
            ]
            
            print(f"  Evaluating on {len(eval_domains)} representative domains...")
            
            # Collect examples by domain
            domain_examples = {domain: [] for domain in eval_domains}
            
            for i, example in enumerate(val_dataset):
                if i >= 10000:  # Limit total examples
                    break
                
                text = example['text']
                meta = example.get('meta', {})
                pile_set_name = meta.get('pile_set_name', 'unknown')
                
                # Map to our eval domains
                if pile_set_name in eval_domains:
                    domain_examples[pile_set_name].append(text)
                
                if (i + 1) % 1000 == 0:
                    print(f"    Collected {i+1} validation examples...")
            
            # Evaluate each domain
            for domain_name in eval_domains:
                if not domain_examples[domain_name]:
                    print(f"    Warning: No examples for {domain_name}, skipping")
                    continue
                
                print(f"  Evaluating on {domain_name} domain...")
                
                # Encode domain examples
                domain_ids = []
                for text in domain_examples[domain_name][:100]:  # Limit per domain
                    for tok in text.strip().split():
                        domain_ids.append(stoi.get(tok, unk_id))
                    
                    # Limit to reasonable size
                    if len(domain_ids) >= 50000:  # 50K tokens per domain
                        break
                
                if len(domain_ids) < config.n_seq:
                    print(f"    Warning: Not enough data for {domain_name}, skipping")
                    continue
                
                domain_data = torch.tensor(domain_ids, dtype=torch.long)
                
                # Evaluate
                total_loss = 0.0
                num_batches = 0
                
                with torch.no_grad():
                    for i in range(0, len(domain_data) - config.n_seq, config.n_seq):
                        x = domain_data[i:i+config.n_seq].unsqueeze(0).to(device)
                        y = domain_data[i+1:i+1+config.n_seq].to(device)
                        
                        if x.size(1) != config.n_seq:
                            continue
                        
                        logits = model(x)
                        loss = criterion(logits.view(-1, logits.size(-1)), y)
                        
                        if not torch.isnan(loss) and not torch.isinf(loss):
                            total_loss += loss.item()
                            num_batches += 1
                
                if num_batches > 0:
                    avg_loss = total_loss / num_batches
                    perplexity = math.exp(min(avg_loss, 20))
                    domain_perplexities[domain_name] = perplexity
                    print(f"    {domain_name}: PPL={perplexity:.2f}")
                else:
                    print(f"    Warning: No valid batches for {domain_name}")
        
        except Exception as e:
            print(f"  Warning: Could not measure domain perplexities: {e}")
            print(f"  Using overall perplexity as fallback")
            # Fallback to overall perplexity
            domain_perplexities = {
                'overall': self.results[list(self.results.keys())[-1]].final_perplexity if self.results else 0.0
            }
        
        model.train()
        return domain_perplexities
    
    def _create_model(self, config: BenchmarkConfig, vocab_size: int) -> nn.Module:
        """
        Create model based on configuration.
        
        Args:
            config: benchmark configuration
            vocab_size: vocabulary size
        
        Returns:
            model instance
        """
        if config.model_name.startswith('transformer'):
            # Create Transformer baseline
            return self._create_transformer_baseline(config, vocab_size)
        else:
            # Create ResNet-BK model using dataclasses.replace
            model_config = replace(
                FULL_CONFIG,
                vocab_size=vocab_size,
                d_model=config.d_model,
                n_layers=config.n_layers,
                n_seq=config.n_seq,
                use_analytic_gradient=config.use_analytic_gradient,
                use_koopman=config.use_koopman,
                use_physics_informed=config.use_physics_informed,
                use_quantization=config.use_quantization,
                use_pruning=config.use_pruning,
                use_mixed_precision=config.use_mixed_precision,
                use_adaptive_computation=config.use_act,
                use_multi_scale=config.use_multi_scale,
                use_learned_sparsity=config.use_sparse_bk,
                use_curriculum_learning=config.use_curriculum,
                use_active_learning=config.use_active_learning,
            )
            
            return ConfigurableResNetBK(model_config)
    
    def _create_transformer_baseline(self, config: BenchmarkConfig, vocab_size: int) -> nn.Module:
        """
        Create Transformer baseline for comparison.
        
        Args:
            config: benchmark configuration
            vocab_size: vocabulary size
        
        Returns:
            Transformer model
        """
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        
        class TransformerLM(nn.Module):
            def __init__(self, vocab_size, d_model, n_layers, n_seq, nhead=4):
                super().__init__()
                self.d_model = d_model
                self.n_seq = n_seq
                
                self.token_embedding = nn.Embedding(vocab_size, d_model)
                self.position_embedding = nn.Embedding(n_seq, d_model)
                
                encoder_layer = TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_model * 4,
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer = TransformerEncoder(encoder_layer, num_layers=n_layers)
                
                self.lm_head = nn.Linear(d_model, vocab_size)
            
            def forward(self, x):
                B, N = x.shape
                
                # Embeddings
                token_emb = self.token_embedding(x)
                pos_ids = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
                pos_emb = self.position_embedding(pos_ids)
                
                h = token_emb + pos_emb
                
                # Transformer
                # Create causal mask
                mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
                h = self.transformer(h, mask=mask, is_causal=True)
                
                # LM head
                logits = self.lm_head(h)
                
                return logits
        
        return TransformerLM(vocab_size, config.d_model, config.n_layers, config.n_seq)

    
    def compare_to_other_datasets(self, 
                                   wikitext2_results_path: str = None,
                                   wikitext103_results_path: str = None,
                                   penn_treebank_results_path: str = None,
                                   c4_results_path: str = None):
        """
        Compare The Pile results to other dataset results.
        
        Args:
            wikitext2_results_path: path to WikiText-2 results JSON
            wikitext103_results_path: path to WikiText-103 results JSON
            penn_treebank_results_path: path to Penn Treebank results JSON
            c4_results_path: path to C4 results JSON
        """
        print("\n" + "=" * 80)
        print("The Pile Cross-Dataset Comparison")
        print("=" * 80)
        
        # Load comparison results
        comparison_data = {}
        
        if wikitext2_results_path and Path(wikitext2_results_path).exists():
            with open(wikitext2_results_path, 'r') as f:
                comparison_data['wikitext2'] = json.load(f)
        
        if wikitext103_results_path and Path(wikitext103_results_path).exists():
            with open(wikitext103_results_path, 'r') as f:
                comparison_data['wikitext103'] = json.load(f)
        
        if penn_treebank_results_path and Path(penn_treebank_results_path).exists():
            with open(penn_treebank_results_path, 'r') as f:
                comparison_data['penn_treebank'] = json.load(f)
        
        if c4_results_path and Path(c4_results_path).exists():
            with open(c4_results_path, 'r') as f:
                comparison_data['c4'] = json.load(f)
        
        for model_name, pile_results in self.results.items():
            print(f"\nModel: {model_name}")
            print("-" * 80)
            
            # Dataset comparison table
            print(f"\n{'Dataset':<20} {'Tokens':>15} {'Vocab':>10} {'PPL':>10} {'Time (min)':>12}")
            print("-" * 80)
            
            # The Pile
            print(f"{'The Pile':<20} {pile_results.total_tokens:>15,} "
                  f"{pile_results.vocab_size:>10,} {pile_results.final_perplexity:>10.2f} "
                  f"{pile_results.training_time/60:>12.1f}")
            
            # C4
            if 'c4' in comparison_data:
                c4 = comparison_data['c4']
                c4_tokens = c4.get('total_tokens', 0)
                c4_vocab = c4.get('vocab_size', 0)
                c4_ppl = c4.get('final_perplexity', 0)
                c4_time = c4.get('training_time', 0)
                print(f"{'C4':<20} {c4_tokens:>15,} {c4_vocab:>10,} "
                      f"{c4_ppl:>10.2f} {c4_time/60:>12.1f}")
            
            # WikiText-103
            if 'wikitext103' in comparison_data:
                wt103 = comparison_data['wikitext103']
                wt103_tokens = wt103.get('total_tokens', 0)
                wt103_vocab = wt103.get('vocab_size', 0)
                wt103_ppl = wt103.get('final_perplexity', 0)
                wt103_time = wt103.get('training_time', 0)
                print(f"{'WikiText-103':<20} {wt103_tokens:>15,} {wt103_vocab:>10,} "
                      f"{wt103_ppl:>10.2f} {wt103_time/60:>12.1f}")
            
            # WikiText-2
            if 'wikitext2' in comparison_data:
                wt2 = comparison_data['wikitext2']
                wt2_tokens = wt2.get('total_tokens', 0)
                wt2_vocab = wt2.get('vocab_size', 0)
                wt2_ppl = wt2.get('final_perplexity', 0)
                wt2_time = wt2.get('training_time', 0)
                print(f"{'WikiText-2':<20} {wt2_tokens:>15,} {wt2_vocab:>10,} "
                      f"{wt2_ppl:>10.2f} {wt2_time/60:>12.1f}")
            
            # Penn Treebank
            if 'penn_treebank' in comparison_data:
                ptb = comparison_data['penn_treebank']
                ptb_tokens = ptb.get('total_tokens', 0)
                ptb_vocab = ptb.get('vocab_size', 0)
                ptb_ppl = ptb.get('final_perplexity', 0)
                ptb_time = ptb.get('training_time', 0)
                print(f"{'Penn Treebank':<20} {ptb_tokens:>15,} {ptb_vocab:>10,} "
                      f"{ptb_ppl:>10.2f} {ptb_time/60:>12.1f}")
            
            # Dataset characteristics
            print(f"\nDataset Characteristics:")
            print(f"  The Pile: 22 diverse domains (825 GiB total)")
            print(f"    - Academic: ArXiv, PubMed, PhilPapers")
            print(f"    - Code: Github, StackExchange")
            print(f"    - Books: Books3, Gutenberg, BookCorpus2")
            print(f"    - Web: Pile-CC, OpenWebText2, HackerNews")
            print(f"    - Legal: FreeLaw, USPTO")
            print(f"    - Conversational: Ubuntu IRC, OpenSubtitles")
            print(f"    - And more...")
            print(f"  C4: Web-crawled text (diverse but less structured)")
            print(f"  WikiText: Wikipedia articles (encyclopedic, formal)")
            print(f"  Penn Treebank: Financial news (Wall Street Journal)")
            print(f"\n  The Pile is the most comprehensive and diverse dataset")
            print(f"  Performance on The Pile demonstrates true generalization capability")
            
            # Scale comparison
            print(f"\nScale Comparison:")
            print(f"  The Pile tokens: {pile_results.total_tokens:,}")
            if 'c4' in comparison_data:
                c4_tokens = comparison_data['c4'].get('total_tokens', 1)
                scale_ratio = pile_results.total_tokens / c4_tokens
                print(f"  The Pile is {scale_ratio:.1f}× larger than C4 subset")
            if 'wikitext103' in comparison_data:
                wt103_tokens = comparison_data['wikitext103'].get('total_tokens', 1)
                scale_ratio = pile_results.total_tokens / wt103_tokens
                print(f"  The Pile is {scale_ratio:.1f}× larger than WikiText-103")
            if 'wikitext2' in comparison_data:
                wt2_tokens = comparison_data['wikitext2'].get('total_tokens', 1)
                scale_ratio = pile_results.total_tokens / wt2_tokens
                print(f"  The Pile is {scale_ratio:.1f}× larger than WikiText-2")
        
        print("\n" + "=" * 80 + "\n")
        
        # Save comparison
        comparison_summary = {
            'pile': {name: asdict(res) for name, res in self.results.items()},
            'c4': comparison_data.get('c4'),
            'wikitext103': comparison_data.get('wikitext103'),
            'wikitext2': comparison_data.get('wikitext2'),
            'penn_treebank': comparison_data.get('penn_treebank'),
        }
        
        comparison_file = self.output_dir / "pile_cross_dataset_comparison.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison_summary, f, indent=2)
        print(f"Cross-dataset comparison saved to {comparison_file}")
    
    def plot_training_curves(self, model_names: Optional[List[str]] = None):
        """
        Plot training curves for comparison.
        
        Args:
            model_names: list of model names to plot (None = all)
        """
        if not HAS_MATPLOTLIB:
            print("Warning: matplotlib not available, skipping plot generation")
            return
        
        if model_names is None:
            model_names = list(self.results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss curves
        ax = axes[0, 0]
        for name in model_names:
            if name in self.results:
                results = self.results[name]
                epochs = range(1, len(results.epoch_losses) + 1)
                ax.plot(epochs, results.epoch_losses, marker='o', label=name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss (The Pile)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Perplexity curves
        ax = axes[0, 1]
        for name in model_names:
            if name in self.results:
                results = self.results[name]
                epochs = range(1, len(results.epoch_perplexities) + 1)
                ax.plot(epochs, results.epoch_perplexities, marker='o', label=name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Perplexity')
        ax.set_title('Perplexity (The Pile)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Time per epoch
        ax = axes[1, 0]
        for name in model_names:
            if name in self.results:
                results = self.results[name]
                epochs = range(1, len(results.epoch_times) + 1)
                ax.plot(epochs, results.epoch_times, marker='o', label=name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Time per Epoch (The Pile)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Domain perplexities (bar chart)
        ax = axes[1, 1]
        for name in model_names:
            if name in self.results:
                results = self.results[name]
                domains = list(results.domain_perplexities.keys())
                ppls = list(results.domain_perplexities.values())
                x_pos = np.arange(len(domains))
                width = 0.35 if len(model_names) > 1 else 0.7
                offset = (list(model_names).index(name) - len(model_names)/2 + 0.5) * width
                ax.bar(x_pos + offset, ppls, width, label=name, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(domains, rotation=45, ha='right')
        ax.set_ylabel('Perplexity')
        ax.set_title('Domain-Specific Perplexities (The Pile)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        plot_file = self.output_dir / "pile_training_curves.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {plot_file}")
        plt.close()



def main():
    """Run comprehensive The Pile benchmark."""
    print("The Pile Comprehensive Benchmark")
    print("=" * 80)
    print("The Pile: 825 GiB diverse dataset with 22 domains")
    print("Training on 1B token subset to test scalability and domain generalization")
    print("=" * 80 + "\n")
    
    # Create benchmark
    benchmark = PileBenchmark(output_dir="benchmark_results/pile")
    
    # Common configuration
    common_config = {
        'd_model': 64,
        'n_layers': 4,
        'n_seq': 128,
        'batch_size': 32,
        'epochs': 2,  # Fewer epochs due to very large dataset
        'lr': 1e-3,
        'weight_decay': 0.01,
        'grad_clip': 0.5,
        'device': 'cuda',
        'seed': 42,
        'data_limit': 1_000_000_000,  # 1B tokens
    }
    
    # 1. ResNet-BK Baseline (no optimizations)
    print("\n[1/2] Running ResNet-BK Baseline on The Pile (1B tokens)...")
    resnet_baseline_config = BenchmarkConfig(
        model_name='resnet_bk_baseline',
        **common_config,
        use_analytic_gradient=False,
    )
    benchmark.run_benchmark(resnet_baseline_config)
    
    # 2. ResNet-BK with All Optimizations
    print("\n[2/2] Running ResNet-BK with All Optimizations on The Pile (1B tokens)...")
    resnet_full_config = BenchmarkConfig(
        model_name='resnet_bk_full',
        **common_config,
        use_analytic_gradient=True,
        use_mixed_precision=True,
        use_act=True,
        use_multi_scale=True,
        use_sparse_bk=True,
    )
    benchmark.run_benchmark(resnet_full_config)
    
    # Compare to other datasets if available
    print("\n" + "=" * 80)
    print("CROSS-DATASET COMPARISON")
    print("=" * 80)
    
    wt2_path = "benchmark_results/wikitext2/resnet_bk_full_results.json"
    wt103_path = "benchmark_results/wikitext103/resnet_bk_full_wikitext103_results.json"
    ptb_path = "benchmark_results/penn_treebank/resnet_bk_full_penn_treebank_results.json"
    c4_path = "benchmark_results/c4/resnet_bk_full_c4_results.json"
    
    benchmark.compare_to_other_datasets(
        wikitext2_results_path=wt2_path if Path(wt2_path).exists() else None,
        wikitext103_results_path=wt103_path if Path(wt103_path).exists() else None,
        penn_treebank_results_path=ptb_path if Path(ptb_path).exists() else None,
        c4_results_path=c4_path if Path(c4_path).exists() else None
    )
    
    # Plot training curves
    benchmark.plot_training_curves()
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {benchmark.output_dir}")
    print("\nKey Findings:")
    for name, results in benchmark.results.items():
        print(f"  {name}:")
        print(f"    - Final Perplexity: {results.final_perplexity:.2f}")
        print(f"    - Training Time: {results.training_time/60:.1f} minutes")
        print(f"    - Total Tokens: {results.total_tokens:,}")
        print(f"    - Domain Perplexities:")
        for domain, ppl in sorted(results.domain_perplexities.items(), key=lambda x: x[1])[:5]:
            print(f"      - {domain}: {ppl:.2f}")
        if len(results.domain_perplexities) > 5:
            print(f"      - ... and {len(results.domain_perplexities) - 5} more domains")
    
    print("\nDataset Analysis:")
    print("  The Pile represents the most comprehensive and diverse language dataset")
    print("  22 domains covering academic, code, books, web, legal, conversational text")
    print("  Performance on The Pile demonstrates true generalization capability")
    print("  Domain-specific perplexities reveal model strengths and weaknesses")
    print("  Higher perplexity on The Pile vs other datasets is expected due to diversity")
    print("\nSignificance:")
    print("  Training on 1B tokens from The Pile validates scalability")
    print("  Domain-specific evaluation shows where model excels and struggles")
    print("  This benchmark completes the comprehensive dataset evaluation suite")


if __name__ == '__main__':
    main()
