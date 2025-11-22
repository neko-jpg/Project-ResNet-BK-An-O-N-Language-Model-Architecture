import argparse
import json
import time
import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.phase4.integrated_model import Phase4IntegratedModel
from src.models.phase3.config import Phase3Config
from src.models.phase3.integrated_model import Phase3IntegratedModel

# Define a Phase4 config that extends Phase3
@dataclass
class Phase4Config(Phase3Config):
    # Add Phase 4 specific config
    n_candidates: int = 3
    bulk_dim: int = 5
    ads_radius: float = 1.0
    enable_emotion: bool = True
    enable_dream: bool = True
    enable_holographic: bool = True
    enable_quantum: bool = True
    enable_topological: bool = True
    enable_ethics: bool = True

def measure_perplexity(model: nn.Module, test_data_path: str, batch_size: int = 4, seq_len: int = 128, device: str = "cpu") -> float:
    """
    Measures Perplexity on the test set.
    """
    print(f"Loading test data from {test_data_path}...")
    data = torch.load(test_data_path)
    input_ids_list = data['input_ids'] # List of lists

    model.eval()
    total_nll = 0.0
    total_tokens = 0

    print(f"Running inference on {len(input_ids_list)} documents...")

    # Limit for benchmark speed
    input_ids_list = input_ids_list[:100]

    with torch.no_grad():
        for i in range(0, len(input_ids_list), batch_size):
            batch_docs = input_ids_list[i:i+batch_size]

            # Pad and collate
            max_len = min(max(len(d) for d in batch_docs), seq_len)
            batch_tensor = torch.zeros(len(batch_docs), max_len, dtype=torch.long).to(device)

            valid_batch = False
            for j, doc in enumerate(batch_docs):
                l = min(len(doc), max_len)
                if l > 1:
                    batch_tensor[j, :l] = torch.tensor(doc[:l], dtype=torch.long)
                    valid_batch = True

            if not valid_batch:
                continue

            # Forward pass
            input_ids = batch_tensor[:, :-1]
            targets = batch_tensor[:, 1:]

            outputs = model(input_ids)
            logits = outputs['logits'] # (B, L, V)

            # Compute loss
            # Flatten
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction='sum', ignore_index=0)

            total_nll += loss.item()
            total_tokens += (targets != 0).sum().item()

            if i % 20 == 0:
                print(f"Processed {i}/{len(input_ids_list)} docs...")

    if total_tokens == 0:
        return float('inf')

    ppl = torch.exp(torch.tensor(total_nll / total_tokens)).item()
    return ppl

def measure_throughput(model: nn.Module, batch_size: int = 1, seq_len: int = 128, device: str = "cpu") -> float:
    """
    Measures throughput in tokens/sec.
    """
    model.eval()
    dummy_input = torch.randint(0, 1000, (batch_size, seq_len)).to(device)

    # Warmup
    with torch.no_grad():
        model(dummy_input)

    start_time = time.time()
    num_batches = 10
    total_tokens = batch_size * seq_len * num_batches

    with torch.no_grad():
        for _ in range(num_batches):
            model(dummy_input)

    end_time = time.time()
    duration = end_time - start_time
    throughput = total_tokens / duration

    return throughput

def run_benchmark(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running benchmark on {device}...")

    # 1. Initialize Model
    vocab_size = 50257
    d_model = 64 # Small for CPU testing

    config_p3 = Phase3Config(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=2,
        max_seq_len=1024,
        d_koopman=d_model*2
    )

    phase3_model = Phase3IntegratedModel(config_p3)

    model = Phase4IntegratedModel(
        phase3_model=phase3_model,
        enable_emotion=True,
        enable_dream=True,
        enable_holographic=True,
        enable_quantum=True,
        enable_topological=True
    ).to(device)

    print("Model initialized.")

    metrics = {}

    # 2. Measure Throughput
    print("Measuring throughput...")
    throughput = measure_throughput(model, device=device, seq_len=128)
    metrics['throughput_tokens_per_sec'] = throughput
    print(f"Throughput: {throughput:.2f} tokens/sec")

    # 3. Measure Perplexity (if data available)
    data_path = Path("data/wikitext2/test.pt")
    if data_path.exists():
        print("Measuring perplexity on WikiText-2 (subset)...")
        ppl = measure_perplexity(model, str(data_path), device=device)
        metrics['perplexity_wikitext2'] = ppl
        print(f"Perplexity: {ppl:.2f}")
    else:
        print("WikiText-2 not found, skipping perplexity.")
        metrics['perplexity_wikitext2'] = None

    # 5. Save Results
    output_dir = Path("results/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "phase4_evaluation_metrics.json"

    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    run_benchmark(args)
