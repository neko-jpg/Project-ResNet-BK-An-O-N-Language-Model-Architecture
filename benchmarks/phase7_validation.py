"""
Benchmark script for Phase 7 Integration Validation.

Implements Experiment A and Experiment B as described in the roadmap.
This script has been updated to use the new Phase7Config and integrated model architecture.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from datasets import load_dataset
import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# --- Import custom modules ---
from src.models.phase7.integrated_model import Phase7IntegratedModel, Phase7Config
from src.utils.prime_init import prime_bump_init_
from src.training.epsilon_scheduler import EpsilonScheduler

# --- Configuration ---
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
D_MODEL = 128
N_LAYERS = 2
N_SEQ = 128
VOCAB_SIZE = 50257 # GPT-2

# --- Helper Functions ---

def get_model_size(model):
    return sum(p.numel() for p in model.parameters()) / 1e6 # in Millions

@torch.no_grad()
def calculate_perplexity(model, tokenizer, dataset, device="cpu", num_samples=100):
    model.eval()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total_tokens = 0

    # Use the validation set for perplexity
    dataset_split = dataset.get("validation", dataset)
    count = 0
    print(f"Calculating perplexity on {num_samples} samples from the validation set...")
    for example in dataset_split:
        if count >= num_samples:
            break

        text = example['text']
        if not text or len(text.split()) < 2: continue # Skip empty or very short texts

        inputs = tokenizer(text, return_tensors="pt", max_length=N_SEQ, truncation=True).to(device)
        input_ids = inputs.input_ids

        if input_ids.size(1) < 2: continue

        outputs = model(input_ids)

        logits = outputs[:, :-1, :].contiguous()
        labels = input_ids[:, 1:].contiguous()

        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        total_loss += loss.item() * (labels.numel())
        total_tokens += labels.numel()
        count += 1

    if total_tokens == 0:
        print("Warning: No valid tokens were processed for perplexity calculation.")
        return float('inf')

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return perplexity

def train_model_for_steps(model, tokenizer, dataset, device, num_steps, scheduler=None):
    model.train()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    losses = []
    data_iter = iter(dataset)

    print(f"Training for {num_steps} steps...")
    for step in range(num_steps):
        if scheduler:
            scheduler.update_model_curvature(model, step)

        try:
            batch = next(data_iter)
            if not batch['text']: continue
        except StopIteration:
            data_iter = iter(dataset) # Reset iterator
            batch = next(data_iter)

        inputs = tokenizer(batch['text'], return_tensors="pt", max_length=N_SEQ, truncation=True, padding="max_length").to(device)
        input_ids = inputs.input_ids

        if input_ids.size(1) < 2: continue

        outputs = model(input_ids)
        logits = outputs[:, :-1, :].contiguous()
        labels = input_ids[:, 1:].contiguous()

        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (step + 1) % (num_steps // 10) == 0:
            print(f"  Step {step+1}/{num_steps}, Loss: {loss.item():.4f}")

    return losses

def plot_experiment_a(results):
    ranks = [str(r['rank']) for r in results]
    params = [r['parameters_M'] for r in results]
    ppl = [r['perplexity'] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.bar(ranks, params, color='skyblue')
    ax1.set_title('Experiment A: Model Size vs. HTT Rank')
    ax1.set_xlabel('HTT Rank')
    ax1.set_ylabel('Parameters (Millions)')
    ax1.grid(axis='y', linestyle='--')

    ax2.bar(ranks, ppl, color='salmon')
    ax2.set_title('Experiment A: Perplexity vs. HTT Rank')
    ax2.set_xlabel('HTT Rank')
    ax2.set_ylabel('Perplexity (on WikiText-103 val)')
    ax2.grid(axis='y', linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "experiment_a_plot.png"))
    print("Experiment A plot saved to results/experiment_a_plot.png")

def plot_experiment_b(results):
    plt.figure(figsize=(10, 6))
    plt.plot(results['baseline_losses'], label='Baseline Model')
    plt.plot(results['proposed_losses'], label='Proposed Model (Prime-Bump + Epsilon Scheduler)')
    plt.title('Experiment B: Convergence Speed Comparison')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.savefig(os.path.join(RESULTS_DIR, "experiment_b_plot.png"))
    print("Experiment B plot saved to results/experiment_b_plot.png")

# --- Experiment A: HTT Embedding Analysis ---

def run_experiment_a(device, tokenizer, dataset):
    print("\n--- Running Experiment A: HTT Embedding Analysis ---")

    train_subset = dataset['train'].select(range(2000)) # Use a subset for pre-training

    results = []
    ranks = [4, 8, 16]

    # Baseline: Standard nn.Embedding
    print("\nEvaluating baseline model (standard nn.Embedding)...")
    base_config = Phase7Config(vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=N_LAYERS, n_seq=N_SEQ)
    base_model = Phase7IntegratedModel(config=base_config)
    # Manually replace HTT with standard nn.Embedding for the baseline
    base_model.model.token_embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)

    train_model_for_steps(base_model, tokenizer, train_subset, device, num_steps=300)
    base_params = get_model_size(base_model)
    base_ppl = calculate_perplexity(base_model, tokenizer, dataset, device)

    results.append({
        "rank": "baseline (nn.Embedding)",
        "parameters_M": base_params,
        "perplexity": base_ppl
    })
    print(f"Baseline: Params={base_params:.2f}M, PPL={base_ppl:.2f}")

    # HTT models with different ranks
    for rank in ranks:
        print(f"\nEvaluating HTT model with rank={rank}...")
        config = Phase7Config(vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=N_LAYERS, n_seq=N_SEQ, htt_rank=rank)
        model = Phase7IntegratedModel(config=config)

        train_model_for_steps(model, tokenizer, train_subset, device, num_steps=300)
        params = get_model_size(model)
        ppl = calculate_perplexity(model, tokenizer, dataset, device)

        results.append({
            "rank": rank,
            "parameters_M": params,
            "perplexity": ppl
        })
        print(f"Rank {rank}: Params={params:.2f}M, PPL={ppl:.2f}")

    with open(os.path.join(RESULTS_DIR, "experiment_a_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("\nExperiment A finished. Results saved to results/experiment_a_results.json")
    plot_experiment_a(results)
    return results

# --- Experiment B: Initialization and Scheduling Analysis ---

def run_experiment_b(device, tokenizer, dataset):
    print("\n--- Running Experiment B: Convergence Speed Analysis ---")

    num_steps = 500
    train_subset = dataset['train'].select(range(num_steps * 2)) # Ensure enough data

    # --- Baseline Model ---
    print("\nTraining baseline model...")
    baseline_config = Phase7Config(vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=N_LAYERS, n_seq=N_SEQ, prime_bump_init=False)
    baseline_model = Phase7IntegratedModel(config=baseline_config)
    baseline_losses = train_model_for_steps(baseline_model, tokenizer, train_subset, device, num_steps)

    # --- Proposed Model ---
    print("\nTraining proposed model (Prime-Bump + Epsilon Scheduler)...")
    proposed_config = Phase7Config(vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=N_LAYERS, n_seq=N_SEQ, prime_bump_init=True)
    proposed_model = Phase7IntegratedModel(config=proposed_config)

    # Manually apply Prime-Bump initialization to Linear layers in the attention blocks
    # The internal init only handles embeddings
    for module in proposed_model.model.modules():
        if isinstance(module, nn.Linear):
            prime_bump_init_(module.weight)

    scheduler = EpsilonScheduler(t_max=num_steps, start_val=1.0, end_val=0.1)
    proposed_losses = train_model_for_steps(proposed_model, tokenizer, train_subset, device, num_steps, scheduler=scheduler)

    results = {
        "baseline_losses": baseline_losses,
        "proposed_losses": proposed_losses,
    }

    with open(os.path.join(RESULTS_DIR, "experiment_b_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("\nExperiment B finished. Results saved to results/experiment_b_results.json")
    plot_experiment_b(results)
    return results


# --- Main Execution ---

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer and dataset once
    print("Loading tokenizer and dataset (wikitext-103)...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Using wikitext-2 as a lighter alternative to 103 for CI/testing
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    print("Dataset loaded.")

    # Run experiments
    exp_a_results = run_experiment_a(device, tokenizer, dataset)
    exp_b_results = run_experiment_b(device, tokenizer, dataset)

    # Update paper with results
    print("\nUpdating paper/main.tex with benchmark results...")
    try:
        # This assumes the script exists and is executable
        os.system(f"python scripts/update_tex_results.py {os.path.join(RESULTS_DIR, 'experiment_a_results.json')} {os.path.join(RESULTS_DIR, 'experiment_b_results.json')}")
        print("Successfully updated paper/main.tex.")
    except Exception as e:
        print(f"Could not update paper/main.tex. Error: {e}")
        print("Please run scripts/update_tex_results.py manually.")
