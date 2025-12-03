import torch
import torch.nn as nn
import time
import argparse
import json
import os
import psutil
from src.models.phase8.config import Phase8Config
from src.models.phase8.integrated_model import Phase8IntegratedModel
from src.models.phase7.integrated_model import Phase7IntegratedModel, Phase7Config

def get_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size_mb(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / (1024 * 1024)

def benchmark_throughput(model, config, batch_size=4, seq_len=128, num_batches=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(2):
            _ = model(input_ids)
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_batches):
            _ = model(input_ids)
    end_time = time.time()
    
    total_tokens = batch_size * seq_len * num_batches
    duration = end_time - start_time
    throughput = total_tokens / duration
    
    return throughput

def run_sanity_check(model, config):
    print("Running sanity check (convergence test)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    # ResNetBK requires exact sequence length match
    seq_len = getattr(config, 'n_seq', 128)
    input_ids = torch.randint(0, config.vocab_size, (2, seq_len)).to(device)
    targets = torch.randint(0, config.vocab_size, (2, seq_len)).to(device)
    
    initial_loss = 0
    final_loss = 0
    
    for i in range(5):
        optimizer.zero_grad()
        logits, _ = model(input_ids, return_diagnostics=False)
        loss = loss_fn(logits.view(-1, config.vocab_size), targets.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        print(f"  Step {i+1}: Loss = {loss.item():.4f}")
        if i == 0: initial_loss = loss.item()
        final_loss = loss.item()
        
    return initial_loss, final_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=str, default="10b_sim", choices=["10b_sim", "small"], help="Model scale to simulate")
    args = parser.parse_args()
    
    print(f"=== Phase 8 Comprehensive Benchmark (Scale: {args.scale}) ===")
    
    # Configuration
    if args.scale == "10b_sim":
        # Simulating 10B topology but scaled down d_model for memory if needed,
        # or use actual 10B config if we just want to measure params/buffers without running full forward
        # For this script, we'll use a scaled version that FITS in memory but has the structural properties.
        # To measure theoretical 10B size, we calculate math.

        # 10B Config (Theoretical)
        real_10b_config = Phase8Config(
            vocab_size=50257,
            d_model=5120,
            n_layers=31,
            htt_rank=128,
            quantized_htt=True,
            use_bk_hyperbolic=True,
            use_ar_ssm_fusion=True
        )

        # Runnable Config (Small)
        run_config = Phase8Config(
            vocab_size=10000,
            d_model=512,
            n_layers=4,
            htt_rank=32,
            quantized_htt=True,
            use_bk_hyperbolic=True,
            use_ar_ssm_fusion=True,
            use_bitnet=True  # Enable BitNet for 1.58-bit weights
        )
    else:
        run_config = Phase8Config(
            vocab_size=1000,
            d_model=128,
            n_layers=2,
            htt_rank=16,
            quantized_htt=True
        )
        real_10b_config = run_config

    # 1. Theoretical Compression Verification (Math only for 10B)
    print("\n--- 1. Model Size Verification ---")
    
    # Calculate FP16 size for 10B
    # Standard: 10B * 2 bytes = 20GB
    # Quantized HTT:
    #   Vocab=50257, D=5120
    #   Std Emb: 50257 * 5120 * 2 bytes = 514 MB
    #   HTT (Rank 128): ~1% of that -> 5MB
    #   BitNet Layers: 1.58 bit per weight.
    #   Weight = 10B - Embedding.
    #   10B * 1.58 bits / 8 = 1.97 GB
    #   Total ~ 2 GB.
    
    print(f"Target 10B Model Config:")
    print(f"  d_model={real_10b_config.d_model}, n_layers={real_10b_config.n_layers}, htt_rank={real_10b_config.htt_rank}")
    
    # Instantiate actual model (Small runnable version) to check Quantization logic
    print(f"\nInstantiating Runnable Phase 8 Model (Small)...")
    model_p8 = Phase8IntegratedModel(run_config)
    size_p8_mb = get_model_size_mb(model_p8)
    params_p8 = count_parameters(model_p8)
    print(f"  Phase 8 (Quantized HTT) Size: {size_p8_mb:.2f} MB, Params: {params_p8:,}")
    
    # Compare with Phase 7 (Standard HTT, FP32 weights)
    print(f"Instantiating Phase 7 Model (Standard)...")
    config_p7 = Phase7Config(
        vocab_size=run_config.vocab_size,
        d_model=run_config.d_model,
        n_layers=run_config.n_layers,
        htt_rank=run_config.htt_rank
    )
    model_p7 = Phase7IntegratedModel(config_p7)
    size_p7_mb = get_model_size_mb(model_p7)
    params_p7 = count_parameters(model_p7)
    print(f"  Phase 7 (Standard HTT)  Size: {size_p7_mb:.2f} MB, Params: {params_p7:,}")
    
    compression_ratio = size_p8_mb / size_p7_mb
    print(f"  Measured Compression Ratio (Small Scale): {compression_ratio:.4f} ({100*(1-compression_ratio):.1f}% reduction)")
    
    # 2. Throughput Benchmark
    print("\n--- 2. Throughput Benchmark (Tokens/sec) ---")
    tp_p8 = benchmark_throughput(model_p8, run_config)
    print(f"  Phase 8 Throughput: {tp_p8:.2f} tokens/sec")
    
    tp_p7 = benchmark_throughput(model_p7, config_p7)
    print(f"  Phase 7 Throughput: {tp_p7:.2f} tokens/sec")
    
    speedup = tp_p8 / tp_p7
    print(f"  Speedup Factor: {speedup:.2f}x")
    
    # 3. Usefulness / Sanity Check
    print("\n--- 3. Usefulness / Convergence Sanity Check ---")
    init_loss, final_loss = run_sanity_check(model_p8, run_config)

    print(f"  Initial Loss: {init_loss:.4f}")
    print(f"  Final Loss:   {final_loss:.4f}")
    if final_loss < init_loss:
        print("  RESULT: PASS (Loss is decreasing)")
    else:
        print("  RESULT: WARNING (Loss did not decrease in 5 steps)")

    # 4. Diagnostics Check
    print("\n--- 4. Diagnostics Availability ---")
    seq_len = getattr(run_config, 'n_seq', 128)
    input_ids = torch.randint(0, run_config.vocab_size, (1, seq_len))
    if torch.cuda.is_available(): input_ids = input_ids.cuda()
    model_p8.eval()
    with torch.no_grad():
        _, diag = model_p8(input_ids, return_diagnostics=True)

    if diag:
        print("  Diagnostics returned successfully.")
        p8_diag = diag.get('phase8', {})
        print(f"  Entailment Violation Rate: {p8_diag.get('entailment_violation_rate')}")
        print(f"  Topology Betti Numbers: {p8_diag.get('betti_numbers')}")
    else:
        print("  RESULT: FAIL (No diagnostics returned)")

if __name__ == "__main__":
    main()
