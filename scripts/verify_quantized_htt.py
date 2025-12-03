#!/usr/bin/env python3
"""
Verification Script for Quantized Holographic Tensor Train (QHTT) Embedding

Goal:
    Demonstrate that QHTT can restore the 99% compression ratio even when
    high rank (e.g., 64) is used for real-world data fidelity.

Scenario:
    1. Instantiate a High-Rank HTT (simulating real-world requirements).
       - This normally degrades compression from 99% to ~60%.
    2. Convert to QHTT (Quantized HTT) using Logarithmic Quantization.
    3. Verify that compression ratio returns to >90% (target 99%).
    4. Check reconstruction fidelity.

Author: Project MUSE Team
"""

import sys
import os
import torch
import torch.nn as nn
import math

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.phase1.htt_embedding import HolographicTTEmbedding
from src.models.phase8.quantized_htt import QuantizedHolographicTTEmbedding
from src.models.phase8.quantization import LogarithmicQuantizer

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def format_mb(bytes_val):
    return f"{bytes_val / (1024*1024):.2f} MB"

def verify_solution():
    print(f"{Colors.BOLD}{Colors.BLUE}=== QHTT Verification: Solving the Real-World Compression Dilemma ==={Colors.END}\n")

    # Configuration
    # Simulating a realistic model size where high rank is needed
    vocab_size = 50257  # GPT-2 size
    d_model = 1024      # Large model dimension

    # 1. The Problem: High Rank for Real Data
    # To capture real data complexity, we need higher rank (e.g., 64 instead of 8 or 16)
    high_rank = 64
    print(f"{Colors.BOLD}Step 1: Simulating High-Fidelity Requirement (Rank={high_rank}){Colors.END}")
    print(f"  Configuration: Vocab={vocab_size}, D_Model={d_model}")

    # Standard Embedding (Baseline)
    standard_emb = nn.Embedding(vocab_size, d_model)
    std_params = vocab_size * d_model
    std_size_mb = std_params * 4 / (1024*1024) # FP32
    print(f"  Standard Embedding Size: {Colors.YELLOW}{std_size_mb:.2f} MB{Colors.END}")

    # High-Rank HTT
    htt_model = HolographicTTEmbedding(vocab_size, d_model, rank=high_rank)
    htt_params = htt_model.get_parameter_counts()[1]
    htt_size_mb = htt_params * 4 / (1024*1024) # FP32

    htt_compression = (1 - htt_size_mb / std_size_mb) * 100

    print(f"  High-Rank HTT Size:      {htt_size_mb:.2f} MB")
    print(f"  Current Compression:     {Colors.RED}{htt_compression:.2f}%{Colors.END} (Below 90% target)")
    print("  -> This simulates the user's problem: Compression drops significantly with real data settings.\n")

    # 2. The Solution: Quantized HTT
    print(f"{Colors.BOLD}Step 2: Applying Manifold-Aware Logarithmic Quantization (Phase 8){Colors.END}")
    print("  Converting to QHTT (8-bit Log Quantization)...")

    # Convert
    qhtt_model = QuantizedHolographicTTEmbedding.from_htt(htt_model, bits=8)

    # 3. Verification
    stats = qhtt_model.get_compression_stats()
    qhtt_size_mb = stats['qhtt_mb']
    new_compression = stats['reduction_percentage']

    print(f"\n{Colors.BOLD}Step 3: Results{Colors.END}")
    print(f"  QHTT Size (Storage):     {Colors.GREEN}{qhtt_size_mb:.2f} MB{Colors.END}")
    print(f"  New Compression Ratio:   {Colors.GREEN}{new_compression:.2f}%{Colors.END}")

    # 4. Check Fidelity
    print(f"\n{Colors.BOLD}Step 4: Checking Fidelity (Reconstruction Error){Colors.END}")

    # Create dummy input
    input_ids = torch.randint(0, vocab_size, (8, 32))

    # Forward pass
    with torch.no_grad():
        out_htt = htt_model(input_ids)
        out_qhtt = qhtt_model(input_ids)

    # Error
    diff = (out_htt - out_qhtt).abs().mean()
    rel_error = diff / out_htt.abs().mean() * 100

    print(f"  Mean Reconstruction Error: {diff:.6f}")
    print(f"  Relative Error:            {rel_error:.4f}%")

    # Conclusion
    print(f"\n{Colors.BOLD}=== Conclusion ==={Colors.END}")
    if new_compression > 90.0:
        print(f"{Colors.GREEN}SUCCESS: Restored >90% compression ratio while maintaining High Rank!{Colors.END}")
    else:
        print(f"{Colors.RED}FAILURE: Compression ratio is still too low.{Colors.END}")

if __name__ == "__main__":
    verify_solution()
