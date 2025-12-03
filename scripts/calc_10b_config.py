#!/usr/bin/env python3
"""
Calculate configuration for a 10B parameter model and estimate compression.
"""
import argparse

def format_size(bytes_val):
    if bytes_val < 1024:
        return f"{bytes_val} B"
    elif bytes_val < 1024**2:
        return f"{bytes_val/1024:.2f} KB"
    elif bytes_val < 1024**3:
        return f"{bytes_val/1024**2:.2f} MB"
    else:
        return f"{bytes_val/1024**3:.2f} GB"

def format_params(num):
    if num < 1000:
        return str(num)
    elif num < 1000**2:
        return f"{num/1000:.2f}K"
    elif num < 1000**3:
        return f"{num/1000**2:.2f}M"
    else:
        return f"{num/1000**3:.2f}B"

def calc_10b_stats(d_model=4096, n_layers=48, vocab_size=50257, n_seq=2048, htt_rank=16):
    print(f"Configuration: d_model={d_model}, n_layers={n_layers}, vocab={vocab_size}")
    
    # --- Dense Model Stats (FP16) ---
    # Embedding
    emb_params = vocab_size * d_model
    pos_params = n_seq * d_model
    
    # Layer
    # Attn: 4 * d_model^2 (W_q, W_k, W_v, W_o) + 4 * d_model (bias)
    attn_params = 4 * d_model**2 + 4 * d_model
    # FFN: 2 * 4 * d_model^2 (up, down) + ... usually 8*d^2
    ffn_params = 8 * d_model**2 + 5 * d_model # biases
    # LayerNorm: 4 * d_model
    ln_params = 4 * d_model
    
    layer_params = attn_params + ffn_params + ln_params
    
    total_dense_params = emb_params + pos_params + n_layers * layer_params + (d_model * vocab_size) # LM Head
    
    dense_memory_bytes = total_dense_params * 2 # FP16 = 2 bytes
    
    print(f"\n[Dense Baseline (FP16)]")
    print(f"Total Params: {format_params(total_dense_params)}")
    print(f"Memory Size:  {format_size(dense_memory_bytes)}")
    
    # --- Phase 8 Compressed Stats ---
    # 1. HTT Embedding (Extreme Compression)
    # Assume 4 cores for vocab, rank=16
    # Param count is roughly O(4 * vocab^(1/4) * rank^2) which is negligible
    # Let's estimate HTT params as ~100KB total for vocab
    htt_params = 100_000 # Estimate
    
    # 2. BitNet Linear Layers (1.58 bit)
    # Weights are 1.58 bit, but stored as 2 bit or packed int8.
    
    # FFN Compression (Low Rank)
    # Dense FFN: 8 * d_model^2
    # Low Rank FFN: 2 * (d_model * rank + 4*d_model * rank) = 2 * 5 * d * r = 10 * d * r
    # rank=64, d=4096 => 10 * 4096 * 64 = 2.6M params per layer
    # Dense: 8 * 4096^2 = 134M params per layer
    # Reduction: ~50x
    
    low_rank_ffn_params = n_layers * (10 * d_model * 64) # rank=64
    
    # Attention Weights (Low Rank)
    # Dense: 4 * d_model^2
    # Low Rank: 4 * (2 * d_model * rank) = 8 * d * r
    # rank=64 => 8 * 4096 * 64 = 2M params per layer
    low_rank_attn_params = n_layers * (8 * d_model * 64)
    
    weight_params = 0 # All low rank now
    
    other_params = n_layers * (ln_params + 9 * d_model) # Biases + LN
    
    # LM Head is also HTT? If so, negligible. If dense BitNet:
    lm_head_params = d_model * vocab_size
    
    # Total "Active" Params (for computation, effectively the dense count)
    # But "Stored" Params is what we care about for compression ratio
    
    # Compressed Memory
    # Low Rank Attn Weights: FP16
    low_rank_attn_memory = low_rank_attn_params * 2
    
    # Low Rank FFN Weights: FP16
    low_rank_ffn_memory = low_rank_ffn_params * 2 
    
    # HTT Embedding Memory (FP32 for cores)
    htt_memory = htt_params * 4 
    
    # Other Params (FP16)
    other_memory = other_params * 2
    
    # LM Head: If HTT, small. If BitNet, 0.25 bytes/param.
    # Let's assume HTT Decoder for LM Head too (Phase 7 integrated model does this)
    lm_head_memory = htt_memory
    
    total_compressed_memory = low_rank_attn_memory + low_rank_ffn_memory + htt_memory + other_memory + lm_head_memory
    
    print(f"\n[Phase 8 Compressed (BitNet + HTT + LowRankFFN + LowRankAttn)]")
    print(f"LowRank Attn (FP16):  {format_params(low_rank_attn_params)}")
    print(f"LowRank FFN (FP16):   {format_params(low_rank_ffn_params)}")
    print(f"HTT Params:           {format_params(htt_params)}")
    print(f"Total Memory:         {format_size(total_compressed_memory)}")
    
    # --- Compression Ratio ---
    ratio = 1.0 - (total_compressed_memory / dense_memory_bytes)
    print(f"\nCompression Ratio: {ratio*100:.4f}%")
    
    if ratio > 0.99:
        print("✅ > 99% Compression Achieved!")
    else:
        print(f"❌ < 99% Compression ({ratio*100:.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=4096)
    parser.add_argument("--n_layers", type=int, default=48)
    args = parser.parse_args()
    
    calc_10b_stats(d_model=args.d_model, n_layers=args.n_layers)
