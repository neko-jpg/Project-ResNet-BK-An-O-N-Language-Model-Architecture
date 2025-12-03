
import torch
import triton
import triton.language as tl

@triton.jit
def htt_fused_contraction_kernel(
    # Input pointers
    idx1_ptr, idx2_ptr,  # Token indices (B, L)
    core1_ptr, core2_ptr,  # Quantized TT cores (int8)
    scale1_ptr, scale2_ptr, # Scales (float)
    output_ptr,  # Output embeddings (float)
    # Dimensions
    batch_size, seq_len,
    v1, v2, rank, d1, d2, d_model,
    # Strides
    stride_idx_b, stride_idx_l,
    stride_c1_v, stride_c1_r, stride_c1_d,
    stride_c2_v, stride_c2_r, stride_c2_d,
    stride_out_b, stride_out_l, stride_out_d,
    # Block sizes
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused Holographic Tensor Train Contraction Kernel.
    Performs dequantization and contraction in one go.
    
    Core1: (v1, rank, d1) int8
    Core2: (v2, rank, d2) int8
    
    out[b, l, d] = sum_r ( (Core1[idx1, r, d1_idx] * s1) * (Core2[idx2, r, d2_idx] * s2) )
    """
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate batch and sequence indices
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len
    
    # Check bounds
    if batch_idx >= batch_size:
        return
    
    # Load token indices
    idx1_offset = batch_idx * stride_idx_b + seq_idx * stride_idx_l
    idx2_offset = batch_idx * stride_idx_b + seq_idx * stride_idx_l
    
    idx1 = tl.load(idx1_ptr + idx1_offset)
    idx2 = tl.load(idx2_ptr + idx2_offset)
    
    # Clamp indices
    idx1 = tl.minimum(tl.maximum(idx1, 0), v1 - 1)
    idx2 = tl.minimum(tl.maximum(idx2, 0), v2 - 1)
    
    # Output offset
    out_offset = batch_idx * stride_out_b + seq_idx * stride_out_l
    
    # Load scales (assuming scalar scale for now, or per-tensor)
    # If scales are pointers, load them.
    s1 = tl.load(scale1_ptr)
    s2 = tl.load(scale2_ptr)
    
    # Process output dimensions in blocks
    for d_block_start in range(0, d_model, BLOCK_SIZE):
        d_block_end = tl.minimum(d_block_start + BLOCK_SIZE, d_model)
        d_range = d_block_end - d_block_start
        
        # Initialize accumulator
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        
        # Iterate over output dimensions in this block
        for d_local in range(d_range):
            d_global = d_block_start + d_local
            
            # Calculate d1, d2 indices
            d2_size = (d_model + d1 - 1) // d1
            d1_idx = d_global // d2_size
            d2_idx = d_global % d2_size
            
            if d1_idx >= d1 or d2_idx >= d2:
                continue
            
            # Contract over rank dimension
            rank_sum = 0.0
            for r in range(rank):
                # Load Core1 (int8)
                c1_offset = (idx1 * stride_c1_v + 
                            r * stride_c1_r + 
                            d1_idx * stride_c1_d)
                c1_int = tl.load(core1_ptr + c1_offset)
                
                # Load Core2 (int8)
                c2_offset = (idx2 * stride_c2_v + 
                            r * stride_c2_r + 
                            d2_idx * stride_c2_d)
                c2_int = tl.load(core2_ptr + c2_offset)
                
                # Dequantize and multiply
                # val = (c1 * s1) * (c2 * s2) = (c1 * c2) * (s1 * s2)
                rank_sum += (c1_int.to(tl.float32) * c2_int.to(tl.float32))
            
            # Apply scales once per sum
            rank_sum = rank_sum * (s1 * s2)
            
            # Store in accumulator
            acc = tl.where(d_local < d_range, rank_sum, acc)
        
        # Store block to output
        for d_local in range(d_range):
            d_global = d_block_start + d_local
            if d_global < d_model:
                out_ptr_offset = out_offset + d_global * stride_out_d
                tl.store(output_ptr + out_ptr_offset, acc[d_local])

def htt_fused_contraction(
    idx1: torch.Tensor,
    idx2: torch.Tensor,
    core1: torch.Tensor, # int8
    core2: torch.Tensor, # int8
    scale1: torch.Tensor, # float scalar
    scale2: torch.Tensor, # float scalar
    d_model: int,
    block_size: int = 32,
) -> torch.Tensor:
    """
    Fused HTT contraction with int8 inputs.
    """
    B, L = idx1.shape
    v1, rank, d1 = core1.shape
    v2, rank2, d2 = core2.shape
    
    assert rank == rank2
    
    output = torch.zeros((B, L, d_model), device=idx1.device, dtype=torch.float32) # Output is float
    
    grid = (B * L,)
    
    htt_fused_contraction_kernel[grid](
        idx1, idx2,
        core1, core2,
        scale1, scale2,
        output,
        B, L,
        v1, v2, rank, d1, d2, d_model,
        idx1.stride(0), idx1.stride(1),
        core1.stride(0), core1.stride(1), core1.stride(2),
        core2.stride(0), core2.stride(1), core2.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_SIZE=block_size,
    )
    
    return output
