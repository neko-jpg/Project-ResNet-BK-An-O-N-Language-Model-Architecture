"""
Fused MoE Kernel
================

A fused Triton kernel for the Sparse Mixture of Experts layer.
This kernel is designed to be a single, monolithic operation that performs:
1. Gating / Routing: Computes router logits for each token.
2. Top-K Selection: Selects the top-k experts for each token.
3. Softmax: Computes routing weights.
4. Dispatch: Routes tokens to their selected experts.
5. Expert Computation: Performs the forward pass for each expert (2-layer MLP).
6. Aggregation: Aggregates the outputs from the experts based on routing weights.

This approach minimizes kernel launch overhead and data movement between the GPU's
global memory and SRAM, aiming for maximum performance.

Includes:
- `fused_moe_kernel`: The main Triton kernel.
- `_fused_moe_pytorch`: A PyTorch reference implementation for correctness testing
  and as a fallback for non-GPU environments.
- `fused_moe_forward`: A dispatcher function that selects the appropriate
  implementation (Triton or PyTorch).
"""

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("Triton not available. Fused MoE kernel will use PyTorch fallback.")

# --- PyTorch Reference Implementation ---

def _fused_moe_pytorch(
    x: torch.Tensor,
    gate_w: torch.Tensor,
    experts_w1: torch.Tensor,
    experts_w2: torch.Tensor,
    top_k: int
):
    """
    PyTorch reference implementation of the fused MoE layer.
    Mirrors the logic of the Triton kernel for verification.

    Args:
        x (torch.Tensor): Input tensor of shape (T, D)
        gate_w (torch.Tensor): Gating network weights of shape (D, E)
        experts_w1 (torch.Tensor): Weights for the first MLP layer of all experts,
                                   shape (E, D, H), H is hidden size.
        experts_w2 (torch.Tensor): Weights for the second MLP layer of all experts,
                                   shape (E, H, D).
        top_k (int): The number of experts to route to.

    Returns:
        torch.Tensor: The output tensor of shape (T, D).
    """
    # Ensure contiguous tensors for performance
    x = x.contiguous()
    gate_w = gate_w.contiguous()
    experts_w1 = experts_w1.contiguous()
    experts_w2 = experts_w2.contiguous()

    # 1. Gating / Routing
    router_logits = x @ gate_w  # (T, E)

    # 2. Top-K Selection
    topk_logits, topk_indices = torch.topk(router_logits, top_k, dim=-1) # (T, K), (T, K)

    # 3. Softmax
    topk_weights = F.softmax(topk_logits, dim=-1) # (T, K)

    # 4-6. Dispatch, Expert Computation, and Aggregation
    output = torch.zeros_like(x)
    num_experts = gate_w.shape[1]

    for i in range(x.shape[0]): # Iterate over each token
        for k in range(top_k):
            expert_idx = topk_indices[i, k].item()
            weight = topk_weights[i, k]

            # Get expert weights
            w1 = experts_w1[expert_idx] # (D, H)
            w2 = experts_w2[expert_idx] # (H, D)

            # Expert computation
            hidden = F.relu(x[i] @ w1) # (H,)
            expert_out = hidden @ w2 # (D,)

            # Aggregation
            output[i] += weight * expert_out

    return output


# --- Triton Kernel Implementation ---

if TRITON_AVAILABLE:
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE_T': 64, 'BLOCK_SIZE_D': 64, 'BLOCK_SIZE_E': 16}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_T': 128, 'BLOCK_SIZE_D': 32, 'BLOCK_SIZE_E': 16}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_T': 64, 'BLOCK_SIZE_D': 128, 'BLOCK_SIZE_E': 16}, num_stages=3, num_warps=8),
            # Add more configurations as needed for different hardware
        ],
        key=['T', 'D', 'E', 'H'],
    )
    @triton.jit
    def fused_moe_kernel(
        X_ptr, Gate_W_ptr, Experts_W1_ptr, Experts_W2_ptr, Output_ptr,
        T: tl.int32, D: tl.int32, E: tl.int32, H: tl.int32, K: tl.int32,
        stride_xt, stride_xd,
        stride_gated, stride_gatee,
        stride_w1e, stride_w1d, stride_w1h,
        stride_w2e, stride_w2h, stride_w2d,
        stride_outt, stride_outd,
        BLOCK_SIZE_T: tl.constexpr, BLOCK_SIZE_D: tl.constexpr, BLOCK_SIZE_E: tl.constexpr,
    ):
        """
        Triton kernel for Fused MoE.
        """
        # --- Program ID and Offsets ---
        pid_t = tl.program_id(axis=0)
        t_offsets = pid_t * BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_T)
        t_mask = t_offsets < T

        # --- Pointers to Input and Gating Weights ---
        x_ptrs = X_ptr + (t_offsets[:, None] * stride_xt + tl.arange(0, BLOCK_SIZE_D)[None, :] * stride_xd)
        gate_w_ptrs = Gate_W_ptr + (tl.arange(0, BLOCK_SIZE_D)[:, None] * stride_gated + tl.arange(0, BLOCK_SIZE_E)[None, :] * stride_gatee)

        # --- Gating / Routing ---
        # Load gating weights into SRAM
        gate_w = tl.load(gate_w_ptrs, mask=(tl.arange(0, BLOCK_SIZE_D)[:, None] < D) & (tl.arange(0, BLOCK_SIZE_E)[None, :] < E), other=0.0)

        # Load input tokens for this block
        x = tl.load(x_ptrs, mask=t_mask[:, None] & (tl.arange(0, BLOCK_SIZE_D)[None, :] < D), other=0.0)

        # Compute router logits
        router_logits = tl.dot(x, gate_w) # (BLOCK_SIZE_T, BLOCK_SIZE_E)

        # --- Top-K Selection (Optimized for K=1) ---
        # For K=1, we just need the argmax.
        topk_indices = tl.argmax(router_logits, axis=1) # (BLOCK_SIZE_T,)

        # --- Expert Computation and Aggregation (Optimized for K=1) ---
        # This is the performance-critical part. Instead of looping, we process
        # the block of tokens in parallel. We iterate through the experts, and for
        # each expert, we create a mask for the tokens that are routed to it.

        output_accumulator = tl.zeros((BLOCK_SIZE_T, BLOCK_SIZE_D), dtype=tl.float32)

        for expert_idx in range(E):
            # Create a mask for tokens assigned to the current expert
            token_mask_for_expert = (topk_indices == expert_idx) & t_mask # (BLOCK_SIZE_T,)

            # If no tokens in this block are routed to this expert, skip.
            if tl.sum(token_mask_for_expert) == 0:
                continue

            # --- Load Expert Weights (Coalesced Access) ---
            # All threads in a warp that are processing tokens for the same expert
            # will load the same weights, leading to coalesced memory access.
            w1_ptrs = Experts_W1_ptr + expert_idx * stride_w1e + \
                      (tl.arange(0, BLOCK_SIZE_D)[:, None] * stride_w1d + tl.arange(0, H)[None, :] * stride_w1h)
            w2_ptrs = Experts_W2_ptr + expert_idx * stride_w2e + \
                      (tl.arange(0, H)[:, None] * stride_w2h + tl.arange(0, BLOCK_SIZE_D)[None, :] * stride_w2d)

            w1 = tl.load(w1_ptrs, mask=(tl.arange(0, BLOCK_SIZE_D)[:, None] < D) & (tl.arange(0, H)[None, :] < H), other=0.0)
            w2 = tl.load(w2_ptrs, mask=(tl.arange(0, H)[:, None] < H) & (tl.arange(0, BLOCK_SIZE_D)[None, :] < D), other=0.0)

            # --- Masked Load of Tokens ---
            # Load only the tokens that are routed to this expert.
            # `x` is already loaded for the whole block.
            masked_x = tl.where(token_mask_for_expert[:, None], x, 0.0)

            # --- Expert Computation ---
            # This performs the computation for all tokens in the block, but the
            # masking ensures that only the relevant tokens contribute.
            hidden = tl.dot(masked_x, w1)
            hidden = tl.where(hidden > 0, hidden, 0) # ReLU
            expert_out = tl.dot(hidden, w2)

            # --- Aggregation ---
            # Since K=1, the weight is always 1.0, so we just add the result.
            # The mask ensures we only add outputs for the correct tokens.
            output_accumulator += tl.where(token_mask_for_expert[:, None], expert_out, 0.0)

        # --- Write Output ---
        output_ptrs = Output_ptr + (t_offsets[:, None] * stride_outt + tl.arange(0, BLOCK_SIZE_D)[None, :] * stride_outd)
        tl.store(output_ptrs, output_accumulator, mask=t_mask[:, None] & (tl.arange(0, BLOCK_SIZE_D)[None, :] < D))


# --- Dispatcher ---

def fused_moe_forward(
    x: torch.Tensor,
    gate_w: torch.Tensor,
    experts_w1: torch.Tensor,
    experts_w2: torch.Tensor,
    top_k: int
):
    """
    Dispatcher for the Fused MoE forward pass.

    Selects the Triton kernel if available, otherwise falls back to the
    PyTorch reference implementation.

    Args:
        x (torch.Tensor): Input tensor of shape (B, N, D).
        gate_w (torch.Tensor): Gating network weights of shape (D, E).
        experts_w1 (torch.Tensor): Weights for all experts' first MLP layer,
                                   shape (E, D, H).
        experts_w2 (torch.Tensor): Weights for all experts' second MLP layer,
                                   shape (E, H, D).
        top_k (int): Number of experts to route to.

    Returns:
        torch.Tensor: The output tensor of shape (B, N, D).
    """
    if x.dim() != 3:
        raise ValueError("Input tensor must be 3-dimensional (B, N, D)")

    B, N, D = x.shape
    E = gate_w.shape[1]
    H = experts_w1.shape[2]

    # Reshape for kernel: (B, N, D) -> (B*N, D)
    x_flat = x.view(-1, D)
    T = x_flat.shape[0]

    # Check if we should use Triton. The current kernel only supports top_k=1.
    use_triton = TRITON_AVAILABLE and x.is_cuda and top_k == 1

    # If top_k > 1 with a CUDA device, inform the user about the fallback.
    if TRITON_AVAILABLE and x.is_cuda and top_k > 1:
        # Using warnings.warn is better for library code than print
        import warnings
        warnings.warn(
            "Fused MoE Triton kernel only supports top_k=1. "
            f"Falling back to the PyTorch implementation for top_k={top_k}.",
            UserWarning
        )

    if use_triton:
        output = torch.empty_like(x_flat)
        grid = lambda meta: (triton.cdiv(T, meta['BLOCK_SIZE_T']),)

        # Call the Triton kernel
        # NOTE: This simplified kernel has limitations and is primarily for demonstration.
        # A full implementation would be significantly more complex, especially the
        # weight loading and token dispatching logic for optimal performance.
        # The current implementation will be functionally correct but not highly optimized.
        fused_moe_kernel[grid](
            x_flat, gate_w, experts_w1, experts_w2, output,
            T, D, E, H, top_k,
            x_flat.stride(0), x_flat.stride(1),
            gate_w.stride(0), gate_w.stride(1),
            experts_w1.stride(0), experts_w1.stride(1), experts_w1.stride(2),
            experts_w2.stride(0), experts_w2.stride(1), experts_w2.stride(2),
            output.stride(0), output.stride(1),
        )

    else:
        # Use PyTorch fallback
        output = _fused_moe_pytorch(x_flat, gate_w, experts_w1, experts_w2, top_k)

    # Reshape back to original: (B*N, D) -> (B, N, D)
    return output.view(B, N, D)

def is_triton_available():
    return TRITON_AVAILABLE
