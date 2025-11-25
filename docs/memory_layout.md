# Developer Documentation: Memory Layout Optimization

## 1. Objective

The memory layout of the core model components has been redesigned with one primary objective:

- **Maximize Data-Throughput to Custom Triton Kernels**: Specifically for `HybridHyperbolicAttention`, the goal is to eliminate high-cost PyTorch operations like `.permute()` and `.contiguous()` from the main training loop. This ensures that data is fed to the GPU kernels in the most efficient format possible, which is critical for performance at long sequence lengths.

## 2. Standard Memory Layout

To achieve this, a project-wide standard memory layout has been established for all tensors that pass through the model's recurrent blocks, whenever `use_hybrid_attention` is enabled.

The official layout is:

`[Batch Size, Number of Heads, Sequence Length, Head Dimension]`
`(B, H, N, D_h)`

## 3. Scope of Application

- This new layout is **only active** when the model configuration specifies `use_hybrid_attention: true`.
- When `use_hybrid_attention: false`, the model reverts to the legacy `[Batch Size, Sequence Length, Dimension]` `(B, N, D)` layout for full backward compatibility.

## 4. Implementation Details

- **Entry/Exit Points**: The `LanguageModel.forward()` method is responsible for reshaping the initial embeddings into the `(B, H, N, D_h)` layout and reshaping the final output back to `(B, N, D)` before the LM head.
- **Native Modules**: Core modules like `MoEResNetBKLayer` (when `use_hybrid_attention: true`), `HybridHyperbolicAttention`, and `HolographicMemory` have been refactored to handle this layout natively.
- **Backward Compatibility**: Modules that have not been refactored (e.g., the legacy Birman-Schwinger path within `MoEResNetBKLayer`) contain internal "adapter" logic to temporarily convert the data to the legacy `(B, N, D)` format for computation.

This change is fundamental to achieving the performance goals of Phase 7 and beyond. All future development on core model components should adhere to this standard layout.
