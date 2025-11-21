# Benchmark Commands for Phase 3

This document lists the commands to run benchmarks for the Phase 3 components.

## 1. Complex Matrix Multiplication Benchmark (Triton vs PyTorch)

Measures the speedup of the custom Triton kernel for complex matrix multiplication.

```bash
python scripts/benchmark_complex_matmul.py
```

**Expected Output:**
- Speedup > 1.25x (on GPU)
- MSE Error < 1e-5

## 2. Symplectic Step Benchmark (Triton vs PyTorch)

Measures the speedup and energy conservation of the symplectic leapfrog integrator.

```bash
python scripts/benchmark_symplectic_step.py
```

**Expected Output:**
- Speedup > 1.20x (on GPU)
- Energy Drift < 5e-5

## 3. Full Phase 3 Model Benchmark

Measures Throughput, VRAM usage, and Perplexity (Simulated).

```bash
python scripts/benchmark_phase3.py --preset small
```

**Expected Output:**
- Throughput > 85% of Phase 2 baseline
- VRAM < 8.0 GB (for base model)

## 4. Training Demo

Runs a short training loop to verify integration and diagnostics.

```bash
python scripts/train_phase3.py --epochs 1 --batch_size 4
```

## 5. Visualization

Generates plots for Koopman eigenvalues and Energy drift.

```bash
python scripts/visualize_phase3.py
```
