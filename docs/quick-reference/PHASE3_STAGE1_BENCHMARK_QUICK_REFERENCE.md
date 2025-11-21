# Phase 3 Stage 1 Benchmark - Quick Reference

## Overview

Phase 3 Stage 1 (Complex Dynamics Foundation) benchmark implementation and test results.

## Quick Start

### Run Basic Test

```bash
python scripts/test_phase3_stage1_simple.py
```

### Run Full Benchmark (when dataloader is fixed)

```bash
# With Phase 2 comparison
python scripts/benchmark_phase3_stage1.py

# Without Phase 2 comparison
python scripts/benchmark_phase3_stage1.py --skip-phase2

# Quick test (limited batches)
python scripts/benchmark_phase3_stage1.py --max-ppl-batches 10
```

## Test Results Summary

### ✓ Passed Tests

| Test | Target | Result | Status |
|------|--------|--------|--------|
| Model Creation | - | 81M parameters | ✓ PASS |
| Forward Pass | No NaN/Inf | No NaN/Inf | ✓ PASS |
| Backward Pass | 1e-6 ≤ grad ≤ 1e3 | 6.77e-04 ≤ grad ≤ 1.01 | ✓ PASS |
| Memory Usage | < 8GB | 1.37 GB | ✓ PASS |

### ✅ Completed Tests

| Test | Target | Result | Status |
|------|--------|--------|--------|
| Perplexity | Phase 2 + 3% | 54430.93 (untrained) | ✓ MEASURED |
| VRAM Usage | < 8GB | 1.21 GB | ✓ PASS |
| Throughput | - | 137,386 tokens/sec | ✓ MEASURED |

### ⏳ Pending Tests

| Test | Target | Status |
|------|--------|--------|
| Phase 2 Comparison | PPL, VRAM | Requires Phase 2 model |
| Post-Training PPL | Phase 2 + 3% | Requires training |

## Model Specifications

```python
Phase3Stage1Config(
    vocab_size=50257,
    d_model=512,
    n_layers=6,
    n_seq=1024,
    use_complex32=True
)
```

- **Parameters**: 80,937,553 (~81M)
- **Memory Layout**: Planar format (real/imag separated)
- **Data Type**: complex32 (float16 × 2)

## Performance Metrics

### Memory Usage (Batch=2, Seq=1024)

- **Current VRAM**: 0.52 GB
- **Peak VRAM**: 1.37 GB
- **Target**: < 8 GB
- **Status**: ✓ Well within target

### Numerical Stability

- **NaN Count**: 0
- **Inf Count**: 0
- **Stability Rate**: 100%
- **Status**: ✓ Excellent

### Gradient Health

- **Min Gradient Norm**: 6.77e-04
- **Max Gradient Norm**: 1.01
- **Target Range**: [1e-6, 1e3]
- **Status**: ✓ All gradients healthy

## Stage 1 Completion Criteria

| Criterion | Target | Status |
|-----------|--------|--------|
| Perplexity | Phase 2 + 3% | ⏳ PENDING |
| VRAM | ≤ 52% of Phase 2 | ⏳ PENDING |
| Numerical Stability | 0% NaN | ✓ ACHIEVED |
| Gradient Health | 1e-6 ≤ grad ≤ 1e3 | ✓ ACHIEVED |
| Memory Layout | Planar format | ✓ ACHIEVED |

## Fixed Issues

### ✅ Dataloader Bug (Fixed 2025-11-21)

**Issue**: `attention_mask` length mismatch in WikiText-2 dataloader

**Fix**: Added `remove_columns=tokenized.column_names` to `group_texts` mapping

**Status**: ✅ Resolved - Full benchmark now operational

## Next Steps

1. Fix dataloader bug
2. Execute full benchmark on WikiText-2
3. Compare with Phase 2 model
4. Generate comprehensive JSON report
5. Verify all Stage 1 completion criteria

## Files

### Scripts

- `scripts/benchmark_phase3_stage1.py` - Full benchmark script
- `scripts/test_phase3_stage1_simple.py` - Basic test script

### Reports

- `results/benchmarks/PHASE3_TASK7_完了報告_日本語.md` - Japanese report
- `results/benchmarks/PHASE3_TASK7_COMPLETION_SUMMARY.md` - English report
- `results/benchmarks/phase3_stage1_basic_test.json` - JSON results

### Documentation

- `docs/quick-reference/PHASE3_STAGE1_BENCHMARK_QUICK_REFERENCE.md` - This file

## Requirements

- Requirements: 1.18, 1.19, 1.20
- PyTorch: 2.5.1+
- CUDA: Optional (CPU fallback available)

## Contact

For issues or questions, refer to:
- `.kiro/specs/phase3-physics-transcendence/tasks.md`
- `.kiro/specs/phase3-physics-transcendence/design.md`

---

**Last Updated**: 2025-11-21  
**Author**: Project MUSE Team
