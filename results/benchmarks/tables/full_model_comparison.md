# Full Model Performance Comparison

## Memory Validation Results

| Model | Vocab | d_model | Layers | Batch | SeqLen | Baseline VRAM | Phase 1 VRAM | Reduction | Saved |
|-------|-------|---------|--------|-------|--------|---------------|--------------|-----------|-------|
| Small | 10K | 512 | 4 | 2 | 1024 | 708.35 MB | 673.82 MB | 4.88% | 34.53 MB |
| Large | 50K | 1024 | 6 | 1 | 512 | 2093.20 MB | 1707.18 MB | 18.44% | 386.02 MB |

## 8GB VRAM Target Validation

| Model | Peak VRAM | Target | Status |
|-------|-----------|--------|--------|
| Small | 673.82 MB | < 7.2 GB | ✅ PASS |
| Large | 1707.18 MB | < 7.2 GB | ✅ PASS |

**Key Findings**:
- Small models: 4.88% reduction (other layers dominate)
- Large models: 18.44% reduction (HTT effect more pronounced)
- **All configurations PASS 8GB VRAM target**

## Memory Breakdown (Large Model)

```
Baseline (2093 MB):
├── Embeddings: 196 MB (9.4%)
├── AR-SSM/Attention: ~800 MB (38%)
├── FFN: ~600 MB (29%)
└── Activations: ~497 MB (24%)

Phase 1 (1707 MB):
├── HTT Embeddings: 1 MB (0.06%)  ← 195 MB saved
├── AR-SSM/Attention: ~800 MB (47%)
├── FFN: ~600 MB (35%)
└── Activations: ~306 MB (18%)  ← 191 MB saved (Gradient Checkpointing)
```

**HTT Contribution**: ~50% of total reduction (195 MB out of 386 MB)
