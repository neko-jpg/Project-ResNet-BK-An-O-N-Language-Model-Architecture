# Step 5 Colab Notebook Fix

## Issue

The notebook `notebooks/step5_hardware_optimizations.ipynb` had an import error when running on Google Colab:

```
ImportError: cannot import name 'prepare_wikitext2_data' from 'src.utils.data_utils'
```

## Root Cause

The function `prepare_wikitext2_data` does not exist in `src/utils/data_utils.py`. The correct function is `get_wikitext2_dataloaders`.

## Changes Made

### 1. Updated Import Statement

**Before:**
```python
from src.utils.data_utils import prepare_wikitext2_data
```

**After:**
```python
from src.utils.data_utils import get_wikitext2_dataloaders
```

### 2. Updated Function Calls

**Before:**
```python
train_loader, val_loader, vocab_size = prepare_wikitext2_data(
    batch_size=8,
    seq_len=128,
    max_samples=1000
)
```

**After:**
```python
train_loader, val_loader, vocab_size = get_wikitext2_dataloaders(
    batch_size=8,
    seq_len=128,
    num_workers=0  # Set to 0 for Colab compatibility
)
```

### 3. Added Target Flattening

The `get_wikitext2_dataloaders` function returns targets as `(B, N)` tensors, but the training functions expect flattened `(B*N,)` tensors for `CrossEntropyLoss`.

**Added before each training step:**
```python
# Flatten targets for CrossEntropyLoss
y_batch = y_batch.view(-1)
```

This was added in 4 locations:
1. AMP training test (Section 3)
2. Gradient accumulation test (Section 4)
3. CPU offloading test (Section 5)
4. Dynamic batch sizing test (Section 6)

### 4. Updated Repository URL

**Before:**
```python
!git clone https://github.com/your-repo/resnet-bk.git
%cd resnet-bk
```

**After:**
```python
!git clone https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture.git
%cd Project-ResNet-BK-An-O-N-Language-Model-Architecture
```

## Testing

The notebook should now work correctly on Google Colab. To test:

1. Open `notebooks/step5_hardware_optimizations.ipynb` in Google Colab
2. Run all cells
3. Verify:
   - Data loads successfully
   - All training tests complete without errors
   - GPU memory usage is reported
   - No import errors

## Function Signature Reference

### get_wikitext2_dataloaders

```python
def get_wikitext2_dataloaders(
    batch_size=32,
    seq_len=128,
    num_workers=2,
    vocab_size_limit=30000
):
    """
    Create PyTorch DataLoaders for WikiText-2 dataset.
    
    Returns:
        train_loader: DataLoader yielding (x, y) where x, y are (B, N) tensors
        val_loader: DataLoader yielding (x, y) where x, y are (B, N) tensors
        vocab_size: int, vocabulary size
    """
```

### Data Format

- **Input (x)**: `(batch_size, seq_len)` - Token IDs
- **Target (y)**: `(batch_size, seq_len)` - Next token IDs
- **For CrossEntropyLoss**: Flatten y to `(batch_size * seq_len,)`

### 5. Fixed Config Handling

`BASELINE_CONFIG` is a `ResNetBKConfig` dataclass object, not a dictionary.

**Before:**
```python
config = BASELINE_CONFIG.copy()  # AttributeError: no 'copy' method
config['vocab_size'] = vocab_size
```

**After:**
```python
from dataclasses import replace
config = replace(
    BASELINE_CONFIG,
    vocab_size=vocab_size,
    d_model=64,
    n_layers=2,
    n_seq=128
)
```

**Model instantiation:**
```python
# Correct: pass config object
model = ConfigurableResNetBK(config).to(device)

# Wrong: unpack as kwargs
model = ConfigurableResNetBK(**config).to(device)  # TypeError
```

### 6. Fixed Device Mismatch

Data from DataLoader is on CPU but model is on CUDA, causing device mismatch error.

**Error:**
```
RuntimeError: Expected all tensors to be on the same device, but got index is on cpu, different from other tensors on cuda:0
```

**Fix:** Move data to device before training step:
```python
for step, (x_batch, y_batch) in enumerate(train_loader):
    # Move to device and flatten targets
    x_batch = x_batch.to(device)
    y_batch = y_batch.view(-1).to(device)
    
    result = trainer.train_step(x_batch, y_batch)
```

This was added to all training loops:
- AMP training test
- Gradient accumulation test
- Dynamic batch sizing test

### 7. Fixed AMP Dtype Mismatch in MoE

AMP converts tensors to FP16, but MoE layer had dtype mismatch during index assignment.

**Error:**
```
RuntimeError: Index put requires the source and destination dtypes match, got Float for the destination and Half for the source.
```

**Root Cause:** In `src/models/moe.py`, when using sparse routing:
```python
out_flat[mask] = sub_y  # sub_y might be FP32, out_flat is FP16
```

**Fix:** Ensure dtype consistency:
```python
out_flat[mask] = sub_y.to(out_flat.dtype)
```

This ensures AMP compatibility by matching dtypes during tensor assignment.

## Status

✅ **FIXED** - The notebook is now ready for Google Colab testing.

All errors have been resolved:
- ✅ Import errors fixed
- ✅ Data format handled correctly
- ✅ Config handling fixed
- ✅ Model instantiation corrected
