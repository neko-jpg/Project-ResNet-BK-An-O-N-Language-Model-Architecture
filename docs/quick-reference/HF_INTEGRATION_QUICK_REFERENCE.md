# Hugging Face Integration Quick Reference

## Quick Start

### Install Dependencies
```bash
pip install transformers torch
pip install onnx onnxruntime  # For ONNX export
```

### Load Pre-trained Model
```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("resnet-bk/resnet-bk-1b")
```

### Load via PyTorch Hub
```python
import torch

model = torch.hub.load('resnet-bk/resnet-bk', 'resnet_bk_1b', pretrained=True)
```

### Create Custom Model
```python
from src.models.hf_resnet_bk import create_resnet_bk_for_hf

model = create_resnet_bk_for_hf(
    "1B",
    use_birman_schwinger=True,
    use_scattering_router=True,
    use_prime_bump=True,
)
```

## Export Models

### ONNX Export
```python
from src.models.onnx_export import export_to_onnx

export_to_onnx(model, "model.onnx", verify=True)
```

### TensorRT Export
```python
from src.models.onnx_export import export_to_tensorrt

export_to_tensorrt("model.onnx", "model.trt", fp16=True)
```

### Full Deployment Pipeline
```python
from src.models.onnx_export import export_model_for_deployment

paths = export_model_for_deployment(
    model,
    output_dir="exports/",
    export_onnx=True,
    export_tensorrt=True,
)
```

## Upload to Hub

### Single Model
```bash
python scripts/upload_to_hf_hub.py \
    --model_path checkpoints/model.pt \
    --repo_id username/resnet-bk-1b \
    --model_size 1B \
    --token YOUR_TOKEN
```

### Batch Upload
```bash
python scripts/upload_to_hf_hub.py \
    --batch \
    --checkpoint_dir checkpoints/ \
    --repo_prefix username/resnet-bk \
    --token YOUR_TOKEN
```

## Model Sizes

| Size | Parameters | d_model | n_layers | experts |
|------|-----------|---------|----------|---------|
| 1M   | ~8M       | 128     | 4        | 2       |
| 10M  | ~22M      | 256     | 6        | 4       |
| 100M | ~100M     | 512     | 12       | 8       |
| 1B   | ~1B       | 1024    | 24       | 16      |
| 10B  | ~10B      | 2048    | 32       | 32      |

## Configuration Options

```python
from src.models.hf_resnet_bk import ResNetBKConfig

config = ResNetBKConfig(
    # Basic
    vocab_size=30000,
    d_model=256,
    n_layers=8,
    n_seq=2048,
    
    # MoE
    num_experts=4,
    top_k=1,
    
    # Birman-Schwinger
    use_birman_schwinger=True,
    epsilon=1.0,
    use_mourre=True,
    use_lap=True,
    
    # Scattering Router
    use_scattering_router=True,
    scattering_scale=0.1,
    
    # Prime-Bump
    use_prime_bump=True,
    prime_bump_scale=0.02,
    k_max=3,
    
    # Advanced
    use_semiseparable=False,
    use_act=False,
)
```

## Files Created

- `src/models/hf_resnet_bk.py` - HF integration
- `hubconf.py` - PyTorch Hub config
- `src/models/onnx_export.py` - ONNX/TensorRT export
- `scripts/upload_to_hf_hub.py` - Hub upload utilities
- `examples/hf_integration_demo.py` - Demo script
- `tests/test_hf_integration.py` - Test suite
- `docs/HUGGINGFACE_INTEGRATION.md` - Full documentation

## See Also

- Full documentation: `docs/HUGGINGFACE_INTEGRATION.md`
- Completion report: `TASK_26_HF_INTEGRATION_COMPLETION.md`
- Run demos: `python examples/hf_integration_demo.py`
- Run tests: `python tests/test_hf_integration.py`
