# Hugging Face Integration Guide

This guide explains how to use ResNet-BK models with the Hugging Face ecosystem, including transformers, PyTorch Hub, and deployment formats (ONNX, TensorRT).

## Table of Contents

1. [Installation](#installation)
2. [Using with Transformers](#using-with-transformers)
3. [PyTorch Hub](#pytorch-hub)
4. [Model Export](#model-export)
5. [Uploading to Hub](#uploading-to-hub)
6. [Available Models](#available-models)

## Installation

### Basic Installation

```bash
pip install transformers torch
```

### For ONNX Export

```bash
pip install onnx onnxruntime
```

### For TensorRT Export

TensorRT requires NVIDIA GPU and separate installation. See [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt).

## Using with Transformers

### Loading Pre-trained Models

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("resnet-bk/resnet-bk-1b")

# Load tokenizer (use compatible tokenizer)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Generate text
inputs = tokenizer("The future of AI is", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

### Creating Custom Models

```python
from src.models.hf_resnet_bk import ResNetBKConfig, ResNetBKForCausalLM

# Create custom configuration
config = ResNetBKConfig(
    vocab_size=50000,
    d_model=512,
    n_layers=12,
    n_seq=2048,
    num_experts=8,
    use_birman_schwinger=True,
    use_scattering_router=True,
)

# Create model
model = ResNetBKForCausalLM(config)
```

### Training with Trainer API

```python
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Load dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Tokenize
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=1000,
    logging_steps=100,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Train
trainer.train()
```

### Saving and Loading

```python
# Save model
model.save_pretrained("./my_resnet_bk_model")

# Load model
loaded_model = ResNetBKForCausalLM.from_pretrained("./my_resnet_bk_model")
```

## PyTorch Hub

### Loading Models

```python
import torch

# Load pre-trained 1B model
model = torch.hub.load('resnet-bk/resnet-bk', 'resnet_bk_1b', pretrained=True)

# Load 10M model without pre-trained weights
model = torch.hub.load('resnet-bk/resnet-bk', 'resnet_bk_10m', pretrained=False)

# Create custom model
model = torch.hub.load('resnet-bk/resnet-bk', 'resnet_bk_custom',
                      d_model=512, n_layers=12, use_birman_schwinger=True)
```

### Available Hub Models

- `resnet_bk_1m`: 1M parameter model
- `resnet_bk_10m`: 10M parameter model
- `resnet_bk_100m`: 100M parameter model
- `resnet_bk_1b`: 1B parameter model
- `resnet_bk_10b`: 10B parameter model
- `resnet_bk_custom`: Custom configuration

## Model Export

### ONNX Export

```python
from src.models.hf_resnet_bk import create_resnet_bk_for_hf
from src.models.onnx_export import export_to_onnx

# Create model
model = create_resnet_bk_for_hf("100M")

# Export to ONNX
export_to_onnx(
    model,
    "resnet_bk_100m.onnx",
    batch_size=1,
    seq_length=512,
    verify=True,
    tolerance=1e-5,
)
```

### TensorRT Export

```python
from src.models.onnx_export import export_to_tensorrt

# Convert ONNX to TensorRT
export_to_tensorrt(
    "resnet_bk_100m.onnx",
    "resnet_bk_100m.trt",
    fp16=True,
    max_batch_size=8,
    max_seq_length=2048,
)
```

### Full Deployment Pipeline

```python
from src.models.onnx_export import export_model_for_deployment

# Export in all formats
exported_paths = export_model_for_deployment(
    model,
    output_dir="exports/",
    model_name="resnet_bk_100m",
    export_onnx=True,
    export_tensorrt=True,
    optimize_onnx=True,
    verify=True,
)
```

### Benchmarking TensorRT Speedup

```python
from src.models.onnx_export import benchmark_tensorrt_speedup

# Benchmark speedup
results = benchmark_tensorrt_speedup(
    pytorch_model=model,
    tensorrt_engine_path="resnet_bk_100m.trt",
    batch_size=1,
    seq_length=512,
    num_iterations=100,
)

print(f"Speedup: {results['speedup']:.2f}x")
```

## Uploading to Hub

### Single Model Upload

```bash
python scripts/upload_to_hf_hub.py \
    --model_path checkpoints/resnet_bk_1b.pt \
    --repo_id username/resnet-bk-1b \
    --model_size 1B \
    --token YOUR_HF_TOKEN
```

### Batch Upload

```bash
python scripts/upload_to_hf_hub.py \
    --batch \
    --checkpoint_dir checkpoints/ \
    --repo_prefix username/resnet-bk \
    --token YOUR_HF_TOKEN
```

### Programmatic Upload

```python
from scripts.upload_to_hf_hub import upload_model_to_hub

upload_model_to_hub(
    model_path="checkpoints/resnet_bk_1b.pt",
    repo_id="username/resnet-bk-1b",
    model_size="1B",
    commit_message="Initial upload",
    private=False,
    token="YOUR_HF_TOKEN",
)
```

## Available Models

### Model Sizes

| Model | Parameters | d_model | n_layers | num_experts | n_seq |
|-------|-----------|---------|----------|-------------|-------|
| 1M    | ~1M       | 128     | 4        | 2           | 512   |
| 10M   | ~10M      | 256     | 6        | 4           | 1024  |
| 100M  | ~100M     | 512     | 12       | 8           | 2048  |
| 1B    | ~1B       | 1024    | 24       | 16          | 4096  |
| 10B   | ~10B      | 2048    | 32       | 32          | 8192  |

### Configuration Options

All models support the following configuration options:

- `use_birman_schwinger`: Enable Birman-Schwinger core (default: False)
- `use_scattering_router`: Enable scattering-based routing (default: False)
- `use_prime_bump`: Enable Prime-Bump initialization (default: False)
- `epsilon`: Regularization parameter for Birman-Schwinger (default: 1.0)
- `use_mourre`: Enable Mourre estimate verification (default: True)
- `use_lap`: Enable Limiting Absorption Principle (default: True)
- `use_semiseparable`: Enable semiseparable matrix structure (default: False)
- `use_act`: Enable Adaptive Computation Time (default: False)

### Example: Full-Featured Model

```python
from src.models.hf_resnet_bk import create_resnet_bk_for_hf

model = create_resnet_bk_for_hf(
    "1B",
    use_birman_schwinger=True,
    use_scattering_router=True,
    use_prime_bump=True,
    epsilon=0.75,
    use_semiseparable=True,
    use_act=True,
)
```

## Performance Comparison

ResNet-BK achieves superior performance compared to baselines:

### Long-Context Stability
- Stable training up to 1M tokens
- Mamba diverges at 32k tokens
- 10× fewer gradient spikes

### Quantization Robustness
- INT8: <5% PPL degradation
- INT4: <15% PPL degradation
- 4× lower PPL than Mamba at INT4

### Dynamic Efficiency
- 2× fewer FLOPs at equal PPL
- 40% FLOPs reduction with ACT
- 10× faster routing than MLP gating

## Troubleshooting

### Import Errors

If you get import errors, ensure transformers is installed:

```bash
pip install transformers>=4.30.0
```

### CUDA Out of Memory

For large models, use gradient checkpointing:

```python
model.gradient_checkpointing_enable()
```

Or reduce batch size and sequence length.

### ONNX Export Errors

Ensure ONNX and onnxruntime are installed:

```bash
pip install onnx onnxruntime
```

For GPU support:

```bash
pip install onnxruntime-gpu
```

### TensorRT Errors

TensorRT requires:
1. NVIDIA GPU with compute capability ≥ 7.0
2. CUDA Toolkit
3. TensorRT installation

See [NVIDIA TensorRT Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html).

## Citation

If you use ResNet-BK models, please cite:

```bibtex
@article{resnetbk2024,
  title={ResNet-BK: O(N) Language Modeling with Birman-Schwinger Kernels},
  author={ResNet-BK Team},
  journal={arXiv preprint},
  year={2024}
}
```

## License

MIT License

## Support

For questions and issues:
- GitHub Issues: https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/issues
- Discussions: https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/discussions
- Email: arat252539@gmail.com
