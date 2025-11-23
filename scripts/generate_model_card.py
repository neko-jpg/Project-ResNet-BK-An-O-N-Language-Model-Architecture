import argparse
import json
import os
import yaml
from datetime import datetime

TEMPLATE = """---
language:
- en
- ja
license: mit
tags:
- pytorch
- resnet-bk
- moe
- physics-inspired
datasets:
- cosmopedia
- wikitext
metrics:
- perplexity
model-index:
- name: {model_name}
  results: []
---

# Model Card for {model_name}

## Model Details

### Model Description

**ResNet-BK** is a O(N) complexity language model based on Birman-Schwinger operator theory.

- **Developed by:** Project MUSE
- **Model type:** Language Model (Physics-Informed)
- **Language(s):** English, Japanese
- **License:** MIT
- **Parameters:** {params} (Approx)
- **Architecture:**
  - d_model: {d_model}
  - n_layers: {n_layers}
  - context_length: {seq_length}

## Uses

Experimental research in physics-inspired AI.

## Training

- **Training Hardware:** NVIDIA RTX 3080 (10GB) / Consumer Hardware
- **Training Data:** Cosmopedia, WikiText, Code
- **Speed:** {speed} tokens/sec (Approx)

## Bias, Risks, and Limitations

This model is experimental. It may produce hallucinations or biased content.

## Citation

```bibtex
@article{{resnet_bk_2025,
  title={{ResNet-BK: A Memory-Efficient Language Model Based on Birman-Schwinger Operator Theory}},
  author={{Arai, Teppei}},
  year={{2025}}
}}
```

*Card generated automatically on {date}*
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to config yaml")
    parser.add_argument("--metrics", help="Path to metrics jsonl")
    parser.add_argument("--output", default="MODEL_CARD.md")
    parser.add_argument("--name", default="ResNet-BK-Experimental")
    args = parser.parse_args()

    d_model = "Unknown"
    n_layers = "Unknown"
    seq_length = "Unknown"

    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                cfg = yaml.safe_load(f)
                if cfg:
                    d_model = cfg.get('d_model', d_model)
                    n_layers = cfg.get('n_layers', n_layers)
                    seq_length = cfg.get('max_seq_len', cfg.get('n_seq', seq_length))
        except Exception as e:
            print(f"Error reading config: {e}")

    speed = "Unknown"
    if args.metrics and os.path.exists(args.metrics):
        # Read last line
        try:
            with open(args.metrics, 'r') as f:
                lines = f.readlines()
                if lines:
                    last = json.loads(lines[-1])
                    speed_val = last.get('speed', 0)
                    if isinstance(speed_val, (int, float)):
                        speed = f"{speed_val:.2f}"
                    else:
                        speed = str(speed_val)
        except Exception as e:
            print(f"Error reading metrics: {e}")

    params = "Unknown" # Requires model load to be accurate, placeholder

    content = TEMPLATE.format(
        model_name=args.name,
        d_model=d_model,
        n_layers=n_layers,
        seq_length=seq_length,
        speed=speed,
        params=params,
        date=datetime.now().strftime("%Y-%m-%d")
    )

    with open(args.output, 'w') as f:
        f.write(content)

    print(f"Model card generated: {args.output}")

if __name__ == "__main__":
    main()
