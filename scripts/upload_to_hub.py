"""
Upload Mamba-Killer ResNet-BK checkpoints to Hugging Face Hub

Uploads trained models to Hugging Face Hub for easy sharing and reproducibility.

Usage:
    python scripts/upload_to_hub.py --checkpoint ./checkpoints/best.pt --model_name mamba-killer-1b
    python scripts/upload_to_hub.py --checkpoint_dir ./checkpoints --upload_all
"""

import argparse
import os
import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional
import logging

try:
    from huggingface_hub import HfApi, create_repo, upload_file, upload_folder
    HF_AVAILABLE = True
except ImportError:
    print("Error: huggingface_hub not installed.")
    print("Install with: pip install huggingface_hub")
    HF_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HuggingFaceUploader:
    """Handles uploading models to Hugging Face Hub."""
    
    def __init__(
        self,
        repo_id: str,
        token: Optional[str] = None,
        private: bool = False
    ):
        """
        Initialize uploader.
        
        Args:
            repo_id: Repository ID (username/model-name)
            token: Hugging Face API token
            private: Whether to create private repository
        """
        if not HF_AVAILABLE:
            raise ImportError("huggingface_hub is required")
        
        self.repo_id = repo_id
        self.token = token or os.environ.get('HF_TOKEN')
        self.private = private
        
        if not self.token:
            raise ValueError("Hugging Face token required. Set HF_TOKEN environment variable or pass token argument.")
        
        self.api = HfApi(token=self.token)
        
        # Create repository if it doesn't exist
        try:
            create_repo(
                repo_id=self.repo_id,
                token=self.token,
                private=self.private,
                exist_ok=True
            )
            logger.info(f"✓ Repository ready: {self.repo_id}")
        except Exception as e:
            logger.error(f"Error creating repository: {e}")
            raise
    
    def upload_checkpoint(
        self,
        checkpoint_path: str,
        model_name: str,
        config: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        Upload checkpoint to Hub.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model_name: Name for the model (e.g., "mamba-killer-1b")
            config: Model configuration
            metrics: Training metrics
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Uploading checkpoint: {checkpoint_path}")
        
        # Load checkpoint to extract metadata
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        
        # Prepare model card
        model_card = self._create_model_card(
            model_name=model_name,
            config=config or checkpoint_data.get('config', {}),
            metrics=metrics or checkpoint_data.get('metrics', {}),
            checkpoint_data=checkpoint_data
        )
        
        # Create temporary directory for upload
        temp_dir = Path(f"./temp_upload_{model_name}")
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Save model card
            model_card_path = temp_dir / "README.md"
            with open(model_card_path, 'w') as f:
                f.write(model_card)
            
            # Save config
            if config or 'config' in checkpoint_data:
                config_path = temp_dir / "config.json"
                with open(config_path, 'w') as f:
                    json.dump(config or checkpoint_data['config'], f, indent=2)
            
            # Save metrics
            if metrics or 'metrics' in checkpoint_data:
                metrics_path = temp_dir / "metrics.json"
                with open(metrics_path, 'w') as f:
                    json.dump(metrics or checkpoint_data['metrics'], f, indent=2)
            
            # Copy checkpoint
            import shutil
            shutil.copy(checkpoint_path, temp_dir / "pytorch_model.pt")
            
            # Upload folder
            self.api.upload_folder(
                folder_path=str(temp_dir),
                repo_id=self.repo_id,
                token=self.token,
                commit_message=f"Upload {model_name}"
            )
            
            logger.info(f"✓ Uploaded to: https://huggingface.co/{self.repo_id}")
            
        finally:
            # Cleanup
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def _create_model_card(
        self,
        model_name: str,
        config: Dict[str, Any],
        metrics: Dict[str, float],
        checkpoint_data: Dict[str, Any]
    ) -> str:
        """Create model card for Hugging Face Hub."""
        
        # Extract key metrics
        ppl = metrics.get('perplexity', metrics.get('ppl', 'N/A'))
        loss = metrics.get('loss', 'N/A')
        
        # Extract model size
        d_model = config.get('model', {}).get('d_model', 'N/A')
        n_layers = config.get('model', {}).get('n_layers', 'N/A')
        n_seq = config.get('model', {}).get('n_seq', 'N/A')
        
        model_card = f"""---
language: en
license: mit
tags:
- language-modeling
- birman-schwinger
- resnet-bk
- mamba-killer
datasets:
- wikitext
- c4
metrics:
- perplexity
---

# {model_name}

Mamba-Killer ResNet-BK model trained with Birman-Schwinger operator theory.

## Model Description

This model implements the ResNet-BK architecture with:
- **Birman-Schwinger Core**: Mathematically rigorous O(N) operator
- **Prime-Bump Initialization**: Spectral initialization based on prime number distribution
- **Scattering-Based Router**: Parameter-free MoE routing using scattering phase
- **Semiseparable Structure**: Memory-efficient O(N log N) attention

## Model Details

- **Model Size**: {d_model}d × {n_layers} layers
- **Sequence Length**: {n_seq} tokens
- **Parameters**: ~{self._estimate_parameters(config)} million
- **Training**: Trained on WikiText-2/103, C4, and The Pile

## Performance

- **Perplexity**: {ppl}
- **Loss**: {loss}
- **Training Steps**: {checkpoint_data.get('step', 'N/A')}
- **Training Epochs**: {checkpoint_data.get('epoch', 'N/A')}

## Usage

```python
import torch
from huggingface_hub import hf_hub_download

# Download checkpoint
checkpoint_path = hf_hub_download(
    repo_id="{self.repo_id}",
    filename="pytorch_model.pt"
)

# Load model
checkpoint = torch.load(checkpoint_path)
model_state = checkpoint['model_state_dict']

# Use with your ResNet-BK implementation
# model.load_state_dict(model_state)
```

## Training Details

- **Framework**: PyTorch 2.1.0
- **Hardware**: NVIDIA T4 GPU (Google Colab)
- **Optimizer**: AdamW
- **Learning Rate**: {config.get('training', {}).get('learning_rate', 'N/A')}
- **Batch Size**: {config.get('training', {}).get('batch_size', 'N/A')}

## Citation

If you use this model, please cite:

```bibtex
@article{{mamba-killer-2024,
  title={{Mamba-Killer: Ultra-Scale ResNet-BK with Birman-Schwinger Theory}},
  author={{Your Name}},
  journal={{arXiv preprint arXiv:XXXX.XXXXX}},
  year={{2024}}
}}
```

## License

MIT License

## Contact

For questions and feedback, please open an issue on [GitHub](https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture).
"""
        return model_card
    
    def _estimate_parameters(self, config: Dict[str, Any]) -> str:
        """Estimate number of parameters in millions."""
        try:
            model_config = config.get('model', {})
            d_model = model_config.get('d_model', 256)
            n_layers = model_config.get('n_layers', 8)
            vocab_size = model_config.get('vocab_size', 50257)
            
            # Rough estimate
            params = vocab_size * d_model  # Embeddings
            params += n_layers * (4 * d_model * d_model)  # Layers
            params_millions = params / 1e6
            
            return f"{params_millions:.1f}"
        except:
            return "N/A"


def main():
    parser = argparse.ArgumentParser(
        description="Upload Mamba-Killer checkpoints to Hugging Face Hub"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to checkpoint file'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        help='Directory containing checkpoints'
    )
    parser.add_argument(
        '--upload_all',
        action='store_true',
        help='Upload all checkpoints in directory'
    )
    parser.add_argument(
        '--repo_id',
        type=str,
        required=True,
        help='Hugging Face repository ID (username/model-name)'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='mamba-killer',
        help='Model name'
    )
    parser.add_argument(
        '--token',
        type=str,
        help='Hugging Face API token (or set HF_TOKEN env var)'
    )
    parser.add_argument(
        '--private',
        action='store_true',
        help='Create private repository'
    )
    
    args = parser.parse_args()
    
    if not args.checkpoint and not args.checkpoint_dir:
        parser.error("Must specify either --checkpoint or --checkpoint_dir")
    
    # Create uploader
    uploader = HuggingFaceUploader(
        repo_id=args.repo_id,
        token=args.token,
        private=args.private
    )
    
    # Upload checkpoint(s)
    if args.checkpoint:
        uploader.upload_checkpoint(
            checkpoint_path=args.checkpoint,
            model_name=args.model_name
        )
    elif args.upload_all and args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        
        logger.info(f"Found {len(checkpoints)} checkpoints to upload")
        
        for i, checkpoint_path in enumerate(checkpoints):
            model_name = f"{args.model_name}_{checkpoint_path.stem}"
            logger.info(f"Uploading {i+1}/{len(checkpoints)}: {checkpoint_path}")
            
            uploader.upload_checkpoint(
                checkpoint_path=str(checkpoint_path),
                model_name=model_name
            )


if __name__ == '__main__':
    main()
