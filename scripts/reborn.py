#!/usr/bin/env python3
"""
MUSE Reborn Ritual (Mystic Evolution)
Resets the model's body while preserving its soul (embeddings).
"""

import torch
import argparse
import os
import sys

# Add src to path
sys.path.append(os.getcwd())

try:
    from src.models.configurable_resnet_bk import ConfigurableResNetBK, ResNetBKConfig
except ImportError:
    print("Error importing model definitions.")
    sys.exit(1)

def reborn(checkpoint_path, output_path):
    print(f"🔁 Starting Reborn Ritual...")
    print(f"   Source (Elder): {checkpoint_path}")

    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"❌ Error loading source: {e}")
        return

    config_data = ckpt.get('config')
    if not config_data:
        print("❌ No config found in checkpoint.")
        return

    # Convert config to object if dict
    if isinstance(config_data, dict):
        config = ResNetBKConfig(**config_data)
    else:
        config = config_data

    # 1. Extract Soul (Embeddings)
    old_state = ckpt['model_state_dict']
    embeddings = None
    embed_key = None

    # Heuristic search for embedding weight
    # Common names: token_embeddings.weight, embeddings.word_embeddings.weight
    for k in old_state.keys():
        if ('token_embedding' in k or 'embed_tokens' in k) and 'weight' in k:
            embed_key = k
            embeddings = old_state[k].clone()
            break

    if embeddings is None:
        print("❌ Could not find Embedding Soul (token_embeddings).")
        return

    print(f"   ✨ Extracted Soul: {embed_key} {embeddings.shape}")

    # 2. Create New Body (Random Init)
    print("   🧬 Constructing new vessel...")
    new_model_wrapper = ConfigurableResNetBK(config)
    new_state = new_model_wrapper.model.state_dict()

    # 3. Transmigrate Soul
    if embed_key in new_state:
        new_state[embed_key] = embeddings
        print("   ✨ Soul Transmigration Complete.")
    else:
        print(f"⚠️ Architecture mismatch? New model has keys: {list(new_state.keys())[:5]}...")
        return

    # 4. Save
    new_ckpt = {
        'model_state_dict': new_state,
        'config': config_data,
        'args': ckpt.get('args', {}),
        'reborn_from': checkpoint_path,
        'epoch': 0,
        'step': 0,
        'optimizer_state_dict': None, # Reset optimizer
        'training_info': {'reborn': True}
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(new_ckpt, output_path)
    print(f"👶 Reborn Model saved to: {output_path}")
    print("   Ready for new curriculum.")

def main():
    parser = argparse.ArgumentParser(description="MUSE Reborn Ritual")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to elder model')
    parser.add_argument('--output', type=str, default='checkpoints/reborn_muse.pt', help='Output path for new model')

    args = parser.parse_args()

    reborn(args.checkpoint, args.output)

if __name__ == "__main__":
    main()
