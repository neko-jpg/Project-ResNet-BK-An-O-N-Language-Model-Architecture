#!/usr/bin/env python3
"""
MUSE Model Merger (Creative Evolution)
Synthesize new models by merging two existing checkpoints.
"""

import torch
import argparse
import copy
import os

def merge_models(path_a, path_b, output_path, alpha=0.5):
    print(f"🧬 Merging Models...")
    print(f"   Model A: {path_a}")
    print(f"   Model B: {path_b}")
    print(f"   Alpha: {alpha} (A) / {1-alpha} (B)")

    try:
        ckpt_a = torch.load(path_a, map_location='cpu')
        ckpt_b = torch.load(path_b, map_location='cpu')
    except Exception as e:
        print(f"❌ Error loading checkpoints: {e}")
        return

    # Check config compatibility
    # We compare critical keys in config dict/object
    conf_a = ckpt_a.get('config', {})
    conf_b = ckpt_b.get('config', {})

    # If config is object, convert to dict roughly
    if hasattr(conf_a, '__dict__'): conf_a = vars(conf_a)
    if hasattr(conf_b, '__dict__'): conf_b = vars(conf_b)

    keys_to_check = ['d_model', 'n_layers', 'n_heads', 'n_seq']
    for k in keys_to_check:
        val_a = conf_a.get(k)
        val_b = conf_b.get(k)
        if val_a != val_b:
            print(f"❌ Configuration mismatch: {k} (A={val_a} vs B={val_b})")
            print("   Cannot merge models with different architectures.")
            return

    # Merge State Dicts
    state_a = ckpt_a['model_state_dict']
    state_b = ckpt_b['model_state_dict']

    state_new = copy.deepcopy(state_a)

    with torch.no_grad():
        for key in state_a:
            if key in state_b:
                # Interpolate
                # theta = alpha * A + (1-alpha) * B
                theta_a = state_a[key]
                theta_b = state_b[key]

                # Ensure shapes match
                if theta_a.shape != theta_b.shape:
                    print(f"⚠️ Shape mismatch for {key}, skipping merge for this layer.")
                    continue

                state_new[key] = alpha * theta_a + (1 - alpha) * theta_b
            else:
                print(f"⚠️ Key {key} not found in Model B, keeping A.")

    # Create new checkpoint
    ckpt_new = {
        'model_state_dict': state_new,
        'config': ckpt_a['config'], # Inherit config from A
        'args': ckpt_a.get('args', {}),
        'merged_from': [path_a, path_b],
        'merge_alpha': alpha
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(ckpt_new, output_path)
    print(f"✅ Merged model saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge two ResNet-BK models")
    parser.add_argument('--model_a', type=str, required=True, help='Path to first checkpoint')
    parser.add_argument('--model_b', type=str, required=True, help='Path to second checkpoint')
    parser.add_argument('--output', type=str, default='checkpoints/merged_model.pt', help='Output path')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for Model A (0.0 to 1.0)')

    args = parser.parse_args()

    merge_models(args.model_a, args.model_b, args.output, args.alpha)

if __name__ == "__main__":
    main()
