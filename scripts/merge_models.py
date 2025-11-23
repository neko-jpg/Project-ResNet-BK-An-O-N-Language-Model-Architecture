#!/usr/bin/env python3
"""
MUSE Model Merger (Creative Evolution)
Synthesize new models by merging two existing checkpoints.
Supports:
1. Linear Interpolation (Lerp)
2. Layer-wise Merge (Franken-merge)
3. Task Vector / Trait Addition
"""

import torch
import argparse
import copy
import os

def merge_models(path_a, path_b, output_path, method="lerp", alpha=0.5, split_layer=None):
    print(f"🧬 Merging Models...")
    print(f"   Model A: {path_a}")
    print(f"   Model B: {path_b}")
    print(f"   Method: {method}")

    try:
        ckpt_a = torch.load(path_a, map_location='cpu')
        ckpt_b = torch.load(path_b, map_location='cpu')
    except Exception as e:
        print(f"❌ Error loading checkpoints: {e}")
        return

    # Check config compatibility
    conf_a = ckpt_a.get('config', {})
    conf_b = ckpt_b.get('config', {})
    if hasattr(conf_a, '__dict__'): conf_a = vars(conf_a)
    if hasattr(conf_b, '__dict__'): conf_b = vars(conf_b)

    # Basic check
    if conf_a.get('d_model') != conf_b.get('d_model'):
        print("❌ Architecture mismatch (d_model).")
        return

    state_a = ckpt_a['model_state_dict']
    state_b = ckpt_b['model_state_dict']
    state_new = copy.deepcopy(state_a)

    if method == "lerp":
        print(f"   Mixing: {alpha:.2f} * A + {1-alpha:.2f} * B")
        with torch.no_grad():
            for key in state_a:
                if key in state_b:
                    theta_a = state_a[key]
                    theta_b = state_b[key]
                    if theta_a.shape == theta_b.shape:
                        state_new[key] = alpha * theta_a + (1 - alpha) * theta_b

    elif method == "layer_wise":
        n_layers = conf_a.get('n_layers', 12)
        split = split_layer if split_layer is not None else n_layers // 2
        print(f"   Layer Split at {split}. Layers 0-{split} from A, {split}-{n_layers} from B.")

        with torch.no_grad():
            for key in state_a:
                # Identify layer index from key name
                # e.g., "layers.5.attention..."
                layer_idx = -1
                parts = key.split('.')
                for p in parts:
                    if p.isdigit():
                        layer_idx = int(p)
                        break

                if layer_idx != -1:
                    if layer_idx >= split:
                        # Use B
                        if key in state_b:
                            state_new[key] = state_b[key]
                else:
                    # Non-layer weights (Embeddings, Final Norm)
                    # Use A for input embeddings, B for output head?
                    # Strategy: Embeddings from A, Head from B
                    if "token_embedding" in key:
                        state_new[key] = state_a[key]
                    elif "head" in key or "final_norm" in key:
                        if key in state_b:
                            state_new[key] = state_b[key]

    elif method == "trait_add":
        print(f"   Trait Addition: A + {alpha:.2f} * (B - A)")
        # This assumes B is a finetune of A, or share same base.
        # We add B's deviation to A.
        with torch.no_grad():
            for key in state_a:
                if key in state_b:
                    theta_a = state_a[key]
                    theta_b = state_b[key]
                    if theta_a.shape == theta_b.shape:
                        diff = theta_b - theta_a
                        state_new[key] = theta_a + alpha * diff

    # Save
    ckpt_new = {
        'model_state_dict': state_new,
        'config': ckpt_a['config'],
        'args': ckpt_a.get('args', {}),
        'merged_from': [path_a, path_b],
        'merge_method': method
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(ckpt_new, output_path)
    print(f"✅ Merged model saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Merge two ResNet-BK models")
    parser.add_argument('--model_a', type=str, required=True, help='Path to first checkpoint')
    parser.add_argument('--model_b', type=str, required=True, help='Path to second checkpoint')
    parser.add_argument('--output', type=str, default='checkpoints/merged_model.pt', help='Output path')
    parser.add_argument('--method', type=str, choices=['lerp', 'layer_wise', 'trait_add'], default='lerp')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight')
    parser.add_argument('--split_layer', type=int, default=None, help='Layer index to split at')

    args = parser.parse_args()

    merge_models(args.model_a, args.model_b, args.output, args.method, args.alpha, args.split_layer)

if __name__ == "__main__":
    main()
