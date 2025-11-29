#!/usr/bin/env python3
"""
Phase 8 Forward Pass デバッグ
実際の訓練と同じ条件でforward passをテストする
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn

from scripts.train_phase8 import Phase8Config, Phase8Model

def debug_forward():
    """Forward passをステップバイステップでデバッグ"""
    print("="*60)
    print("Phase 8 Forward Pass Debug")
    print("="*60)
    
    # 設定（train_phase8.pyと同じ）
    config = Phase8Config(
        d_model=1024,
        n_layers=24,
        num_heads=16,
        max_seq_len=512,
        curvature=0.01,
        use_hyperbolic_ssm=False
    )
    
    # モデル作成
    model = Phase8Model(config)
    model.train()  # eval() -> train() に変更
    print("Model in TRAIN mode")
    
    # 入力データ（訓練と同じ：ランダムなtoken IDs）
    B, T = 2, 512
    input_ids = torch.randint(0, config.vocab_size, (B, T))
    labels = input_ids.clone()
    
    print(f"\nInput shape: {input_ids.shape}")
    print(f"Input range: [{input_ids.min().item()}, {input_ids.max().item()}]")
    
    # ステップバイステップでforward
    with torch.no_grad():
        # 1. Embedding
        print("\n1. Embedding...")
        x = model.embed_high(model.embed_low(input_ids))
        print(f"   After embedding: shape={x.shape}, min={x.min().item():.4f}, max={x.max().item():.4f}")
        if torch.isnan(x).any():
            print("   ❌ NaN detected in embedding!")
            return False
        
        # 2. Positional Encoding
        print("\n2. Positional Encoding...")
        x = x + model.pos_embed[:, :T, :]
        print(f"   After pos_embed: shape={x.shape}, min={x.min().item():.4f}, max={x.max().item():.4f}")
        if torch.isnan(x).any():
            print("   ❌ NaN detected after positional encoding!")
            return False
        
        # 3. Transformer Layers
        print("\n3. Transformer Layers...")
        for i, layer in enumerate(model.layers):
            x_before = x.clone()
            x = layer(x)
            print(f"   Layer {i}: min={x.min().item():.4f}, max={x.max().item():.4f}, " +
                  f"mean={x.mean().item():.4f}, std={x.std().item():.4f}")
            
            if torch.isnan(x).any():
                print(f"   ❌ NaN detected in layer {i}!")
                print(f"      Input to layer: min={x_before.min().item():.4f}, max={x_before.max().item():.4f}")
                
                # レイヤー内部をチェック
                print("\n   Debugging layer internals...")
                x_test = x_before
                
                # LayerNorm
                x_ln = layer.ln1(x_test)
                print(f"      After LN1: min={x_ln.min().item():.4f}, max={x_ln.max().item():.4f}")
                if torch.isnan(x_ln).any():
                    print("      ❌ NaN in LayerNorm1!")
                    return False
                
                # Attention
                try:
                    attn_out = layer.attn(x_ln)
                    if isinstance(attn_out, tuple):
                        attn_out = attn_out[0]
                    print(f"      After Attention: min={attn_out.min().item():.4f}, max={attn_out.max().item():.4f}")
                    if torch.isnan(attn_out).any():
                        print("      ❌ NaN in Attention!")
                        return False
                except Exception as e:
                    print(f"      ❌ Error in Attention: {e}")
                    return False
                
                return False
        
        # 4. Final LayerNorm
        print("\n4. Final LayerNorm...")
        x = model.ln_f(x)
        print(f"   After ln_f: min={x.min().item():.4f}, max={x.max().item():.4f}")
        if torch.isnan(x).any():
            print("   ❌ NaN detected in final LayerNorm!")
            return False
        
        # 5. LM Head
        print("\n5. LM Head...")
        logits = model.lm_head(x)
        print(f"   Logits: shape={logits.shape}, min={logits.min().item():.4f}, max={logits.max().item():.4f}")
        if torch.isnan(logits).any():
            print("   ❌ NaN detected in logits!")
            return False
        
        # 6. Loss
        print("\n6. Computing Loss...")
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
        print(f"   Loss: {loss.item():.4f}")
        if torch.isnan(loss):
            print("   ❌ NaN detected in loss!")
            return False
    
    print("\n✓ Forward pass completed successfully!")
    return True

if __name__ == "__main__":
    success = debug_forward()
    sys.exit(0 if success else 1)
