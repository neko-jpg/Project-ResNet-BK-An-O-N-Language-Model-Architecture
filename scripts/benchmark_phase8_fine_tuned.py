#!/usr/bin/env python3
"""
Phase 8 Fine-Tuned Benchmark
8GB VRAM制限内で最適なチャットAI構成を細かく探索
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import json
import time
from dataclasses import dataclass, asdict

try:
    from src.models.phase8.hyperbolic_ssm import HyperbolicSSM, HyperbolicSSMConfig
    from src.models.phase8.linear_attention import TangentSpaceLinearAttention, LinearAttentionConfig
    PHASE8_AVAILABLE = True
except:
    PHASE8_AVAILABLE = False
    HyperbolicSSMConfig = None
    LinearAttentionConfig = None


@dataclass
class ModelConfig:
    d_model: int
    n_layers: int
    num_heads: int
    seq_len: int = 512
    batch_size: int = 1
    vocab_size: int = 50257
    use_hyperbolic_ssm: bool = True
    use_linear_attention: bool = True
    gradient_checkpointing: bool = True
    mixed_precision: bool = True
    
    def estimate_parameters(self) -> float:
        embed_params = self.vocab_size * self.d_model
        layer_params = 0
        
        if self.use_hyperbolic_ssm:
            layer_params += self.d_model * self.d_model * 3
        if self.use_linear_attention:
            layer_params += self.d_model * self.d_model * 4
        
        ffn_hidden = self.d_model * 4
        layer_params += self.d_model * ffn_hidden * 2
        layer_params += self.d_model * 4
        
        total_params = embed_params + layer_params * self.n_layers
        total_params += self.d_model * self.vocab_size
        
        return total_params / 1_000_000


class Phase8Layer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        if config.use_hyperbolic_ssm and PHASE8_AVAILABLE:
            ssm_config = HyperbolicSSMConfig(
                d_model=config.d_model,
                d_state=config.d_model // 4,
                curvature=1.0
            )
            self.ssm = HyperbolicSSM(ssm_config)
        else:
            self.ssm = None
        
        if config.use_linear_attention and PHASE8_AVAILABLE:
            attn_config = LinearAttentionConfig(
                d_model=config.d_model,
                num_heads=config.num_heads,
                curvature=1.0
            )
            self.attn = TangentSpaceLinearAttention(attn_config)
        else:
            self.attn = nn.MultiheadAttention(
                config.d_model, config.num_heads, batch_first=True
            )
        
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(),
            nn.Linear(config.d_model * 4, config.d_model)
        )
        
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ssm is not None:
            ssm_out = self.ssm(self.ln1(x))
            if isinstance(ssm_out, tuple):
                ssm_out = ssm_out[0]
            x = x + ssm_out
        
        residual = x
        x = self.ln1(x)
        if isinstance(self.attn, nn.MultiheadAttention):
            x, _ = self.attn(x, x, x, need_weights=False)
        else:
            attn_out = self.attn(x)
            if isinstance(attn_out, tuple):
                attn_out = attn_out[0]
            x = attn_out
        x = residual + x
        x = x + self.ffn(self.ln2(x))
        return x


class Phase8Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([Phase8Layer(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        for layer in self.layers:
            if self.config.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        x = self.ln_f(x)
        return self.lm_head(x)


def test_config(config: ModelConfig) -> dict:
    if not torch.cuda.is_available():
        return None
    
    device = torch.device('cuda')
    
    try:
        model = Phase8Model(config).to(device)
        if config.mixed_precision:
            model = model.half()
        
        input_ids = torch.randint(0, config.vocab_size, 
                                  (config.batch_size, config.seq_len), device=device)
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        with torch.amp.autocast('cuda', enabled=config.mixed_precision):
            _ = model(input_ids)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        params_m = config.estimate_parameters()
        
        result = {
            'd_model': config.d_model,
            'n_layers': config.n_layers,
            'seq_len': config.seq_len,
            'parameters_billions': round(params_m / 1000, 3),
            'peak_memory_gb': round(peak_memory, 3),
            'status': 'SUCCESS'
        }
        
        del model, input_ids
        torch.cuda.empty_cache()
        return result
        
    except RuntimeError as e:
        if 'out of memory' in str(e):
            torch.cuda.empty_cache()
            return {'status': 'OOM', 'd_model': config.d_model, 'n_layers': config.n_layers}
        else:
            return {'status': 'ERROR'}


def main():
    print("="*60)
    print("Phase 8 Fine-Tuned Benchmark (8GB VRAM)")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    gpu_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\nGPU: {gpu_name}")
    print(f"VRAM: {total_memory:.2f} GB")
    print(f"\nPhase 8 Available: {PHASE8_AVAILABLE}")
    
    # 細かく探索
    configs = []
    
    # d_model=2048で層数を細かく
    print("\n[1] d_model=2048 sweep...")
    for n_layers in [24, 28, 32, 36, 40, 44, 48]:
        configs.append(ModelConfig(d_model=2048, n_layers=n_layers, num_heads=16))
    
    # d_model=2304で試す
    print("[2] d_model=2304 sweep...")
    for n_layers in [24, 28, 32, 36, 40]:
        configs.append(ModelConfig(d_model=2304, n_layers=n_layers, num_heads=18))
    
    # d_model=2560で試す
    print("[3] d_model=2560 sweep...")
    for n_layers in [20, 24, 28, 32, 36]:
        configs.append(ModelConfig(d_model=2560, n_layers=n_layers, num_heads=20))
    
    # d_model=3072で試す
    print("[4] d_model=3072 sweep...")
    for n_layers in [16, 20, 24, 28, 32]:
        configs.append(ModelConfig(d_model=3072, n_layers=n_layers, num_heads=24))
    
    # 長文対応
    print("[5] Long context sweep...")
    for seq_len in [1024, 2048, 4096]:
        configs.append(ModelConfig(d_model=2048, n_layers=24, num_heads=16, seq_len=seq_len))
    
    results = []
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] d={config.d_model}, L={config.n_layers}, seq={config.seq_len}", end=" ")
        result = test_config(config)
        if result:
            results.append(result)
            if result['status'] == 'SUCCESS':
                print(f"OK {result['parameters_billions']:.2f}B, {result['peak_memory_gb']:.2f}GB")
            elif result['status'] == 'OOM':
                print("OOM")
                # OOMしたらそのd_modelでの探索を打ち切り
                if i < len(configs) and configs[i].d_model == config.d_model:
                    print(f"  Skipping remaining d_model={config.d_model} configs...")
                    while i < len(configs) and configs[i].d_model == config.d_model:
                        i += 1
            else:
                print("ERROR")
    
    # 結果保存
    output_file = Path('results/benchmarks/PHASE8_FINE_TUNED_BENCHMARK.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'gpu': gpu_name,
        'total_memory_gb': round(total_memory, 2),
        'phase8_available': PHASE8_AVAILABLE,
        'results': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # サマリー
    successful = [r for r in results if r['status'] == 'SUCCESS']
    if successful:
        successful.sort(key=lambda x: x['parameters_billions'], reverse=True)
        
        print(f"\n{'='*60}")
        print("Results Summary")
        print(f"{'='*60}")
        print(f"\n{'Params (B)':<12} {'VRAM (GB)':<12} {'Config'}")
        print("-" * 50)
        
        for r in successful[:15]:
            print(f"{r['parameters_billions']:<12.2f} {r['peak_memory_gb']:<12.2f} "
                  f"d={r['d_model']}, L={r['n_layers']}, seq={r['seq_len']}")
        
        max_config = successful[0]
        print(f"\n** Maximum Configuration:")
        print(f"   Parameters: {max_config['parameters_billions']:.2f}B")
        print(f"   VRAM: {max_config['peak_memory_gb']:.2f} GB")
        print(f"   Config: d_model={max_config['d_model']}, n_layers={max_config['n_layers']}, seq={max_config['seq_len']}")
        
        phase7_params = 1.83
        phase7_vram = 6.89
        print(f"\n** Phase 7 Comparison:")
        print(f"   Parameters: {phase7_params:.2f}B -> {max_config['parameters_billions']:.2f}B "
              f"({(max_config['parameters_billions']/phase7_params - 1)*100:+.1f}%)")
        print(f"   VRAM: {phase7_vram:.2f}GB -> {max_config['peak_memory_gb']:.2f}GB "
              f"({(max_config['peak_memory_gb']/phase7_vram - 1)*100:+.1f}%)")
        
        # 8GB以内の最大構成
        within_8gb = [r for r in successful if r['peak_memory_gb'] <= 7.5]
        if within_8gb:
            max_8gb = within_8gb[0]
            print(f"\n** Maximum within 8GB VRAM:")
            print(f"   Parameters: {max_8gb['parameters_billions']:.2f}B")
            print(f"   VRAM: {max_8gb['peak_memory_gb']:.2f} GB")
            print(f"   Config: d_model={max_8gb['d_model']}, n_layers={max_8gb['n_layers']}")
    
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
