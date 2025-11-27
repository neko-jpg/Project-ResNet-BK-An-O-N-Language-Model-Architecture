#!/usr/bin/env python3
"""
8GB VRAM 限界プッシュベンチマーク
VRAMを7GB程度まで使い切る最大パラメータを探索
"""
import gc
import json
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from datetime import datetime
from pathlib import Path

def get_vram():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(0) / 1024**3
    return 0

def clear_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class LowRankLinear(nn.Module):
    def __init__(self, in_f, out_f, rank):
        super().__init__()
        self.down = nn.Linear(in_f, rank, bias=False)
        self.up = nn.Linear(rank, out_f, bias=True)
    
    def forward(self, x):
        return self.up(self.down(x))


class EfficientBlock(nn.Module):
    def __init__(self, d_model, ffn_rank=None):
        super().__init__()
        if ffn_rank is None:
            ffn_rank = max(d_model // 8, 64)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_down = LowRankLinear(d_model, d_model * 4, ffn_rank)
        self.ffn_up = LowRankLinear(d_model * 4, d_model, ffn_rank)
        self.ssm_proj = nn.Linear(d_model, d_model)
        self.ssm_gate = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        h = self.norm1(x)
        gate = torch.sigmoid(self.ssm_gate(h))
        h = self.ssm_proj(h) * gate
        x = x + h
        h = self.norm2(x)
        h = torch.nn.functional.gelu(self.ffn_down(h))
        h = self.ffn_up(h)
        return x + h


class MaxChatModel(nn.Module):
    """最大パラメータチャットモデル"""
    def __init__(self, vocab_size, d_model, n_layers, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        
        # 低ランク埋め込み
        embed_rank = max(d_model // 4, 128)
        self.embed_down = nn.Embedding(vocab_size, embed_rank)
        self.embed_up = nn.Linear(embed_rank, d_model, bias=False)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        
        self.blocks = nn.ModuleList([EfficientBlock(d_model) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        
        head_rank = max(d_model // 8, 64)
        self.head_down = nn.Linear(d_model, head_rank, bias=False)
        self.head_up = nn.Linear(head_rank, vocab_size, bias=True)
    
    def forward(self, input_ids):
        B, L = input_ids.shape
        x = self.embed_up(self.embed_down(input_ids))
        pos = torch.arange(L, device=input_ids.device)
        x = x + self.pos_embed(pos)
        
        for block in self.blocks:
            if self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        
        x = self.final_norm(x)
        return self.head_up(self.head_down(x))


def test_model(d_model, n_layers, batch_size, seq_len, vocab_size=50257):
    clear_mem()
    try:
        model = MaxChatModel(vocab_size, d_model, n_layers, seq_len).cuda().half()
        model.train()
        
        params = sum(p.numel() for p in model.parameters())
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda')
        
        with torch.cuda.amp.autocast(enabled=True):
            out = model(input_ids)
            loss = out.mean()
            loss.backward()
        
        torch.cuda.synchronize()
        vram = get_vram()
        
        del model, out, loss, input_ids
        clear_mem()
        
        return {"success": True, "params_m": params/1e6, "vram_gb": vram, 
                "d_model": d_model, "n_layers": n_layers, "batch": batch_size, "seq": seq_len}
    except Exception as e:
        clear_mem()
        return {"success": False, "error": str(e)[:60]}


def find_max_layers(d_model, batch_size, seq_len, max_layers=48):
    """最大レイヤー数を探索"""
    best = None
    for n_layers in range(4, max_layers + 1, 2):
        print(f"  layers={n_layers}...", end=" ", flush=True)
        r = test_model(d_model, n_layers, batch_size, seq_len)
        
        if r["success"]:
            print(f"✓ {r['params_m']:.1f}M, {r['vram_gb']:.2f}GB")
            best = r
            if r["vram_gb"] > 6.5:  # 7GB近くまで使ったら停止
                break
        else:
            print(f"✗ {r.get('error', 'OOM')[:30]}")
            break
    
    return best


def main():
    print("=" * 60)
    print("8GB VRAM 限界プッシュベンチマーク")
    print("=" * 60)
    
    props = torch.cuda.get_device_properties(0)
    print(f"\nGPU: {props.name}")
    print(f"VRAM: {props.total_memory / 1024**3:.2f} GB")
    
    results = {"timestamp": datetime.now().isoformat(), "tests": []}
    
    # 大きなd_modelで限界を探る
    print("\n" + "=" * 60)
    print("[1] d_model=2048 で最大レイヤー数を探索")
    print("=" * 60)
    r = find_max_layers(2048, 1, 512, max_layers=64)
    if r:
        results["tests"].append({"desc": "d=2048", **r})
    
    print("\n" + "=" * 60)
    print("[2] d_model=3072 で最大レイヤー数を探索")
    print("=" * 60)
    r = find_max_layers(3072, 1, 512, max_layers=48)
    if r:
        results["tests"].append({"desc": "d=3072", **r})
    
    print("\n" + "=" * 60)
    print("[3] d_model=4096 で最大レイヤー数を探索")
    print("=" * 60)
    r = find_max_layers(4096, 1, 512, max_layers=32)
    if r:
        results["tests"].append({"desc": "d=4096", **r})
    
    print("\n" + "=" * 60)
    print("[4] d_model=1536, batch=2 で最大レイヤー数を探索")
    print("=" * 60)
    r = find_max_layers(1536, 2, 512, max_layers=48)
    if r:
        results["tests"].append({"desc": "d=1536 batch=2", **r})
    
    print("\n" + "=" * 60)
    print("[5] d_model=2048, seq=1024 で最大レイヤー数を探索")
    print("=" * 60)
    r = find_max_layers(2048, 1, 1024, max_layers=48)
    if r:
        results["tests"].append({"desc": "d=2048 seq=1024", **r})
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("結果サマリー")
    print("=" * 60)
    
    if results["tests"]:
        best = max(results["tests"], key=lambda x: x["params_m"])
        results["best"] = best
        
        print(f"\n最大パラメータ構成:")
        print(f"  d_model: {best['d_model']}")
        print(f"  n_layers: {best['n_layers']}")
        print(f"  batch_size: {best['batch']}")
        print(f"  seq_len: {best['seq']}")
        print(f"  パラメータ: {best['params_m']:.1f}M ({best['params_m']/1000:.2f}B)")
        print(f"  VRAM使用: {best['vram_gb']:.2f}GB")
        
        # 全結果表示
        print("\n全テスト結果:")
        for t in sorted(results["tests"], key=lambda x: -x["params_m"]):
            print(f"  {t['desc']}: {t['params_m']:.1f}M, d={t['d_model']}, L={t['n_layers']}, VRAM={t['vram_gb']:.2f}GB")
    
    # 保存
    out_path = Path("results/benchmarks/gpu_max_push_8gb.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n保存: {out_path}")
    return results


if __name__ == "__main__":
    main()
