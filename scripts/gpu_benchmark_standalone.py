#!/usr/bin/env python3
"""
スタンドアロン GPU ベンチマーク (8GB VRAM)
外部依存なしで最大パラメータ数を測定
"""
import gc
import json
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from datetime import datetime
from pathlib import Path

def get_gpu_info():
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "total_vram_gb": props.total_memory / 1024**3,
        "compute_capability": f"{props.major}.{props.minor}",
    }

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
    """低ランク線形層 (パラメータ圧縮)"""
    def __init__(self, in_f, out_f, rank):
        super().__init__()
        self.down = nn.Linear(in_f, rank, bias=False)
        self.up = nn.Linear(rank, out_f, bias=True)
    
    def forward(self, x):
        return self.up(self.down(x))


class EfficientBlock(nn.Module):
    """効率的Transformerブロック"""
    def __init__(self, d_model, ffn_rank=None):
        super().__init__()
        if ffn_rank is None:
            ffn_rank = max(d_model // 8, 32)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 低ランクFFN
        self.ffn_down = LowRankLinear(d_model, d_model * 4, ffn_rank)
        self.ffn_up = LowRankLinear(d_model * 4, d_model, ffn_rank)
        
        # 簡易SSM
        self.ssm_proj = nn.Linear(d_model, d_model)
        self.ssm_gate = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        # SSM
        h = self.norm1(x)
        gate = torch.sigmoid(self.ssm_gate(h))
        h = self.ssm_proj(h) * gate
        x = x + h
        
        # FFN
        h = self.norm2(x)
        h = torch.nn.functional.gelu(self.ffn_down(h))
        h = self.ffn_up(h)
        x = x + h
        
        return x


class ChatModel(nn.Module):
    """チャットAI用効率モデル"""
    def __init__(self, vocab_size, d_model, n_layers, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        self.use_checkpoint = True
        
        # 低ランク埋め込み
        embed_rank = max(d_model // 4, 64)
        self.embed_down = nn.Embedding(vocab_size, embed_rank)
        self.embed_up = nn.Linear(embed_rank, d_model, bias=False)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        
        # ブロック
        self.blocks = nn.ModuleList([EfficientBlock(d_model) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        
        # 低ランク出力
        head_rank = max(d_model // 8, 32)
        self.head_down = nn.Linear(d_model, head_rank, bias=False)
        self.head_up = nn.Linear(head_rank, vocab_size, bias=True)
    
    def forward(self, input_ids):
        B, L = input_ids.shape
        
        x = self.embed_up(self.embed_down(input_ids))
        pos = torch.arange(L, device=input_ids.device)
        x = x + self.pos_embed(pos)
        
        for block in self.blocks:
            if self.training and self.use_checkpoint:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        
        x = self.final_norm(x)
        return self.head_up(self.head_down(x))


def test_model(d_model, n_layers, batch_size, seq_len, vocab_size=32000):
    """モデルテスト"""
    clear_mem()
    try:
        model = ChatModel(vocab_size, d_model, n_layers, seq_len).cuda().half()
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


def find_max(n_layers, batch_size, seq_len, base=128, max_d=2048, step=64):
    """最大d_modelを探索"""
    low, high = base, max_d
    best = None
    
    while low <= high:
        mid = ((low + high) // 2 // step) * step
        if mid < base:
            mid = base
        
        print(f"  d={mid}...", end=" ", flush=True)
        r = test_model(mid, n_layers, batch_size, seq_len)
        
        if r["success"]:
            print(f"✓ {r['params_m']:.1f}M, {r['vram_gb']:.2f}GB")
            best = r
            low = mid + step
        else:
            print(f"✗ {r.get('error', 'OOM')[:30]}")
            high = mid - step
    
    return best


def main():
    print("=" * 60)
    print("8GB VRAM 最大パラメータベンチマーク")
    print("=" * 60)
    
    gpu = get_gpu_info()
    print(f"\nGPU: {gpu.get('name', 'N/A')}")
    print(f"VRAM: {gpu.get('total_vram_gb', 0):.2f} GB")
    
    results = {"timestamp": datetime.now().isoformat(), "gpu": gpu, "tests": []}
    
    configs = [
        (6, 1, 512, "Chat 6層 batch=1 seq=512"),
        (6, 2, 512, "Chat 6層 batch=2 seq=512"),
        (8, 1, 512, "Chat 8層 batch=1 seq=512"),
        (12, 1, 512, "Chat 12層 batch=1 seq=512"),
        (6, 1, 1024, "Long 6層 batch=1 seq=1024"),
        (6, 1, 2048, "VeryLong 6層 batch=1 seq=2048"),
        (6, 4, 256, "HighBatch 6層 batch=4 seq=256"),
    ]
    
    print("\n" + "=" * 60)
    
    for n_layers, batch, seq, desc in configs:
        print(f"\n[{desc}]")
        r = find_max(n_layers, batch, seq)
        if r:
            results["tests"].append({"desc": desc, **r})
            print(f"  → MAX: {r['params_m']:.1f}M @ d={r['d_model']}")
    
    # 推奨設定
    print("\n" + "=" * 60)
    print("推奨設定")
    print("=" * 60)
    
    if results["tests"]:
        best = max(results["tests"], key=lambda x: x["params_m"])
        results["recommended"] = best
        print(f"\n最大構成: {best['desc']}")
        print(f"  d_model: {best['d_model']}")
        print(f"  n_layers: {best['n_layers']}")
        print(f"  パラメータ: {best['params_m']:.1f}M")
        print(f"  VRAM: {best['vram_gb']:.2f}GB")
        
        # チャット用推奨
        chat_tests = [t for t in results["tests"] if "Chat" in t["desc"] and t["batch"] >= 1]
        if chat_tests:
            chat_best = max(chat_tests, key=lambda x: x["params_m"])
            print(f"\nチャットAI推奨:")
            print(f"  d_model: {chat_best['d_model']}")
            print(f"  n_layers: {chat_best['n_layers']}")
            print(f"  batch_size: {chat_best['batch']}")
            print(f"  seq_len: {chat_best['seq']}")
            print(f"  パラメータ: {chat_best['params_m']:.1f}M")
    
    # 保存
    out_path = Path("results/benchmarks/gpu_max_params_8gb.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n保存: {out_path}")
    return results


if __name__ == "__main__":
    main()
