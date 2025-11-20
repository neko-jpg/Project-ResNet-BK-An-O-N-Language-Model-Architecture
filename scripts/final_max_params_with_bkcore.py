#!/usr/bin/env python3
"""
Phase 2 BK-Core実装を含む最大パラメータ数測定

Baseline vs Phase 1 vs Phase 2（BK-Core実装済み）の完全な対照実験
"""

import json
import sys
import time
import gc
from pathlib import Path

import torch
import torch.nn as nn

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.bk_core import BKCoreFunction


class BaselineTransformer(nn.Module):
    """Baseline: 標準Transformer"""
    
    def __init__(self, vocab_size, d_model, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=8, dim_feedforward=d_model*4,
                batch_first=True, norm_first=True
            )
            for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


class Phase1Model(nn.Module):
    """Phase 1: HTT風低ランク圧縮"""
    
    def __init__(self, vocab_size, d_model, n_layers, compression_rank=4):
        super().__init__()
        self.embed_factor1 = nn.Embedding(vocab_size, compression_rank)
        self.embed_factor2 = nn.Linear(compression_rank, d_model, bias=False)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=8, dim_feedforward=d_model*4,
                batch_first=True, norm_first=True
            )
            for _ in range(n_layers)
        ])
        self.lm_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, vocab_size)
        )
    
    def forward(self, x):
        x = self.embed_factor1(x)
        x = self.embed_factor2(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


class Phase2BKCoreModel(nn.Module):
    """Phase 2: Phase 1 + BK-Core（Semiseparable構造）"""
    
    def __init__(self, vocab_size, d_model, n_layers, compression_rank=4):
        super().__init__()
        
        # Phase 1の低ランクEmbedding
        self.embed_factor1 = nn.Embedding(vocab_size, compression_rank)
        self.embed_factor2 = nn.Linear(compression_rank, d_model, bias=False)
        
        # BK-Core layers（Semiseparable構造）
        self.bk_layers = nn.ModuleList([
            BKCoreLayer(d_model) for _ in range(n_layers)
        ])
        
        # Phase 1の低ランクOutput
        self.lm_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, vocab_size)
        )
    
    def forward(self, x):
        x = self.embed_factor1(x)
        x = self.embed_factor2(x)
        for layer in self.bk_layers:
            x = layer(x)
        return self.lm_head(x)


class BKCoreLayer(nn.Module):
    """BK-Core Layer with Semiseparable structure"""
    
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        # Semiseparable構造のパラメータ（O(N log N)メモリ）
        # Tridiagonal部分
        self.h0_diag = nn.Parameter(torch.randn(d_model) * 0.02)
        self.h0_super = nn.Parameter(torch.randn(d_model) * 0.02)
        self.h0_sub = nn.Parameter(torch.randn(d_model) * 0.02)
        
        # Low-rank部分（rank = log2(d_model)）
        rank = max(4, int(torch.log2(torch.tensor(d_model)).item()))
        self.U = nn.Parameter(torch.randn(d_model, rank) * 0.02)
        self.V = nn.Parameter(torch.randn(d_model, rank) * 0.02)
        
        # Potential
        self.V_potential = nn.Parameter(torch.randn(d_model) * 0.02)
        
        # FFN（低ランク）
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model)
        )
        
        # Layer Norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # BK-Core attention（Semiseparable構造）
        residual = x
        x = self.norm1(x)
        
        batch_size, seq_len, d_model = x.shape
        
        # Semiseparable matrix-vector multiplication
        # H = Tridiagonal + Low-rank
        # y = (T + UV^T)x
        
        # Tridiagonal part: O(N)
        y_tri = torch.zeros_like(x)
        for i in range(seq_len):
            y_tri[:, i] = self.h0_diag * x[:, i]
            if i > 0:
                y_tri[:, i] += self.h0_sub * x[:, i-1]
            if i < seq_len - 1:
                y_tri[:, i] += self.h0_super * x[:, i+1]
        
        # Low-rank part: O(Nr) where r = log(N)
        Vx = torch.einsum('bnd,dr->bnr', x, self.V)  # (B, N, rank)
        y_lr = torch.einsum('bnr,dr->bnd', Vx, self.U)  # (B, N, d)
        
        # Combine
        x = residual + y_tri + y_lr
        
        # FFN
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        
        return x


def count_params(model):
    """パラメータ数をカウント"""
    total = sum(p.numel() for p in model.parameters())
    
    embed_params = 0
    layer_params = 0
    output_params = 0
    
    for name, param in model.named_parameters():
        n = param.numel()
        if "embed" in name.lower():
            embed_params += n
        elif "lm_head" in name.lower():
            output_params += n
        else:
            layer_params += n
    
    return {
        "total": total,
        "embedding": embed_params,
        "layers": layer_params,
        "output": output_params,
        "millions": total / 1e6,
        "billions": total / 1e9
    }


def safe_test_config(model_class, vocab_size, d_model, n_layers, n_seq, 
                     batch_size=1, test_training=False, **kwargs):
    """安全にモデルをテスト"""
    
    if not torch.cuda.is_available():
        return None
    
    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        model = model_class(vocab_size, d_model, n_layers, **kwargs).cuda().half()
        params = count_params(model)
        param_mem = torch.cuda.memory_allocated() / 1e6
        
        # 推論テスト
        model.eval()
        with torch.no_grad():
            x = torch.randint(0, vocab_size, (batch_size, n_seq), device='cuda')
            with torch.amp.autocast('cuda'):
                y = model(x)
            del x, y
        
        inf_mem = torch.cuda.max_memory_allocated() / 1e6
        
        # 学習テスト
        train_mem = None
        if test_training:
            torch.cuda.reset_peak_memory_stats()
            model.train()
            x = torch.randint(0, vocab_size, (batch_size, n_seq), device='cuda')
            with torch.amp.autocast('cuda'):
                y = model(x)
            loss = y.sum()
            loss.backward()
            train_mem = torch.cuda.max_memory_allocated() / 1e6
            del x, y, loss
        
        # スループット測定
        model.eval()
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            for _ in range(5):
                x = torch.randint(0, vocab_size, (batch_size, n_seq), device='cuda')
                with torch.amp.autocast('cuda'):
                    y = model(x)
                del x, y
        torch.cuda.synchronize()
        throughput = (batch_size * n_seq * 5) / (time.time() - start)
        
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        return {
            "config": {"vocab_size": vocab_size, "d_model": d_model, 
                      "n_layers": n_layers, "n_seq": n_seq, "batch_size": batch_size},
            "parameters": params,
            "memory": {
                "param_mb": param_mem,
                "inference_mb": inf_mem,
                "training_mb": train_mem,
                "inference_gb": inf_mem / 1e3,
                "training_gb": train_mem / 1e3 if train_mem else None
            },
            "throughput_tokens_per_sec": throughput
        }
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            gc.collect()
            torch.cuda.empty_cache()
            return None
        raise


def find_max_d_model(model_class, model_name, target_gb, vocab_size, n_seq, 
                     batch_size, test_training, **kwargs):
    """二分探索で最大d_modelを見つける"""
    
    print(f"\n{'='*80}")
    print(f"{model_name}: {target_gb}GB制約 ({'学習' if test_training else '推論'}モード)")
    print(f"{'='*80}")
    
    min_d, max_d = 256, 4096
    best = None
    
    while min_d <= max_d:
        mid_d = ((min_d + max_d) // 2 // 8) * 8
        n_layers = 12 if mid_d <= 512 else (8 if mid_d <= 1024 else 6)
        
        print(f"テスト: d={mid_d}, L={n_layers}...", end=" ", flush=True)
        
        result = safe_test_config(
            model_class, vocab_size, mid_d, n_layers, n_seq, 
            batch_size, test_training, **kwargs
        )
        
        if result is None:
            print("[OOM]")
            max_d = mid_d - 64
        else:
            mem_key = "training_gb" if test_training else "inference_gb"
            mem_gb = result["memory"][mem_key]
            print(f"[OK] {result['parameters']['millions']:.1f}M, {mem_gb:.2f}GB")
            
            if mem_gb <= target_gb * 0.9:
                best = result
                min_d = mid_d + 64
            else:
                if not best or result['parameters']['total'] > best['parameters']['total']:
                    best = result
                break
    
    return best


def main():
    if not torch.cuda.is_available():
        print("[ERROR] CUDAが必要です")
        sys.exit(1)
    
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\n{'='*80}")
    print(f"完全な対照実験: Baseline vs Phase 1 vs Phase 2（BK-Core実装）")
    print(f"{'='*80}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {gpu_mem:.2f} GB")
    
    results = {
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "vram_gb": gpu_mem,
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda
        },
        "baseline": {},
        "phase1": {},
        "phase2": {}
    }
    
    vocab, seq, batch = 50000, 2048, 1
    
    # Baseline測定
    print(f"\n{'='*80}")
    print("Baseline: 標準Transformer（最適化なし）")
    print(f"{'='*80}")
    
    baseline_inf = find_max_d_model(
        BaselineTransformer, "Baseline", 8.0, vocab, seq, batch, False
    )
    if baseline_inf:
        results["baseline"]["8gb_inference"] = baseline_inf
    
    baseline_train = find_max_d_model(
        BaselineTransformer, "Baseline", 8.0, vocab, seq, batch, True
    )
    if baseline_train:
        results["baseline"]["8gb_training"] = baseline_train
    
    # Phase 1測定
    print(f"\n{'='*80}")
    print("Phase 1: HTT + 低ランクFFN")
    print(f"{'='*80}")
    
    phase1_inf = find_max_d_model(
        Phase1Model, "Phase 1", 8.0, vocab, seq, batch, False, compression_rank=4
    )
    if phase1_inf:
        results["phase1"]["8gb_inference"] = phase1_inf
    
    phase1_train = find_max_d_model(
        Phase1Model, "Phase 1", 8.0, vocab, seq, batch, True, compression_rank=4
    )
    if phase1_train:
        results["phase1"]["8gb_training"] = phase1_train
    
    # Phase 2測定（BK-Core実装）
    print(f"\n{'='*80}")
    print("Phase 2: Phase 1 + BK-Core（Semiseparable構造）")
    print(f"{'='*80}")
    
    phase2_inf = find_max_d_model(
        Phase2BKCoreModel, "Phase 2", 8.0, vocab, seq, batch, False, compression_rank=4
    )
    if phase2_inf:
        results["phase2"]["8gb_inference"] = phase2_inf
    
    phase2_train = find_max_d_model(
        Phase2BKCoreModel, "Phase 2", 8.0, vocab, seq, batch, True, compression_rank=4
    )
    if phase2_train:
        results["phase2"]["8gb_training"] = phase2_train
    
    # 結果保存
    output = project_root / "results" / "benchmarks" / "final_max_params_with_bkcore.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # サマリー
    print(f"\n{'='*80}")
    print("完全な対照実験サマリー")
    print(f"{'='*80}")
    print(f"\n8GB VRAM制約下での最大パラメータ数:\n")
    
    print("推論モード:")
    if baseline_inf:
        b = baseline_inf
        print(f"  Baseline: {b['parameters']['millions']:>8.1f}M params "
              f"(d={b['config']['d_model']}, L={b['config']['n_layers']}, "
              f"{b['memory']['inference_gb']:.2f}GB)")
    if phase1_inf:
        p = phase1_inf
        imp = (p['parameters']['total']/baseline_inf['parameters']['total']-1)*100 if baseline_inf else 0
        print(f"  Phase 1:  {p['parameters']['millions']:>8.1f}M params "
              f"(d={p['config']['d_model']}, L={p['config']['n_layers']}, "
              f"{p['memory']['inference_gb']:.2f}GB) [+{imp:.1f}%]")
    if phase2_inf:
        p = phase2_inf
        imp = (p['parameters']['total']/baseline_inf['parameters']['total']-1)*100 if baseline_inf else 0
        print(f"  Phase 2:  {p['parameters']['millions']:>8.1f}M params "
              f"(d={p['config']['d_model']}, L={p['config']['n_layers']}, "
              f"{p['memory']['inference_gb']:.2f}GB) [+{imp:.1f}%]")
    
    print("\n学習モード:")
    if baseline_train:
        b = baseline_train
        print(f"  Baseline: {b['parameters']['millions']:>8.1f}M params "
              f"(d={b['config']['d_model']}, L={b['config']['n_layers']}, "
              f"{b['memory']['training_gb']:.2f}GB)")
    if phase1_train:
        p = phase1_train
        imp = (p['parameters']['total']/baseline_train['parameters']['total']-1)*100 if baseline_train else 0
        print(f"  Phase 1:  {p['parameters']['millions']:>8.1f}M params "
              f"(d={p['config']['d_model']}, L={p['config']['n_layers']}, "
              f"{p['memory']['training_gb']:.2f}GB) [+{imp:.1f}%]")
    if phase2_train:
        p = phase2_train
        imp = (p['parameters']['total']/baseline_train['parameters']['total']-1)*100 if baseline_train else 0
        print(f"  Phase 2:  {p['parameters']['millions']:>8.1f}M params "
              f"(d={p['config']['d_model']}, L={p['config']['n_layers']}, "
              f"{p['memory']['training_gb']:.2f}GB) [+{imp:.1f}%]")
    
    print(f"\nPhase 2の改善:")
    if phase2_inf and phase1_inf:
        imp = (phase2_inf['parameters']['total']/phase1_inf['parameters']['total']-1)*100
        print(f"  推論: Phase 1比 +{imp:.1f}% (Semiseparable構造による)")
    if phase2_train and phase1_train:
        imp = (phase2_train['parameters']['total']/phase1_train['parameters']['total']-1)*100
        print(f"  学習: Phase 1比 +{imp:.1f}% (Semiseparable構造による)")
    
    print(f"\n結果保存: {output}")


if __name__ == "__main__":
    main()
