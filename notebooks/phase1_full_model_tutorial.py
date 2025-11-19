# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Phase 1: Complete Model Integration Tutorial
#
# このノートブックでは、Phase 1のすべてのコンポーネントを統合した完全なモデルを探索します。
#
# ## 学習目標
# 1. Phase 1モデルファクトリの使用方法を学ぶ
# 2. プリセット設定を理解する
# 3. パフォーマンス検証を実行する
# 4. 安定性監視を統合する
# 5. エラーハンドリングとリカバリを実装する

# %% [markdown]
# ## セットアップ

# %%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List
import time

# Phase 1コンポーネントのインポート
import sys
sys.path.append('..')
from src.models.phase1 import (
    create_phase1_model,
    Phase1Config,
    get_preset_config,
    BKStabilityMonitor,
)
from src.models.phase1.errors import VRAMExhaustedError, NumericalInstabilityError
from src.models.phase1.recovery import Phase1ErrorRecovery

# プロット設定
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# デバイス設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %% [markdown]
# ## 1. Phase 1設定の作成

# %%
# カスタム設定の作成
custom_config = Phase1Config(
    # AR-SSM設定
    ar_ssm_enabled=True,
    ar_ssm_max_rank=32,
    ar_ssm_min_rank=4,
    ar_ssm_gate_hidden_dim=128,
    ar_ssm_l1_regularization=0.001,
    ar_ssm_use_fused_scan=False,  # CPUでも動作するようにFalse
    
    # HTT Embedding設定
    htt_enabled=True,
    htt_rank=16,
    htt_num_cores=2,
    htt_phase_encoding=True,
    htt_compression_target=0.1,  # 90%圧縮
    
    # LNSカーネル設定（オプション）
    lns_enabled=False,  # 実験的、推論専用
    
    # 安定性監視設定
    stability_monitoring_enabled=True,
    stability_threshold=1e-6,
    schatten_s1_bound=100.0,
    schatten_s2_bound=50.0,
    gradient_norm_threshold=10.0,
    
    # メモリ最適化設定
    use_gradient_checkpointing=True,
    checkpoint_ar_ssm=True,
    checkpoint_htt=False,
    
    # パフォーマンス目標
    target_vram_gb=8.0,
    target_ppl_degradation=0.05,  # 5%最大
    target_speedup=3.0,  # Fused Scanで3倍
)

print("カスタム設定:")
print(f"  AR-SSM: {custom_config.ar_ssm_enabled}")
print(f"  HTT: {custom_config.htt_enabled}")
print(f"  LNS: {custom_config.lns_enabled}")
print(f"  Stability Monitoring: {custom_config.stability_monitoring_enabled}")
print(f"  Target VRAM: {custom_config.target_vram_gb}GB")

# %% [markdown]
# ## 2. プリセット設定の使用

# %%
# 利用可能なプリセット
presets = ['8gb', '10gb', '24gb', 'inference', 'max_efficiency']

print("利用可能なプリセット設定:\n")
for preset_name in presets:
    try:
        preset_config = get_preset_config(preset_name)
        print(f"{preset_name}:")
        print(f"  AR-SSM max_rank: {preset_config.ar_ssm_max_rank}")
        print(f"  HTT rank: {preset_config.htt_rank}")
        print(f"  Target VRAM: {preset_config.target_vram_gb}GB")
        print(f"  Gradient Checkpointing: {preset_config.use_gradient_checkpointing}")
        print()
    except Exception as e:
        print(f"{preset_name}: Error - {e}\n")

# %%
# 8GB設定を使用
config_8gb = get_preset_config('8gb')
print("8GB設定を使用:")
print(f"  AR-SSM max_rank: {config_8gb.ar_ssm_max_rank}")
print(f"  HTT rank: {config_8gb.htt_rank}")

# %% [markdown]
# ## 3. Phase 1モデルの作成（簡易版）

# %%
# 注意: create_phase1_model()は完全な言語モデルを作成します
# このデモでは、個別のコンポーネントを使用します

from src.models.phase1 import AdaptiveRankSemiseparableLayer, HolographicTTEmbedding

# モデルパラメータ
vocab_size = 10000
d_model = 256
seq_len = 128

# HTT Embedding
htt_embedding = HolographicTTEmbedding(
    vocab_size=vocab_size,
    d_model=d_model,
    rank=config_8gb.htt_rank,
    num_cores=config_8gb.htt_num_cores,
    phase_encoding=config_8gb.htt_phase_encoding,
).to(device)

# AR-SSM Layer
ar_ssm = AdaptiveRankSemiseparableLayer(
    d_model=d_model,
    max_rank=config_8gb.ar_ssm_max_rank,
    min_rank=config_8gb.ar_ssm_min_rank,
    gate_hidden_dim=config_8gb.ar_ssm_gate_hidden_dim or d_model // 4,
    l1_regularization=config_8gb.ar_ssm_l1_regularization,
    use_fused_scan=config_8gb.ar_ssm_use_fused_scan,
).to(device)

print(f"Phase 1コンポーネント作成完了:")
print(f"  HTT Embedding: {sum(p.numel() for p in htt_embedding.parameters()):,} params")
print(f"  AR-SSM Layer: {sum(p.numel() for p in ar_ssm.parameters()):,} params")

# %% [markdown]
# ## 4. 順伝播の実行

# %%
# サンプル入力
batch_size = 4
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

# 順伝播
with torch.no_grad():
    # Embedding
    embeddings = htt_embedding(input_ids)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # AR-SSM
    ar_ssm_output = ar_ssm(embeddings)
    print(f"AR-SSM output shape: {ar_ssm_output.shape}")

# %% [markdown]
# ## 5. 安定性監視の統合

# %%
# 安定性モニターの作成
stability_monitor = BKStabilityMonitor(
    stability_threshold=config_8gb.stability_threshold,
    schatten_s1_bound=config_8gb.schatten_s1_bound,
    schatten_s2_bound=config_8gb.schatten_s2_bound,
    gradient_norm_threshold=config_8gb.gradient_norm_threshold,
)

print("安定性モニター作成完了")
print(f"  Stability threshold: {stability_monitor.stability_threshold}")
print(f"  Schatten S1 bound: {stability_monitor.schatten_s1_bound}")
print(f"  Schatten S2 bound: {stability_monitor.schatten_s2_bound}")

# %%
# 模擬的な安定性チェック
# 注意: 実際のBK-Core出力が必要ですが、ここではダミーデータを使用

# ダミーのBK-Core診断データ
dummy_G_ii = torch.randn(batch_size, seq_len, device=device, dtype=torch.complex64)
dummy_v = torch.randn(batch_size, seq_len, device=device)
dummy_epsilon = 0.1

# 安定性チェック
stability_info = stability_monitor.check_stability(
    G_ii=dummy_G_ii,
    v=dummy_v,
    epsilon=dummy_epsilon,
)

print("\n安定性チェック結果:")
print(f"  Is stable: {stability_info['is_stable']}")
print(f"  Det condition: {stability_info['det_condition']:.6e}")
print(f"  Schatten S1: {stability_info['schatten_s1']:.6e}")
print(f"  Schatten S2: {stability_info['schatten_s2']:.6e}")
if stability_info['warnings']:
    print(f"  Warnings: {stability_info['warnings']}")
if stability_info['actions']:
    print(f"  Recommended actions: {stability_info['actions']}")

# %% [markdown]
# ## 6. パフォーマンス測定

# %%
def measure_performance(embedding, ar_ssm, batch_size: int, seq_len: int, num_iterations: int = 50):
    """パフォーマンスを測定"""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # ウォームアップ
    for _ in range(10):
        with torch.no_grad():
            emb = embedding(input_ids)
            _ = ar_ssm(emb)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # メモリ測定
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    
    # ベンチマーク
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            emb = embedding(input_ids)
            _ = ar_ssm(emb)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed_time = time.time() - start_time
    tokens_per_sec = (batch_size * seq_len * num_iterations) / elapsed_time
    
    if device.type == 'cuda':
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
    else:
        peak_memory_mb = 0
    
    return {
        'tokens_per_sec': tokens_per_sec,
        'peak_memory_mb': peak_memory_mb,
        'avg_time_ms': (elapsed_time / num_iterations) * 1000,
    }

# %%
# 異なる設定でパフォーマンスを測定
configs_to_test = [
    {'batch_size': 1, 'seq_len': 128},
    {'batch_size': 2, 'seq_len': 128},
    {'batch_size': 4, 'seq_len': 128},
    {'batch_size': 4, 'seq_len': 256},
]

print("パフォーマンス測定:\n")
print(f"{'Batch':>6} {'Seq':>6} {'Tokens/sec':>12} {'Memory (MB)':>12} {'Time (ms)':>10}")
print("-" * 60)

perf_results = []
for config in configs_to_test:
    result = measure_performance(htt_embedding, ar_ssm, **config)
    perf_results.append({**config, **result})
    print(f"{config['batch_size']:6d} {config['seq_len']:6d} "
          f"{result['tokens_per_sec']:12.1f} {result['peak_memory_mb']:12.2f} "
          f"{result['avg_time_ms']:10.2f}")

# %% [markdown]
# ## 7. エラーハンドリングとリカバリ

# %%
# エラーリカバリの設定
recovery = Phase1ErrorRecovery()

# 模擬的なVRAM不足エラー
def simulate_vram_error():
    """VRAM不足エラーをシミュレート"""
    try:
        # 意図的に大きなテンソルを作成
        if device.type == 'cuda':
            large_tensor = torch.randn(10000, 10000, device=device)
            raise VRAMExhaustedError(
                current_mb=9000,
                limit_mb=8000,
                suggestions=[
                    "Reduce batch size",
                    "Reduce sequence length",
                    "Enable gradient checkpointing",
                ]
            )
    except VRAMExhaustedError as e:
        print(f"VRAM不足エラーをキャッチ:")
        print(f"  Current: {e.current_mb}MB")
        print(f"  Limit: {e.limit_mb}MB")
        print(f"  Suggestions:")
        for suggestion in e.suggestions:
            print(f"    - {suggestion}")
        
        # リカバリを試行
        print("\nリカバリを試行中...")
        # 注意: 実際のリカバリにはモデルとconfigが必要
        print("  1. 勾配チェックポイントを有効化")
        print("  2. AR-SSMランクを削減")
        print("  3. バッチサイズを削減")

# %%
if device.type == 'cuda':
    simulate_vram_error()
else:
    print("CUDA not available, skipping VRAM error simulation")

# %% [markdown]
# ## 8. 診断情報の収集

# %%
def collect_diagnostics(embedding, ar_ssm, input_ids):
    """診断情報を収集"""
    diagnostics = {}
    
    with torch.no_grad():
        # Embedding診断
        emb = embedding(input_ids)
        diagnostics['embedding_norm_mean'] = torch.norm(emb, dim=-1).mean().item()
        diagnostics['embedding_norm_std'] = torch.norm(emb, dim=-1).std().item()
        
        # AR-SSM診断
        gates = ar_ssm.estimate_rank_gate(emb)
        diagnostics['ar_ssm_effective_rank'] = gates.sum(dim=-1).mean().item()
        diagnostics['ar_ssm_gate_sparsity'] = (gates < 0.1).float().mean().item()
        
        # メモリ診断
        if device.type == 'cuda':
            diagnostics['current_memory_mb'] = torch.cuda.memory_allocated() / 1024**2
            diagnostics['peak_memory_mb'] = torch.cuda.max_memory_allocated() / 1024**2
        else:
            diagnostics['current_memory_mb'] = 0
            diagnostics['peak_memory_mb'] = 0
    
    return diagnostics

# %%
# 診断情報の収集
input_ids = torch.randint(0, vocab_size, (4, 128), device=device)
diagnostics = collect_diagnostics(htt_embedding, ar_ssm, input_ids)

print("診断情報:")
for key, value in diagnostics.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

# %% [markdown]
# ## 9. 可視化ダッシュボード

# %%
def create_dashboard(perf_results, diagnostics):
    """パフォーマンスダッシュボードを作成"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # スループット
    batch_sizes = [r['batch_size'] for r in perf_results]
    throughputs = [r['tokens_per_sec'] for r in perf_results]
    axes[0, 0].bar(range(len(batch_sizes)), throughputs, color='steelblue')
    axes[0, 0].set_xticks(range(len(batch_sizes)))
    axes[0, 0].set_xticklabels([f"B{r['batch_size']}\nS{r['seq_len']}" for r in perf_results])
    axes[0, 0].set_ylabel('Tokens/sec')
    axes[0, 0].set_title('Throughput')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # メモリ使用量
    if device.type == 'cuda':
        memories = [r['peak_memory_mb'] for r in perf_results]
        axes[0, 1].bar(range(len(batch_sizes)), memories, color='coral')
        axes[0, 1].set_xticks(range(len(batch_sizes)))
        axes[0, 1].set_xticklabels([f"B{r['batch_size']}\nS{r['seq_len']}" for r in perf_results])
        axes[0, 1].set_ylabel('Memory (MB)')
        axes[0, 1].set_title('Peak Memory Usage')
        axes[0, 1].axhline(y=8000, color='r', linestyle='--', label='8GB limit')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
    else:
        axes[0, 1].text(0.5, 0.5, 'CUDA not available', ha='center', va='center')
        axes[0, 1].set_title('Peak Memory Usage')
    
    # 診断メトリクス
    diag_keys = ['ar_ssm_effective_rank', 'ar_ssm_gate_sparsity', 
                 'embedding_norm_mean', 'embedding_norm_std']
    diag_values = [diagnostics.get(k, 0) for k in diag_keys]
    diag_labels = ['Eff. Rank', 'Gate Sparsity', 'Emb. Norm Mean', 'Emb. Norm Std']
    
    axes[1, 0].bar(range(len(diag_keys)), diag_values, color='mediumseagreen')
    axes[1, 0].set_xticks(range(len(diag_keys)))
    axes[1, 0].set_xticklabels(diag_labels, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].set_title('Diagnostic Metrics')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # レイテンシ
    latencies = [r['avg_time_ms'] for r in perf_results]
    axes[1, 1].bar(range(len(batch_sizes)), latencies, color='mediumpurple')
    axes[1, 1].set_xticks(range(len(batch_sizes)))
    axes[1, 1].set_xticklabels([f"B{r['batch_size']}\nS{r['seq_len']}" for r in perf_results])
    axes[1, 1].set_ylabel('Time (ms)')
    axes[1, 1].set_title('Average Latency')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()

# %%
print("パフォーマンスダッシュボード:")
create_dashboard(perf_results, diagnostics)

# %% [markdown]
# ## まとめ
#
# このチュートリアルでは、以下を学びました：
#
# 1. **Phase 1設定**: カスタム設定とプリセット設定の作成
# 2. **コンポーネント統合**: HTT EmbeddingとAR-SSM Layerの組み合わせ
# 3. **安定性監視**: BK安定性モニターの統合
# 4. **パフォーマンス測定**: スループット、メモリ、レイテンシの測定
# 5. **エラーハンドリング**: VRAM不足エラーのキャッチとリカバリ
# 6. **診断情報**: 有効ランク、ゲートスパース性などのメトリクス
# 7. **可視化**: パフォーマンスダッシュボードの作成
#
# ### 次のステップ
#
# - 実際の言語モデリングタスクでPerplexityを評価
# - 8GB VRAM制約での完全な訓練を実行
# - Phase 2の複素数値演算への拡張を準備
# - カスタムTritonカーネルでさらなる最適化
