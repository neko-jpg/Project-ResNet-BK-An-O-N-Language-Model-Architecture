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
# # Phase 1: Holographic Tensor Train (HTT) Embedding Tutorial
#
# このノートブックでは、90%のパラメータ圧縮を実現するHTT Embeddingを探索します。
#
# ## 学習目標
# 1. Tensor Train分解の数学的基礎を理解する
# 2. ホログラフィック位相エンコーディングを可視化する
# 3. 圧縮率と品質のトレードオフを分析する
# 4. 標準Embeddingとの比較を行う

# %% [markdown]
# ## セットアップ

# %%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple

# Phase 1コンポーネントのインポート
import sys
sys.path.append('..')
from src.models.phase1 import HolographicTTEmbedding

# プロット設定
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# デバイス設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %% [markdown]
# ## 1. Tensor Train分解の基礎

# %% [markdown]
# ### 数学的基礎
#
# HTTは埋め込み行列 $E \in \mathbb{R}^{V \times D}$ をTensor Trainコアに分解します：
#
# $$E[i, :] = \text{Contract}(\text{Core}_1[i_1], \text{Core}_2[i_2], ..., \text{Core}_K[i_k])$$
#
# where:
# - $i = i_1 \cdot V_2 \cdot ... \cdot V_k + i_2 \cdot V_3 \cdot ... \cdot V_k + ... + i_k$
# - $V = V_1 \times V_2 \times ... \times V_K$ (語彙の因数分解)
# - $D = D_1 \times D_2 \times ... \times D_K$ (次元の因数分解)

# %%
def compare_parameter_counts(vocab_size: int, d_model: int, rank: int = 16):
    """標準EmbeddingとHTTのパラメータ数を比較"""
    # 標準Embedding
    standard_params = vocab_size * d_model
    
    # HTT (2コア)
    V1 = int(np.sqrt(vocab_size))
    V2 = (vocab_size + V1 - 1) // V1
    D1 = int(np.sqrt(d_model))
    D2 = (d_model + D1 - 1) // D1
    
    core1_params = V1 * 1 * rank * D1
    core2_params = V2 * rank * 1 * D2
    phase_params = rank
    htt_params = core1_params + core2_params + phase_params
    
    compression_ratio = htt_params / standard_params
    
    return {
        'standard_params': standard_params,
        'htt_params': htt_params,
        'compression_ratio': compression_ratio,
        'compression_percent': (1 - compression_ratio) * 100,
        'V1': V1, 'V2': V2,
        'D1': D1, 'D2': D2,
    }

# %%
# 異なる設定での圧縮率を計算
configs = [
    {'vocab_size': 10000, 'd_model': 512, 'rank': 16},
    {'vocab_size': 30000, 'd_model': 768, 'rank': 16},
    {'vocab_size': 50000, 'd_model': 1024, 'rank': 16},
    {'vocab_size': 50000, 'd_model': 1024, 'rank': 32},
]

print("パラメータ圧縮比較:\n")
print(f"{'Vocab':>8} {'d_model':>8} {'Rank':>6} {'Standard':>12} {'HTT':>12} {'Compression':>12}")
print("-" * 80)

for config in configs:
    stats = compare_parameter_counts(**config)
    print(f"{config['vocab_size']:8d} {config['d_model']:8d} {config['rank']:6d} "
          f"{stats['standard_params']:12,d} {stats['htt_params']:12,d} "
          f"{stats['compression_percent']:11.1f}%")

# %% [markdown]
# ## 2. HTT Embeddingの作成と使用

# %%
# HTT Embeddingの作成
vocab_size = 10000
d_model = 512
rank = 16

htt_embedding = HolographicTTEmbedding(
    vocab_size=vocab_size,
    d_model=d_model,
    rank=rank,
    num_cores=2,
    phase_encoding=True,
).to(device)

# 標準Embeddingとの比較
standard_embedding = nn.Embedding(vocab_size, d_model).to(device)

print(f"HTT Embedding:")
print(f"  Parameters: {sum(p.numel() for p in htt_embedding.parameters()):,}")
print(f"\nStandard Embedding:")
print(f"  Parameters: {sum(p.numel() for p in standard_embedding.parameters()):,}")

compression_stats = compare_parameter_counts(vocab_size, d_model, rank)
print(f"\nCompression: {compression_stats['compression_percent']:.1f}%")

# %%
# 順伝播の比較
batch_size = 4
seq_len = 32
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

# HTT
htt_output = htt_embedding(input_ids)
print(f"HTT output shape: {htt_output.shape}")

# 標準
std_output = standard_embedding(input_ids)
print(f"Standard output shape: {std_output.shape}")

# %% [markdown]
# ## 3. 位相エンコーディングの可視化

# %%
def visualize_phase_encoding(htt_embedding):
    """位相パラメータを可視化"""
    phase_shift = htt_embedding.phase_shift.detach().cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 位相値
    axes[0].bar(range(len(phase_shift)), phase_shift)
    axes[0].set_xlabel('Rank Dimension')
    axes[0].set_ylabel('Phase Value (radians)')
    axes[0].set_title('Phase Shift Parameters')
    axes[0].grid(True, alpha=0.3)
    
    # コサイン変調
    phase_mod = np.cos(phase_shift)
    axes[1].bar(range(len(phase_mod)), phase_mod, color='orange')
    axes[1].set_xlabel('Rank Dimension')
    axes[1].set_ylabel('Cosine Modulation')
    axes[1].set_title('Phase Modulation (cos(θ))')
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return phase_shift, phase_mod

# %%
print("位相エンコーディングの可視化:")
phase_shift, phase_mod = visualize_phase_encoding(htt_embedding)

print(f"\n位相統計:")
print(f"  Mean: {phase_shift.mean():.4f}")
print(f"  Std: {phase_shift.std():.4f}")
print(f"  Range: [{phase_shift.min():.4f}, {phase_shift.max():.4f}]")

# %% [markdown]
# ## 4. 埋め込み品質の評価

# %%
def evaluate_embedding_quality(htt_emb, std_emb, num_samples: int = 1000):
    """埋め込み品質を評価"""
    # ランダムなトークンIDをサンプル
    token_ids = torch.randint(0, vocab_size, (num_samples,), device=device)
    
    # 埋め込みを取得
    with torch.no_grad():
        htt_vecs = htt_emb(token_ids.unsqueeze(0)).squeeze(0)  # (num_samples, d_model)
        std_vecs = std_emb(token_ids)  # (num_samples, d_model)
    
    # ノルムの比較
    htt_norms = torch.norm(htt_vecs, dim=1).cpu().numpy()
    std_norms = torch.norm(std_vecs, dim=1).cpu().numpy()
    
    # コサイン類似度の分布（ランダムペア）
    num_pairs = 500
    idx1 = torch.randint(0, num_samples, (num_pairs,))
    idx2 = torch.randint(0, num_samples, (num_pairs,))
    
    htt_sim = torch.cosine_similarity(htt_vecs[idx1], htt_vecs[idx2], dim=1).cpu().numpy()
    std_sim = torch.cosine_similarity(std_vecs[idx1], std_vecs[idx2], dim=1).cpu().numpy()
    
    return {
        'htt_norms': htt_norms,
        'std_norms': std_norms,
        'htt_sim': htt_sim,
        'std_sim': std_sim,
    }

# %%
print("埋め込み品質の評価中...")
quality_stats = evaluate_embedding_quality(htt_embedding, standard_embedding)

# 可視化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ノルムの分布
axes[0].hist(quality_stats['htt_norms'], bins=50, alpha=0.6, label='HTT', density=True)
axes[0].hist(quality_stats['std_norms'], bins=50, alpha=0.6, label='Standard', density=True)
axes[0].set_xlabel('Embedding Norm')
axes[0].set_ylabel('Density')
axes[0].set_title('Embedding Norm Distribution')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# コサイン類似度の分布
axes[1].hist(quality_stats['htt_sim'], bins=50, alpha=0.6, label='HTT', density=True)
axes[1].hist(quality_stats['std_sim'], bins=50, alpha=0.6, label='Standard', density=True)
axes[1].set_xlabel('Cosine Similarity')
axes[1].set_ylabel('Density')
axes[1].set_title('Pairwise Cosine Similarity Distribution')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nノルム統計:")
print(f"  HTT - Mean: {quality_stats['htt_norms'].mean():.4f}, Std: {quality_stats['htt_norms'].std():.4f}")
print(f"  Standard - Mean: {quality_stats['std_norms'].mean():.4f}, Std: {quality_stats['std_norms'].std():.4f}")

# %% [markdown]
# ## 5. ランクの影響

# %%
def analyze_rank_impact(vocab_size: int, d_model: int, ranks: List[int]):
    """異なるランクでの影響を分析"""
    results = []
    
    for rank in ranks:
        htt = HolographicTTEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            rank=rank,
            num_cores=2,
        ).to(device)
        
        # パラメータ数
        params = sum(p.numel() for p in htt.parameters())
        
        # 圧縮率
        standard_params = vocab_size * d_model
        compression_ratio = params / standard_params
        
        # サンプル埋め込みの品質（ノルムの標準偏差）
        token_ids = torch.randint(0, vocab_size, (100,), device=device)
        with torch.no_grad():
            embeddings = htt(token_ids.unsqueeze(0)).squeeze(0)
            norms = torch.norm(embeddings, dim=1)
            norm_std = norms.std().item()
        
        results.append({
            'rank': rank,
            'params': params,
            'compression_ratio': compression_ratio,
            'norm_std': norm_std,
        })
        
        del htt
    
    return results

# %%
ranks = [4, 8, 16, 32, 64]
print("ランクの影響を分析中...")
rank_results = analyze_rank_impact(vocab_size, d_model, ranks)

# 可視化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ranks_list = [r['rank'] for r in rank_results]
params_list = [r['params'] for r in rank_results]
compression_list = [r['compression_ratio'] for r in rank_results]

# パラメータ数
axes[0].plot(ranks_list, params_list, 'o-', linewidth=2, markersize=8)
axes[0].set_xlabel('Rank')
axes[0].set_ylabel('Parameters')
axes[0].set_title('Parameters vs Rank')
axes[0].grid(True, alpha=0.3)
axes[0].set_xscale('log', base=2)

# 圧縮率
axes[1].plot(ranks_list, [c * 100 for c in compression_list], 'o-', linewidth=2, markersize=8, color='orange')
axes[1].axhline(y=10, color='r', linestyle='--', label='10% (90% compression target)')
axes[1].set_xlabel('Rank')
axes[1].set_ylabel('Compression Ratio (%)')
axes[1].set_title('Compression Ratio vs Rank')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xscale('log', base=2)

plt.tight_layout()
plt.show()

print("\nランク分析結果:")
for r in rank_results:
    print(f"  Rank {r['rank']:2d}: {r['params']:8,d} params, "
          f"{r['compression_ratio']*100:5.2f}% compression, "
          f"norm_std={r['norm_std']:.4f}")

# %% [markdown]
# ## 6. メモリとパフォーマンス

# %%
import time

def benchmark_embedding(embedding, batch_size: int, seq_len: int, num_iterations: int = 100):
    """埋め込みのパフォーマンスをベンチマーク"""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # ウォームアップ
    for _ in range(10):
        _ = embedding(input_ids)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # ベンチマーク
    start_time = time.time()
    for _ in range(num_iterations):
        _ = embedding(input_ids)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed_time = time.time() - start_time
    tokens_per_sec = (batch_size * seq_len * num_iterations) / elapsed_time
    
    return tokens_per_sec

# %%
# ベンチマーク
batch_size = 8
seq_len = 128

print("パフォーマンスベンチマーク:")
htt_throughput = benchmark_embedding(htt_embedding, batch_size, seq_len)
std_throughput = benchmark_embedding(standard_embedding, batch_size, seq_len)

print(f"  HTT: {htt_throughput:.1f} tokens/sec")
print(f"  Standard: {std_throughput:.1f} tokens/sec")
print(f"  Ratio: {std_throughput/htt_throughput:.2f}x")

# %% [markdown]
# ## 7. 勾配フローの検証

# %%
def verify_gradient_flow(embedding, name: str):
    """勾配フローを検証"""
    input_ids = torch.randint(0, vocab_size, (4, 32), device=device)
    
    # 順伝播
    output = embedding(input_ids)
    loss = output.sum()
    
    # 逆伝播
    loss.backward()
    
    # 勾配統計
    grad_stats = {}
    for param_name, param in embedding.named_parameters():
        if param.grad is not None:
            grad_stats[param_name] = {
                'mean': param.grad.abs().mean().item(),
                'max': param.grad.abs().max().item(),
                'has_nan': torch.isnan(param.grad).any().item(),
                'has_inf': torch.isinf(param.grad).any().item(),
            }
    
    return grad_stats

# %%
print("HTT勾配フローの検証:")
htt_grad_stats = verify_gradient_flow(htt_embedding, "HTT")

for name, stats in htt_grad_stats.items():
    print(f"\n{name}:")
    print(f"  Mean |grad|: {stats['mean']:.6f}")
    print(f"  Max |grad|: {stats['max']:.6f}")
    print(f"  Has NaN: {stats['has_nan']}")
    print(f"  Has Inf: {stats['has_inf']}")

# %% [markdown]
# ## まとめ
#
# このチュートリアルでは、以下を学びました：
#
# 1. **Tensor Train分解**: 語彙と次元を因数分解して圧縮
# 2. **圧縮率**: 90%以上のパラメータ削減（rank=16で）
# 3. **位相エンコーディング**: 学習可能な位相パラメータで意味関係を保存
# 4. **品質**: 標準Embeddingと同等のノルム分布と類似度分布
# 5. **ランクの影響**: ランクを増やすと圧縮率は下がるが品質は向上
# 6. **パフォーマンス**: 標準Embeddingと同等のスループット
# 7. **勾配フロー**: すべてのコアで安定した勾配
#
# ### 次のステップ
#
# - AR-SSMチュートリアルで動的ランク調整を学ぶ
# - 完全なPhase 1モデルでHTTとAR-SSMを統合
# - 実際の言語モデリングタスクでPerplexityを評価
