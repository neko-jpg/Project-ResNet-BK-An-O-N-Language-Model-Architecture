#!/usr/bin/env python3
"""
HTT Embedding Compression Verification Script

このスクリプトは、HTT Embeddingによる実際のパラメータ削減率を検証します。
標準的なEmbedding層と比較して、90%削減が達成されているかを確認します。

実行方法:
    python scripts/verify_htt_compression.py

Author: Project MUSE Team
Date: 2025-11-19
"""

import sys
import os
import torch
import torch.nn as nn

# プロジェクトルートをPythonパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.phase1.htt_embedding import HolographicTTEmbedding, verify_compression_ratio


# 色付き出力用
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}\n")


def print_success(text):
    try:
        print(f"{Colors.GREEN}✓ {text}{Colors.END}")
    except UnicodeEncodeError:
        print(f"{Colors.GREEN}[OK] {text}{Colors.END}")


def print_info(text):
    try:
        print(f"{Colors.CYAN}ℹ {text}{Colors.END}")
    except UnicodeEncodeError:
        print(f"{Colors.CYAN}[INFO] {text}{Colors.END}")


def format_number(num):
    """数値を読みやすい形式にフォーマット"""
    if num >= 1_000_000:
        return f"{num/1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num/1_000:.2f}K"
    else:
        return str(num)


def verify_single_embedding(vocab_size, d_model, rank):
    """単一のEmbedding設定を検証"""
    print(f"\n{Colors.BOLD}設定: vocab_size={format_number(vocab_size)}, "
          f"d_model={d_model}, rank={rank}{Colors.END}")
    print("-" * 80)
    
    # 標準Embedding
    standard_embedding = nn.Embedding(vocab_size, d_model)
    standard_params = sum(p.numel() for p in standard_embedding.parameters())
    
    # HTT Embedding
    htt_embedding = HolographicTTEmbedding(
        vocab_size=vocab_size,
        d_model=d_model,
        rank=rank,
        phase_encoding=True
    )
    htt_params = sum(p.numel() for p in htt_embedding.parameters())
    
    # 圧縮率計算
    compression_ratio = htt_params / standard_params
    reduction_percentage = (1 - compression_ratio) * 100
    
    # 結果表示
    print(f"標準Embedding:     {format_number(standard_params):>10} パラメータ")
    print(f"HTT Embedding:     {format_number(htt_params):>10} パラメータ")
    print(f"削減量:            {format_number(standard_params - htt_params):>10} パラメータ")
    print(f"圧縮率:            {compression_ratio:>10.4f} ({reduction_percentage:.1f}%削減)")
    
    # 検証結果の詳細
    result = verify_compression_ratio(htt_embedding, target_ratio=0.1)
    
    # 目標達成確認
    if result['meets_target']:
        print_success(f"目標達成: {reduction_percentage:.1f}% >= 90%削減")
    else:
        print(f"{Colors.YELLOW}⚠ 目標未達成: {reduction_percentage:.1f}% < 90%削減{Colors.END}")
    
    # メモリ使用量
    standard_memory_mb = (standard_params * 4) / (1024 ** 2)  # float32
    htt_memory_mb = (htt_params * 4) / (1024 ** 2)
    memory_saved_mb = standard_memory_mb - htt_memory_mb
    
    print(f"\nメモリ使用量 (float32):")
    print(f"  標準Embedding:   {standard_memory_mb:>8.2f} MB")
    print(f"  HTT Embedding:   {htt_memory_mb:>8.2f} MB")
    print(f"  削減量:          {memory_saved_mb:>8.2f} MB ({(memory_saved_mb/standard_memory_mb)*100:.1f}%)")
    
    return {
        'vocab_size': vocab_size,
        'd_model': d_model,
        'rank': rank,
        'standard_params': standard_params,
        'htt_params': htt_params,
        'compression_ratio': compression_ratio,
        'reduction_percentage': reduction_percentage,
        'meets_target': result['meets_target'],
        'memory_saved_mb': memory_saved_mb,
    }


def test_typical_configurations():
    """典型的なモデル設定でテスト"""
    print_header("HTT Embedding パラメータ削減率の検証")
    
    print_info("典型的な言語モデルの設定で検証します")
    
    # テスト設定
    configurations = [
        # (vocab_size, d_model, rank, description)
        (50000, 512, 16, "小規模モデル (GPT-2 Small相当)"),
        (50000, 768, 16, "中規模モデル (BERT Base相当)"),
        (50000, 1024, 16, "中規模モデル (GPT-2 Medium相当)"),
        (50000, 1280, 16, "大規模モデル (GPT-2 Large相当)"),
        (50000, 1600, 32, "超大規模モデル (GPT-2 XL相当)"),
    ]
    
    results = []
    
    for vocab_size, d_model, rank, description in configurations:
        print(f"\n{Colors.CYAN}【{description}】{Colors.END}")
        result = verify_single_embedding(vocab_size, d_model, rank)
        results.append(result)
    
    # サマリー表示
    print_header("検証結果サマリー")
    
    print(f"\n{Colors.BOLD}パラメータ削減率:{Colors.END}")
    print(f"{'設定':<30} {'標準':<12} {'HTT':<12} {'削減率':<10} {'目標達成'}")
    print("-" * 80)
    
    for i, (config, result) in enumerate(zip(configurations, results)):
        vocab_size, d_model, rank, description = config
        status = "✓" if result['meets_target'] else "✗"
        try:
            print(f"{description:<30} "
                  f"{format_number(result['standard_params']):<12} "
                  f"{format_number(result['htt_params']):<12} "
                  f"{result['reduction_percentage']:>6.1f}%   "
                  f"{status}")
        except UnicodeEncodeError:
            status_text = "OK" if result['meets_target'] else "NG"
            print(f"{description:<30} "
                  f"{format_number(result['standard_params']):<12} "
                  f"{format_number(result['htt_params']):<12} "
                  f"{result['reduction_percentage']:>6.1f}%   "
                  f"{status_text}")
    
    # 統計情報
    avg_reduction = sum(r['reduction_percentage'] for r in results) / len(results)
    min_reduction = min(r['reduction_percentage'] for r in results)
    max_reduction = max(r['reduction_percentage'] for r in results)
    all_meet_target = all(r['meets_target'] for r in results)
    
    print(f"\n{Colors.BOLD}統計情報:{Colors.END}")
    print(f"  平均削減率:     {avg_reduction:.1f}%")
    print(f"  最小削減率:     {min_reduction:.1f}%")
    print(f"  最大削減率:     {max_reduction:.1f}%")
    print(f"  目標達成率:     {sum(r['meets_target'] for r in results)}/{len(results)} 設定")
    
    # メモリ削減量の合計
    total_memory_saved = sum(r['memory_saved_mb'] for r in results)
    print(f"  合計メモリ削減: {total_memory_saved:.2f} MB")
    
    # 最終判定
    print()
    if all_meet_target and avg_reduction >= 90.0:
        print_success(f"すべての設定で90%以上の削減を達成しました！ (平均: {avg_reduction:.1f}%)")
        return True
    else:
        print(f"{Colors.YELLOW}⚠ 一部の設定で目標未達成です{Colors.END}")
        return False


def test_extreme_configurations():
    """極端な設定でのテスト"""
    print_header("極端な設定での検証")
    
    print_info("非常に大きな語彙サイズや次元でテストします")
    
    configurations = [
        # (vocab_size, d_model, rank, description)
        (100000, 1024, 16, "大語彙 (100K tokens)"),
        (50000, 2048, 32, "大次元 (2048 dim)"),
        (32000, 4096, 64, "超大次元 (4096 dim, LLaMA相当)"),
    ]
    
    results = []
    
    for vocab_size, d_model, rank, description in configurations:
        print(f"\n{Colors.CYAN}【{description}】{Colors.END}")
        result = verify_single_embedding(vocab_size, d_model, rank)
        results.append(result)
    
    # サマリー
    print(f"\n{Colors.BOLD}極端な設定での削減率:{Colors.END}")
    for config, result in zip(configurations, results):
        description = config[3]
        print(f"  {description:<30} {result['reduction_percentage']:>6.1f}%削減")
    
    avg_reduction = sum(r['reduction_percentage'] for r in results) / len(results)
    print(f"\n  平均削減率: {avg_reduction:.1f}%")
    
    return all(r['meets_target'] for r in results)


def test_rank_sensitivity():
    """ランクパラメータの感度分析"""
    print_header("ランクパラメータの感度分析")
    
    print_info("異なるランク値での圧縮率を比較します")
    
    vocab_size = 50000
    d_model = 1024
    ranks = [4, 8, 16, 32, 64]
    
    print(f"\n設定: vocab_size={format_number(vocab_size)}, d_model={d_model}")
    print(f"\n{'Rank':<10} {'HTTパラメータ':<15} {'削減率':<12} {'目標達成'}")
    print("-" * 50)
    
    for rank in ranks:
        htt_embedding = HolographicTTEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            rank=rank,
            phase_encoding=True
        )
        
        standard_params = vocab_size * d_model
        htt_params = sum(p.numel() for p in htt_embedding.parameters())
        reduction = (1 - htt_params / standard_params) * 100
        meets_target = reduction >= 90.0
        
        status = "✓" if meets_target else "✗"
        try:
            print(f"{rank:<10} {format_number(htt_params):<15} {reduction:>6.1f}%     {status}")
        except UnicodeEncodeError:
            status_text = "OK" if meets_target else "NG"
            print(f"{rank:<10} {format_number(htt_params):<15} {reduction:>6.1f}%     {status_text}")
    
    print(f"\n{Colors.BOLD}推奨ランク:{Colors.END}")
    print(f"  - 90%以上削減: rank <= 16")
    print(f"  - 95%以上削減: rank <= 8")
    print(f"  - 品質重視:     rank >= 16")


def main():
    """メイン実行関数"""
    print_header("HTT Embedding 90%削減検証スクリプト")
    
    # 典型的な設定でテスト
    typical_success = test_typical_configurations()
    
    # 極端な設定でテスト
    extreme_success = test_extreme_configurations()
    
    # ランク感度分析
    test_rank_sensitivity()
    
    # 最終結果
    print_header("最終結果")
    
    if typical_success and extreme_success:
        print_success("すべてのテストで90%以上の削減を達成しました！")
        print(f"\n{Colors.GREEN}{Colors.BOLD}HTT Embeddingは目標の90%パラメータ削減を達成しています。{Colors.END}")
        print(f"{Colors.GREEN}Phase 2への移行準備が完了しました。{Colors.END}\n")
        return 0
    else:
        print(f"\n{Colors.YELLOW}一部のテストで目標未達成です。{Colors.END}")
        print(f"{Colors.YELLOW}ランクパラメータの調整を検討してください。{Colors.END}\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
