"""
HTT Embedding圧縮率比較ベンチマーク

Phase 1とPhase 8のHTT Embeddingの圧縮率を比較します。
Phase 8はPhase 7を継承しており、Phase 7はPhase 1のHTT Embeddingを使用しているため、
理論的には同じ圧縮率になるはずです。

このスクリプトは以下を検証します:
1. Phase 1のHTT Embedding圧縮率
2. Phase 8（Phase 7経由）のHTT Embedding圧縮率
3. 両者が同一であることの確認
"""

import torch
import json
from pathlib import Path
from typing import Dict, Any

from src.models.phase1.htt_embedding import (
    HolographicTTEmbedding,
    verify_compression_ratio,
    calculate_htt_memory_savings,
)
from src.models.phase8.integrated_model import create_phase8_model
from src.models.phase8.config import Phase8Config


def benchmark_phase1_htt(
    vocab_size: int = 50257,
    d_model: int = 256,
    rank: int = 16,
) -> Dict[str, Any]:
    """
    Phase 1のHTT Embeddingをベンチマーク
    
    Args:
        vocab_size: 語彙サイズ
        d_model: モデル次元
        rank: TTランク
    
    Returns:
        結果辞書
    """
    print(f"\n{'='*60}")
    print(f"Phase 1 HTT Embedding Benchmark")
    print(f"{'='*60}")
    print(f"Config: vocab_size={vocab_size}, d_model={d_model}, rank={rank}")
    
    # HTT Embeddingを作成
    htt_embedding = HolographicTTEmbedding(
        vocab_size=vocab_size,
        d_model=d_model,
        rank=rank,
        phase_encoding=True,
    )
    
    # 圧縮率を検証
    compression_info = verify_compression_ratio(htt_embedding)
    
    # メモリ削減量を計算
    memory_info = calculate_htt_memory_savings(
        vocab_size=vocab_size,
        d_model=d_model,
        rank=rank,
    )
    
    # 結果を表示
    print(f"\n--- Parameter Count ---")
    print(f"Standard Embedding: {compression_info['standard_params']:,} params")
    print(f"HTT Embedding:      {compression_info['tt_params']:,} params")
    print(f"Reduction:          {compression_info['parameter_reduction']:,} params")
    print(f"Compression Ratio:  {compression_info['compression_ratio']:.6f}")
    print(f"Compression:        {compression_info['compression_percentage']:.2f}%")
    
    print(f"\n--- Memory Usage ---")
    print(f"Standard Embedding: {memory_info['standard_memory_mb']:.2f} MB")
    print(f"HTT Embedding:      {memory_info['htt_memory_mb']:.2f} MB")
    print(f"Memory Saved:       {memory_info['memory_saved_mb']:.2f} MB")
    print(f"Memory Reduction:   {memory_info['memory_saved_percentage']:.2f}%")
    
    # 実際のパラメータ数を確認
    actual_params = sum(p.numel() for p in htt_embedding.parameters())
    print(f"\n--- Actual Parameter Count ---")
    print(f"Calculated:  {compression_info['tt_params']:,} params")
    print(f"Actual:      {actual_params:,} params")
    print(f"Match:       {actual_params == compression_info['tt_params']}")
    
    return {
        'phase': 'Phase 1',
        'vocab_size': vocab_size,
        'd_model': d_model,
        'rank': rank,
        'standard_params': compression_info['standard_params'],
        'htt_params': compression_info['tt_params'],
        'actual_params': actual_params,
        'compression_ratio': compression_info['compression_ratio'],
        'compression_percentage': compression_info['compression_percentage'],
        'memory_standard_mb': memory_info['standard_memory_mb'],
        'memory_htt_mb': memory_info['htt_memory_mb'],
        'memory_saved_mb': memory_info['memory_saved_mb'],
        'memory_saved_percentage': memory_info['memory_saved_percentage'],
    }


def benchmark_phase8_htt(
    vocab_size: int = 50257,
    d_model: int = 256,
    rank: int = 16,
) -> Dict[str, Any]:
    """
    Phase 8のHTT Embeddingをベンチマーク
    
    Args:
        vocab_size: 語彙サイズ
        d_model: モデル次元
        rank: TTランク
    
    Returns:
        結果辞書
    """
    print(f"\n{'='*60}")
    print(f"Phase 8 HTT Embedding Benchmark (via Phase 7)")
    print(f"{'='*60}")
    print(f"Config: vocab_size={vocab_size}, d_model={d_model}, rank={rank}")
    
    # Phase 8モデルを作成（最小構成）
    config = Phase8Config(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=1,  # 最小レイヤー数
        htt_rank=rank,
        use_bk_hyperbolic=False,  # Phase 8拡張を無効化
        use_ar_ssm_fusion=False,
        enable_entailment_cones=False,
        enable_persistent_homology=False,
        enable_sheaf_attention=False,
    )
    
    model = create_phase8_model(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=1,
        htt_rank=rank,
        use_bk_hyperbolic=False,
        use_ar_ssm_fusion=False,
    )
    
    # Phase 7のHTT Embeddingにアクセス
    htt_embedding = model.phase7_model.htt_embedding
    
    # 圧縮率を検証
    compression_info = verify_compression_ratio(htt_embedding)
    
    # メモリ削減量を計算
    memory_info = calculate_htt_memory_savings(
        vocab_size=vocab_size,
        d_model=d_model,
        rank=rank,
    )
    
    # 結果を表示
    print(f"\n--- Parameter Count ---")
    print(f"Standard Embedding: {compression_info['standard_params']:,} params")
    print(f"HTT Embedding:      {compression_info['tt_params']:,} params")
    print(f"Reduction:          {compression_info['parameter_reduction']:,} params")
    print(f"Compression Ratio:  {compression_info['compression_ratio']:.6f}")
    print(f"Compression:        {compression_info['compression_percentage']:.2f}%")
    
    print(f"\n--- Memory Usage ---")
    print(f"Standard Embedding: {memory_info['standard_memory_mb']:.2f} MB")
    print(f"HTT Embedding:      {memory_info['htt_memory_mb']:.2f} MB")
    print(f"Memory Saved:       {memory_info['memory_saved_mb']:.2f} MB")
    print(f"Memory Reduction:   {memory_info['memory_saved_percentage']:.2f}%")
    
    # 実際のパラメータ数を確認
    actual_params = sum(p.numel() for p in htt_embedding.parameters())
    print(f"\n--- Actual Parameter Count ---")
    print(f"Calculated:  {compression_info['tt_params']:,} params")
    print(f"Actual:      {actual_params:,} params")
    print(f"Match:       {actual_params == compression_info['tt_params']}")
    
    # Phase 8全体のパラメータ数も表示
    total_params = model.get_total_parameter_count()
    phase7_params = model.get_phase7_parameter_count()
    phase8_extension_params = model.get_phase8_extension_parameter_count()
    
    print(f"\n--- Phase 8 Model Parameters ---")
    print(f"Total:            {total_params:,} params")
    print(f"Phase 7 (core):   {phase7_params:,} params")
    print(f"Phase 8 (ext):    {phase8_extension_params:,} params")
    print(f"HTT Embedding:    {actual_params:,} params ({actual_params/total_params*100:.2f}% of total)")
    
    return {
        'phase': 'Phase 8 (via Phase 7)',
        'vocab_size': vocab_size,
        'd_model': d_model,
        'rank': rank,
        'standard_params': compression_info['standard_params'],
        'htt_params': compression_info['tt_params'],
        'actual_params': actual_params,
        'compression_ratio': compression_info['compression_ratio'],
        'compression_percentage': compression_info['compression_percentage'],
        'memory_standard_mb': memory_info['standard_memory_mb'],
        'memory_htt_mb': memory_info['htt_memory_mb'],
        'memory_saved_mb': memory_info['memory_saved_mb'],
        'memory_saved_percentage': memory_info['memory_saved_percentage'],
        'total_model_params': total_params,
        'phase7_params': phase7_params,
        'phase8_extension_params': phase8_extension_params,
    }


def compare_compression_ratios(
    phase1_result: Dict[str, Any],
    phase8_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Phase 1とPhase 8の圧縮率を比較
    
    Args:
        phase1_result: Phase 1の結果
        phase8_result: Phase 8の結果
    
    Returns:
        比較結果
    """
    print(f"\n{'='*60}")
    print(f"Compression Ratio Comparison")
    print(f"{'='*60}")
    
    # 圧縮率の比較
    ratio_diff = abs(phase1_result['compression_ratio'] - phase8_result['compression_ratio'])
    ratio_match = ratio_diff < 1e-6  # 浮動小数点誤差を考慮
    
    print(f"\n--- Compression Ratio ---")
    print(f"Phase 1: {phase1_result['compression_ratio']:.6f}")
    print(f"Phase 8: {phase8_result['compression_ratio']:.6f}")
    print(f"Difference: {ratio_diff:.9f}")
    print(f"Match: {ratio_match} (tolerance: 1e-6)")
    
    # パラメータ数の比較
    params_diff = abs(phase1_result['htt_params'] - phase8_result['htt_params'])
    params_match = params_diff == 0
    
    print(f"\n--- HTT Parameters ---")
    print(f"Phase 1: {phase1_result['htt_params']:,} params")
    print(f"Phase 8: {phase8_result['htt_params']:,} params")
    print(f"Difference: {params_diff:,} params")
    print(f"Match: {params_match}")
    
    # 実際のパラメータ数の比較
    actual_diff = abs(phase1_result['actual_params'] - phase8_result['actual_params'])
    actual_match = actual_diff == 0
    
    print(f"\n--- Actual Parameters ---")
    print(f"Phase 1: {phase1_result['actual_params']:,} params")
    print(f"Phase 8: {phase8_result['actual_params']:,} params")
    print(f"Difference: {actual_diff:,} params")
    print(f"Match: {actual_match}")
    
    # メモリ削減量の比較
    memory_diff = abs(phase1_result['memory_saved_mb'] - phase8_result['memory_saved_mb'])
    memory_match = memory_diff < 0.01  # 0.01 MB以下の差は許容
    
    print(f"\n--- Memory Savings ---")
    print(f"Phase 1: {phase1_result['memory_saved_mb']:.2f} MB")
    print(f"Phase 8: {phase8_result['memory_saved_mb']:.2f} MB")
    print(f"Difference: {memory_diff:.4f} MB")
    print(f"Match: {memory_match} (tolerance: 0.01 MB)")
    
    # 総合判定
    all_match = ratio_match and params_match and actual_match and memory_match
    
    print(f"\n{'='*60}")
    print(f"Overall Result: {'✓ PASS' if all_match else '✗ FAIL'}")
    print(f"{'='*60}")
    
    if all_match:
        print("\n✓ Phase 1とPhase 8のHTT Embedding圧縮率は同一です。")
        print("  Phase 8はPhase 7を継承し、Phase 7はPhase 1のHTT Embeddingを")
        print("  使用しているため、期待通りの結果です。")
    else:
        print("\n✗ Phase 1とPhase 8のHTT Embedding圧縮率に差異があります。")
        print("  実装を確認してください。")
    
    return {
        'compression_ratio_match': ratio_match,
        'compression_ratio_diff': ratio_diff,
        'params_match': params_match,
        'params_diff': params_diff,
        'actual_params_match': actual_match,
        'actual_params_diff': actual_diff,
        'memory_match': memory_match,
        'memory_diff': memory_diff,
        'all_match': all_match,
    }


def run_comprehensive_benchmark():
    """
    包括的なベンチマークを実行
    """
    print("\n" + "="*60)
    print("HTT Embedding Compression Ratio Comparison Benchmark")
    print("Phase 1 vs Phase 8 (via Phase 7)")
    print("="*60)
    
    # テスト設定
    test_configs = [
        {'vocab_size': 50257, 'd_model': 256, 'rank': 16},
        {'vocab_size': 50257, 'd_model': 512, 'rank': 16},
        {'vocab_size': 50257, 'd_model': 1024, 'rank': 16},
        {'vocab_size': 32000, 'd_model': 256, 'rank': 8},
    ]
    
    all_results = []
    
    for i, config in enumerate(test_configs, 1):
        print(f"\n{'#'*60}")
        print(f"Test Case {i}/{len(test_configs)}")
        print(f"{'#'*60}")
        
        # Phase 1ベンチマーク
        phase1_result = benchmark_phase1_htt(**config)
        
        # Phase 8ベンチマーク
        phase8_result = benchmark_phase8_htt(**config)
        
        # 比較
        comparison = compare_compression_ratios(phase1_result, phase8_result)
        
        # 結果を保存
        all_results.append({
            'config': config,
            'phase1': phase1_result,
            'phase8': phase8_result,
            'comparison': comparison,
        })
    
    # 結果をJSONに保存
    output_dir = Path('results/benchmarks')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'htt_compression_comparison.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")
    
    # サマリーを表示
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    
    all_passed = all(r['comparison']['all_match'] for r in all_results)
    passed_count = sum(1 for r in all_results if r['comparison']['all_match'])
    
    print(f"\nTotal Test Cases: {len(all_results)}")
    print(f"Passed: {passed_count}/{len(all_results)}")
    print(f"Overall: {'✓ ALL PASS' if all_passed else '✗ SOME FAILED'}")
    
    if all_passed:
        print("\n✓ すべてのテストケースでPhase 1とPhase 8のHTT Embedding圧縮率が")
        print("  同一であることが確認されました。")
    else:
        print("\n✗ 一部のテストケースで差異が検出されました。")
        print("  詳細は上記の結果を確認してください。")
    
    return all_results


if __name__ == '__main__':
    results = run_comprehensive_benchmark()
