"""
Phase 2 Long Context Test Script

このスクリプトは、Phase 2モデルの長期依存関係処理能力を検証します。

Test Objectives:
    1. VRAM使用量の測定（Seq=1024, 2048, 4096）
    2. 勾配消失の検証（末尾→先頭の勾配ノルム）
    3. 数値安定性の確認

Success Criteria:
    - Seq=4096でVRAM使用量が8.0GB未満
    - 末尾→先頭の勾配ノルムが1e-5以上
    - NaN/Infが発生しないこと

Requirements:
    - Task 13: 長期依存関係テストの実装
    - Task 13.1: VRAM使用量測定の実装
    - Task 13.2: 勾配消失検証の実装
    - Requirements: 7.4, 7.5

Usage:
    # 基本的な使用方法
    python scripts/test_long_context.py
    
    # カスタム設定
    python scripts/test_long_context.py --batch-size 2 --max-seq-len 8192
    
    # 結果をJSONに保存
    python scripts/test_long_context.py --output results/long_context_test.json

Author: Project MUSE Team
Date: 2025-01-20
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
import numpy as np

# Phase 2コンポーネント
from src.models.phase2 import Phase2IntegratedModel, create_phase2_model


class LongContextTester:
    """
    Phase 2モデルの長期依存関係テストを実行するクラス
    
    このクラスは以下の機能を提供します:
    - VRAM使用量の測定
    - 勾配消失の検証
    - 数値安定性のチェック
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        vocab_size: int = 50257,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Args:
            model: Phase2IntegratedModel
            device: 計算デバイス
            vocab_size: 語彙サイズ
            batch_size: バッチサイズ
            dtype: データ型（fp16推奨）
        """
        self.model = model.to(device)
        self.device = device
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.dtype = dtype
        
        # モデルをdtypeに変換
        if dtype == torch.float16:
            self.model = self.model.half()
        
        print(f"\n{'='*80}")
        print("LongContextTester initialized")
        print(f"{'='*80}")
        print(f"Model: {model.__class__.__name__}")
        print(f"Device: {device}")
        print(f"Batch size: {batch_size}")
        print(f"Data type: {dtype}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"{'='*80}\n")
    
    def measure_vram_usage(
        self,
        seq_lengths: List[int] = [1024, 2048, 4096],
    ) -> Dict[int, Dict[str, float]]:
        """
        Task 13.1: VRAM使用量測定の実装
        
        シーケンス長ごとのVRAM使用量を測定します。
        
        測定条件:
            - Batch=1
            - Seq=[1024, 2048, 4096]
            - fp16
        
        合格基準:
            - Seq=4096で8.0GB未満
        
        Args:
            seq_lengths: テストするシーケンス長のリスト
        
        Returns:
            結果の辞書 {seq_len: {'vram_mb': float, 'vram_gb': float, 'passed': bool}}
        
        Requirements: 7.4
        """
        print(f"\n{'='*80}")
        print("Task 13.1: VRAM Usage Measurement")
        print(f"{'='*80}")
        print(f"Testing sequence lengths: {seq_lengths}")
        print(f"Batch size: {self.batch_size}")
        print(f"Data type: {self.dtype}")
        print(f"{'='*80}\n")
        
        results = {}
        
        for seq_len in seq_lengths:
            print(f"\n--- Testing Seq={seq_len} ---")
            
            # メモリをクリア
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            # ダミーデータ生成
            input_ids = torch.randint(
                0, self.vocab_size,
                (self.batch_size, seq_len),
                device=self.device
            )
            
            # モデルの状態をリセット
            self.model.reset_state()
            
            try:
                # Forward pass
                with torch.no_grad():
                    start_time = time.time()
                    logits = self.model(input_ids)
                    forward_time = time.time() - start_time
                
                # VRAM使用量を測定
                if self.device.type == 'cuda':
                    vram_bytes = torch.cuda.max_memory_allocated()
                    vram_mb = vram_bytes / (1024 ** 2)
                    vram_gb = vram_bytes / (1024 ** 3)
                else:
                    # CPUの場合はメモリ測定をスキップ
                    vram_mb = 0.0
                    vram_gb = 0.0
                    warnings.warn("CPU mode: VRAM measurement skipped", UserWarning)
                
                # 合格判定（Seq=4096で8.0GB未満）
                if seq_len == 4096:
                    passed = vram_gb < 8.0
                else:
                    passed = True  # 他のシーケンス長は参考値
                
                # 結果を記録
                results[seq_len] = {
                    'vram_bytes': int(vram_bytes) if self.device.type == 'cuda' else 0,
                    'vram_mb': float(vram_mb),
                    'vram_gb': float(vram_gb),
                    'forward_time_sec': float(forward_time),
                    'passed': passed,
                    'output_shape': list(logits.shape),
                }
                
                # 結果を表示
                print(f"  VRAM usage: {vram_mb:.2f} MB ({vram_gb:.4f} GB)")
                print(f"  Forward time: {forward_time:.4f} sec")
                print(f"  Output shape: {logits.shape}")
                print(f"  Status: {'✓ PASSED' if passed else '✗ FAILED'}")
                
                if seq_len == 4096:
                    if passed:
                        print(f"  ✓ SUCCESS: VRAM usage {vram_gb:.4f} GB < 8.0 GB")
                    else:
                        print(f"  ✗ FAILURE: VRAM usage {vram_gb:.4f} GB >= 8.0 GB")
                
            except RuntimeError as e:
                print(f"  ✗ ERROR: {e}")
                results[seq_len] = {
                    'vram_bytes': 0,
                    'vram_mb': 0.0,
                    'vram_gb': 0.0,
                    'forward_time_sec': 0.0,
                    'passed': False,
                    'error': str(e),
                }
            
            # メモリをクリア
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # サマリーを表示
        print(f"\n{'='*80}")
        print("VRAM Usage Summary")
        print(f"{'='*80}")
        print(f"{'Seq Length':<12} {'VRAM (MB)':<12} {'VRAM (GB)':<12} {'Status':<10}")
        print(f"{'-'*80}")
        for seq_len, result in results.items():
            status = '✓ PASS' if result['passed'] else '✗ FAIL'
            print(f"{seq_len:<12} {result['vram_mb']:<12.2f} {result['vram_gb']:<12.4f} {status:<10}")
        print(f"{'='*80}\n")
        
        return results
    
    def verify_gradient_flow(
        self,
        seq_len: int = 4096,
        min_gradient_norm: float = 1e-5,
    ) -> Dict[str, Any]:
        """
        Task 13.2: 勾配消失検証の実装
        
        末尾から先頭への勾配ノルムを計算し、勾配消失が発生していないことを確認します。
        
        測定:
            - Seq=4096の末尾から先頭への勾配ノルムを計算
        
        合格基準:
            - 勾配ノルムが1e-5以上残っていること
        
        Args:
            seq_len: テストするシーケンス長
            min_gradient_norm: 最小勾配ノルム閾値
        
        Returns:
            結果の辞書
        
        Requirements: 7.5
        """
        print(f"\n{'='*80}")
        print("Task 13.2: Gradient Flow Verification")
        print(f"{'='*80}")
        print(f"Sequence length: {seq_len}")
        print(f"Minimum gradient norm threshold: {min_gradient_norm}")
        print(f"{'='*80}\n")
        
        # メモリをクリア
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # モデルを訓練モードに設定
        self.model.train()
        
        # モデルの状態をリセット
        self.model.reset_state()
        
        # ダミーデータ生成
        input_ids = torch.randint(
            0, self.vocab_size,
            (self.batch_size, seq_len),
            device=self.device
        )
        labels = torch.randint(
            0, self.vocab_size,
            (self.batch_size, seq_len),
            device=self.device
        )
        
        # 勾配をゼロ化
        self.model.zero_grad()
        
        try:
            # Forward pass
            logits = self.model(input_ids)
            
            # 損失計算（末尾のトークンのみ）
            # 末尾から先頭への勾配フローを測定するため、末尾の損失を使用
            loss = nn.functional.cross_entropy(
                logits[:, -1, :],  # 末尾のトークンのみ
                labels[:, -1],
            )
            
            # Backward pass
            loss.backward()
            
            # 各層の勾配ノルムを計算
            layer_gradient_norms = []
            for i, block in enumerate(self.model.blocks):
                # 各ブロックの勾配ノルムを計算
                block_grad_norm = 0.0
                num_params = 0
                for name, param in block.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        block_grad_norm += grad_norm ** 2
                        num_params += 1
                
                if num_params > 0:
                    block_grad_norm = (block_grad_norm ** 0.5) / num_params
                
                layer_gradient_norms.append({
                    'layer': i,
                    'gradient_norm': float(block_grad_norm),
                })
            
            # Embedding層の勾配ノルム
            emb_grad_norm = 0.0
            if self.model.token_embedding.weight.grad is not None:
                emb_grad_norm = self.model.token_embedding.weight.grad.norm().item()
            
            # 先頭層（Embedding直後）の勾配ノルム
            first_layer_grad_norm = layer_gradient_norms[0]['gradient_norm'] if layer_gradient_norms else 0.0
            
            # 末尾層の勾配ノルム
            last_layer_grad_norm = layer_gradient_norms[-1]['gradient_norm'] if layer_gradient_norms else 0.0
            
            # 勾配消失の判定
            passed = first_layer_grad_norm >= min_gradient_norm
            
            # NaN/Infチェック
            has_nan = any(
                torch.isnan(param.grad).any().item() or torch.isinf(param.grad).any().item()
                for param in self.model.parameters()
                if param.grad is not None
            )
            
            # 結果を記録
            result = {
                'seq_len': seq_len,
                'loss': float(loss.item()),
                'embedding_grad_norm': float(emb_grad_norm),
                'first_layer_grad_norm': float(first_layer_grad_norm),
                'last_layer_grad_norm': float(last_layer_grad_norm),
                'layer_gradient_norms': layer_gradient_norms,
                'min_gradient_norm': float(min(
                    [ln['gradient_norm'] for ln in layer_gradient_norms] + [emb_grad_norm]
                )),
                'max_gradient_norm': float(max(
                    [ln['gradient_norm'] for ln in layer_gradient_norms] + [emb_grad_norm]
                )),
                'mean_gradient_norm': float(np.mean(
                    [ln['gradient_norm'] for ln in layer_gradient_norms] + [emb_grad_norm]
                )),
                'has_nan_or_inf': has_nan,
                'passed': passed and not has_nan,
            }
            
            # 結果を表示
            print(f"Loss: {loss.item():.6f}")
            print(f"Embedding gradient norm: {emb_grad_norm:.6e}")
            print(f"First layer gradient norm: {first_layer_grad_norm:.6e}")
            print(f"Last layer gradient norm: {last_layer_grad_norm:.6e}")
            print(f"Min gradient norm: {result['min_gradient_norm']:.6e}")
            print(f"Max gradient norm: {result['max_gradient_norm']:.6e}")
            print(f"Mean gradient norm: {result['mean_gradient_norm']:.6e}")
            print(f"Has NaN/Inf: {has_nan}")
            
            print(f"\n--- Layer-wise Gradient Norms ---")
            for ln in layer_gradient_norms:
                print(f"  Layer {ln['layer']}: {ln['gradient_norm']:.6e}")
            
            print(f"\n--- Gradient Flow Status ---")
            if passed:
                print(f"✓ SUCCESS: First layer gradient norm {first_layer_grad_norm:.6e} >= {min_gradient_norm:.6e}")
            else:
                print(f"✗ FAILURE: First layer gradient norm {first_layer_grad_norm:.6e} < {min_gradient_norm:.6e}")
            
            if has_nan:
                print(f"✗ WARNING: NaN or Inf detected in gradients")
            else:
                print(f"✓ No NaN or Inf in gradients")
            
        except RuntimeError as e:
            print(f"✗ ERROR: {e}")
            result = {
                'seq_len': seq_len,
                'passed': False,
                'error': str(e),
            }
        
        # モデルを評価モードに戻す
        self.model.eval()
        
        # メモリをクリア
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        print(f"\n{'='*80}\n")
        
        return result
    
    def run_full_test(
        self,
        seq_lengths: List[int] = [1024, 2048, 4096],
        gradient_test_seq_len: int = 4096,
    ) -> Dict[str, Any]:
        """
        完全なテストスイートを実行
        
        Args:
            seq_lengths: VRAM測定用のシーケンス長リスト
            gradient_test_seq_len: 勾配検証用のシーケンス長
        
        Returns:
            全テスト結果の辞書
        """
        print(f"\n{'='*80}")
        print("Phase 2 Long Context Test Suite")
        print(f"{'='*80}\n")
        
        # Task 13.1: VRAM使用量測定
        vram_results = self.measure_vram_usage(seq_lengths)
        
        # Task 13.2: 勾配消失検証
        gradient_results = self.verify_gradient_flow(gradient_test_seq_len)
        
        # 総合結果
        all_passed = (
            all(r['passed'] for r in vram_results.values()) and
            gradient_results['passed']
        )
        
        results = {
            'vram_test': vram_results,
            'gradient_test': gradient_results,
            'all_passed': all_passed,
            'summary': {
                'vram_4096_gb': vram_results.get(4096, {}).get('vram_gb', 0.0),
                'vram_4096_passed': vram_results.get(4096, {}).get('passed', False),
                'gradient_norm': gradient_results.get('first_layer_grad_norm', 0.0),
                'gradient_passed': gradient_results.get('passed', False),
                'has_nan_or_inf': gradient_results.get('has_nan_or_inf', True),
            }
        }
        
        # 最終サマリー
        print(f"\n{'='*80}")
        print("Final Summary")
        print(f"{'='*80}")
        print(f"VRAM Test (Seq=4096): {'✓ PASSED' if results['summary']['vram_4096_passed'] else '✗ FAILED'}")
        print(f"  VRAM usage: {results['summary']['vram_4096_gb']:.4f} GB")
        print(f"Gradient Test (Seq=4096): {'✓ PASSED' if results['summary']['gradient_passed'] else '✗ FAILED'}")
        print(f"  Gradient norm: {results['summary']['gradient_norm']:.6e}")
        print(f"NaN/Inf Check: {'✓ PASSED' if not results['summary']['has_nan_or_inf'] else '✗ FAILED'}")
        print(f"\nOverall Status: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
        print(f"{'='*80}\n")
        
        return results


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Phase 2 Long Context Test Script"
    )
    
    # モデル設定
    parser.add_argument(
        '--preset',
        type=str,
        default='small',
        choices=['small', 'base', 'large'],
        help='Model preset (default: small)'
    )
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=50257,
        help='Vocabulary size (default: 50257)'
    )
    parser.add_argument(
        '--d-model',
        type=int,
        default=None,
        help='Model dimension (default: preset-dependent)'
    )
    parser.add_argument(
        '--n-layers',
        type=int,
        default=None,
        help='Number of layers (default: preset-dependent)'
    )
    
    # テスト設定
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size (default: 1)'
    )
    parser.add_argument(
        '--seq-lengths',
        type=int,
        nargs='+',
        default=[1024, 2048, 4096],
        help='Sequence lengths to test (default: 1024 2048 4096)'
    )
    parser.add_argument(
        '--max-seq-len',
        type=int,
        default=4096,
        help='Maximum sequence length for gradient test (default: 4096)'
    )
    parser.add_argument(
        '--min-gradient-norm',
        type=float,
        default=1e-5,
        help='Minimum gradient norm threshold (default: 1e-5)'
    )
    
    # デバイス設定
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (default: cuda if available)'
    )
    parser.add_argument(
        '--fp32',
        action='store_true',
        help='Use fp32 instead of fp16'
    )
    
    # 出力設定
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file path (default: None)'
    )
    
    args = parser.parse_args()
    
    # デバイス設定
    device = torch.device(args.device)
    dtype = torch.float32 if args.fp32 else torch.float16
    
    print(f"\n{'='*80}")
    print("Phase 2 Long Context Test")
    print(f"{'='*80}")
    print(f"Preset: {args.preset}")
    print(f"Device: {device}")
    print(f"Data type: {dtype}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence lengths: {args.seq_lengths}")
    print(f"Max sequence length: {args.max_seq_len}")
    print(f"{'='*80}\n")
    
    # モデル作成
    print("Creating Phase 2 model...")
    model_kwargs = {
        'vocab_size': args.vocab_size,
    }
    if args.d_model is not None:
        model_kwargs['d_model'] = args.d_model
    if args.n_layers is not None:
        model_kwargs['n_layers'] = args.n_layers
    
    model = create_phase2_model(
        preset=args.preset,
        **model_kwargs
    )
    
    print(f"Model created: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # テスター作成
    tester = LongContextTester(
        model=model,
        device=device,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        dtype=dtype,
    )
    
    # テスト実行
    results = tester.run_full_test(
        seq_lengths=args.seq_lengths,
        gradient_test_seq_len=args.max_seq_len,
    )
    
    # 結果を保存
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    # 終了コード
    sys.exit(0 if results['all_passed'] else 1)


if __name__ == '__main__':
    main()
