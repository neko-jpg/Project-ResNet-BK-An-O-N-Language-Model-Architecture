#!/usr/bin/env python3
"""
現在の設計で対応可能な最大パラメータ数を検証するスクリプト

このスクリプトは以下を検証します：
1. 8GB VRAM制約下での最大モデルサイズ
2. 10GB VRAM制約下での最大モデルサイズ
3. 各設定でのメモリ使用量とスループット
4. スケーラビリティの限界

結果をJSON形式で保存し、論文への記載を推奨します。
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.phase1.config import Phase1Config
from src.models.phase1.factory import create_phase1_model


class MaxParameterTester:
    """最大パラメータ数をテストするクラス"""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.results = {
            "environment": self._get_environment_info(),
            "configurations": [],
            "max_parameters": {}
        }
        
        if not torch.cuda.is_available():
            print("⚠️ CUDAが利用できません。このスクリプトはGPUが必要です。")
            sys.exit(1)
        
        # GPU情報を取得
        self.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {self.gpu_memory_gb:.2f} GB")
        
    def _get_environment_info(self) -> Dict[str, Any]:
        """環境情報を取得"""
        info = {
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
            
        return info
    
    def test_configuration(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_seq: int,
        batch_size: int = 1,
        htt_rank: int = 4,
        ar_ssm_max_rank: int = 8,
        use_fp16: bool = True,
        use_gradient_checkpointing: bool = True,
        test_training: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        指定された設定でモデルをテストし、メモリ使用量を測定
        
        Returns:
            成功した場合は結果の辞書、OOMの場合はNone
        """
        config = Phase1Config(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_seq=n_seq,
            htt_rank=htt_rank,
            ar_ssm_max_rank=ar_ssm_max_rank,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        
        try:
            # メモリをクリア
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # モデルを作成
            model = create_phase1_model(config, device=self.device)
            
            if use_fp16:
                model = model.half()
            
            # パラメータ数を計算
            total_params = sum(p.numel() for p in model.parameters())
            param_memory_mb = torch.cuda.memory_allocated() / 1e6
            
            # 推論テスト
            model.eval()
            x = torch.randint(0, vocab_size, (batch_size, n_seq), device=self.device)
            
            with torch.no_grad():
                if use_fp16:
                    with torch.cuda.amp.autocast():
                        y = model(x)
                else:
                    y = model(x)
            
            inference_memory_mb = torch.cuda.max_memory_allocated() / 1e6
            
            # 学習テスト（オプション）
            training_memory_mb = None
            if test_training:
                torch.cuda.reset_peak_memory_stats()
                model.train()
                
                x = torch.randint(0, vocab_size, (batch_size, n_seq), device=self.device)
                
                if use_fp16:
                    with torch.cuda.amp.autocast():
                        y = model(x)
                else:
                    y = model(x)
                
                loss = y.sum()
                loss.backward()
                
                training_memory_mb = torch.cuda.max_memory_allocated() / 1e6
            
            # スループットを測定
            model.eval()
            torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(10):
                    x = torch.randint(0, vocab_size, (batch_size, n_seq), device=self.device)
                    if use_fp16:
                        with torch.cuda.amp.autocast():
                            y = model(x)
                    else:
                        y = model(x)
            
            torch.cuda.synchronize()
            elapsed_time = time.time() - start_time
            throughput = (batch_size * n_seq * 10) / elapsed_time
            
            # クリーンアップ
            del model, x, y
            if test_training:
                del loss
            torch.cuda.empty_cache()
            
            result = {
                "config": {
                    "vocab_size": vocab_size,
                    "d_model": d_model,
                    "n_layers": n_layers,
                    "n_seq": n_seq,
                    "batch_size": batch_size,
                    "htt_rank": htt_rank,
                    "ar_ssm_max_rank": ar_ssm_max_rank,
                    "use_fp16": use_fp16,
                    "use_gradient_checkpointing": use_gradient_checkpointing,
                },
                "parameters": {
                    "total": total_params,
                    "millions": total_params / 1e6,
                    "billions": total_params / 1e9,
                },
                "memory": {
                    "parameter_mb": param_memory_mb,
                    "inference_peak_mb": inference_memory_mb,
                    "training_peak_mb": training_memory_mb,
                    "inference_peak_gb": inference_memory_mb / 1e3,
                    "training_peak_gb": training_memory_mb / 1e3 if training_memory_mb else None,
                },
                "performance": {
                    "throughput_tokens_per_sec": throughput,
                },
                "status": "SUCCESS"
            }
            
            return result
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                return None
            else:
                raise
    
    def binary_search_max_parameters(
        self,
        target_vram_gb: float = 8.0,
        vocab_size: int = 50000,
        n_seq: int = 2048,
        batch_size: int = 1,
        test_training: bool = False
    ) -> Dict[str, Any]:
        """
        二分探索で最大パラメータ数を見つける
        
        Args:
            target_vram_gb: 目標VRAM制約（GB）
            vocab_size: 語彙サイズ
            n_seq: シーケンス長
            batch_size: バッチサイズ
            test_training: 学習時のメモリもテストするか
        """
        print(f"\n{'='*80}")
        print(f"二分探索: {target_vram_gb}GB VRAM制約下での最大パラメータ数")
        print(f"{'='*80}")
        print(f"設定: vocab={vocab_size}, seq={n_seq}, batch={batch_size}")
        print(f"モード: {'Training' if test_training else 'Inference'}")
        
        # 探索範囲を設定
        # 小さいモデルから始める
        min_d_model = 256
        max_d_model = 4096
        
        best_config = None
        best_params = 0
        
        # d_modelで二分探索
        while min_d_model <= max_d_model:
            mid_d_model = (min_d_model + max_d_model) // 2
            
            # レイヤー数を調整（d_modelに応じて）
            if mid_d_model <= 512:
                n_layers = 12
            elif mid_d_model <= 1024:
                n_layers = 8
            else:
                n_layers = 6
            
            print(f"\nテスト中: d_model={mid_d_model}, n_layers={n_layers}")
            
            result = self.test_configuration(
                vocab_size=vocab_size,
                d_model=mid_d_model,
                n_layers=n_layers,
                n_seq=n_seq,
                batch_size=batch_size,
                test_training=test_training
            )
            
            if result is None:
                # OOM - より小さいモデルを試す
                print(f"  ❌ OOM")
                max_d_model = mid_d_model - 64
            else:
                # 成功 - メモリ使用量をチェック
                memory_key = "training_peak_gb" if test_training else "inference_peak_gb"
                memory_gb = result["memory"][memory_key]
                
                print(f"  ✅ 成功: {result['parameters']['millions']:.1f}M params, "
                      f"{memory_gb:.2f}GB VRAM")
                
                if memory_gb <= target_vram_gb * 0.9:  # 90%以下を目標
                    # まだ余裕がある - より大きいモデルを試す
                    best_config = result
                    best_params = result['parameters']['total']
                    min_d_model = mid_d_model + 64
                else:
                    # ギリギリ - これを保存して終了
                    if result['parameters']['total'] > best_params:
                        best_config = result
                        best_params = result['parameters']['total']
                    break
        
        if best_config:
            print(f"\n{'='*80}")
            print(f"最大構成を発見:")
            print(f"  パラメータ数: {best_config['parameters']['millions']:.1f}M "
                  f"({best_config['parameters']['billions']:.3f}B)")
            print(f"  d_model: {best_config['config']['d_model']}")
            print(f"  n_layers: {best_config['config']['n_layers']}")
            memory_key = "training_peak_gb" if test_training else "inference_peak_gb"
            print(f"  VRAM使用量: {best_config['memory'][memory_key]:.2f}GB / {target_vram_gb}GB")
            print(f"  スループット: {best_config['performance']['throughput_tokens_per_sec']:.1f} tokens/s")
            print(f"{'='*80}")
        
        return best_config
    
    def test_standard_configurations(self) -> List[Dict[str, Any]]:
        """標準的な構成をテスト"""
        print(f"\n{'='*80}")
        print("標準構成のテスト")
        print(f"{'='*80}")
        
        configurations = [
            # Small models
            {"name": "Small (論文)", "vocab_size": 10000, "d_model": 512, "n_layers": 6, "n_seq": 512, "batch_size": 2},
            {"name": "Small (実用)", "vocab_size": 50000, "d_model": 512, "n_layers": 6, "n_seq": 2048, "batch_size": 1},
            
            # Medium models
            {"name": "Medium", "vocab_size": 50000, "d_model": 768, "n_layers": 8, "n_seq": 2048, "batch_size": 1},
            
            # Large models
            {"name": "Large", "vocab_size": 50000, "d_model": 1024, "n_layers": 12, "n_seq": 2048, "batch_size": 1},
            
            # Extra Large models
            {"name": "XL", "vocab_size": 50000, "d_model": 1536, "n_layers": 12, "n_seq": 2048, "batch_size": 1},
            {"name": "XXL", "vocab_size": 50000, "d_model": 2048, "n_layers": 12, "n_seq": 2048, "batch_size": 1},
        ]
        
        results = []
        
        for config in configurations:
            print(f"\n{config['name']}:")
            print(f"  vocab={config['vocab_size']}, d={config['d_model']}, "
                  f"layers={config['n_layers']}, seq={config['n_seq']}, batch={config['batch_size']}")
            
            # 推論テスト
            result_inference = self.test_configuration(
                vocab_size=config['vocab_size'],
                d_model=config['d_model'],
                n_layers=config['n_layers'],
                n_seq=config['n_seq'],
                batch_size=config['batch_size'],
                test_training=False
            )
            
            if result_inference:
                print(f"  推論: {result_inference['parameters']['millions']:.1f}M params, "
                      f"{result_inference['memory']['inference_peak_gb']:.2f}GB VRAM, "
                      f"{result_inference['performance']['throughput_tokens_per_sec']:.1f} tokens/s")
                
                # 学習テスト
                result_training = self.test_configuration(
                    vocab_size=config['vocab_size'],
                    d_model=config['d_model'],
                    n_layers=config['n_layers'],
                    n_seq=config['n_seq'],
                    batch_size=config['batch_size'],
                    test_training=True
                )
                
                if result_training:
                    print(f"  学習: {result_training['memory']['training_peak_gb']:.2f}GB VRAM")
                    result_inference['memory']['training_peak_mb'] = result_training['memory']['training_peak_mb']
                    result_inference['memory']['training_peak_gb'] = result_training['memory']['training_peak_gb']
                else:
                    print(f"  学習: ❌ OOM")
                
                result_inference['name'] = config['name']
                results.append(result_inference)
            else:
                print(f"  推論: ❌ OOM")
        
        return results
    
    def run_all_tests(self):
        """すべてのテストを実行"""
        print(f"\n{'='*80}")
        print("最大パラメータ数の包括的テスト")
        print(f"{'='*80}")
        
        # 標準構成をテスト
        standard_results = self.test_standard_configurations()
        self.results["configurations"] = standard_results
        
        # 8GB制約下での最大パラメータ数（推論）
        max_8gb_inference = self.binary_search_max_parameters(
            target_vram_gb=8.0,
            vocab_size=50000,
            n_seq=2048,
            batch_size=1,
            test_training=False
        )
        if max_8gb_inference:
            self.results["max_parameters"]["8gb_inference"] = max_8gb_inference
        
        # 8GB制約下での最大パラメータ数（学習）
        max_8gb_training = self.binary_search_max_parameters(
            target_vram_gb=8.0,
            vocab_size=50000,
            n_seq=2048,
            batch_size=1,
            test_training=True
        )
        if max_8gb_training:
            self.results["max_parameters"]["8gb_training"] = max_8gb_training
        
        # 10GB制約下での最大パラメータ数（推論）
        if self.gpu_memory_gb >= 10.0:
            max_10gb_inference = self.binary_search_max_parameters(
                target_vram_gb=10.0,
                vocab_size=50000,
                n_seq=2048,
                batch_size=1,
                test_training=False
            )
            if max_10gb_inference:
                self.results["max_parameters"]["10gb_inference"] = max_10gb_inference
        
        self._print_summary()
    
    def _print_summary(self):
        """結果のサマリーを表示"""
        print(f"\n{'='*80}")
        print("テスト結果サマリー")
        print(f"{'='*80}")
        
        print("\n標準構成:")
        for config in self.results["configurations"]:
            print(f"  {config['name']}: {config['parameters']['millions']:.1f}M params")
            print(f"    推論: {config['memory']['inference_peak_gb']:.2f}GB")
            if config['memory'].get('training_peak_gb'):
                print(f"    学習: {config['memory']['training_peak_gb']:.2f}GB")
        
        print("\n最大パラメータ数:")
        for key, config in self.results["max_parameters"].items():
            print(f"  {key}: {config['parameters']['millions']:.1f}M params "
                  f"({config['parameters']['billions']:.3f}B)")
            memory_key = "training_peak_gb" if "training" in key else "inference_peak_gb"
            print(f"    VRAM: {config['memory'][memory_key]:.2f}GB")
            print(f"    d_model={config['config']['d_model']}, "
                  f"n_layers={config['config']['n_layers']}")
    
    def save_results(self, output_path: str = "results/benchmarks/max_parameters_test.json"):
        """結果をJSONファイルに保存"""
        output_file = project_root / output_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n結果を保存しました: {output_file}")


def main():
    """メイン関数"""
    tester = MaxParameterTester()
    tester.run_all_tests()
    tester.save_results()


if __name__ == "__main__":
    main()
