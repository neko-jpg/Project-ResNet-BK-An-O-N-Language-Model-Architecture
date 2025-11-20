#!/usr/bin/env python3
"""
Phase 2可視化スクリプト

Phase 2モデルの学習過程と診断情報を可視化します。

使用例:
    # 学習ログから可視化
    python scripts/visualize_phase2.py --log-dir results/phase2_training --output-dir results/visualizations
    
    # 特定の可視化のみ実行
    python scripts/visualize_phase2.py --log-dir results/phase2_training --only learning_curves
    
    # インタラクティブモード
    python scripts/visualize_phase2.py --log-dir results/phase2_training --interactive

Requirements: 7.6, 10.7
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

# スタイル設定
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10


class Phase2Visualizer:
    """
    Phase 2可視化クラス
    
    学習ログから診断情報を読み込み、各種グラフを生成します。
    """
    
    def __init__(self, log_dir: Path, output_dir: Path):
        """
        Args:
            log_dir: 学習ログディレクトリ
            output_dir: 出力ディレクトリ
        """
        self.log_dir = Path(log_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ログデータ
        self.training_log = None
        self.diagnostics_log = None
        
        # データをロード
        self._load_logs()
    
    def _load_logs(self):
        """ログファイルをロード"""
        # 学習ログ（Loss, Perplexity等）
        training_log_path = self.log_dir / "training_log.json"
        if training_log_path.exists():
            with open(training_log_path, 'r') as f:
                self.training_log = json.load(f)
            print(f"✓ 学習ログをロード: {training_log_path}")
        else:
            warnings.warn(f"学習ログが見つかりません: {training_log_path}")
            self.training_log = {}
        
        # 診断ログ（Γ, SNR, 共鳴情報等）
        diagnostics_log_path = self.log_dir / "diagnostics_log.json"
        if diagnostics_log_path.exists():
            with open(diagnostics_log_path, 'r') as f:
                self.diagnostics_log = json.load(f)
            print(f"✓ 診断ログをロード: {diagnostics_log_path}")
        else:
            warnings.warn(f"診断ログが見つかりません: {diagnostics_log_path}")
            self.diagnostics_log = {}
    
    def visualize_all(self):
        """すべての可視化を実行"""
        print("\n" + "="*60)
        print("Phase 2 可視化スクリプト")
        print("="*60)
        
        # 14.1: 学習曲線
        self.plot_learning_curves()
        
        # 14.2: Γ変化
        self.plot_gamma_evolution()
        
        # 14.3: SNR統計
        self.plot_snr_statistics()
        
        # 14.4: 共鳴情報
        self.plot_resonance_info()
        
        print("\n" + "="*60)
        print(f"✓ すべての可視化が完了しました")
        print(f"  出力ディレクトリ: {self.output_dir}")
        print("="*60)
    
    # ========================================
    # 14.1: 学習曲線可視化
    # ========================================
    
    def plot_learning_curves(self):
        """
        学習曲線を可視化
        
        - Loss曲線
        - Perplexity曲線
        
        Requirements: 7.6
        """
        print("\n[14.1] 学習曲線を可視化中...")
        
        if not self.training_log:
            warnings.warn("学習ログがないため、学習曲線をスキップします")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Loss曲線
        if 'loss' in self.training_log:
            losses = self.training_log['loss']
            steps = list(range(len(losses)))
            
            axes[0].plot(steps, losses, linewidth=2, color='#2E86AB', label='Training Loss')
            axes[0].set_xlabel('Training Step')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training Loss Curve', fontweight='bold')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # 移動平均を追加
            if len(losses) > 10:
                window = min(50, len(losses) // 10)
                moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
                axes[0].plot(
                    range(window-1, len(losses)),
                    moving_avg,
                    linewidth=2,
                    color='#A23B72',
                    linestyle='--',
                    label=f'Moving Avg (window={window})'
                )
                axes[0].legend()
        
        # Perplexity曲線
        if 'perplexity' in self.training_log:
            ppls = self.training_log['perplexity']
            steps = list(range(len(ppls)))
            
            axes[1].plot(steps, ppls, linewidth=2, color='#F18F01', label='Perplexity')
            axes[1].set_xlabel('Training Step')
            axes[1].set_ylabel('Perplexity')
            axes[1].set_title('Perplexity Curve', fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # 移動平均を追加
            if len(ppls) > 10:
                window = min(50, len(ppls) // 10)
                moving_avg = np.convolve(ppls, np.ones(window)/window, mode='valid')
                axes[1].plot(
                    range(window-1, len(ppls)),
                    moving_avg,
                    linewidth=2,
                    color='#C73E1D',
                    linestyle='--',
                    label=f'Moving Avg (window={window})'
                )
                axes[1].legend()
        
        plt.tight_layout()
        output_path = self.output_dir / "learning_curves.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ 学習曲線を保存: {output_path}")
    
    # ========================================
    # 14.2: Γ変化可視化
    # ========================================
    
    def plot_gamma_evolution(self):
        """
        Γ（忘却率）の時間変化を可視化
        
        - 各層のΓ値の時間変化
        - ヒートマップ
        
        Requirements: 7.6
        """
        print("\n[14.2] Γ変化を可視化中...")
        
        if not self.diagnostics_log or 'gamma_values' not in self.diagnostics_log:
            warnings.warn("Γデータがないため、Γ変化をスキップします")
            return
        
        gamma_data = self.diagnostics_log['gamma_values']
        
        # データ構造: List[Dict[str, List[float]]]
        # gamma_data[step]['layer_0'] = [gamma values for batch]
        
        # 各層のΓ平均値を時系列で取得
        num_steps = len(gamma_data)
        layer_names = list(gamma_data[0].keys()) if gamma_data else []
        num_layers = len(layer_names)
        
        if num_layers == 0:
            warnings.warn("層データがありません")
            return
        
        # 時系列データを構築
        gamma_evolution = np.zeros((num_layers, num_steps))
        for step_idx, step_data in enumerate(gamma_data):
            for layer_idx, layer_name in enumerate(layer_names):
                if layer_name in step_data:
                    # バッチ平均を取る
                    gamma_evolution[layer_idx, step_idx] = np.mean(step_data[layer_name])
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # 1. 時系列プロット
        for layer_idx, layer_name in enumerate(layer_names):
            axes[0].plot(
                range(num_steps),
                gamma_evolution[layer_idx, :],
                linewidth=2,
                label=layer_name,
                alpha=0.8
            )
        
        axes[0].set_xlabel('Training Step')
        axes[0].set_ylabel('Γ (Decay Rate)')
        axes[0].set_title('Γ Evolution Across Layers', fontweight='bold')
        axes[0].legend(loc='best', ncol=2)
        axes[0].grid(True, alpha=0.3)
        
        # 2. ヒートマップ
        im = axes[1].imshow(
            gamma_evolution,
            aspect='auto',
            cmap='YlOrRd',
            interpolation='nearest'
        )
        axes[1].set_xlabel('Training Step')
        axes[1].set_ylabel('Layer')
        axes[1].set_title('Γ Heatmap (Layers × Time)', fontweight='bold')
        axes[1].set_yticks(range(num_layers))
        axes[1].set_yticklabels(layer_names)
        
        # カラーバー
        cbar = plt.colorbar(im, ax=axes[1])
        cbar.set_label('Γ Value', rotation=270, labelpad=20)
        
        plt.tight_layout()
        output_path = self.output_dir / "gamma_evolution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Γ変化を保存: {output_path}")
        
        # 統計情報を出力
        print(f"  - Γ平均値: {gamma_evolution.mean():.6f}")
        print(f"  - Γ標準偏差: {gamma_evolution.std():.6f}")
        print(f"  - Γ最小値: {gamma_evolution.min():.6f}")
        print(f"  - Γ最大値: {gamma_evolution.max():.6f}")
    
    # ========================================
    # 14.3: SNR統計可視化
    # ========================================
    
    def plot_snr_statistics(self):
        """
        SNR統計を可視化
        
        - SNR分布のヒストグラム
        - SNRの時間変化
        
        Requirements: 7.6
        """
        print("\n[14.3] SNR統計を可視化中...")
        
        if not self.diagnostics_log or 'snr_stats' not in self.diagnostics_log:
            warnings.warn("SNRデータがないため、SNR統計をスキップします")
            return
        
        snr_data = self.diagnostics_log['snr_stats']
        
        # データ構造: List[Dict[str, float]]
        # snr_data[step] = {'mean_snr': ..., 'std_snr': ..., 'min_snr': ..., 'max_snr': ...}
        
        num_steps = len(snr_data)
        
        # 時系列データを抽出
        mean_snr = [step_data.get('mean_snr', 0) for step_data in snr_data]
        std_snr = [step_data.get('std_snr', 0) for step_data in snr_data]
        min_snr = [step_data.get('min_snr', 0) for step_data in snr_data]
        max_snr = [step_data.get('max_snr', 0) for step_data in snr_data]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. SNR時系列（平均±標準偏差）
        steps = list(range(num_steps))
        axes[0, 0].plot(steps, mean_snr, linewidth=2, color='#2E86AB', label='Mean SNR')
        axes[0, 0].fill_between(
            steps,
            np.array(mean_snr) - np.array(std_snr),
            np.array(mean_snr) + np.array(std_snr),
            alpha=0.3,
            color='#2E86AB',
            label='±1 Std Dev'
        )
        axes[0, 0].axhline(y=2.0, color='red', linestyle='--', linewidth=1, label='Threshold (2.0)')
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('SNR')
        axes[0, 0].set_title('Mean SNR Evolution', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. SNR範囲（Min-Max）
        axes[0, 1].fill_between(
            steps,
            min_snr,
            max_snr,
            alpha=0.4,
            color='#F18F01',
            label='SNR Range'
        )
        axes[0, 1].plot(steps, mean_snr, linewidth=2, color='#C73E1D', label='Mean SNR')
        axes[0, 1].axhline(y=2.0, color='red', linestyle='--', linewidth=1, label='Threshold (2.0)')
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('SNR')
        axes[0, 1].set_title('SNR Range (Min-Max)', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. SNR分布ヒストグラム（最終ステップ）
        if mean_snr:
            # 全ステップのSNR値を収集（簡易版：平均値のみ）
            all_snr_values = mean_snr
            axes[1, 0].hist(all_snr_values, bins=50, color='#A23B72', alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(x=2.0, color='red', linestyle='--', linewidth=2, label='Threshold (2.0)')
            axes[1, 0].set_xlabel('SNR Value')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('SNR Distribution (All Steps)', fontweight='bold')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. 低SNR比率の時間変化
        # 低SNR比率 = SNR < 2.0 の割合（簡易推定）
        low_snr_ratio = [(1.0 if snr < 2.0 else 0.0) for snr in mean_snr]
        axes[1, 1].plot(steps, low_snr_ratio, linewidth=2, color='#C73E1D', marker='o', markersize=3)
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Low SNR Ratio')
        axes[1, 1].set_title('Low SNR Ratio Evolution (SNR < 2.0)', fontweight='bold')
        axes[1, 1].set_ylim(-0.1, 1.1)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "snr_statistics.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ SNR統計を保存: {output_path}")
        
        # 統計情報を出力
        if mean_snr:
            print(f"  - 最終SNR平均: {mean_snr[-1]:.4f}")
            print(f"  - 最終SNR標準偏差: {std_snr[-1]:.4f}")
            print(f"  - 全体SNR平均: {np.mean(mean_snr):.4f}")
    
    # ========================================
    # 14.4: 共鳴情報可視化
    # ========================================
    
    def plot_resonance_info(self):
        """
        共鳴情報を可視化
        
        - 共鳴エネルギーのヒートマップ
        - 共鳴マスクの可視化
        
        Requirements: 10.7
        """
        print("\n[14.4] 共鳴情報を可視化中...")
        
        if not self.diagnostics_log or 'resonance_info' not in self.diagnostics_log:
            warnings.warn("共鳴データがないため、共鳴情報をスキップします")
            return
        
        resonance_data = self.diagnostics_log['resonance_info']
        
        # データ構造: List[Dict[str, Any]]
        # resonance_data[step] = {
        #     'diag_energy': [...],
        #     'resonance_mask': [...],
        #     'num_resonant': float,
        #     'total_energy': float
        # }
        
        num_steps = len(resonance_data)
        
        # 時系列データを抽出
        num_resonant = [step_data.get('num_resonant', 0) for step_data in resonance_data]
        total_energy = [step_data.get('total_energy', 0) for step_data in resonance_data]
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. 共鳴成分数の時間変化
        ax1 = fig.add_subplot(gs[0, 0])
        steps = list(range(num_steps))
        ax1.plot(steps, num_resonant, linewidth=2, color='#2E86AB', marker='o', markersize=3)
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Number of Resonant Modes')
        ax1.set_title('Resonant Modes Evolution', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. 総共鳴エネルギーの時間変化
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(steps, total_energy, linewidth=2, color='#F18F01', marker='s', markersize=3)
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Total Resonance Energy')
        ax2.set_title('Total Resonance Energy Evolution', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. 共鳴エネルギーヒートマップ（最終ステップ）
        if resonance_data and 'diag_energy' in resonance_data[-1]:
            diag_energy = resonance_data[-1]['diag_energy']
            
            # diag_energyが2D配列の場合（複数ヘッド）
            if isinstance(diag_energy, list) and len(diag_energy) > 0:
                if isinstance(diag_energy[0], list):
                    # 2D: (num_heads, head_dim)
                    energy_matrix = np.array(diag_energy)
                else:
                    # 1D: (head_dim,) → 1行に変換
                    energy_matrix = np.array(diag_energy).reshape(1, -1)
                
                ax3 = fig.add_subplot(gs[1, :])
                im = ax3.imshow(
                    energy_matrix,
                    aspect='auto',
                    cmap='viridis',
                    interpolation='nearest'
                )
                ax3.set_xlabel('Dimension')
                ax3.set_ylabel('Head')
                ax3.set_title('Resonance Energy Heatmap (Final Step)', fontweight='bold')
                
                # カラーバー
                cbar = plt.colorbar(im, ax=ax3)
                cbar.set_label('Energy', rotation=270, labelpad=20)
        
        # 4. 共鳴マスクの可視化（最終ステップ）
        if resonance_data and 'resonance_mask' in resonance_data[-1]:
            resonance_mask = resonance_data[-1]['resonance_mask']
            
            # resonance_maskが2D配列の場合
            if isinstance(resonance_mask, list) and len(resonance_mask) > 0:
                if isinstance(resonance_mask[0], list):
                    # 2D: (num_heads, head_dim)
                    mask_matrix = np.array(resonance_mask, dtype=float)
                else:
                    # 1D: (head_dim,) → 1行に変換
                    mask_matrix = np.array(resonance_mask, dtype=float).reshape(1, -1)
                
                ax4 = fig.add_subplot(gs[2, :])
                im = ax4.imshow(
                    mask_matrix,
                    aspect='auto',
                    cmap='RdYlGn',
                    interpolation='nearest',
                    vmin=0,
                    vmax=1
                )
                ax4.set_xlabel('Dimension')
                ax4.set_ylabel('Head')
                ax4.set_title('Resonance Mask (Final Step, Green=Active)', fontweight='bold')
                
                # カラーバー
                cbar = plt.colorbar(im, ax=ax4)
                cbar.set_label('Mask Value', rotation=270, labelpad=20)
        
        output_path = self.output_dir / "resonance_info.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ 共鳴情報を保存: {output_path}")
        
        # 統計情報を出力
        if num_resonant:
            print(f"  - 最終共鳴成分数: {num_resonant[-1]:.2f}")
            print(f"  - 最終総エネルギー: {total_energy[-1]:.6f}")
            print(f"  - 平均共鳴成分数: {np.mean(num_resonant):.2f}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Phase 2可視化スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # すべての可視化を実行
  python scripts/visualize_phase2.py --log-dir results/phase2_training
  
  # 出力ディレクトリを指定
  python scripts/visualize_phase2.py --log-dir results/phase2_training --output-dir results/viz
  
  # 特定の可視化のみ実行
  python scripts/visualize_phase2.py --log-dir results/phase2_training --only learning_curves
        """
    )
    
    parser.add_argument(
        '--log-dir',
        type=str,
        default='results/phase2_training',
        help='学習ログディレクトリ（デフォルト: results/phase2_training）'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/visualizations',
        help='出力ディレクトリ（デフォルト: results/visualizations）'
    )
    
    parser.add_argument(
        '--only',
        type=str,
        choices=['learning_curves', 'gamma', 'snr', 'resonance'],
        help='特定の可視化のみ実行'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='インタラクティブモード（グラフを表示）'
    )
    
    args = parser.parse_args()
    
    # 可視化実行
    visualizer = Phase2Visualizer(
        log_dir=Path(args.log_dir),
        output_dir=Path(args.output_dir)
    )
    
    if args.only:
        # 特定の可視化のみ
        if args.only == 'learning_curves':
            visualizer.plot_learning_curves()
        elif args.only == 'gamma':
            visualizer.plot_gamma_evolution()
        elif args.only == 'snr':
            visualizer.plot_snr_statistics()
        elif args.only == 'resonance':
            visualizer.plot_resonance_info()
    else:
        # すべての可視化
        visualizer.visualize_all()
    
    # インタラクティブモード
    if args.interactive:
        print("\nインタラクティブモードで表示中...")
        plt.show()


if __name__ == "__main__":
    main()
