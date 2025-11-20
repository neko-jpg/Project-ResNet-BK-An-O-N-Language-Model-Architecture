"""
Phase 2 Diagnostics Example

このスクリプトは、Phase 2モデルの診断機能を示します。

主な内容:
1. Γ値（忘却率）の監視
2. SNR統計の取得
3. 共鳴情報の可視化
4. 安定性メトリクスの追跡
5. Fast Weightsのエネルギー分析

Requirements: 11.10
Author: Project MUSE Team
Date: 2025-01-20
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import json

from src.models.phase2 import Phase2IntegratedModel, create_phase2_model, Phase2Config


def setup_output_dir() -> Path:
    """出力ディレクトリを設定"""
    output_dir = Path("results/phase2_diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def example_1_gamma_monitoring():
    """
    例1: Γ値（忘却率）の監視
    
    各層のΓ値を時系列で監視します。
    """
    print("=" * 60)
    print("例1: Γ値（忘却率）の監視")
    print("=" * 60)
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nデバイス: {device}")
    
    # モデル作成
    print("\nモデルを作成中...")
    config = Phase2Config(
        vocab_size=1000,
        d_model=128,
        n_layers=3,  # 3層で可視化
        n_seq=64,
        num_heads=4,
        head_dim=32,
        base_decay=0.01,
        adaptive_decay=True,
    )
    model = create_phase2_model(config=config, device=device)
    model.eval()
    
    # 複数のシーケンスでΓ値を収集
    num_sequences = 10
    gamma_history = {i: [] for i in range(config.n_layers)}
    
    print(f"\n{num_sequences}個のシーケンスでΓ値を収集中...")
    
    with torch.no_grad():
        for seq_idx in range(num_sequences):
            # ランダムな入力
            input_ids = torch.randint(0, 1000, (1, config.n_seq)).to(device)
            
            # 状態をリセット
            model.reset_state()
            
            # 診断情報付きforward pass
            logits, diagnostics = model(input_ids, return_diagnostics=True)
            
            # 各層のΓ値を記録
            for layer_idx, gamma in enumerate(diagnostics['gamma_values']):
                if gamma is not None:
                    gamma_mean = gamma.mean().item()
                    gamma_history[layer_idx].append(gamma_mean)
            
            if (seq_idx + 1) % 5 == 0:
                print(f"  処理済み: {seq_idx + 1}/{num_sequences}")
    
    # 統計を表示
    print("\nΓ値の統計:")
    for layer_idx in range(config.n_layers):
        if gamma_history[layer_idx]:
            values = np.array(gamma_history[layer_idx])
            print(f"\n  Layer {layer_idx}:")
            print(f"    平均: {values.mean():.6f}")
            print(f"    標準偏差: {values.std():.6f}")
            print(f"    最小値: {values.min():.6f}")
            print(f"    最大値: {values.max():.6f}")
    
    # 可視化
    output_dir = setup_output_dir()
    
    plt.figure(figsize=(12, 6))
    for layer_idx in range(config.n_layers):
        if gamma_history[layer_idx]:
            plt.plot(gamma_history[layer_idx], marker='o', label=f'Layer {layer_idx}')
    
    plt.xlabel('Sequence Index')
    plt.ylabel('Γ (Decay Rate)')
    plt.title('Γ値の時系列変化')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = output_dir / "gamma_monitoring.png"
    plt.savefig(save_path, dpi=150)
    print(f"\n可視化を保存: {save_path}")
    plt.close()
    
    return gamma_history


def example_2_snr_statistics():
    """
    例2: SNR統計の取得
    
    Signal-to-Noise Ratio統計を収集して分析します。
    """
    print("\n" + "=" * 60)
    print("例2: SNR統計の取得")
    print("=" * 60)
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nデバイス: {device}")
    
    # モデル作成
    print("\nモデルを作成中...")
    config = Phase2Config(
        vocab_size=1000,
        d_model=128,
        n_layers=3,
        n_seq=64,
        num_heads=4,
        head_dim=32,
        snr_threshold=2.0,
    )
    model = create_phase2_model(config=config, device=device)
    model.eval()
    
    # SNR統計を収集
    num_sequences = 20
    snr_history = {i: {'mean_snr': [], 'std_snr': [], 'min_snr': [], 'max_snr': []} 
                   for i in range(config.n_layers)}
    
    print(f"\n{num_sequences}個のシーケンスでSNR統計を収集中...")
    
    with torch.no_grad():
        for seq_idx in range(num_sequences):
            # ランダムな入力
            input_ids = torch.randint(0, 1000, (1, config.n_seq)).to(device)
            
            # 状態をリセット
            model.reset_state()
            
            # 診断情報付きforward pass
            logits, diagnostics = model(input_ids, return_diagnostics=True)
            
            # 各層のSNR統計を記録
            for layer_idx, snr_stats in enumerate(diagnostics['snr_stats']):
                if snr_stats:
                    for key in ['mean_snr', 'std_snr', 'min_snr', 'max_snr']:
                        if key in snr_stats:
                            snr_history[layer_idx][key].append(snr_stats[key])
            
            if (seq_idx + 1) % 10 == 0:
                print(f"  処理済み: {seq_idx + 1}/{num_sequences}")
    
    # 統計を表示
    print("\nSNR統計:")
    for layer_idx in range(config.n_layers):
        print(f"\n  Layer {layer_idx}:")
        for key, values in snr_history[layer_idx].items():
            if values:
                arr = np.array(values)
                print(f"    {key}:")
                print(f"      全体平均: {arr.mean():.4f}")
                print(f"      全体標準偏差: {arr.std():.4f}")
    
    # 可視化
    output_dir = setup_output_dir()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SNR統計の時系列変化', fontsize=14)
    
    metrics = ['mean_snr', 'std_snr', 'min_snr', 'max_snr']
    titles = ['Mean SNR', 'Std SNR', 'Min SNR', 'Max SNR']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        for layer_idx in range(config.n_layers):
            if snr_history[layer_idx][metric]:
                ax.plot(snr_history[layer_idx][metric], 
                       marker='o', label=f'Layer {layer_idx}', alpha=0.7)
        
        ax.set_xlabel('Sequence Index')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = output_dir / "snr_statistics.png"
    plt.savefig(save_path, dpi=150)
    print(f"\n可視化を保存: {save_path}")
    plt.close()
    
    return snr_history


def example_3_resonance_visualization():
    """
    例3: 共鳴情報の可視化
    
    Memory Resonance Layerの共鳴パターンを可視化します。
    """
    print("\n" + "=" * 60)
    print("例3: 共鳴情報の可視化")
    print("=" * 60)
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nデバイス: {device}")
    
    # モデル作成
    print("\nモデルを作成中...")
    config = Phase2Config(
        vocab_size=1000,
        d_model=128,
        n_layers=3,
        n_seq=64,
        num_heads=4,
        head_dim=32,
        resonance_enabled=True,
        resonance_energy_threshold=0.1,
    )
    model = create_phase2_model(config=config, device=device)
    model.eval()
    
    # 共鳴情報を収集
    print("\n共鳴情報を収集中...")
    
    input_ids = torch.randint(0, 1000, (1, config.n_seq)).to(device)
    model.reset_state()
    
    with torch.no_grad():
        logits, diagnostics = model(input_ids, return_diagnostics=True)
    
    # 共鳴情報を表示
    print("\n共鳴情報:")
    for layer_idx, resonance_info in enumerate(diagnostics['resonance_info']):
        print(f"\n  Layer {layer_idx}:")
        if resonance_info:
            for key, value in resonance_info.items():
                if isinstance(value, torch.Tensor):
                    if value.numel() == 1:
                        print(f"    {key}: {value.item():.4f}")
                    else:
                        print(f"    {key}: 形状={value.shape}, 平均={value.mean().item():.4f}, "
                              f"標準偏差={value.std().item():.4f}")
                else:
                    print(f"    {key}: {value}")
        else:
            print("    共鳴情報なし")
    
    # 共鳴エネルギーのヒートマップを可視化
    output_dir = setup_output_dir()
    
    fig, axes = plt.subplots(1, config.n_layers, figsize=(15, 4))
    if config.n_layers == 1:
        axes = [axes]
    
    fig.suptitle('共鳴エネルギーのヒートマップ', fontsize=14)
    
    for layer_idx, (ax, resonance_info) in enumerate(zip(axes, diagnostics['resonance_info'])):
        if resonance_info and 'diag_energy' in resonance_info:
            diag_energy = resonance_info['diag_energy'].cpu().numpy()
            
            # (B, H, D_h) → (H, D_h) に平均
            if len(diag_energy.shape) == 3:
                diag_energy = diag_energy.mean(axis=0)
            
            im = ax.imshow(diag_energy, aspect='auto', cmap='viridis')
            ax.set_title(f'Layer {layer_idx}')
            ax.set_xlabel('Dimension')
            ax.set_ylabel('Head')
            plt.colorbar(im, ax=ax)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Layer {layer_idx}')
    
    plt.tight_layout()
    
    save_path = output_dir / "resonance_heatmap.png"
    plt.savefig(save_path, dpi=150)
    print(f"\n可視化を保存: {save_path}")
    plt.close()
    
    return diagnostics['resonance_info']


def example_4_stability_tracking():
    """
    例4: 安定性メトリクスの追跡
    
    Lyapunov安定性メトリクスを時系列で追跡します。
    """
    print("\n" + "=" * 60)
    print("例4: 安定性メトリクスの追跡")
    print("=" * 60)
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nデバイス: {device}")
    
    # モデル作成
    print("\nモデルを作成中...")
    config = Phase2Config(
        vocab_size=1000,
        d_model=128,
        n_layers=3,
        n_seq=64,
        num_heads=4,
        head_dim=32,
    )
    model = create_phase2_model(config=config, device=device)
    model.eval()
    
    # 安定性メトリクスを収集
    num_sequences = 15
    stability_history = {i: {'energy': [], 'dE_dt': [], 'is_stable': []} 
                        for i in range(config.n_layers)}
    
    print(f"\n{num_sequences}個のシーケンスで安定性メトリクスを収集中...")
    
    with torch.no_grad():
        for seq_idx in range(num_sequences):
            # ランダムな入力
            input_ids = torch.randint(0, 1000, (1, config.n_seq)).to(device)
            
            # 状態をリセット
            model.reset_state()
            
            # 診断情報付きforward pass
            logits, diagnostics = model(input_ids, return_diagnostics=True)
            
            # 各層の安定性メトリクスを記録
            for layer_idx, stability in enumerate(diagnostics['stability_metrics']):
                if stability:
                    if 'energy' in stability:
                        stability_history[layer_idx]['energy'].append(stability['energy'])
                    if 'dE_dt' in stability:
                        stability_history[layer_idx]['dE_dt'].append(stability['dE_dt'])
                    if 'is_stable' in stability:
                        stability_history[layer_idx]['is_stable'].append(stability['is_stable'])
            
            if (seq_idx + 1) % 5 == 0:
                print(f"  処理済み: {seq_idx + 1}/{num_sequences}")
    
    # 統計を表示
    print("\n安定性メトリクス:")
    for layer_idx in range(config.n_layers):
        print(f"\n  Layer {layer_idx}:")
        
        if stability_history[layer_idx]['energy']:
            energies = np.array(stability_history[layer_idx]['energy'])
            print(f"    エネルギー:")
            print(f"      平均: {energies.mean():.6f}")
            print(f"      標準偏差: {energies.std():.6f}")
        
        if stability_history[layer_idx]['dE_dt']:
            dE_dts = np.array(stability_history[layer_idx]['dE_dt'])
            print(f"    dE/dt:")
            print(f"      平均: {dE_dts.mean():.6f}")
            print(f"      標準偏差: {dE_dts.std():.6f}")
        
        if stability_history[layer_idx]['is_stable']:
            stables = np.array(stability_history[layer_idx]['is_stable'])
            stable_ratio = stables.mean()
            print(f"    安定性:")
            print(f"      安定率: {stable_ratio * 100:.1f}%")
    
    # 可視化
    output_dir = setup_output_dir()
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('安定性メトリクスの時系列変化', fontsize=14)
    
    # エネルギー
    ax = axes[0]
    for layer_idx in range(config.n_layers):
        if stability_history[layer_idx]['energy']:
            ax.plot(stability_history[layer_idx]['energy'], 
                   marker='o', label=f'Layer {layer_idx}', alpha=0.7)
    ax.set_xlabel('Sequence Index')
    ax.set_ylabel('Energy ||W||²')
    ax.set_title('Fast Weightsのエネルギー')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # dE/dt
    ax = axes[1]
    for layer_idx in range(config.n_layers):
        if stability_history[layer_idx]['dE_dt']:
            ax.plot(stability_history[layer_idx]['dE_dt'], 
                   marker='o', label=f'Layer {layer_idx}', alpha=0.7)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Stability Threshold')
    ax.set_xlabel('Sequence Index')
    ax.set_ylabel('dE/dt')
    ax.set_title('エネルギー微分（Lyapunov条件: dE/dt ≤ 0）')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = output_dir / "stability_tracking.png"
    plt.savefig(save_path, dpi=150)
    print(f"\n可視化を保存: {save_path}")
    plt.close()
    
    return stability_history


def example_5_comprehensive_report():
    """
    例5: 包括的な診断レポート
    
    すべての診断情報を収集してJSONレポートを生成します。
    """
    print("\n" + "=" * 60)
    print("例5: 包括的な診断レポート")
    print("=" * 60)
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nデバイス: {device}")
    
    # モデル作成
    print("\nモデルを作成中...")
    config = Phase2Config(
        vocab_size=1000,
        d_model=128,
        n_layers=3,
        n_seq=64,
        num_heads=4,
        head_dim=32,
    )
    model = create_phase2_model(config=config, device=device)
    model.eval()
    
    # 診断情報を収集
    print("\n診断情報を収集中...")
    
    input_ids = torch.randint(0, 1000, (2, config.n_seq)).to(device)
    model.reset_state()
    
    with torch.no_grad():
        logits, diagnostics = model(input_ids, return_diagnostics=True)
    
    # モデル統計を取得
    model_stats = model.get_statistics()
    
    # レポートを作成
    report = {
        'model_config': config.to_dict(),
        'model_statistics': {
            'num_parameters': model_stats['num_parameters'],
            'num_trainable_parameters': model_stats['num_trainable_parameters'],
            'num_layers': model_stats['num_layers'],
            'd_model': model_stats['d_model'],
            'vocab_size': model_stats['vocab_size'],
            'n_seq': model_stats['n_seq'],
        },
        'diagnostics': {
            'gamma_values': [],
            'snr_stats': [],
            'resonance_info': [],
            'stability_metrics': [],
        },
        'block_statistics': model_stats['block_stats'],
    }
    
    # 診断情報を整形
    for layer_idx in range(config.n_layers):
        # Γ値
        if layer_idx < len(diagnostics['gamma_values']):
            gamma = diagnostics['gamma_values'][layer_idx]
            if gamma is not None:
                report['diagnostics']['gamma_values'].append({
                    'layer': layer_idx,
                    'mean': gamma.mean().item(),
                    'std': gamma.std().item(),
                    'min': gamma.min().item(),
                    'max': gamma.max().item(),
                })
        
        # SNR統計
        if layer_idx < len(diagnostics['snr_stats']):
            snr_stats = diagnostics['snr_stats'][layer_idx]
            if snr_stats:
                report['diagnostics']['snr_stats'].append({
                    'layer': layer_idx,
                    **snr_stats
                })
        
        # 共鳴情報
        if layer_idx < len(diagnostics['resonance_info']):
            resonance_info = diagnostics['resonance_info'][layer_idx]
            if resonance_info:
                resonance_summary = {'layer': layer_idx}
                for key, value in resonance_info.items():
                    if isinstance(value, torch.Tensor):
                        if value.numel() == 1:
                            resonance_summary[key] = value.item()
                        else:
                            resonance_summary[key] = {
                                'shape': list(value.shape),
                                'mean': value.mean().item(),
                                'std': value.std().item(),
                            }
                    else:
                        resonance_summary[key] = value
                report['diagnostics']['resonance_info'].append(resonance_summary)
        
        # 安定性メトリクス
        if layer_idx < len(diagnostics['stability_metrics']):
            stability = diagnostics['stability_metrics'][layer_idx]
            if stability:
                stability_summary = {'layer': layer_idx}
                for key, value in stability.items():
                    if isinstance(value, (int, float, bool)):
                        stability_summary[key] = value
                    elif isinstance(value, torch.Tensor):
                        stability_summary[key] = value.item()
                report['diagnostics']['stability_metrics'].append(stability_summary)
    
    # レポートを保存
    output_dir = setup_output_dir()
    report_path = output_dir / "comprehensive_report.json"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n包括的レポートを保存: {report_path}")
    
    # サマリーを表示
    print("\n=== 診断レポートサマリー ===")
    print(f"\nモデル設定:")
    print(f"  - パラメータ数: {report['model_statistics']['num_parameters']:,}")
    print(f"  - レイヤー数: {report['model_statistics']['num_layers']}")
    print(f"  - モデル次元: {report['model_statistics']['d_model']}")
    
    print(f"\nΓ値:")
    for gamma_info in report['diagnostics']['gamma_values']:
        print(f"  Layer {gamma_info['layer']}: 平均={gamma_info['mean']:.6f}, "
              f"標準偏差={gamma_info['std']:.6f}")
    
    print(f"\nSNR統計:")
    for snr_info in report['diagnostics']['snr_stats']:
        print(f"  Layer {snr_info['layer']}: mean_snr={snr_info.get('mean_snr', 'N/A')}")
    
    print(f"\n安定性:")
    for stability_info in report['diagnostics']['stability_metrics']:
        is_stable = stability_info.get('is_stable', 'N/A')
        print(f"  Layer {stability_info['layer']}: is_stable={is_stable}")
    
    return report


def main():
    """メイン関数"""
    print("\n" + "=" * 60)
    print("Phase 2 Diagnostics Examples")
    print("=" * 60)
    
    # 例1: Γ値の監視
    gamma_history = example_1_gamma_monitoring()
    
    # 例2: SNR統計の取得
    snr_history = example_2_snr_statistics()
    
    # 例3: 共鳴情報の可視化
    resonance_info = example_3_resonance_visualization()
    
    # 例4: 安定性メトリクスの追跡
    stability_history = example_4_stability_tracking()
    
    # 例5: 包括的な診断レポート
    report = example_5_comprehensive_report()
    
    print("\n" + "=" * 60)
    print("すべての例が正常に完了しました!")
    print("=" * 60)
    print("\n診断結果は results/phase2_diagnostics/ に保存されました。")


if __name__ == "__main__":
    # シード設定（再現性のため）
    torch.manual_seed(42)
    
    # メイン実行
    main()
