#!/usr/bin/env python3
"""
Phase 2可視化デモスクリプト

Phase 2の可視化機能をデモンストレーションします。
サンプルデータを生成して、各種グラフを作成します。

使用例:
    python examples/phase2_visualization_demo.py
"""

import json
import numpy as np
from pathlib import Path


def generate_sample_logs(output_dir: Path):
    """
    サンプルログデータを生成
    
    実際の学習では、train_phase2.pyがこれらのログを生成します。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_steps = 1000
    num_layers = 6
    num_heads = 8
    head_dim = 64
    
    # ========================================
    # 1. 学習ログ（Loss, Perplexity）
    # ========================================
    
    # Loss: 指数減衰 + ノイズ
    initial_loss = 5.0
    final_loss = 1.5
    losses = []
    for step in range(num_steps):
        decay = np.exp(-step / 200)
        loss = final_loss + (initial_loss - final_loss) * decay
        loss += np.random.normal(0, 0.1)  # ノイズ
        losses.append(max(0.1, loss))  # 負にならないように
    
    # Perplexity: exp(Loss)
    perplexities = [np.exp(loss) for loss in losses]
    
    training_log = {
        'loss': losses,
        'perplexity': perplexities,
    }
    
    with open(output_dir / 'training_log.json', 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print(f"✓ 学習ログを生成: {output_dir / 'training_log.json'}")
    
    # ========================================
    # 2. 診断ログ（Γ, SNR, 共鳴情報）
    # ========================================
    
    gamma_values = []
    snr_stats = []
    resonance_info = []
    
    for step in range(num_steps):
        # Γ値: 各層で異なる減衰率、時間とともに変化
        step_gamma = {}
        for layer_idx in range(num_layers):
            # 層ごとに異なる基底減衰率
            base_gamma = 0.01 + 0.02 * layer_idx / num_layers
            # 時間とともに増加（学習が進むと忘却が強化される）
            time_factor = 1.0 + 0.5 * step / num_steps
            # バッチ内のばらつき
            batch_gamma = [
                base_gamma * time_factor + np.random.normal(0, 0.005)
                for _ in range(4)  # バッチサイズ4
            ]
            step_gamma[f'layer_{layer_idx}'] = batch_gamma
        
        gamma_values.append(step_gamma)
        
        # SNR統計: 時間とともに改善
        mean_snr = 1.0 + 2.0 * step / num_steps + np.random.normal(0, 0.1)
        std_snr = 0.5 + 0.3 * np.random.random()
        min_snr = max(0.1, mean_snr - 2 * std_snr)
        max_snr = mean_snr + 2 * std_snr
        
        snr_stats.append({
            'mean_snr': mean_snr,
            'std_snr': std_snr,
            'min_snr': min_snr,
            'max_snr': max_snr,
        })
        
        # 共鳴情報: 時間とともに共鳴成分が増加
        num_resonant = 10 + 20 * step / num_steps + np.random.normal(0, 2)
        total_energy = 0.5 + 1.5 * step / num_steps + np.random.normal(0, 0.1)
        
        # 対角エネルギー（最終ステップのみ詳細データ）
        if step == num_steps - 1:
            diag_energy = [
                [np.random.exponential(0.2) for _ in range(head_dim)]
                for _ in range(num_heads)
            ]
            resonance_mask = [
                [1 if energy > 0.1 else 0 for energy in head_energies]
                for head_energies in diag_energy
            ]
        else:
            diag_energy = []
            resonance_mask = []
        
        resonance_info.append({
            'num_resonant': num_resonant,
            'total_energy': total_energy,
            'diag_energy': diag_energy,
            'resonance_mask': resonance_mask,
        })
    
    diagnostics_log = {
        'gamma_values': gamma_values,
        'snr_stats': snr_stats,
        'resonance_info': resonance_info,
    }
    
    with open(output_dir / 'diagnostics_log.json', 'w') as f:
        json.dump(diagnostics_log, f, indent=2)
    
    print(f"✓ 診断ログを生成: {output_dir / 'diagnostics_log.json'}")


def main():
    """メイン関数"""
    print("="*60)
    print("Phase 2 可視化デモ")
    print("="*60)
    
    # サンプルログを生成
    log_dir = Path('results/phase2_demo_logs')
    print(f"\n[1] サンプルログを生成中...")
    generate_sample_logs(log_dir)
    
    # 可視化スクリプトを実行
    print(f"\n[2] 可視化スクリプトを実行中...")
    import subprocess
    result = subprocess.run([
        'python',
        'scripts/visualize_phase2.py',
        '--log-dir', str(log_dir),
        '--output-dir', 'results/phase2_demo_visualizations'
    ])
    
    if result.returncode == 0:
        print("\n" + "="*60)
        print("✓ デモが完了しました！")
        print("  可視化結果: results/phase2_demo_visualizations/")
        print("="*60)
    else:
        print("\n✗ 可視化スクリプトの実行に失敗しました")


if __name__ == "__main__":
    main()
