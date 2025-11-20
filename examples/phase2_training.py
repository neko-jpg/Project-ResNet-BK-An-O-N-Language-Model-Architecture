"""
Phase 2 Training Example

このスクリプトは、Phase 2モデルの学習方法を示します。

主な内容:
1. 小規模データセットでの学習
2. 診断情報の取得とロギング
3. Γ値の監視
4. SNR統計の追跡
5. 学習曲線の可視化

Requirements: 11.10
Author: Project MUSE Team
Date: 2025-01-20
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple
import json
from pathlib import Path

from src.models.phase2 import Phase2IntegratedModel, create_phase2_model, Phase2Config


class TinyTextDataset(Dataset):
    """
    小規模なテキストデータセット（デモ用）
    
    ランダムなシーケンスを生成します。
    """
    
    def __init__(
        self,
        vocab_size: int = 1000,
        seq_len: int = 64,
        num_samples: int = 100,
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        
        # ランダムシードを固定
        np.random.seed(42)
        
        # データを生成
        self.data = []
        for _ in range(num_samples):
            # ランダムなシーケンスを生成
            seq = np.random.randint(0, vocab_size, size=seq_len + 1)
            self.data.append(seq)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        seq = self.data[idx]
        # 入力と目標を分離
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        target_ids = torch.tensor(seq[1:], dtype=torch.long)
        return input_ids, target_ids


def create_dataloader(
    vocab_size: int = 1000,
    seq_len: int = 64,
    num_samples: int = 100,
    batch_size: int = 4,
) -> DataLoader:
    """
    データローダーを作成
    
    Args:
        vocab_size: 語彙サイズ
        seq_len: シーケンス長
        num_samples: サンプル数
        batch_size: バッチサイズ
    
    Returns:
        DataLoader
    """
    dataset = TinyTextDataset(vocab_size, seq_len, num_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    return dataloader


def train_one_epoch(
    model: Phase2IntegratedModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    collect_diagnostics: bool = False,
) -> Tuple[float, Dict]:
    """
    1エポックの学習
    
    Args:
        model: Phase2IntegratedModel
        dataloader: データローダー
        optimizer: オプティマイザー
        device: デバイス
        epoch: エポック番号
        collect_diagnostics: 診断情報を収集するか
    
    Returns:
        avg_loss: 平均損失
        diagnostics: 診断情報の辞書
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # 診断情報を収集
    diagnostics = {
        'gamma_values': [],
        'snr_stats': [],
        'resonance_info': [],
        'stability_metrics': [],
        'losses': [],
    }
    
    for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
        # デバイスに移動
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        # 状態をリセット（各バッチで独立）
        model.reset_state()
        
        # Forward pass
        if collect_diagnostics and batch_idx == 0:
            # 最初のバッチのみ診断情報を収集
            logits, batch_diagnostics = model(input_ids, return_diagnostics=True)
            
            # 診断情報を保存
            diagnostics['gamma_values'].append([
                gamma.mean().item() if gamma is not None else 0.0
                for gamma in batch_diagnostics['gamma_values']
            ])
            diagnostics['snr_stats'].append(batch_diagnostics['snr_stats'])
            diagnostics['resonance_info'].append(batch_diagnostics['resonance_info'])
            diagnostics['stability_metrics'].append(batch_diagnostics['stability_metrics'])
        else:
            logits = model(input_ids)
        
        # 損失計算
        # logits: (B, N, V), target_ids: (B, N)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # 勾配クリッピング（安定性のため）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # パラメータ更新
        optimizer.step()
        
        # 統計を記録
        total_loss += loss.item()
        num_batches += 1
        diagnostics['losses'].append(loss.item())
        
        # 進捗表示
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    return avg_loss, diagnostics


def evaluate(
    model: Phase2IntegratedModel,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """
    評価
    
    Args:
        model: Phase2IntegratedModel
        dataloader: データローダー
        device: デバイス
    
    Returns:
        avg_loss: 平均損失
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for input_ids, target_ids in dataloader:
            # デバイスに移動
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            # 状態をリセット
            model.reset_state()
            
            # Forward pass
            logits = model(input_ids)
            
            # 損失計算
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1)
            )
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


def example_1_basic_training():
    """
    例1: 基本的な学習ループ
    
    小規模データセットでPhase 2モデルを学習します。
    """
    print("=" * 60)
    print("例1: 基本的な学習ループ")
    print("=" * 60)
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nデバイス: {device}")
    
    # モデル作成
    print("\nモデルを作成中...")
    config = Phase2Config(
        vocab_size=1000,
        d_model=128,
        n_layers=2,
        n_seq=64,
        num_heads=4,
        head_dim=32,
        base_decay=0.01,
        hebbian_eta=0.1,
    )
    model = create_phase2_model(config=config, device=device)
    
    # データローダー作成
    print("\nデータローダーを作成中...")
    train_loader = create_dataloader(
        vocab_size=config.vocab_size,
        seq_len=config.n_seq,
        num_samples=100,
        batch_size=4,
    )
    val_loader = create_dataloader(
        vocab_size=config.vocab_size,
        seq_len=config.n_seq,
        num_samples=20,
        batch_size=4,
    )
    
    # オプティマイザー設定
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    # 学習ループ
    num_epochs = 5
    print(f"\n学習開始（{num_epochs}エポック）...")
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)
        
        # 学習
        train_loss, _ = train_one_epoch(
            model, train_loader, optimizer, device, epoch,
            collect_diagnostics=False
        )
        train_losses.append(train_loss)
        
        # 評価
        val_loss = evaluate(model, val_loader, device)
        val_losses.append(val_loss)
        
        print(f"\nEpoch {epoch + 1} 結果:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
    
    print("\n学習完了!")
    print(f"最終 Train Loss: {train_losses[-1]:.4f}")
    print(f"最終 Val Loss: {val_losses[-1]:.4f}")
    
    return model, train_losses, val_losses


def example_2_training_with_diagnostics():
    """
    例2: 診断情報付き学習
    
    Γ値、SNR統計、共鳴情報を収集しながら学習します。
    """
    print("\n" + "=" * 60)
    print("例2: 診断情報付き学習")
    print("=" * 60)
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nデバイス: {device}")
    
    # モデル作成
    print("\nモデルを作成中...")
    config = Phase2Config(
        vocab_size=1000,
        d_model=128,
        n_layers=2,
        n_seq=64,
        num_heads=4,
        head_dim=32,
        base_decay=0.01,
        hebbian_eta=0.1,
    )
    model = create_phase2_model(config=config, device=device)
    
    # データローダー作成
    print("\nデータローダーを作成中...")
    train_loader = create_dataloader(
        vocab_size=config.vocab_size,
        seq_len=config.n_seq,
        num_samples=100,
        batch_size=4,
    )
    
    # オプティマイザー設定
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    # 学習ループ
    num_epochs = 3
    print(f"\n学習開始（{num_epochs}エポック）...")
    
    all_diagnostics = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)
        
        # 診断情報を収集しながら学習
        train_loss, diagnostics = train_one_epoch(
            model, train_loader, optimizer, device, epoch,
            collect_diagnostics=True
        )
        
        all_diagnostics.append(diagnostics)
        
        print(f"\nEpoch {epoch + 1} 結果:")
        print(f"  Train Loss: {train_loss:.4f}")
        
        # Γ値の統計
        if diagnostics['gamma_values']:
            gamma_means = diagnostics['gamma_values'][0]
            print(f"\n  Γ値（各層の平均）:")
            for layer_idx, gamma_mean in enumerate(gamma_means):
                print(f"    Layer {layer_idx}: {gamma_mean:.6f}")
        
        # SNR統計
        if diagnostics['snr_stats']:
            snr_stats_list = diagnostics['snr_stats'][0]
            print(f"\n  SNR統計:")
            for layer_idx, snr_stats in enumerate(snr_stats_list):
                if snr_stats:
                    print(f"    Layer {layer_idx}:")
                    for key, value in snr_stats.items():
                        print(f"      {key}: {value:.4f}")
        
        # 安定性メトリクス
        if diagnostics['stability_metrics']:
            stability_list = diagnostics['stability_metrics'][0]
            print(f"\n  安定性メトリクス:")
            for layer_idx, stability in enumerate(stability_list):
                if stability:
                    print(f"    Layer {layer_idx}:")
                    for key, value in stability.items():
                        if isinstance(value, (int, float)):
                            print(f"      {key}: {value}")
                        else:
                            print(f"      {key}: {value}")
    
    print("\n学習完了!")
    
    return model, all_diagnostics


def example_3_save_and_load():
    """
    例3: モデルの保存と読み込み
    
    学習したモデルを保存し、後で読み込みます。
    """
    print("\n" + "=" * 60)
    print("例3: モデルの保存と読み込み")
    print("=" * 60)
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # モデル作成と簡単な学習
    print("\nモデルを作成して学習中...")
    config = Phase2Config(
        vocab_size=1000,
        d_model=128,
        n_layers=2,
        n_seq=64,
    )
    model = create_phase2_model(config=config, device=device)
    
    # 簡単な学習（1エポックのみ）
    train_loader = create_dataloader(
        vocab_size=config.vocab_size,
        seq_len=config.n_seq,
        num_samples=50,
        batch_size=4,
    )
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    train_loss, _ = train_one_epoch(
        model, train_loader, optimizer, device, 0,
        collect_diagnostics=False
    )
    print(f"Train Loss: {train_loss:.4f}")
    
    # モデルを保存
    save_dir = Path("results/phase2_demo_checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = save_dir / "phase2_example.pt"
    print(f"\nモデルを保存中: {checkpoint_path}")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.to_dict(),
        'train_loss': train_loss,
    }, checkpoint_path)
    
    print("保存完了!")
    
    # モデルを読み込み
    print(f"\nモデルを読み込み中: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 新しいモデルを作成
    loaded_config = Phase2Config.from_dict(checkpoint['config'])
    loaded_model = create_phase2_model(config=loaded_config, device=device)
    
    # 状態を復元
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    
    print("読み込み完了!")
    print(f"保存時の Train Loss: {checkpoint['train_loss']:.4f}")
    
    # 読み込んだモデルで推論
    print("\n読み込んだモデルで推論中...")
    input_ids = torch.randint(0, 1000, (1, 16)).to(device)
    
    with torch.no_grad():
        logits = loaded_model(input_ids)
    
    print(f"出力形状: {logits.shape}")
    print("推論成功!")
    
    return loaded_model


def example_4_learning_rate_scheduling():
    """
    例4: 学習率スケジューリング
    
    学習率スケジューラーを使用した学習を示します。
    """
    print("\n" + "=" * 60)
    print("例4: 学習率スケジューリング")
    print("=" * 60)
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nデバイス: {device}")
    
    # モデル作成
    print("\nモデルを作成中...")
    config = Phase2Config(
        vocab_size=1000,
        d_model=128,
        n_layers=2,
        n_seq=64,
    )
    model = create_phase2_model(config=config, device=device)
    
    # データローダー作成
    train_loader = create_dataloader(
        vocab_size=config.vocab_size,
        seq_len=config.n_seq,
        num_samples=100,
        batch_size=4,
    )
    
    # オプティマイザーとスケジューラー設定
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
    
    # 学習ループ
    num_epochs = 5
    print(f"\n学習開始（{num_epochs}エポック）...")
    
    train_losses = []
    learning_rates = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        print("-" * 40)
        
        # 学習
        train_loss, _ = train_one_epoch(
            model, train_loader, optimizer, device, epoch,
            collect_diagnostics=False
        )
        train_losses.append(train_loss)
        learning_rates.append(scheduler.get_last_lr()[0])
        
        print(f"\nEpoch {epoch + 1} 結果:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Learning Rate: {learning_rates[-1]:.6f}")
        
        # スケジューラーステップ
        scheduler.step()
    
    print("\n学習完了!")
    print(f"最終 Train Loss: {train_losses[-1]:.4f}")
    print(f"最終 Learning Rate: {learning_rates[-1]:.6f}")
    
    return model, train_losses, learning_rates


def main():
    """メイン関数"""
    print("\n" + "=" * 60)
    print("Phase 2 Training Examples")
    print("=" * 60)
    
    # 例1: 基本的な学習ループ
    model1, train_losses1, val_losses1 = example_1_basic_training()
    
    # 例2: 診断情報付き学習
    model2, all_diagnostics = example_2_training_with_diagnostics()
    
    # 例3: モデルの保存と読み込み
    loaded_model = example_3_save_and_load()
    
    # 例4: 学習率スケジューリング
    model4, train_losses4, learning_rates = example_4_learning_rate_scheduling()
    
    print("\n" + "=" * 60)
    print("すべての例が正常に完了しました!")
    print("=" * 60)


if __name__ == "__main__":
    # シード設定（再現性のため）
    torch.manual_seed(42)
    np.random.seed(42)
    
    # メイン実行
    main()
