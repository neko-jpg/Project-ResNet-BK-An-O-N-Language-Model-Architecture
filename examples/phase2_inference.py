"""
Phase 2 Inference Example

このスクリプトは、Phase 2モデルの推論方法を示します。

主な内容:
1. テキスト生成（Greedy Decoding）
2. テキスト生成（Top-k Sampling）
3. Fast Weightsの状態管理
4. バッチ推論
5. 逐次推論（ストリーミング）

Requirements: 11.10
Author: Project MUSE Team
Date: 2025-01-20
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import time

from src.models.phase2 import Phase2IntegratedModel, create_phase2_model, Phase2Config


def greedy_decode(
    model: Phase2IntegratedModel,
    input_ids: torch.Tensor,
    max_length: int = 50,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Greedy Decodingによるテキスト生成
    
    各ステップで最も確率の高いトークンを選択します。
    
    Args:
        model: Phase2IntegratedModel
        input_ids: (B, N) 初期入力トークン
        max_length: 最大生成長
        device: デバイス
    
    Returns:
        generated_ids: (B, max_length) 生成されたトークン
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # 状態をリセット
    model.reset_state()
    
    # 生成されたトークンを保存
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_length - input_ids.size(1)):
            # Forward pass
            logits = model(generated)
            
            # 最後のトークンのロジットを取得
            next_token_logits = logits[:, -1, :]  # (B, V)
            
            # 最も確率の高いトークンを選択
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # (B, 1)
            
            # 生成されたトークンを追加
            generated = torch.cat([generated, next_token], dim=1)
    
    return generated


def top_k_sampling(
    model: Phase2IntegratedModel,
    input_ids: torch.Tensor,
    max_length: int = 50,
    k: int = 10,
    temperature: float = 1.0,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Top-k Samplingによるテキスト生成
    
    上位k個のトークンから確率的にサンプリングします。
    
    Args:
        model: Phase2IntegratedModel
        input_ids: (B, N) 初期入力トークン
        max_length: 最大生成長
        k: Top-kのk値
        temperature: 温度パラメータ（高いほどランダム）
        device: デバイス
    
    Returns:
        generated_ids: (B, max_length) 生成されたトークン
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # 状態をリセット
    model.reset_state()
    
    # 生成されたトークンを保存
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_length - input_ids.size(1)):
            # Forward pass
            logits = model(generated)
            
            # 最後のトークンのロジットを取得
            next_token_logits = logits[:, -1, :] / temperature  # (B, V)
            
            # Top-kフィルタリング
            top_k_logits, top_k_indices = torch.topk(next_token_logits, k, dim=-1)
            
            # 確率に変換
            probs = F.softmax(top_k_logits, dim=-1)
            
            # サンプリング
            sampled_indices = torch.multinomial(probs, num_samples=1)  # (B, 1)
            next_token = torch.gather(top_k_indices, 1, sampled_indices)  # (B, 1)
            
            # 生成されたトークンを追加
            generated = torch.cat([generated, next_token], dim=1)
    
    return generated


def example_1_greedy_generation():
    """
    例1: Greedy Decodingによるテキスト生成
    
    最も確率の高いトークンを選択して生成します。
    """
    print("=" * 60)
    print("例1: Greedy Decodingによるテキスト生成")
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
        n_seq=128,  # 長めのシーケンス
        num_heads=4,
        head_dim=32,
    )
    model = create_phase2_model(config=config, device=device)
    
    # 初期入力（プロンプト）
    prompt = torch.randint(0, 1000, (1, 10)).to(device)
    print(f"\nプロンプト: {prompt[0].tolist()}")
    
    # テキスト生成
    print("\nテキスト生成中...")
    start_time = time.time()
    
    generated = greedy_decode(
        model,
        prompt,
        max_length=50,
        device=device
    )
    
    elapsed_time = time.time() - start_time
    
    print(f"\n生成完了! ({elapsed_time:.2f}秒)")
    print(f"生成されたトークン: {generated[0].tolist()}")
    print(f"生成長: {generated.size(1)}")
    print(f"生成速度: {generated.size(1) / elapsed_time:.2f} tokens/sec")
    
    return generated


def example_2_top_k_sampling():
    """
    例2: Top-k Samplingによるテキスト生成
    
    上位k個のトークンから確率的にサンプリングします。
    """
    print("\n" + "=" * 60)
    print("例2: Top-k Samplingによるテキスト生成")
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
        n_seq=128,
        num_heads=4,
        head_dim=32,
    )
    model = create_phase2_model(config=config, device=device)
    
    # 初期入力（プロンプト）
    prompt = torch.randint(0, 1000, (1, 10)).to(device)
    print(f"\nプロンプト: {prompt[0].tolist()}")
    
    # 複数の温度で生成
    temperatures = [0.5, 1.0, 1.5]
    
    for temp in temperatures:
        print(f"\n--- Temperature = {temp} ---")
        
        # テキスト生成
        generated = top_k_sampling(
            model,
            prompt,
            max_length=50,
            k=10,
            temperature=temp,
            device=device
        )
        
        print(f"生成されたトークン: {generated[0].tolist()}")
    
    return generated


def example_3_batch_inference():
    """
    例3: バッチ推論
    
    複数のプロンプトを同時に処理します。
    """
    print("\n" + "=" * 60)
    print("例3: バッチ推論")
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
        n_seq=128,
        num_heads=4,
        head_dim=32,
    )
    model = create_phase2_model(config=config, device=device)
    
    # 複数のプロンプト
    batch_size = 4
    prompts = torch.randint(0, 1000, (batch_size, 10)).to(device)
    
    print(f"\nバッチサイズ: {batch_size}")
    print(f"プロンプト形状: {prompts.shape}")
    
    # バッチ推論
    print("\nバッチ推論中...")
    start_time = time.time()
    
    generated = greedy_decode(
        model,
        prompts,
        max_length=50,
        device=device
    )
    
    elapsed_time = time.time() - start_time
    
    print(f"\n推論完了! ({elapsed_time:.2f}秒)")
    print(f"生成形状: {generated.shape}")
    print(f"スループット: {batch_size * generated.size(1) / elapsed_time:.2f} tokens/sec")
    
    # 各サンプルの結果を表示
    for i in range(batch_size):
        print(f"\nサンプル {i + 1}:")
        print(f"  プロンプト: {prompts[i].tolist()}")
        print(f"  生成: {generated[i].tolist()}")
    
    return generated


def example_4_state_management():
    """
    例4: Fast Weightsの状態管理
    
    Fast Weight状態を保持しながら逐次的に生成します。
    """
    print("\n" + "=" * 60)
    print("例4: Fast Weightsの状態管理")
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
        n_seq=128,
        num_heads=4,
        head_dim=32,
    )
    model = create_phase2_model(config=config, device=device)
    
    # シナリオ1: 状態をリセットせずに連続生成
    print("\n--- シナリオ1: 状態を保持して連続生成 ---")
    
    prompt1 = torch.randint(0, 1000, (1, 10)).to(device)
    print(f"\nプロンプト1: {prompt1[0].tolist()}")
    
    # 最初の生成
    model.reset_state()  # 初期化
    generated1 = greedy_decode(model, prompt1, max_length=30, device=device)
    print(f"生成1: {generated1[0].tolist()}")
    
    # Fast Weight状態を確認
    print("\nFast Weight状態（生成1後）:")
    for i, block in enumerate(model.blocks):
        if block.fast_weight_state is not None:
            norm = torch.norm(block.fast_weight_state).item()
            print(f"  Layer {i}: ノルム={norm:.4f}")
    
    # 2番目の生成（状態を保持）
    prompt2 = torch.randint(0, 1000, (1, 10)).to(device)
    print(f"\nプロンプト2: {prompt2[0].tolist()}")
    
    generated2 = greedy_decode(model, prompt2, max_length=30, device=device)
    print(f"生成2: {generated2[0].tolist()}")
    
    # Fast Weight状態を確認
    print("\nFast Weight状態（生成2後）:")
    for i, block in enumerate(model.blocks):
        if block.fast_weight_state is not None:
            norm = torch.norm(block.fast_weight_state).item()
            print(f"  Layer {i}: ノルム={norm:.4f}")
    
    # シナリオ2: 状態をリセットして独立生成
    print("\n--- シナリオ2: 状態をリセットして独立生成 ---")
    
    model.reset_state()  # リセット
    print("\n状態をリセットしました")
    
    # Fast Weight状態を確認
    print("\nFast Weight状態（リセット後）:")
    for i, block in enumerate(model.blocks):
        if block.fast_weight_state is not None:
            norm = torch.norm(block.fast_weight_state).item()
            print(f"  Layer {i}: ノルム={norm:.4f}")
        else:
            print(f"  Layer {i}: None")
    
    # 新しい生成
    prompt3 = torch.randint(0, 1000, (1, 10)).to(device)
    print(f"\nプロンプト3: {prompt3[0].tolist()}")
    
    generated3 = greedy_decode(model, prompt3, max_length=30, device=device)
    print(f"生成3: {generated3[0].tolist()}")


def example_5_streaming_inference():
    """
    例5: ストリーミング推論
    
    トークンを1つずつ生成し、リアルタイムで出力します。
    """
    print("\n" + "=" * 60)
    print("例5: ストリーミング推論")
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
        n_seq=128,
        num_heads=4,
        head_dim=32,
    )
    model = create_phase2_model(config=config, device=device)
    model.eval()
    
    # 初期入力（プロンプト）
    prompt = torch.randint(0, 1000, (1, 10)).to(device)
    print(f"\nプロンプト: {prompt[0].tolist()}")
    
    # 状態をリセット
    model.reset_state()
    
    # ストリーミング生成
    print("\nストリーミング生成中...")
    print("生成されたトークン: ", end="", flush=True)
    
    generated = prompt.clone()
    max_length = 50
    
    with torch.no_grad():
        for step in range(max_length - prompt.size(1)):
            # Forward pass
            logits = model(generated)
            
            # 最後のトークンのロジットを取得
            next_token_logits = logits[:, -1, :]  # (1, V)
            
            # 最も確率の高いトークンを選択
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # (1, 1)
            
            # 生成されたトークンを追加
            generated = torch.cat([generated, next_token], dim=1)
            
            # リアルタイムで出力
            print(f"{next_token.item()} ", end="", flush=True)
            
            # 少し待機（ストリーミング効果のため）
            time.sleep(0.05)
    
    print("\n\n生成完了!")
    print(f"最終シーケンス: {generated[0].tolist()}")


def example_6_perplexity_evaluation():
    """
    例6: Perplexity評価
    
    テストデータでPerplexityを計算します。
    """
    print("\n" + "=" * 60)
    print("例6: Perplexity評価")
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
    )
    model = create_phase2_model(config=config, device=device)
    model.eval()
    
    # テストデータ
    num_samples = 10
    test_data = []
    for _ in range(num_samples):
        seq = torch.randint(0, 1000, (1, config.n_seq + 1)).to(device)
        test_data.append(seq)
    
    print(f"\nテストサンプル数: {num_samples}")
    
    # Perplexity計算
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for seq in test_data:
            # 入力と目標を分離
            input_ids = seq[:, :-1]
            target_ids = seq[:, 1:]
            
            # 状態をリセット
            model.reset_state()
            
            # Forward pass
            logits = model(input_ids)
            
            # 損失計算
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                reduction='sum'
            )
            
            total_loss += loss.item()
            total_tokens += target_ids.numel()
    
    # Perplexity計算
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    print(f"\n評価結果:")
    print(f"  平均損失: {avg_loss:.4f}")
    print(f"  Perplexity: {perplexity:.4f}")
    print(f"  総トークン数: {total_tokens}")


def main():
    """メイン関数"""
    print("\n" + "=" * 60)
    print("Phase 2 Inference Examples")
    print("=" * 60)
    
    # 例1: Greedy Decodingによるテキスト生成
    generated1 = example_1_greedy_generation()
    
    # 例2: Top-k Samplingによるテキスト生成
    generated2 = example_2_top_k_sampling()
    
    # 例3: バッチ推論
    generated3 = example_3_batch_inference()
    
    # 例4: Fast Weightsの状態管理
    example_4_state_management()
    
    # 例5: ストリーミング推論
    example_5_streaming_inference()
    
    # 例6: Perplexity評価
    example_6_perplexity_evaluation()
    
    print("\n" + "=" * 60)
    print("すべての例が正常に完了しました!")
    print("=" * 60)


if __name__ == "__main__":
    # シード設定（再現性のため）
    torch.manual_seed(42)
    
    # メイン実行
    main()
