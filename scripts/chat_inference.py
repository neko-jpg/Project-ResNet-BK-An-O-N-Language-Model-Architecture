#!/usr/bin/env python3
"""
Phase 7 Chat AI - 推論スクリプト

訓練済みモデルでチャットできます！

Usage:
    make chat-ai                              # 最新のチェックポイントでチャット
    make chat-ai CHECKPOINT=path/to/model.pt  # 指定チェックポイント
    python scripts/chat_inference.py --checkpoint checkpoints/phase7_max_push/best.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F


def load_model(checkpoint_path: str, device: str = "cuda"):
    """チェックポイントからモデルをロード"""
    from scripts.train_phase7_max import Phase7MaxModel, TrainingConfig
    
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # 設定を復元
    config_dict = ckpt.get('config', {})
    config = TrainingConfig()
    for k, v in config_dict.items():
        if hasattr(config, k):
            setattr(config, k, v)
    
    # モデル作成
    model = Phase7MaxModel(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_seq=config.n_seq,
        num_heads=config.num_heads,
        embed_rank=config.embed_rank,
        ffn_rank=config.ffn_rank,
        head_rank=config.head_rank,
        use_checkpoint=False,  # 推論時は不要
    ).to(device)
    
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    model.half()  # FP16で推論
    
    step = ckpt.get('step', 'unknown')
    print(f"✓ Model loaded (step {step})")
    print(f"  d_model: {config.d_model}, n_layers: {config.n_layers}")
    print(f"  Parameters: {model.total_params / 1e6:.1f}M")
    
    return model, config


def load_tokenizer():
    """トークナイザーをロード（GPT-2互換）"""
    try:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        print("✓ GPT-2 tokenizer loaded")
        return tokenizer
    except ImportError:
        print("⚠ transformers not installed. Using simple tokenizer.")
        return None


class SimpleTokenizer:
    """シンプルな文字レベルトークナイザー（フォールバック用）"""
    def __init__(self, vocab_size=50257):
        self.vocab_size = vocab_size
        self.eos_token_id = 50256
    
    def encode(self, text):
        # 簡易的なバイトエンコード
        return [min(b, self.vocab_size - 1) for b in text.encode('utf-8')]
    
    def decode(self, ids):
        # バイトデコード
        try:
            return bytes([i % 256 for i in ids]).decode('utf-8', errors='replace')
        except:
            return "".join(chr(i % 128) for i in ids)


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    device: str = "cuda",
):
    """テキスト生成"""
    # エンコード
    if hasattr(tokenizer, 'encode'):
        if hasattr(tokenizer, '__call__'):
            # HuggingFace tokenizer
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        else:
            # Simple tokenizer
            input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    else:
        input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    
    # 生成ループ
    generated = input_ids.clone()
    
    for _ in range(max_new_tokens):
        # 最大シーケンス長でトランケート
        if generated.shape[1] > model.n_seq:
            context = generated[:, -model.n_seq:]
        else:
            context = generated
        
        # Forward
        with torch.cuda.amp.autocast():
            logits = model(context)
        
        # 最後のトークンのlogitsを取得
        next_logits = logits[:, -1, :] / temperature
        
        # Top-k filtering
        if top_k > 0:
            indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
            next_logits[indices_to_remove] = float('-inf')
        
        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_logits[indices_to_remove] = float('-inf')
        
        # サンプリング
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        generated = torch.cat([generated, next_token], dim=1)
        
        # EOS check
        if hasattr(tokenizer, 'eos_token_id') and next_token.item() == tokenizer.eos_token_id:
            break
    
    # デコード
    output_ids = generated[0].tolist()
    if hasattr(tokenizer, 'decode'):
        return tokenizer.decode(output_ids, skip_special_tokens=True)
    else:
        return tokenizer.decode(output_ids)


def find_latest_checkpoint(checkpoint_dir: str = "checkpoints/phase7_max_push"):
    """最新のチェックポイントを探す"""
    ckpt_dir = Path(checkpoint_dir)
    
    if not ckpt_dir.exists():
        return None
    
    # best.pt を優先
    best_path = ckpt_dir / "best.pt"
    if best_path.exists():
        return str(best_path)
    
    # final.pt
    final_path = ckpt_dir / "final.pt"
    if final_path.exists():
        return str(final_path)
    
    # step_*.pt から最新を探す
    step_files = list(ckpt_dir.glob("step_*.pt"))
    if step_files:
        latest = max(step_files, key=lambda p: int(p.stem.split('_')[1]))
        return str(latest)
    
    return None


def interactive_chat(model, tokenizer, device="cuda"):
    """インタラクティブチャットモード"""
    print("\n" + "=" * 60)
    print("🤖 MUSE Chat AI - Interactive Mode")
    print("=" * 60)
    print("Commands:")
    print("  /quit, /exit - 終了")
    print("  /temp <value> - temperature設定 (default: 0.8)")
    print("  /tokens <value> - 最大生成トークン数 (default: 100)")
    print("  /clear - 会話履歴クリア")
    print("=" * 60 + "\n")
    
    temperature = 0.8
    max_tokens = 100
    history = []
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Bye!")
            break
        
        if not user_input:
            continue
        
        # コマンド処理
        if user_input.startswith('/'):
            parts = user_input.split()
            cmd = parts[0].lower()
            
            if cmd in ['/quit', '/exit']:
                print("👋 Bye!")
                break
            elif cmd == '/temp' and len(parts) > 1:
                try:
                    temperature = float(parts[1])
                    print(f"✓ Temperature set to {temperature}")
                except ValueError:
                    print("❌ Invalid temperature value")
                continue
            elif cmd == '/tokens' and len(parts) > 1:
                try:
                    max_tokens = int(parts[1])
                    print(f"✓ Max tokens set to {max_tokens}")
                except ValueError:
                    print("❌ Invalid token count")
                continue
            elif cmd == '/clear':
                history = []
                print("✓ History cleared")
                continue
            else:
                print(f"❌ Unknown command: {cmd}")
                continue
        
        # プロンプト構築
        # シンプルなチャット形式
        prompt = ""
        for h in history[-3:]:  # 直近3ターンのみ
            prompt += f"User: {h['user']}\nAssistant: {h['assistant']}\n"
        prompt += f"User: {user_input}\nAssistant:"
        
        # 生成
        print("AI: ", end="", flush=True)
        try:
            response = generate(
                model, tokenizer, prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                device=device,
            )
            
            # プロンプト部分を除去
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            
            # 次のUser:以降を除去
            if "User:" in response:
                response = response.split("User:")[0].strip()
            
            print(response)
            
            history.append({
                'user': user_input,
                'assistant': response
            })
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
        
        print()


def main():
    parser = argparse.ArgumentParser(description="MUSE Chat AI")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (default: auto-detect latest)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt (non-interactive mode)")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    args = parser.parse_args()
    
    # デバイスチェック
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, using CPU")
        args.device = "cpu"
    
    # チェックポイント探索
    checkpoint = args.checkpoint or find_latest_checkpoint()
    
    if checkpoint is None:
        print("❌ No checkpoint found!")
        print("\nPlease train a model first:")
        print("  make train-chat")
        print("\nOr specify a checkpoint:")
        print("  python scripts/chat_inference.py --checkpoint path/to/model.pt")
        sys.exit(1)
    
    # モデルロード
    model, config = load_model(checkpoint, args.device)
    
    # トークナイザー
    tokenizer = load_tokenizer()
    if tokenizer is None:
        tokenizer = SimpleTokenizer(config.vocab_size)
    
    # 推論
    if args.prompt:
        # 単発モード
        print(f"\nPrompt: {args.prompt}")
        print("-" * 40)
        response = generate(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            device=args.device,
        )
        print(f"Response: {response}")
    else:
        # インタラクティブモード
        interactive_chat(model, tokenizer, args.device)


if __name__ == "__main__":
    main()
