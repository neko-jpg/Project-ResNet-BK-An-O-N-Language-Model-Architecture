#!/usr/bin/env python3
"""
MUSE Chat AI - Phase 8 対応版
==============================
訓練済みモデルでチャットできます！日本語・英語両対応。

Usage:
    make chat                    # 最新のチェックポイントでチャット
    make chat CHECKPOINT=path    # 指定チェックポイント
    python scripts/chat_inference.py --checkpoint checkpoints/phase8_10b_japanese/best.pt
"""

import argparse
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None


@dataclass
class ChatConfig:
    """チャット設定"""
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    max_new_tokens: int = 256
    repetition_penalty: float = 1.1


def load_model_phase8(checkpoint_path: str, device: str = "cuda"):
    """Phase 8 モデルをロード"""
    
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 設定を復元
    if 'config' in ckpt:
        config_dict = ckpt['config']
        if isinstance(config_dict, dict):
            config = config_dict
        else:
            config = vars(config_dict) if hasattr(config_dict, '__dict__') else {}
    else:
        config = {
            'd_model': 1024,
            'n_layers': 24,
            'n_seq': 512,
            'vocab_size': 50256,
        }
    
    # Phase8IntegratedModel を試す (推奨)
    try:
        from src.models.phase8.integrated_model import Phase8IntegratedModel
        from src.models.phase8.config import Phase8Config
        
        # Phase8Config を作成
        phase8_config = Phase8Config(
            d_model=config.get('d_model', 1024),
            n_layers=config.get('n_layers', 24),
            n_seq=config.get('n_seq', 512),
            vocab_size=config.get('vocab_size', 50256),
            num_heads=config.get('num_heads', 16),
            htt_rank=config.get('htt_rank', 16),
            use_resonant_htt=config.get('use_resonant_htt', True),
            resonant_num_cores=config.get('resonant_num_cores', 4),
            use_zeta_init=config.get('use_zeta_init', True),
            low_rank_ffn=config.get('low_rank_ffn', True),
            low_rank_attention=config.get('low_rank_attention', True),
            low_rank_rank=config.get('low_rank_rank', 32),
            use_bk_hyperbolic=config.get('use_bk_hyperbolic', True),
        )
        
        model = Phase8IntegratedModel(phase8_config).to(device)
        print("✓ Using Phase8IntegratedModel")
        
    except ImportError:
        # フォールバック: 旧LanguageModel
        from src.models.resnet_bk import LanguageModel
        from src.models.config import ResNetBKConfig
        
        import dataclasses
        valid_fields = {f.name for f in dataclasses.fields(ResNetBKConfig)}
        filtered_dict = {k: v for k, v in config.items() if k in valid_fields}
        model_config = ResNetBKConfig(**filtered_dict)
        model = LanguageModel(model_config).to(device)
        print("✓ Using LanguageModel (fallback)")
    
    # 重みをロード
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    elif 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    
    model.eval()
    
    # パラメータ数
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"✓ Model loaded successfully!")
    print(f"  d_model: {config.get('d_model', '?')}, n_layers: {config.get('n_layers', '?')}")
    print(f"  Parameters: {total_params / 1e6:.1f}M")
    
    return model, config


def load_tokenizer(tokenizer_name: str = "rinna/japanese-gpt-neox-3.6b"):
    """トークナイザーをロード（日本語対応）"""
    try:
        from transformers import AutoTokenizer
        
        # 日本語トークナイザーを試す
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            print(f"✓ Japanese tokenizer loaded: {tokenizer_name}")
        except:
            # フォールバック: GPT-2
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            print("✓ GPT-2 tokenizer loaded (fallback)")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer
    except ImportError:
        print("⚠ transformers not installed. Using simple tokenizer.")
        return SimpleTokenizer()


class SimpleTokenizer:
    """シンプルなUTF-8トークナイザー（フォールバック用）"""
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        self.eos_token_id = 2
        self.pad_token_id = 0
    
    def encode(self, text, return_tensors=None, **kwargs):
        # UTF-8バイトエンコード
        ids = [min(b + 3, self.vocab_size - 1) for b in text.encode('utf-8')]
        if return_tensors == "pt":
            return torch.tensor([ids])
        return ids
    
    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids[0], list):
            ids = ids[0]
        # バイトデコード
        try:
            bytes_list = [max(0, i - 3) for i in ids if i > 2]
            return bytes(bytes_list).decode('utf-8', errors='replace')
        except:
            return ""


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    config: ChatConfig,
    device: str = "cuda",
    stream: bool = True,
    n_seq: int = 512,  # シーケンス長 (モデルの固定長)
):
    """テキスト生成（ストリーミング対応）"""
    
    # パッドトークンID
    pad_token_id = getattr(tokenizer, 'pad_token_id', 0) or 0
    
    # エンコード - トークナイザーによって出力形式が異なる
    input_ids = None
    
    # 方法1: tokenizer(prompt) を試す
    try:
        encoded = tokenizer(prompt, return_tensors="pt")
        if hasattr(encoded, 'input_ids'):
            input_ids = encoded.input_ids.to(device)
        elif isinstance(encoded, dict) and 'input_ids' in encoded:
            input_ids = encoded['input_ids'].to(device)
        elif isinstance(encoded, torch.Tensor):
            input_ids = encoded.to(device)
    except Exception as e:
        pass
    
    # 方法2: tokenizer.encode(prompt) を試す
    if input_ids is None:
        try:
            encoded = tokenizer.encode(prompt, return_tensors="pt")
            if isinstance(encoded, torch.Tensor):
                input_ids = encoded.to(device)
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
            elif hasattr(encoded, 'input_ids'):
                input_ids = encoded.input_ids.to(device)
            elif isinstance(encoded, dict) and 'input_ids' in encoded:
                input_ids = encoded['input_ids'].to(device)
        except Exception as e:
            pass
    
    # 方法3: tokenizer.encode(prompt) でリストを取得
    if input_ids is None:
        try:
            encoded = tokenizer.encode(prompt)
            # タプルの場合は最初の要素を使用 (input_ids, attention_mask など)
            if isinstance(encoded, tuple):
                ids = encoded[0]
            elif isinstance(encoded, list):
                ids = encoded
            elif isinstance(encoded, torch.Tensor):
                ids = encoded.tolist()
            else:
                ids = list(encoded)
            
            # ネストされたリストの場合は展開
            if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
                ids = ids[0]
            
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        except Exception as e:
            pass
    
    # 方法4: UTF-8バイトエンコード（最終手段）
    if input_ids is None:
        ids = list(prompt.encode('utf-8'))
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    
    generated = input_ids.clone()
    past_tokens = set(input_ids[0].tolist())
    
    for step in range(config.max_new_tokens):
        # 現在のシーケンス長
        current_len = generated.shape[1]
        
        # n_seq より長い場合は切り詰め
        if current_len > n_seq:
            context = generated[:, -n_seq:]
            actual_len = n_seq
        else:
            # n_seq に満たない場合は左パディング
            context = generated
            actual_len = current_len
            if current_len < n_seq:
                padding = torch.full((1, n_seq - current_len), pad_token_id, dtype=torch.long, device=device)
                context = torch.cat([padding, context], dim=1)
        
        # Forward pass
        with torch.cuda.amp.autocast(enabled=device=="cuda"):
            output = model(context)
            # Phase8IntegratedModel returns (logits, diagnostics) tuple
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
        
        # 最後のトークンのlogits
        next_logits = logits[:, -1, :].float() / config.temperature
        
        # Repetition penalty
        if config.repetition_penalty != 1.0:
            for token_id in past_tokens:
                if token_id < next_logits.shape[-1]:
                    next_logits[0, token_id] /= config.repetition_penalty
        
        # Top-k filtering
        if config.top_k > 0:
            indices_to_remove = next_logits < torch.topk(next_logits, config.top_k)[0][..., -1, None]
            next_logits[indices_to_remove] = float('-inf')
        
        # Top-p (nucleus) filtering
        if config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > config.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_logits[indices_to_remove] = float('-inf')
        
        # サンプリング
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        generated = torch.cat([generated, next_token], dim=1)
        past_tokens.add(next_token.item())
        
        # ストリーミング出力
        if stream:
            token_text = tokenizer.decode([next_token.item()], skip_special_tokens=True)
            print(token_text, end="", flush=True)
        
        # EOS check
        if hasattr(tokenizer, 'eos_token_id') and next_token.item() == tokenizer.eos_token_id:
            break
    
    if stream:
        print()  # 改行
    
    # デコード
    output_ids = generated[0].tolist()
    return tokenizer.decode(output_ids, skip_special_tokens=True)


def find_latest_checkpoint():
    """最新のチェックポイントを探す"""
    search_dirs = [
        "checkpoints/phase8_300m_scaling",  # 300M scaling experiment
        "checkpoints/phase8_10b_japanese",
        "checkpoints/phase8_10b_rtx3080",
        "checkpoints/phase8",
        "checkpoints/phase7_max_push",
    ]
    
    for dir_path in search_dirs:
        ckpt_dir = Path(dir_path)
        if not ckpt_dir.exists():
            continue
        
        # 優先順位: best.pt > final.pt > phase8_10b_final.pt > step_*.pt
        for name in ["best.pt", "final.pt", "phase8_10b_final.pt"]:
            path = ckpt_dir / name
            if path.exists():
                return str(path)
        
        # step_*.pt から最新
        step_files = list(ckpt_dir.glob("step_*.pt")) + list(ckpt_dir.glob("*.pt"))
        if step_files:
            return str(max(step_files, key=lambda p: p.stat().st_mtime))
    
    return None


def interactive_chat(model, tokenizer, config: ChatConfig, device: str = "cuda"):
    """インタラクティブチャットモード"""
    
    print("\n" + "=" * 60)
    print("🤖 MUSE Chat AI - Phase 8 Japanese/English")
    print("=" * 60)
    print("コマンド / Commands:")
    print("  /quit, /exit  - 終了 / Exit")
    print("  /temp <val>   - temperature (現在: {:.1f})".format(config.temperature))
    print("  /tokens <val> - 最大トークン数 (現在: {})".format(config.max_new_tokens))
    print("  /clear        - 履歴クリア / Clear history")
    print("  /system <msg> - システムプロンプト設定")
    print("=" * 60 + "\n")
    
    history: List[Dict[str, str]] = []
    system_prompt = "あなたは親切で知識豊富なAIアシスタントです。"
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 さようなら！ / Goodbye!")
            break
        
        if not user_input:
            continue
        
        # コマンド処理
        if user_input.startswith('/'):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            
            if cmd in ['/quit', '/exit', '/q']:
                print("👋 さようなら！ / Goodbye!")
                break
            elif cmd == '/temp' and len(parts) > 1:
                try:
                    config.temperature = float(parts[1])
                    print(f"✓ Temperature: {config.temperature}")
                except ValueError:
                    print("❌ 無効な値です")
                continue
            elif cmd == '/tokens' and len(parts) > 1:
                try:
                    config.max_new_tokens = int(parts[1])
                    print(f"✓ Max tokens: {config.max_new_tokens}")
                except ValueError:
                    print("❌ 無効な値です")
                continue
            elif cmd == '/clear':
                history = []
                print("✓ 履歴をクリアしました")
                continue
            elif cmd == '/system' and len(parts) > 1:
                system_prompt = parts[1]
                print(f"✓ System prompt: {system_prompt}")
                continue
            else:
                print(f"❌ 不明なコマンド: {cmd}")
                continue
        
        # プロンプト構築（日本語チャット形式）
        prompt = f"### システム:\n{system_prompt}\n\n"
        
        # 履歴（直近3ターン）
        for h in history[-3:]:
            prompt += f"### ユーザー:\n{h['user']}\n\n### アシスタント:\n{h['assistant']}\n\n"
        
        prompt += f"### ユーザー:\n{user_input}\n\n### アシスタント:\n"
        
        # 生成
        print("AI: ", end="", flush=True)
        try:
            response = generate(
                model, tokenizer, prompt,
                config=config,
                device=device,
                stream=True,
            )
            
            # プロンプト部分を除去
            if "### アシスタント:" in response:
                response = response.split("### アシスタント:")[-1].strip()
            if "### ユーザー:" in response:
                response = response.split("### ユーザー:")[0].strip()
            
            history.append({
                'user': user_input,
                'assistant': response
            })
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
        
        print()


def main():
    parser = argparse.ArgumentParser(description="MUSE Chat AI - Phase 8")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (default: auto-detect)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt (non-interactive)")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--tokenizer", type=str, default="rinna/japanese-gpt-neox-3.6b",
                        help="Tokenizer name")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cuda/cpu/auto)")
    args = parser.parse_args()
    
    # デバイス
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, using CPU")
        device = "cpu"
    
    # チェックポイント探索
    checkpoint = args.checkpoint or find_latest_checkpoint()
    
    if checkpoint is None:
        print("❌ チェックポイントが見つかりません！")
        print("\nまず学習を実行してください:")
        print("  make start-japanese   # 日本語10Bモデル学習")
        print("\nまたはチェックポイントを指定:")
        print("  python scripts/chat_inference.py --checkpoint path/to/model.pt")
        sys.exit(1)
    
    # モデルロード
    model, model_config = load_model_phase8(checkpoint, device)
    
    # トークナイザー
    tokenizer = load_tokenizer(args.tokenizer)
    
    # チャット設定
    chat_config = ChatConfig(
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
    )
    
    # 推論
    if args.prompt:
        # 単発モード
        print(f"\nPrompt: {args.prompt}")
        print("-" * 40)
        response = generate(
            model, tokenizer, args.prompt,
            config=chat_config,
            device=device,
            stream=True,
        )
    else:
        # インタラクティブモード
        interactive_chat(model, tokenizer, chat_config, device)


if __name__ == "__main__":
    main()
