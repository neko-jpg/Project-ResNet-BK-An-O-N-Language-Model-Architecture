# Phase 2 Usage Examples - Quick Reference

**最終更新**: 2025-01-20

## 概要

Phase 2統合モデルの使用例のクイックリファレンスです。4つの例ファイルの主要な使用方法を簡潔にまとめています。

---

## 1. 基本使用 (`examples/phase2_basic_usage.py`)

### モデルの作成

```python
from src.models.phase2 import Phase2IntegratedModel, create_phase2_model, Phase2Config

# 方法1: 直接インスタンス化
model = Phase2IntegratedModel(
    vocab_size=1000,
    d_model=128,
    n_layers=2,
    n_seq=64,
)

# 方法2: ファクトリ関数（プリセット）
model = create_phase2_model(preset="small")

# 方法3: ファクトリ関数（カスタム設定）
config = Phase2Config(
    vocab_size=1000,
    d_model=128,
    n_layers=2,
    base_decay=0.02,
    hebbian_eta=0.15,
)
model = create_phase2_model(config=config)
```

### Forward Pass

```python
# 基本的なforward pass
input_ids = torch.randint(0, 1000, (batch_size, seq_len))
logits = model(input_ids)

# 診断情報付きforward pass
logits, diagnostics = model(input_ids, return_diagnostics=True)

# 診断情報の内容
# - gamma_values: 各層のΓ値
# - snr_stats: SNR統計
# - resonance_info: 共鳴情報
# - stability_metrics: 安定性メトリクス
```

### 状態管理

```python
# 状態をリセット（新しいシーケンスの開始時）
model.reset_state()

# 状態を確認
for i, block in enumerate(model.blocks):
    if block.fast_weight_state is not None:
        norm = torch.norm(block.fast_weight_state).item()
        print(f"Layer {i}: ノルム={norm:.4f}")
```

### 統計情報の取得

```python
# モデル全体の統計
stats = model.get_statistics()

# 含まれる情報:
# - num_parameters: パラメータ数
# - num_trainable_parameters: 学習可能パラメータ数
# - num_layers: レイヤー数
# - block_stats: 各ブロックの統計
```

---

## 2. 学習 (`examples/phase2_training.py`)

### データセットの準備

```python
from torch.utils.data import Dataset, DataLoader

class TinyTextDataset(Dataset):
    def __init__(self, vocab_size, seq_len, num_samples):
        self.data = [
            np.random.randint(0, vocab_size, size=seq_len + 1)
            for _ in range(num_samples)
        ]
    
    def __getitem__(self, idx):
        seq = self.data[idx]
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        target_ids = torch.tensor(seq[1:], dtype=torch.long)
        return input_ids, target_ids

# データローダー作成
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
```

### 学習ループ

```python
import torch.optim as optim

# オプティマイザー設定
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# 学習ループ
model.train()
for epoch in range(num_epochs):
    for input_ids, target_ids in dataloader:
        # 状態をリセット（各バッチで独立）
        model.reset_state()
        
        # Forward pass
        logits = model(input_ids)
        
        # 損失計算
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # パラメータ更新
        optimizer.step()
```

### 診断情報の収集

```python
# 診断情報付き学習
logits, diagnostics = model(input_ids, return_diagnostics=True)

# Γ値の監視
for layer_idx, gamma in enumerate(diagnostics['gamma_values']):
    if gamma is not None:
        print(f"Layer {layer_idx} Γ: {gamma.mean().item():.6f}")

# SNR統計
for layer_idx, snr_stats in enumerate(diagnostics['snr_stats']):
    if snr_stats:
        print(f"Layer {layer_idx} SNR: {snr_stats['mean_snr']:.4f}")
```

### モデルの保存と読み込み

```python
# 保存
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config.to_dict(),
    'train_loss': train_loss,
}, 'checkpoint.pt')

# 読み込み
checkpoint = torch.load('checkpoint.pt')
config = Phase2Config.from_dict(checkpoint['config'])
model = create_phase2_model(config=config)
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## 3. 推論 (`examples/phase2_inference.py`)

### Greedy Decoding

```python
def greedy_decode(model, input_ids, max_length):
    model.eval()
    model.reset_state()
    
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_length - input_ids.size(1)):
            logits = model(generated)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
    
    return generated

# 使用例
prompt = torch.randint(0, 1000, (1, 10))
generated = greedy_decode(model, prompt, max_length=50)
```

### Top-k Sampling

```python
def top_k_sampling(model, input_ids, max_length, k=10, temperature=1.0):
    model.eval()
    model.reset_state()
    
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_length - input_ids.size(1)):
            logits = model(generated)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Top-kフィルタリング
            top_k_logits, top_k_indices = torch.topk(next_token_logits, k, dim=-1)
            probs = F.softmax(top_k_logits, dim=-1)
            
            # サンプリング
            sampled_indices = torch.multinomial(probs, num_samples=1)
            next_token = torch.gather(top_k_indices, 1, sampled_indices)
            
            generated = torch.cat([generated, next_token], dim=1)
    
    return generated

# 使用例
generated = top_k_sampling(model, prompt, max_length=50, k=10, temperature=1.0)
```

### バッチ推論

```python
# 複数のプロンプトを同時に処理
batch_prompts = torch.randint(0, 1000, (4, 10))
generated = greedy_decode(model, batch_prompts, max_length=50)
```

### ストリーミング推論

```python
# トークンを1つずつ生成
model.eval()
model.reset_state()

generated = prompt.clone()

with torch.no_grad():
    for step in range(max_length - prompt.size(1)):
        logits = model(generated)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        
        # リアルタイムで出力
        print(f"{next_token.item()} ", end="", flush=True)
```

### Perplexity評価

```python
model.eval()
total_loss = 0.0
total_tokens = 0

with torch.no_grad():
    for input_ids, target_ids in test_loader:
        model.reset_state()
        logits = model(input_ids)
        
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            reduction='sum'
        )
        
        total_loss += loss.item()
        total_tokens += target_ids.numel()

avg_loss = total_loss / total_tokens
perplexity = torch.exp(torch.tensor(avg_loss)).item()
```

---

## 4. 診断 (`examples/phase2_diagnostics.py`)

### Γ値の監視

```python
import matplotlib.pyplot as plt

# Γ値を収集
gamma_history = {i: [] for i in range(n_layers)}

for seq_idx in range(num_sequences):
    input_ids = torch.randint(0, 1000, (1, seq_len))
    model.reset_state()
    
    logits, diagnostics = model(input_ids, return_diagnostics=True)
    
    for layer_idx, gamma in enumerate(diagnostics['gamma_values']):
        if gamma is not None:
            gamma_history[layer_idx].append(gamma.mean().item())

# 可視化
plt.figure(figsize=(12, 6))
for layer_idx in range(n_layers):
    plt.plot(gamma_history[layer_idx], marker='o', label=f'Layer {layer_idx}')
plt.xlabel('Sequence Index')
plt.ylabel('Γ (Decay Rate)')
plt.legend()
plt.savefig('gamma_monitoring.png')
```

### SNR統計の取得

```python
# SNR統計を収集
snr_history = {i: {'mean_snr': [], 'std_snr': []} for i in range(n_layers)}

for seq_idx in range(num_sequences):
    input_ids = torch.randint(0, 1000, (1, seq_len))
    model.reset_state()
    
    logits, diagnostics = model(input_ids, return_diagnostics=True)
    
    for layer_idx, snr_stats in enumerate(diagnostics['snr_stats']):
        if snr_stats:
            snr_history[layer_idx]['mean_snr'].append(snr_stats['mean_snr'])
            snr_history[layer_idx]['std_snr'].append(snr_stats['std_snr'])

# 統計を表示
for layer_idx in range(n_layers):
    mean_snr = np.array(snr_history[layer_idx]['mean_snr']).mean()
    print(f"Layer {layer_idx} 平均SNR: {mean_snr:.4f}")
```

### 共鳴情報の可視化

```python
# 共鳴情報を取得
input_ids = torch.randint(0, 1000, (1, seq_len))
model.reset_state()

logits, diagnostics = model(input_ids, return_diagnostics=True)

# 共鳴エネルギーのヒートマップ
for layer_idx, resonance_info in enumerate(diagnostics['resonance_info']):
    if resonance_info and 'diag_energy' in resonance_info:
        diag_energy = resonance_info['diag_energy'].cpu().numpy()
        
        plt.figure(figsize=(10, 6))
        plt.imshow(diag_energy.mean(axis=0), aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title(f'Layer {layer_idx} 共鳴エネルギー')
        plt.savefig(f'resonance_layer_{layer_idx}.png')
```

### 安定性メトリクスの追跡

```python
# 安定性メトリクスを収集
stability_history = {i: {'energy': [], 'dE_dt': []} for i in range(n_layers)}

for seq_idx in range(num_sequences):
    input_ids = torch.randint(0, 1000, (1, seq_len))
    model.reset_state()
    
    logits, diagnostics = model(input_ids, return_diagnostics=True)
    
    for layer_idx, stability in enumerate(diagnostics['stability_metrics']):
        if stability:
            if 'energy' in stability:
                stability_history[layer_idx]['energy'].append(stability['energy'])
            if 'dE_dt' in stability:
                stability_history[layer_idx]['dE_dt'].append(stability['dE_dt'])

# Lyapunov条件の確認
for layer_idx in range(n_layers):
    dE_dts = np.array(stability_history[layer_idx]['dE_dt'])
    stable_ratio = (dE_dts <= 0).mean()
    print(f"Layer {layer_idx} 安定率: {stable_ratio * 100:.1f}%")
```

### 包括的レポートの生成

```python
import json

# 診断情報を収集
input_ids = torch.randint(0, 1000, (2, seq_len))
model.reset_state()

logits, diagnostics = model(input_ids, return_diagnostics=True)
model_stats = model.get_statistics()

# レポートを作成
report = {
    'model_config': config.to_dict(),
    'model_statistics': {
        'num_parameters': model_stats['num_parameters'],
        'num_layers': model_stats['num_layers'],
    },
    'diagnostics': {
        'gamma_values': [...],
        'snr_stats': [...],
        'resonance_info': [...],
        'stability_metrics': [...],
    },
}

# JSONとして保存
with open('comprehensive_report.json', 'w') as f:
    json.dump(report, f, indent=2)
```

---

## よくある使用パターン

### パターン1: 基本的な学習と評価

```python
# モデル作成
model = create_phase2_model(preset="base")
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# 学習
for epoch in range(num_epochs):
    model.train()
    for input_ids, target_ids in train_loader:
        model.reset_state()
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 評価
    model.eval()
    with torch.no_grad():
        for input_ids, target_ids in val_loader:
            model.reset_state()
            logits = model(input_ids)
            # 評価メトリクスを計算
```

### パターン2: 診断情報付き学習

```python
# 診断情報を収集しながら学習
for epoch in range(num_epochs):
    model.train()
    
    # 最初のバッチで診断情報を収集
    for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
        model.reset_state()
        
        if batch_idx == 0:
            logits, diagnostics = model(input_ids, return_diagnostics=True)
            # Γ値、SNR統計などをロギング
        else:
            logits = model(input_ids)
        
        # 学習処理
```

### パターン3: インタラクティブな推論

```python
# ユーザー入力に基づく生成
model.eval()
model.reset_state()

while True:
    user_input = input("プロンプト: ")
    if user_input == "quit":
        break
    
    # トークン化（簡易版）
    input_ids = tokenize(user_input)
    
    # 生成
    generated = greedy_decode(model, input_ids, max_length=100)
    
    # デトークン化して表示
    output_text = detokenize(generated)
    print(f"生成: {output_text}")
```

---

## トラブルシューティング

### 問題1: VRAM不足

```python
# 解決策1: バッチサイズを減らす
dataloader = DataLoader(dataset, batch_size=2)  # 4 → 2

# 解決策2: シーケンス長を減らす
config = Phase2Config(n_seq=512)  # 1024 → 512

# 解決策3: モデルサイズを減らす
model = create_phase2_model(preset="small")
```

### 問題2: 過減衰警告

```python
# 警告: "Overdamped system detected: Γ/|V| = 600.63"

# 解決策: base_decayを減らす
config = Phase2Config(base_decay=0.005)  # 0.01 → 0.005
```

### 問題3: 安定性違反

```python
# 警告: "Lyapunov stability violated"

# 解決策1: gamma_adjust_rateを増やす
config = Phase2Config(gamma_adjust_rate=0.02)  # 0.01 → 0.02

# 解決策2: 学習率を下げる
optimizer = optim.AdamW(model.parameters(), lr=5e-4)  # 1e-3 → 5e-4
```

---

## 参考資料

- **設計書**: `.kiro/specs/phase2-breath-of-life/design.md`
- **要件定義**: `.kiro/specs/phase2-breath-of-life/requirements.md`
- **実装ガイド**: `docs/PHASE2_IMPLEMENTATION_GUIDE.md`
- **完了サマリー**: `results/benchmarks/TASK16_COMPLETION_SUMMARY.md`

---

**最終更新**: 2025-01-20  
**バージョン**: 1.0
