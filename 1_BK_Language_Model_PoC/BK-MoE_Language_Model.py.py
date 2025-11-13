# ===========================
# Google Colab 用 完成版コード
# ===========================

# ライブラリインストール（最初の実行時だけ少し時間がかかります）
%pip install datasets --quiet

import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
from torch.func import vmap
import torch.nn.functional as F
import numpy as np
import copy
from collections import Counter
from datasets import load_dataset

# --- (✘1) O(N) Tridiagonal Inverse Diagonal Algorithm (by Teppei Arai) ---
# TorchScript / torch.jit を使わない、安全な純Python版
def get_tridiagonal_inverse_diagonal(a, b, c, z):
    """
    a, b, c: 1次元テンソル (N,), (N-1,), (N-1,)
    z:       スカラー (complex tensor)
    戻り値:  diag((T - zI)^-1)  (N,) complex64
    想定している T は tridiagonal:
      a: main diag
      b: super diag
      c: sub diag
    """
    n = a.shape[-1]
    device = a.device

    a_c = a.to(torch.complex128)
    b_c = b.to(torch.complex128)
    c_c = c.to(torch.complex128)
    z_c = z.to(torch.complex128)
    a_shifted = a_c - z_c

    if n == 0:
        return torch.zeros(0, dtype=torch.complex64, device=device)

    # --- theta の再帰（前向き） ---
    theta = []
    theta.append(torch.ones((), dtype=torch.complex128, device=device))  # θ_0
    theta.append(a_shifted[0])                                           # θ_1

    for i in range(1, n):
        theta.append(a_shifted[i] * theta[i] - c_c[i-1] * b_c[i-1] * theta[i-1])

    theta_stack = torch.stack(theta)
    det_T = theta_stack[-1]

    # --- phi の再帰（後ろ向き） ---
    phi = [torch.zeros((), dtype=torch.complex128, device=device) for _ in range(n)]
    phi[n-1] = torch.ones((), dtype=torch.complex128, device=device)

    if n > 1:
        phi[n-2] = a_shifted[-1]
        for i in range(n-3, -1, -1):
            phi[i] = a_shifted[i+1] * phi[i+1] - c_c[i] * b_c[i] * phi[i+2]

    phi_stack = torch.stack(phi)

    eps = torch.tensor(1e-18, dtype=torch.complex128, device=device)
    diag_inv = theta_stack[:-1] * phi_stack / (det_T + eps)

    return diag_inv.to(torch.complex64)

# vmap 化された関数（batched Diag 計算）
vmapped_get_diag = vmap(
    get_tridiagonal_inverse_diagonal, in_dims=(0, 0, 0, None), out_dims=0
)

# --- (✘3) Sparse Mixture of Experts (MoE) Layer ---
class SparseMoELayer(nn.Module):
    def __init__(self, d_model, num_experts=4, top_k=1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),  # FFNを模倣
                nn.ReLU(),
                nn.Linear(d_model * 2, d_model)
            ) for _ in range(num_experts)
        ])
        self.gating_network = nn.Linear(d_model, num_experts)
        self.x_flat = None
        self.gates = None

    def forward(self, x):
        batch_size, n_seq, d_model = x.shape
        self.x_flat = x.reshape(-1, d_model)
        router_logits = self.gating_network(self.x_flat)
        # hard ルーティング (Gumbel-Softmax)
        self.gates = F.gumbel_softmax(router_logits, hard=True, tau=1.0)

        final_output = torch.zeros((batch_size * n_seq, d_model), device=x.device)

        for i in range(self.num_experts):
            expert = self.experts[i]
            gate_for_this_expert = self.gates[:, i].unsqueeze(-1)
            expert_output = expert(self.x_flat)
            final_output += expert_output * gate_for_this_expert

        return final_output.reshape(batch_size, n_seq, d_model)  # (B, N, D)

# --- (✘1 + ✘2 + ✘3) 最終形態: MoE-ResNet-BK Layer ---
class MoEResNetBKLayer(nn.Module):
    def __init__(self, d_model, num_experts=4, top_k=1):
        super().__init__()
        # ポテンシャル v を計算する MLP の代わりに MoE FFN
        self.moe_ffn = SparseMoELayer(d_model, num_experts, top_k)

        # v を計算する MLP（MoE FFN の出力から v を生成）
        self.v_proj = nn.Linear(d_model, 1)

        self.z = torch.tensor(1.0j, dtype=torch.complex64)
        self.output_proj = nn.Linear(2, d_model)

        # (✘2) backward のためのフック（今回は未使用）
        self.v = None
        self.G_ii = None
        self.output_features = None

    def forward(self, x):
        batch_size, n_seq, d_model = x.shape

        # (✘3) FFN (MLP) 部分を MoE で計算
        ffn_output = self.moe_ffn(x)

        # (✘3) v を計算
        self.v = self.v_proj(ffn_output)   # (batch_size, n_seq, 1)
        v_squeezed = self.v.squeeze(-1)    # (batch_size, n_seq)

        # ベースの tridiagonal 行列 H0
        h0_diag = torch.full((batch_size, n_seq), -2.0, device=x.device, dtype=torch.float32)
        h0_sub  = torch.full((batch_size, n_seq-1), 1.0, device=x.device, dtype=torch.float32)
        h0_super= torch.full((batch_size, n_seq-1), 1.0, device=x.device, dtype=torch.float32)

        he_diag = h0_diag + v_squeezed  # 対角をポテンシャルで変形

        self.G_ii = vmapped_get_diag(
            he_diag, h0_super, h0_sub, self.z.to(x.device)
        )  # (batch_size, n_seq) complex

        self.output_features = torch.stack(
            [self.G_ii.real, self.G_ii.imag], dim=-1
        ).to(torch.float32)  # (B, N, 2)

        # (✘1) スペクトル情報から d_model 次元の表現を再構築
        final_output = self.output_proj(self.output_features)  # (B, N, D)

        # ResNet-BK の出力 + MoE FFN の出力（残差的に合成）
        return final_output + ffn_output

# --- (✘2) ResNet-BK ブロック（LayerNorm + 残差） ---
class ResNetBKBlock(nn.Module):
    def __init__(self, d_model, num_experts=4, top_k=1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.bk_layer = MoEResNetBKLayer(d_model, num_experts, top_k)

    def forward(self, x):
        return x + self.bk_layer(self.layer_norm(x))

    # analytic_backward は今回未使用
    def analytic_backward(self, grad_output):
        pass

# --- Nano-GPT 風 言語モデル本体 ---
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_layers=4, n_seq=128, num_experts=4):
        super().__init__()
        self.d_model = d_model
        self.n_seq = n_seq

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(n_seq, d_model)

        # Attention の代わりに ResNet-BK ブロックをスタック
        self.blocks = nn.ModuleList(
            [ResNetBKBlock(d_model, num_experts) for _ in range(n_layers)]
        )

        self.layer_norm_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        batch_size, n_seq = x.shape

        tok_emb = self.token_embedding(x)  # (B, N, D)
        pos = torch.arange(0, n_seq, dtype=torch.long, device=x.device).unsqueeze(0)  # (1, N)
        pos_emb = self.position_embedding(pos)  # (1, N, D)

        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.layer_norm_final(x)
        logits = self.lm_head(x)  # (B, N, vocab_size)
        return logits

# --- HuggingFace datasets を使ったリアルテキスト処理 (WikiText2) ---
def get_data_loader(batch_size, n_seq):
    """
    torchtext ではなく datasets(wikitext-2-raw-v1) を使う版
    戻り値:
      train_data: (seq_len_total, batch_size) の LongTensor
      vocab:      dict (stoi, itos, vocab_size)
      get_batch:  関数 (source, i) -> (data, target)
    """
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        train_texts = dataset["train"]["text"]  # list[str]
    except Exception as e:
        print(f"WikiText2 データセットのロードに失敗しました: {e}")
        print("ネットワーク接続や環境を確認してください。")
        return None, None, None

    # --- 単純な単語分割 & vocab 構築 ---
    counter = Counter()
    for line in train_texts:
        tokens = line.strip().split()
        if tokens:
            counter.update(tokens)

    special_tokens = ["<unk>"]
    stoi = {}
    itos = []

    for sp in special_tokens:
        stoi[sp] = len(itos)
        itos.append(sp)

    for tok, freq in counter.items():
        if tok not in stoi:
            stoi[tok] = len(itos)
            itos.append(tok)

    vocab_size = len(itos)
    unk_id = stoi["<unk>"]

    def encode_texts(texts):
        ids = []
        for line in texts:
            for tok in line.strip().split():
                ids.append(stoi.get(tok, unk_id))
        return torch.tensor(ids, dtype=torch.long)

    train_ids = encode_texts(train_texts)  # 1D LongTensor

    # --- GPT 系と同じ batchify 戦略 ---
    def batchify(data, bsz):
        seq_len = data.size(0) // bsz
        data = data.narrow(0, 0, seq_len * bsz)
        data = data.view(bsz, seq_len).t().contiguous()  # (seq_len, batch_size)
        return data

    train_data = batchify(train_ids, batch_size)

    def get_batch(source, i):
        seq_len = min(n_seq, len(source) - 1 - i)
        data = source[i:i+seq_len]                # (seq_len, batch_size)
        target = source[i+1:i+1+seq_len].reshape(-1)  # (seq_len * batch_size,)
        return data, target

    vocab = {
        "stoi": stoi,
        "itos": itos,
        "vocab_size": vocab_size,
    }

    return train_data, vocab, get_batch

# --- メインの実行ブロック (知能テスト) ---
def run_language_model_test():
    # パラメータ
    D_MODEL    = 64
    N_SEQ      = 128
    BATCH_SIZE = 20
    N_LAYERS   = 4
    NUM_EXPERTS= 4
    EPOCHS     = 3  # PoC のため軽く 3 エポック

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device.type.upper()}")

    # 1. データの準備
    train_data, vocab, get_batch = get_data_loader(BATCH_SIZE, N_SEQ)
    if train_data is None:
        return

    VOCAB_SIZE = vocab["vocab_size"]
    print(f"Vocabulary Size: {VOCAB_SIZE}")

    # 我々の最終形態モデル
    model = LanguageModel(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_seq=N_SEQ,
        num_experts=NUM_EXPERTS
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("--- (Final Step) Language Model 'Intelligence Test' ---")
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()

        for i in range(0, train_data.size(0) - 1, N_SEQ):
            x_batch, y_batch = get_batch(train_data, i)
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # 不完全バッチはスキップ
            if x_batch.size(0) != N_SEQ:
                continue

            optimizer.zero_grad()
            logits = model(x_batch)  # (B, N, V)

            loss = criterion(
                logits.view(-1, logits.size(-1)),  # (B*N, V)
                y_batch                             # (B*N,)
            )

            if torch.isnan(loss):
                print(f"Epoch {epoch}, Batch {i}: Loss is NaN. Skipping update.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        end_time = time.time()
        avg_loss = total_loss / max(1, num_batches)
        perplexity = math.exp(avg_loss)
        print(f"Epoch {epoch}/{EPOCHS} | Time: {end_time - start_time:.2f}s | "
              f"Avg Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")

if __name__ == "__main__":
    run_language_model_test()
