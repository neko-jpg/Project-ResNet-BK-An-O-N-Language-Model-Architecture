# ===========================
# ResNet-BK Ultra v2
# O(N) BK-Core + Hybrid Analytic Grad + Sparse MoE
# Google Colab 用 完成版コード（数値安定化版）
# ===========================

# ライブラリインストール（最初の実行時だけ少し時間がかかります）
%pip install datasets --quiet

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import math
from torch.func import vmap
from collections import Counter
from datasets import load_dataset
from torch.optim.lr_scheduler import CosineAnnealingLR

# ---------------------------------------------
# (✘1) O(N) Tridiagonal Inverse Diagonal (BK-Core)
# ---------------------------------------------
def get_tridiagonal_inverse_diagonal(a, b, c, z):
    """
    a, b, c: 1次元テンソル (N,), (N-1,), (N-1,)
    z:       スカラー (complex tensor)
    戻り値:  diag((T - zI)^-1)  (N,) complex64
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

    # --- theta 再帰（前向き） ---
    theta = []
    theta.append(torch.ones((), dtype=torch.complex128, device=device))  # θ_0
    theta.append(a_shifted[0])                                          # θ_1

    for i in range(1, n):
        theta.append(a_shifted[i] * theta[i] - c_c[i-1] * b_c[i-1] * theta[i-1])

    theta_stack = torch.stack(theta)
    det_T = theta_stack[-1]

    # --- phi 再帰（後ろ向き） ---
    phi = [torch.zeros((), dtype=torch.complex128, device=device) for _ in range(n)]
    phi[n-1] = torch.ones((), dtype=torch.complex128, device=device)

    if n > 1:
        phi[n-2] = a_shifted[-1]
        for i in range(n - 3, -1, -1):
            phi[i] = a_shifted[i+1] * phi[i+1] - c_c[i] * b_c[i] * phi[i+2]

    phi_stack = torch.stack(phi)

    eps = torch.tensor(1e-18, dtype=torch.complex128, device=device)
    diag_inv = theta_stack[:-1] * phi_stack / (det_T + eps)

    # --- 数値安定化: NaN/Inf 除去 + 振幅クリップ ---
    diag_inv = torch.where(torch.isfinite(diag_inv), diag_inv, torch.zeros_like(diag_inv))
    max_mag = 50.0  # resolvent の振幅上限
    mag = diag_inv.abs()
    factor = torch.where(mag > max_mag, max_mag / (mag + 1e-9), torch.ones_like(mag))
    diag_inv = diag_inv * factor

    return diag_inv.to(torch.complex64)


# batched BK-Core
vmapped_get_diag = vmap(
    get_tridiagonal_inverse_diagonal, in_dims=(0, 0, 0, None), out_dims=0
)

# ---------------------------------------------
# (✘2) BK-Core Autograd Function (Hybrid Gradient)
# ---------------------------------------------
class BKCoreFunction(torch.autograd.Function):
    """
    O(N) BK-Core (forward) + O(N) 解析的勾配 (backward) をカプセル化。
    勾配は：
      - 理論系:   dG/dv = -G^2
      - 仮説7系:  dL/dv ~ -(dL/dG) / G^2
    のハイブリッド。
    """
    # 0.0 -> 純粋に理論系
    # 1.0 -> 純粋に仮説7系
    GRAD_BLEND = 0.5

    @staticmethod
    def forward(ctx, he_diag, h0_super, h0_sub, z):
        # G_ii = diag((H - zI)^-1)
        G_ii = vmapped_get_diag(he_diag, h0_super, h0_sub, z)
        ctx.save_for_backward(G_ii)

        # 実数特徴量 (real, imag) に変換
        output_features = torch.stack(
            [G_ii.real, G_ii.imag], dim=-1
        ).to(torch.float32)  # (B, N, 2)

        return output_features

    @staticmethod
    def backward(ctx, grad_output_features):
        """
        grad_output_features: (B, N, 2) 実数勾配
        """
        (G_ii,) = ctx.saved_tensors  # (B, N) complex
        # dL/dG = dL/dRe(G) + i dL/dIm(G)
        grad_G = torch.complex(
            grad_output_features[..., 0],
            grad_output_features[..., 1],
        )

        # --- G^2 と 1/G^2 の両方を安全に扱う ---
        G_sq = G_ii ** 2

        # 1/G^2 用の denominator を位相維持したまま下限クランプ
        denom = G_sq
        denom_mag = denom.abs()
        min_denom = 1e-3  # これより小さいと 1/G^2 が爆発する
        denom = torch.where(
            denom_mag < min_denom,
            denom / (denom_mag + 1e-9) * min_denom,
            denom,
        )

        # --- 理論系勾配 dG/dv = -G^2 ---
        grad_v_analytic = -(grad_G * G_sq).real

        # --- 仮説7系勾配 逆二乗型 ---
        grad_v_h7 = -(grad_G / (denom + 1e-6)).real

        # --- ハイブリッド ---
        alpha = BKCoreFunction.GRAD_BLEND
        grad_v = (1.0 - alpha) * grad_v_analytic + alpha * grad_v_h7

        # --- 数値的安全装置 ---
        grad_v = torch.where(torch.isfinite(grad_v), grad_v, torch.zeros_like(grad_v))
        grad_v = torch.clamp(grad_v, -1000.0, 1000.0)

        grad_he_diag = grad_v.to(torch.float32)

        # he_diag 以外 (h0_super, h0_sub, z) には勾配を流さない
        return grad_he_diag, None, None, None


# ---------------------------------------------
# (✘3) Sparse Mixture of Experts Layer (top-1 routing)
# ---------------------------------------------
class SparseMoELayer(nn.Module):
    def __init__(self, d_model, num_experts=4, top_k=1, dropout_p=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(d_model * 2, d_model),
            )
            for _ in range(num_experts)
        ])
        self.gating_network = nn.Linear(d_model, num_experts)

    def forward(self, x):
        """
        x: (B, N, D)
        戻り値: (B, N, D)
        """
        B, N, D = x.shape
        x_flat = x.reshape(B * N, D)  # (T, D), T = B*N
        router_logits = self.gating_network(x_flat)  # (T, E)

        if self.top_k >= self.num_experts:
            # Dense Mixture (softmax 合成) モード
            gates = F.softmax(router_logits, dim=-1)  # (T, E)
            expert_outputs = []
            for expert in self.experts:
                expert_outputs.append(expert(x_flat))  # (T, D)
            stacked = torch.stack(expert_outputs, dim=1)  # (T, E, D)
            out_flat = torch.sum(stacked * gates.unsqueeze(-1), dim=1)  # (T, D)
        else:
            # Sparse top-1 routing
            if self.top_k != 1:
                raise NotImplementedError("top_k > 1 の sparse routing は未実装です。")

            indices = router_logits.argmax(dim=-1)  # (T,)
            out_flat = torch.zeros_like(x_flat)

            for e, expert in enumerate(self.experts):
                mask = (indices == e)
                if mask.any():
                    sub_x = x_flat[mask]          # (T_e, D)
                    sub_y = expert(sub_x)         # (T_e, D)
                    out_flat[mask] = sub_y

        return out_flat.view(B, N, D)


# ---------------------------------------------
# (✘1 + ✘2 + ✘3) MoE-ResNet-BK Layer
# ---------------------------------------------
class MoEResNetBKLayer(nn.Module):
    def __init__(self, d_model, n_seq, num_experts=4, top_k=1, dropout_p=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_seq = n_seq

        self.moe_ffn = SparseMoELayer(d_model, num_experts, top_k, dropout_p)
        self.v_proj = nn.Linear(d_model, 1)

        # BK-Core 出力 (real, imag) -> d_model
        self.output_proj = nn.Linear(2, d_model)

        # 学習可能スケール（BKブランチの寄与を自動調整）
        self.bk_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        # H0 (離散ラプラシアン相当) を buffer として保持
        self.register_buffer("h0_diag_base", torch.full((1, n_seq), -2.0, dtype=torch.float32))
        self.register_buffer("h0_sub_base",  torch.full((1, n_seq - 1), 1.0, dtype=torch.float32))
        self.register_buffer("h0_super_base",torch.full((1, n_seq - 1), 1.0, dtype=torch.float32))

        # スペクトルシフト z も buffer にしておく
        self.register_buffer("z", torch.tensor(1.0j, dtype=torch.complex64))

        self.bk_core = BKCoreFunction.apply

        # --- 数値安定化パラメータ ---
        self.v_max = 3.0          # ポテンシャル v_i のクリップ範囲
        self.feature_clamp = 10.0 # BK 特徴量 (ReG, ImG) のクリップ範囲

    def forward(self, x):
        """
        x: (B, N, D)
        """
        B, N, D = x.shape
        assert N == self.n_seq, f"Sequence length mismatch: expected {self.n_seq}, got {N}"

        # (✘3) MoE-FFN
        ffn_out = self.moe_ffn(x)               # (B, N, D)

        # ポテンシャル v_i (B, N)
        v = self.v_proj(ffn_out).squeeze(-1)    # (B, N)
        # 数値安定化: ポテンシャルをクリップ
        v = torch.clamp(v, -self.v_max, self.v_max)

        # H0 をバッチ分だけ拡張
        h0_diag  = self.h0_diag_base.expand(B, -1)   # (B, N)
        h0_sub   = self.h0_sub_base.expand(B, -1)    # (B, N-1)
        h0_super = self.h0_super_base.expand(B, -1)  # (B, N-1)

        he_diag = h0_diag + v                       # (B, N)

        # (✘1 + ✘2) BK-Core + ハイブリッド解析勾配
        features = self.bk_core(he_diag, h0_super, h0_sub, self.z)  # (B, N, 2)

        # BK特徴量もクリップ（MoE + 残差での爆発を防ぐ）
        if self.feature_clamp is not None:
            features = torch.clamp(features, -self.feature_clamp, self.feature_clamp)

        spec_out = self.output_proj(features)       # (B, N, D)

        # BK ブランチを学習可能スケールで混ぜる
        return ffn_out + self.bk_scale * spec_out


# ---------------------------------------------
# ResNet-BK Block (LayerNorm + 残差)
# ---------------------------------------------
class ResNetBKBlock(nn.Module):
    def __init__(self, d_model, n_seq, num_experts=4, top_k=1, dropout_p=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.bk_layer = MoEResNetBKLayer(d_model, n_seq, num_experts, top_k, dropout_p)

    def forward(self, x):
        # Pre-Norm 構造
        return x + self.bk_layer(self.layer_norm(x))


# ---------------------------------------------
# Nano-GPT 風 Language Model 本体
# ---------------------------------------------
class LanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=64,
        n_layers=4,
        n_seq=128,
        num_experts=4,
        top_k=1,
        dropout_p=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_seq = n_seq

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(n_seq, d_model)

        self.blocks = nn.ModuleList([
            ResNetBKBlock(
                d_model=d_model,
                n_seq=n_seq,
                num_experts=num_experts,
                top_k=top_k,
                dropout_p=dropout_p,
            )
            for _ in range(n_layers)
        ])

        self.layer_norm_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        x: (batch_size, n_seq)
        """
        batch_size, n_seq = x.shape
        assert n_seq == self.n_seq, f"n_seq mismatch: expected {self.n_seq}, got {n_seq}"

        tok_emb = self.token_embedding(x)  # (B, N, D)

        pos = torch.arange(0, n_seq, dtype=torch.long, device=x.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos)  # (1, N, D)

        h = tok_emb + pos_emb

        for block in self.blocks:
            h = block(h)

        h = self.layer_norm_final(h)
        logits = self.lm_head(h)           # (B, N, vocab_size)
        return logits


# ---------------------------------------------
# WikiText2 DataLoader (datasets版)
# ---------------------------------------------
def get_data_loader(batch_size, n_seq):
    """
    HuggingFace datasets(wikitext-2-raw-v1) を使用した簡易 LM データローダ
    戻り値:
      train_data: (seq_len_total, batch_size) LongTensor
      vocab:      dict (stoi, itos, vocab_size)
      get_batch:  関数 (source, i) -> (data, target)
    """
    try:
        # ★ trust_remote_code は削除 ★
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        train_texts = dataset["train"]["text"]  # list[str]
    except Exception as e:
        print(f"WikiText2 データセットのロードに失敗しました: {e}")
        print("ネットワーク接続や環境を確認してください。")
        return None, None, None

    # --- 単純単語分割 & vocab 構築 ---
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

    # 語彙を頻度順に並べて上限をかける
    VOCAB_LIMIT = 30000
    for tok, freq in counter.most_common(VOCAB_LIMIT - len(special_tokens)):
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

    train_ids = encode_texts(train_texts)

    # トークン数を上限でカット（Colab 無料枠対策）
    MAX_TOKENS = 500_000
    if train_ids.numel() > MAX_TOKENS:
        train_ids = train_ids[:MAX_TOKENS]

    def batchify(data, bsz):
        seq_len = data.size(0) // bsz
        data = data.narrow(0, 0, seq_len * bsz)
        data = data.view(bsz, seq_len).t().contiguous()  # (seq_len, batch_size)
        return data

    train_data = batchify(train_ids, batch_size)

    def get_batch(source, i):
        # target のために -1 が必要
        seq_len = min(n_seq, len(source) - 1 - i)
        data = source[i:i+seq_len]                   # (seq_len, batch_size)
        target = source[i+1:i+1+seq_len].reshape(-1) # (seq_len * batch_size,)
        return data, target

    vocab = {
        "stoi": stoi,
        "itos": itos,
        "vocab_size": vocab_size,
    }

    return train_data, vocab, get_batch


# ---------------------------------------------
# メイン実行 (Language Model "Intelligence Test")
# ---------------------------------------------
def run_language_model_test():
    # ハイパーパラメータ
    D_MODEL     = 64
    N_SEQ       = 128
    BATCH_SIZE  = 20
    N_LAYERS    = 4
    NUM_EXPERTS = 4
    TOP_K       = 1         # MoE を本物の sparse top-1 に
    DROPOUT_P   = 0.1
    EPOCHS      = 3         # 安定化したので 4〜5 に上げてもOK
    LR          = 1e-3
    WEIGHT_DECAY = 0.01

    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device.type.upper()}")

    # 1. データ準備
    train_data, vocab, get_batch = get_data_loader(BATCH_SIZE, N_SEQ)
    if train_data is None:
        return

    VOCAB_SIZE = vocab["vocab_size"]
    print(f"Vocabulary Size: {VOCAB_SIZE}")
    print(f"Train tokens: {train_data.numel()} (after batchify)")

    # 2. モデル構築
    model = LanguageModel(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_seq=N_SEQ,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        dropout_p=DROPOUT_P,
    ).to(device)

    # （オプション）torch.compile
    USE_TORCH_COMPILE = False
    if USE_TORCH_COMPILE:
        try:
            model = torch.compile(model)
            print("Using torch.compile()")
        except Exception as e:
            print(f"torch.compile に失敗したため通常モードで実行します: {e}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()

    # 学習率スケジューラ (コサインアニーリング)
    num_total_steps = (train_data.size(0) // N_SEQ) * EPOCHS
    scheduler = CosineAnnealingLR(optimizer, T_max=num_total_steps, eta_min=LR / 10)

    print("--- ResNet-BK Ultra v2: O(N) + Hybrid Analytic Grad + Sparse MoE ---")
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
    print(f"Total Steps (approx): {num_total_steps}")
    print(f"BKCore GRAD_BLEND = {BKCoreFunction.GRAD_BLEND}")

    model.train()
    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()

        for i in range(0, train_data.size(0) - 1, N_SEQ):
            x_batch, y_batch = get_batch(train_data, i)  # x: (seq_len, batch)
            x_batch = x_batch.t().contiguous()          # (batch, seq_len)

            if x_batch.size(1) != N_SEQ:
                # 不完全バッチはシンプルにスキップ
                continue

            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)  # (B, N, V)

            loss = criterion(
                logits.view(-1, logits.size(-1)),  # (B*N, V)
                y_batch                             # (B*N,)
            )

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Epoch {epoch}, Batch {i}: Loss is NaN/Inf. Skipping update.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()
            global_step += 1

            total_loss += loss.item()
            num_batches += 1

            if global_step % 50 == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"  [Step {global_step}] Epoch {epoch} | "
                      f"Loss: {loss.item():.4f} | LR: {lr:.6f}")

        end_time = time.time()
        avg_loss = total_loss / max(1, num_batches)
        perplexity = math.exp(avg_loss)
        print("=" * 60)
        print(f"Epoch {epoch}/{EPOCHS} | Time: {end_time - start_time:.2f}s | "
              f"Avg Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")
        print("=" * 60)


if __name__ == "__main__":
    run_language_model_test()
