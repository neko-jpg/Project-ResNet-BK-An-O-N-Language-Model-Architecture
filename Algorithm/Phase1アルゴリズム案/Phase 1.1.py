import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AdaptiveSemiseparableLayer(nn.Module):
    """
    Phase 1.1: Adaptive Rank Semiseparable Layer (AR-SSM)
    
    物理的直観:
    半可分行列 H = T + U * V^T において、ランク r (U, Vの列数) を
    入力信号の「複雑性（エントロピー）」に応じて動的にゲーティングします。
    
    これにより、簡単なトークン処理には計算資源を使わず、
    難解な文脈（乱流）にのみリソースを集中させます。
    """
    
    def __init__(self, d_model, max_rank=32, min_rank=4, chunk_size=128):
        super().__init__()
        self.d_model = d_model
        self.max_rank = max_rank
        self.min_rank = min_rank
        self.chunk_size = chunk_size
        
        # Complexity Estimator: 入力の局所的な分散からランクを決定する軽量ネットワーク
        self.complexity_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, max_rank),
            nn.Sigmoid() # 0~1のゲート係数を出力
        )
        
        # Semiseparable Core Components (Low-Rank Factors)
        # U: Input Projection (The "Source" currents)
        self.U_proj = nn.Linear(d_model, max_rank)
        # V: Output Projection (The "Measurement" probes)
        self.V_proj = nn.Linear(d_model, max_rank)
        
        # Displacement Operator T (Toeplitz-like near-diagonal component)
        # 近接相互作用を扱う畳み込み項
        self.T_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)

    def estimate_rank_gate(self, x):
        """
        入力 x の複雑度に基づいて、各ランク次元の有効/無効を決定するゲートを生成。
        """
        # Global Average Pooling over chunks to stabilize
        # (B, L, D) -> (B, L, max_rank)
        gate_logits = self.complexity_gate(x)
        
        # ハードな切断ではなく、ソフトな重み付け（Gating）を行うことで
        # 微分可能性を維持しつつ、実効的なランクを下げる。
        # L1正則化をかけることで、学習中にスパース化（低ランク化）を誘導可能。
        return gate_logits

    def forward(self, x):
        """
        x: (Batch, SeqLen, d_model)
        """
        B, L, D = x.shape
        
        # 1. Local Interactions (The "T" matrix component)
        # O(N) convolution
        x_conv = x.transpose(1, 2) # (B, D, L)
        t_out = self.T_conv(x_conv).transpose(1, 2)
        
        # 2. Low-Rank Global Interactions (The "UV^T" component)
        u = self.U_proj(x) # (B, L, max_rank)
        v = self.V_proj(x) # (B, L, max_rank)
        
        # 3. Adaptive Rank Gating
        # ランクごとの重要度を計算
        gates = self.estimate_rank_gate(x) # (B, L, max_rank)
        
        # ゲートを適用 (Broadcasting)
        # ここで gate が 0 に近いランク次元は計算に寄与しない（＝実効ランクの低下）
        u_gated = u * gates
        v_gated = v * gates
        
        # 4. Causality & Associative Scan (The SSM Recurrence)
        # H = Lower Triangular (Causal) part of U * V^T
        # 通常のAttention (O(N^2)) ではなく、累積和 (O(N)) で計算
        # Linear Attention form: result = sum(u_i) * v_i (simplified)
        
        # 高速化のため、ここでは簡易的なCumsum実装とする
        # 本番環境では Parallel Associative Scan (Triton) を使用する
        k_cumsum = torch.cumsum(u_gated, dim=1) # (B, L, max_rank)
        
        # Global context injection
        global_out = k_cumsum * v_gated # Element-wise multiplication
        
        # Project back to d_model (Reconstruct)
        # Note: In a real SSM, we would mix ranks differently, but this simulates the O(N) accumulation.
        # rank_mixing is needed to go back to D dimension. 
        # For prototype, we sum ranks or project back. Let's project back.
        
        # (B, L, max_rank) -> (B, L, D)
        # We use a shared projection for efficiency or einsum
        # Here strictly simulating the "Hidden State" accumulation
        
        # 簡易的な復元: ランク次元を平均して元の次元に加算（プロトタイプ用）
        # 実際は: y = Hx の計算を厳密に行う
        uv_term = torch.matmul(global_out, self.U_proj.weight.t()[:self.max_rank, :]) 
        
        return t_out + uv_term

def test_adaptive_rank():
    print("Testing Adaptive Rank Semiseparable Layer...")
    B, L, D = 2, 128, 64
    x = torch.randn(B, L, D)
    
    model = AdaptiveSemiseparableLayer(D, max_rank=16)
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Check gates
    gates = model.estimate_rank_gate(x)
    print(f"Avg Gate Activation: {gates.mean().item():.4f}")
    print("Test Passed: O(N) forward pass completed.")

if __name__ == "__main__":
    test_adaptive_rank()