import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
from torch.utils.data import DataLoader, TensorDataset
from torch.func import vmap
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import copy

# --- (✘1) O(N) Tridiagonal Inverse Diagonal Algorithm (by Teppei Arai) ---
# (Step 1 & 2 で検証済みの、安定化された心臓部)
def get_tridiagonal_inverse_diagonal(a, b, c, z):
    n = a.shape[-1]
    device = a.device
    a_c = a.to(torch.complex128)
    b_c = b.to(torch.complex128)
    c_c = c.to(torch.complex128)
    z_c = z.to(torch.complex128)
    a_shifted = a_c - z_c

    if n == 0:
        return torch.zeros(0, dtype=torch.complex64, device=device)

    theta = []
    theta.append(torch.ones((), dtype=torch.complex128, device=device))
    theta.append(a_shifted[0])
    for i in range(1, n):
        theta.append(a_shifted[i] * theta[i] - c_c[i-1] * b_c[i-1] * theta[i-1])
    theta = torch.stack(theta)
    det_T = theta[-1]

    phi = [None] * n
    phi[-1] = torch.ones((), dtype=torch.complex128, device=device)
    if n > 1:
        phi[-2] = a_shifted[-1]
        for i in range(n - 3, -1, -1):
            phi[i] = a_shifted[i+1] * phi[i+1] - c_c[i] * b_c[i] * phi[i+2]
    phi = torch.stack(phi)

    eps = torch.tensor(1e-18, dtype=torch.complex128, device=device)
    diag_inv = theta[:-1] * phi / (det_T + eps)

    return diag_inv.to(torch.complex64)

# --- (✘3) Step 3: Sparse Mixture of Experts (MoE) Layer ---
class SparseMoELayer(nn.Module):
    def __init__(self, d_model, num_experts=4, top_k=1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 1)
            ) for _ in range(num_experts)
        ])
        self.gating_network = nn.Linear(d_model, num_experts)
        
        # (✘2) backward のためのフック
        self.x_flat = None
        self.gates = None

    def forward(self, x):
        batch_size, n_seq, d_model = x.shape
        self.x_flat = x.reshape(-1, d_model) # (B*N, D)
        
        router_logits = self.gating_network(self.x_flat) # (B*N, num_experts)
        
        # (✘2) Gumbel-Softmax (hard=True) は微分可能
        self.gates = F.gumbel_softmax(router_logits, hard=True, tau=1.0) # (B*N, num_experts)

        final_output = torch.zeros((batch_size * n_seq, 1), device=x.device) # (B*N, 1)
        
        for i in range(self.num_experts):
            expert = self.experts[i]
            gate_for_this_expert = self.gates[:, i].unsqueeze(-1) # (B*N, 1)
            
            # (✘3) スパースな計算 (担当するトークンのみ)
            # gate_for_this_expert が 0 のトークンは、結果が0になる
            expert_output = expert(self.x_flat) # (B*N, 1)
            final_output += expert_output * gate_for_this_expert
            
        return final_output.reshape(batch_size, n_seq, 1) # (B, N, 1)


# --- (✘1 + ✘2 + ✘3) 最終形態: MoE-ResNet-BK Layer ---
class MoEResNetBKLayer(nn.Module):
    def __init__(self, d_model, num_experts=4, top_k=1):
        super().__init__()
        # (✘3) MLPをMoEレイヤーに置き換える
        self.moe_mlp = SparseMoELayer(d_model, num_experts, top_k)
        
        self.z = torch.tensor(1.0j, dtype=torch.complex64)
        self.output_proj = nn.Linear(2, d_model)
        
        self.vmapped_get_diag = vmap(
            get_tridiagonal_inverse_diagonal, in_dims=(0, 0, 0, None), out_dims=0
        )
        
        # (✘2) backward のためのフック
        self.v = None
        self.G_ii = None
        self.output_features = None

    def forward(self, x):
        batch_size, n_seq, d_model = x.shape
        
        # (✘3) v をMoEレイヤーで計算
        self.v = self.moe_mlp(x) # (batch_size, n_seq, 1)
        v_squeezed = self.v.squeeze(-1) # (batch_size, n_seq)
        
        h0_diag = torch.full((batch_size, n_seq), -2.0, device=x.device, dtype=torch.float32)
        h0_sub = torch.full((batch_size, n_seq-1), 1.0, device=x.device, dtype=torch.float32)
        h0_super = torch.full((batch_size, n_seq-1), 1.0, device=x.device, dtype=torch.float32)
        
        he_diag = h0_diag + v_squeezed
        
        self.G_ii = self.vmapped_get_diag(
            he_diag, h0_super, h0_sub, self.z.to(x.device)
        )
        
        self.output_features = torch.stack(
            [self.G_ii.real, self.G_ii.imag], dim=-1
        ).to(torch.float32)
        
        final_output = self.output_proj(self.output_features)
        return final_output

# --- 標準の Attention Layer (Autograd) ---
class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_head=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_head, batch_first=True)
    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return attn_output

# --- Toy Model for Benchmarking ---
class ToyClassifier(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.layer.output_proj.out_features if isinstance(layer, MoEResNetBKLayer) else layer.attention.embed_dim, 1)
        
    def forward(self, x):
        x = self.layer(x)
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

# --- (✘1 + ✘2 + ✘3) 最終決戦のベンチマーク ---
def time_model(model, x_batch, y_batch, use_analytic_grad=False):
    device = x_batch.device
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    start_time = time.time()
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    
    y_pred = model(x_batch)
    loss = criterion(y_pred, y_batch)

    if use_analytic_grad:
        # --- (✘2 + ✘3) 最終形態: Analytic Grad (MoE対応) ---
        optimizer.zero_grad()
        
        batch_size, n_seq, _ = model.layer.output_features.shape
        
        # (BUG FIX) .squeeze(-1) を削除。 grad_y_pred の形状は (B, 1) になる
        grad_y_pred = (torch.sigmoid(y_pred) - y_batch) # (B, 1)
        
        # fc層
        P_avg = model.layer.output_proj(model.layer.output_features).mean(dim=1) # (B, D_model)
        
        # (BUG FIX) grad_y_pred の形状が (B, 1) になったため、計算を修正
        grad_fc_w = (grad_y_pred * P_avg).mean(dim=0).unsqueeze(0) # (1, D_model)
        
        # (BUG FIX) grad_fc_b の形状を (1,) にする
        grad_fc_b = grad_y_pred.mean(dim=0) # (1,)
        
        grad_P_avg = grad_y_pred * model.fc.weight.data # (B, D_model)
        
        # pool層
        grad_P = grad_P_avg.unsqueeze(1).expand(-1, n_seq, -1) / n_seq # (B, N, D_model)
        
        # output_proj層
        F = model.layer.output_features # (B, N, 2)
        W_proj = model.layer.output_proj.weight.data # (D_model, 2)
        grad_proj_w = torch.einsum('bni,bnj->ij', grad_P, F) / batch_size # (D_model, 2)
        grad_proj_b = grad_P.sum(dim=(0, 1)) / batch_size
        grad_F = torch.einsum('bni,ij->bnj', grad_P, W_proj) # (B, N, 2)
        
        # G_ii 
        grad_G_ii_complex = torch.complex(grad_F[:, :, 0], grad_F[:, :, 1]) # (B, N)
        G_ii = model.layer.G_ii # (B, N)
        
        # (✘2) 解析的勾配 (G_ii -> v)
        grad_v = - (grad_G_ii_complex / (G_ii**2 + 1e-18)).real
        grad_v = grad_v.unsqueeze(-1) # (B, N, 1)
        
        # (✘3) MoE層の逆伝播
        # v (MoEの出力) に対して、手計算した勾配 'grad_v' を使って逆伝播
        model.layer.v.backward(gradient=grad_v)

        # 手動で計算した勾配をセット
        model.fc.weight.grad = grad_fc_w
        model.fc.bias.grad = grad_fc_b
        model.layer.output_proj.weight.grad = grad_proj_w
        model.layer.output_proj.bias.grad = grad_proj_b
        
    else:
        # --- 標準の Autograd (BP) ---
        loss.backward()

    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    end_time = time.time()
    return (end_time - start_time) * 1000 # ミリ秒で返す

# --- Main Scaling Benchmark ---
def run_scaling_benchmark():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device.type.upper()}")

    D_MODEL = 64
    BATCH_SIZE = 8
    N_SEQS = [64, 256, 1024, 2048]
    
    results = { "ResNet-BK (MoE)": [], "Attention (Autograd)": [] }
    
    print(f"\n--- (✘1 + ✘2 + ✘3) Final Form Benchmark (Forward + Backward Time) ---")
    print(f"D_MODEL={D_MODEL}, BATCH_SIZE={BATCH_SIZE}")
    header = "| N_SEQ      | ResNet-BK (MoE) (ms) | Attention (Autograd) (ms) |"
    print(header)
    print("|-"* (len(header)//2) + "|")

    for n_seq in N_SEQS:
        # モデルの初期化
        resnet_bk_model = ToyClassifier(layer=MoEResNetBKLayer(d_model=D_MODEL, num_experts=4, top_k=1)).to(device)
        attention_model = ToyClassifier(layer=AttentionLayer(d_model=D_MODEL, n_head=4)).to(device)
        
        x = torch.rand(BATCH_SIZE, n_seq, D_MODEL).to(device)
        y = (torch.rand(BATCH_SIZE, 1) > 0.5).float().to(device)
        
        t_bk_avg = 0
        t_attn_avg = 0
        loops = 10 
        
        # (FIX) エラーハンドリングをモデルごとに分離
        t_bk = None
        try:
            # ウォームアップ
            time_model(resnet_bk_model, x, y, use_analytic_grad=True) 
            # (✘1+✘2+✘3) 最終形態
            for _ in range(loops):
                t_bk_avg += time_model(resnet_bk_model, x, y, use_analytic_grad=True)
            t_bk = t_bk_avg / loops
            results["ResNet-BK (MoE)"].append(t_bk)
            
        except Exception as e:
            print(f"ResNet-BK failed at N={n_seq}: {e}")
            results["ResNet-BK (MoE)"].append(None)

        t_attn = None
        try:
            # ウォームアップ
            time_model(attention_model, x, y, use_analytic_grad=False)
            # 比較対象
            for _ in range(loops):
                t_attn_avg += time_model(attention_model, x, y, use_analytic_grad=False)
            t_attn = t_attn_avg / loops
            results["Attention (Autograd)"].append(t_attn)
            
        except Exception as e:
            print(f"Attention failed at N={n_seq}: {e}")
            results["Attention (Autograd)"].append(None)

        # プリント処理
        t_bk_str = f"{t_bk:<20.3f}" if t_bk is not None else f"{'Failed':<20}"
        t_attn_str = f"{t_attn:<25.3f}" if t_attn is not None else f"{'Failed':<25}"
        print(f"| {n_seq:<10} | {t_bk_str} | {t_attn_str} |")

            
    # --- グラフプロット ---
    try:
        plt.figure(figsize=(12, 7))
        
        valid_n_seqs_bk = [N_SEQS[i] for i, t in enumerate(results["ResNet-BK (MoE)"]) if t is not None]
        valid_times_bk = [t for t in results["ResNet-BK (MoE)"] if t is not None]
        
        valid_n_seqs_attn = [N_SEQS[i] for i, t in enumerate(results["Attention (Autograd)"]) if t is not None]
        valid_times_attn = [t for t in results["Attention (Autograd)"] if t is not None]

        if valid_times_bk:
            plt.plot(valid_n_seqs_bk, valid_times_bk, 'bo-', label=f'ResNet-BK (O(N) + Analytic Grad + MoE) - Teppei\'s Model', markersize=8)
            if valid_times_bk[0] > 0:
                scale_n = valid_times_bk[0] / valid_n_seqs_bk[0]
                n_theory = [n for n in N_SEQS if n >= valid_n_seqs_bk[0]]
                o_n_slope = [n * scale_n for n in n_theory]
                plt.plot(n_theory, o_n_slope, 'b:', label='Theory O(N) slope')

        if valid_times_attn:
            plt.plot(valid_n_seqs_attn, valid_times_attn, 'ro-', label=f'Attention (O(N^2) + Autograd) - Standard', markersize=8)
            if valid_times_attn[0] > 0:
                scale_n2 = valid_times_attn[0] / (valid_n_seqs_attn[0]**2)
                n_theory = [n for n in N_SEQS if n >= valid_n_seqs_attn[0]]
                o_n2_slope = [(n**2) * scale_n2 for n in n_theory]
                plt.plot(n_theory, o_n2_slope, 'r:', label='Theory O(N^2) slope')

        plt.title(f'Final Form Benchmark (Step 1+2+3): Total Compute Cost\n(B={BATCH_SIZE}, D={D_MODEL})')
        plt.xlabel('Sequence Length (N)')
        plt.ylabel('Time per Batch (ms)')
        plt.legend()
        plt.grid(True)
        plt.xscale('linear')
        plt.yscale('linear')
        
        plot_filename = 'scaling_benchmark_final_moe.png'
        plt.savefig(plot_filename)
        print(f"\nBenchmark graph saved to {plot_filename}")

    except ImportError:
        print("\nMatplotlib not found. Please run 'pip install matplotlib' to plot the results.")
    except Exception as e:
        print(f"\nFailed to plot graph: {e}")

if __name__ == "__main__":
    run_scaling_benchmark()