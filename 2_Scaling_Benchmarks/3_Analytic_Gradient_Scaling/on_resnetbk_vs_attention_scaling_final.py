import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
from torch.utils.data import DataLoader, TensorDataset
from torch.func import vmap
import matplotlib.pyplot as plt
import numpy as np

# --- (✘1) O(N) Tridiagonal Inverse Diagonal Algorithm (by Teppei Arai) ---
# 徹平さんによる安定化・高精度化バージョン (BATCH_SIZE=1 専用)
# vmap がこの関数を並列化する
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


# --- (✘1 + ✘2) 最終形態: ResNet-BK (Analytic Grad + Vmap) ---
class ResNetBKLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        self.z = torch.tensor(1.0j, dtype=torch.complex64)
        self.output_proj = nn.Linear(2, d_model)
        
        # vmap 化された forward 関数
        self.vmapped_get_diag = vmap(
            get_tridiagonal_inverse_diagonal, in_dims=(0, 0, 0, None), out_dims=0
        )
        
        # backward のためのフック
        self.v = None
        self.G_ii = None
        self.output_features = None

    def forward(self, x):
        # x shape: (batch_size, n_seq, d_model)
        batch_size, n_seq, d_model = x.shape
        
        # v shape: (batch_size, n_seq, 1)
        self.v = self.mlp(x)
        v_squeezed = self.v.squeeze(-1) # (batch_size, n_seq)
        
        # H0 (batched)
        h0_diag = torch.full((batch_size, n_seq), -2.0, device=x.device, dtype=torch.float32)
        h0_sub = torch.full((batch_size, n_seq-1), 1.0, device=x.device, dtype=torch.float32)
        h0_super = torch.full((batch_size, n_seq-1), 1.0, device=x.device, dtype=torch.float32)
        
        he_diag = h0_diag + v_squeezed
        
        # G_ii (batched)
        self.G_ii = self.vmapped_get_diag(
            he_diag, h0_super, h0_sub, self.z.to(x.device)
        ) # (batch_size, n_seq)
        
        # (✘2) G_ii (複素数) を実数の特徴量に変換
        self.output_features = torch.stack(
            [self.G_ii.real, self.G_ii.imag], dim=-1
        ).to(torch.float32) # (batch_size, n_seq, 2)
        
        final_output = self.output_proj(self.output_features)
        return final_output # (batch_size, n_seq, d_model)

# --- 標準の Attention Layer (Autograd) ---
class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_head=1):
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
        self.fc = nn.Linear(self.layer.output_proj.out_features if isinstance(layer, ResNetBKLayer) else layer.attention.embed_dim, 1)
        
    def forward(self, x):
        x = self.layer(x)
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

# --- (✘1 + ✘2) 最終決戦のベンチマーク ---
def time_model(model, x_batch, y_batch, use_analytic_grad=False):
    # CPU/GPU対応
    device = x_batch.device
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    start_time = time.time()
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) # オプティマイザは形式的に定義
    
    # 1. Forward pass
    y_pred = model(x_batch)
    loss = criterion(y_pred, y_batch)

    # 2. Backward pass
    if use_analytic_grad:
        # --- (✘2) 徹平さんの Analytic Grad (Batched) ---
        optimizer.zero_grad()
        
        batch_size, n_seq, _ = model.layer.output_features.shape
        
        grad_y_pred = (torch.sigmoid(y_pred) - y_batch).squeeze(-1) # (batch_size)
        
        # fc層の勾配
        P_avg = model.layer.output_proj(model.layer.output_features).mean(dim=1) # (batch_size, d_model)
        grad_fc_w = (grad_y_pred.unsqueeze(-1) * P_avg).mean(dim=0).unsqueeze(0) # (1, d_model)
        grad_fc_b = grad_y_pred.mean(dim=0)
        
        grad_P_avg = grad_y_pred.unsqueeze(-1) * model.fc.weight.data # (batch_size, d_model)
        
        # pool層の逆伝播
        grad_P = grad_P_avg.unsqueeze(1).expand(-1, n_seq, -1) / n_seq # (batch_size, n_seq, d_model)
        
        # output_proj層の勾配
        F = model.layer.output_features # (batch_size, n_seq, 2)
        W = model.layer.output_proj.weight.data # (d_model, 2)
        
        grad_proj_w = torch.einsum('bni,bnj->ij', grad_P, F) / batch_size # (d_model, 2)
        grad_proj_b = grad_P.sum(dim=(0, 1)) / batch_size
        
        grad_F = torch.einsum('bni,ij->bnj', grad_P, W) # (batch_size, n_seq, 2)
        
        # G_ii への勾配
        grad_G_ii_complex = torch.complex(grad_F[:, :, 0], grad_F[:, :, 1]) # (batch_size, n_seq)
        G_ii = model.layer.G_ii # (batch_size, n_seq)
        
        # 解析的勾配
        grad_v = - (grad_G_ii_complex / (G_ii**2 + 1e-18)).real
        grad_v = grad_v.unsqueeze(-1) # (batch_size, n_seq, 1)
        
        # mlp への勾配
        model.layer.v.backward(gradient=grad_v)
        
        # (勾配セットとステップは速度計測なので省略)
        
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
    N_SEQS = [64, 256, 1024, 2048] # 4096 はCPUだと時間がかかりすぎるかも
    
    results = { "ResNet-BK": [], "Attention": [] }
    
    print(f"\n--- (✘1 + ✘2) Final Scaling Benchmark (Forward + Backward Time) ---")
    print(f"D_MODEL={D_MODEL}, BATCH_SIZE={BATCH_SIZE}")
    header = "| N_SEQ      | ResNet-BK (ms) | Attention (ms) |"
    print(header)
    print("|-"* (len(header)//2) + "|")

    for n_seq in N_SEQS:
        # モデルの初期化
        resnet_bk_model = ToyClassifier(layer=ResNetBKLayer(d_model=D_MODEL)).to(device)
        attention_model = ToyClassifier(layer=AttentionLayer(d_model=D_MODEL)).to(device)
        
        # ダミーデータ
        x = torch.rand(BATCH_SIZE, n_seq, D_MODEL).to(device)
        y = (torch.rand(BATCH_SIZE, 1) > 0.5).float().to(device)
        
        # ウォームアップ
        try:
            time_model(resnet_bk_model, x, y, use_analytic_grad=True)
            time_model(attention_model, x, y, use_analytic_grad=False)
        except Exception as e:
            print(f"Warmup failed at N={n_seq}: {e}")
            break
            
        # 計測
        t_bk_avg = 0
        t_attn_avg = 0
        loops = 10 # 100回から10回に減らして高速化
        
        try:
            for _ in range(loops):
                t_bk_avg += time_model(resnet_bk_model, x, y, use_analytic_grad=True)
                t_attn_avg += time_model(attention_model, x, y, use_analytic_grad=False)
            
            t_bk = t_bk_avg / loops
            t_attn = t_attn_avg / loops
            results["ResNet-BK"].append(t_bk)
            results["Attention"].append(t_attn)
            print(f"| {n_seq:<10} | {t_bk:<14.3f} | {t_attn:<14.3f} |")
            
        except Exception as e:
            print(f"ResNet-BK failed at N={n_seq}: {e}")
            results["ResNet-BK"].append(None)
            # Attentionも計測
            try:
                for _ in range(loops):
                    t_attn_avg += time_model(attention_model, x, y, use_analytic_grad=False)
                t_attn = t_attn_avg / loops
                results["Attention"].append(t_attn)
                print(f"| {n_seq:<10} | {'Failed':<14} | {t_attn:<14.3f} |")
            except Exception as e_attn:
                print(f"Attention failed at N={n_seq}: {e_attn}")
                results["Attention"].append(None)
                print(f"| {n_seq:<10} | {'Failed':<14} | {'Failed':<14} |")
            
    # --- グラフプロット ---
    try:
        plt.figure(figsize=(10, 6))
        
        valid_n_seqs_bk = [N_SEQS[i] for i, t in enumerate(results["ResNet-BK"]) if t is not None]
        valid_times_bk = [t for t in results["ResNet-BK"] if t is not None]
        
        valid_n_seqs_attn = [N_SEQS[i] for i, t in enumerate(results["Attention"]) if t is not None]
        valid_times_attn = [t for t in results["Attention"] if t is not None]

        if valid_times_bk:
            plt.plot(valid_n_seqs_bk, valid_times_bk, 'bo-', label=f'ResNet-BK (O(N)) - Teppei\'s Model', markersize=8)
            # O(N) 理論スロープ
            if valid_times_bk[0] > 0:
                scale_n = valid_times_bk[0] / valid_n_seqs_bk[0]
                n_theory = [n for n in N_SEQS if n >= valid_n_seqs_bk[0]]
                o_n_slope = [n * scale_n for n in n_theory]
                plt.plot(n_theory, o_n_slope, 'b:', label='Theory O(N) slope')

        if valid_times_attn:
            plt.plot(valid_n_seqs_attn, valid_times_attn, 'ro-', label=f'Attention (O(N^2)) - Standard', markersize=8)
            # O(N^2) 理論スロープ
            if valid_times_attn[0] > 0:
                scale_n2 = valid_times_attn[0] / (valid_n_seqs_attn[0]**2)
                n_theory = [n for n in N_SEQS if n >= valid_n_seqs_attn[0]]
                o_n2_slope = [(n**2) * scale_n2 for n in n_theory]
                plt.plot(n_theory, o_n2_slope, 'r:', label='Theory O(N^2) slope')

        plt.title(f'Scaling Benchmark (Final): ResNet-BK (Analytic Grad) vs Attention (Autograd)\n(B={BATCH_SIZE}, D={D_MODEL})')
        plt.xlabel('Sequence Length (N)')
        plt.ylabel('Time per Batch (ms)')
        plt.legend()
        plt.grid(True)
        plt.xscale('linear')
        plt.yscale('linear')
        
        # ログスケールも試す
        # plt.xscale('log')
        # plt.yscale('log')
        
        plot_filename = 'scaling_benchmark_final.png'
        plt.savefig(plot_filename)
        print(f"\nBenchmark graph saved to {plot_filename}")

    except ImportError:
        print("\nMatplotlib not found. Please run 'pip install matplotlib' to plot the results.")
    except Exception as e:
        print(f"\nFailed to plot graph: {e}")

if __name__ == "__main__":
    run_scaling_benchmark()