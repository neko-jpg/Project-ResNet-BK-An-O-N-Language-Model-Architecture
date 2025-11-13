import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
from torch.utils.data import DataLoader, TensorDataset
from torch.func import vmap
import torch.nn.functional as F
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
# self.mlp の代わりとなる、スパースなMLP
class SparseMoELayer(nn.Module):
    def __init__(self, d_model, num_experts=4, top_k=1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 専門家 (Expert) のリスト
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 1) # v_i を出力
            ) for _ in range(num_experts)
        ])
        
        # 振り分け役 (Gating Network / Router)
        self.gating_network = nn.Linear(d_model, num_experts)

    def forward(self, x):
        # x shape: (batch_size, n_seq, d_model)
        batch_size, n_seq, d_model = x.shape
        
        # (✘3) 全てのトークンをフラットにする
        x_flat = x.reshape(-1, d_model) # (B*N, D)
        
        # (✘3) ルーターが、各トークンがどの専門家に行くべきかのスコアを計算
        router_logits = self.gating_network(x_flat) # (B*N, num_experts)
        
        # (✘3) Top-k (k=1) の専門家を選ぶ
        # routing_weights: 選ばれた専門家のスコア (Softmax後)
        # selected_experts: 選ばれた専門家のインデックス
        routing_weights = F.softmax(router_logits, dim=-1)
        top_k_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # (✘3) 非常にスパースな「振り分け行列」を作成
        # B*N 個のトークン x num_experts 個の専門家 の行列
        # (B*N, num_experts)
        gates = F.gumbel_softmax(router_logits, hard=True, tau=1.0) # Gumbel-Softmaxで微分可能なone-hotを生成

        # (✘3) 全専門家の計算をバッチで行い、後でマスクする（PoCのための簡単な実装）
        # 本来のMoEはもっと効率的な実装を行うが、まずは概念実証
        
        final_output = torch.zeros_like(x_flat[..., 0:1]) # (B*N, 1)
        
        for i in range(self.num_experts):
            expert = self.experts[i]
            gate_for_this_expert = gates[:, i].unsqueeze(-1) # (B*N, 1)
            
            # (✘3) この専門家が担当するトークンだけ計算（風）
            # gate_for_this_expert が 0 のトークンは、結果が0になる
            expert_output = expert(x_flat) # (B*N, 1)
            final_output += expert_output * gate_for_this_expert
            
        # (✘3) スパース化による「負荷」の損失項 (Load Balancing Loss)
        # 全トークンが1人の専門家に集中しないようにするペナルティ
        # (これはPoCなので、今回は省略)

        # (B*N, 1) -> (B, N, 1)
        return final_output.reshape(batch_size, n_seq, 1)


# --- (✘1 + ✘2 + ✘3) MoE-ResNet-BK Layer ---
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
        
        # backward のためのフック (Autograd用)
        self.v = None

    def forward(self, x):
        batch_size, n_seq, d_model = x.shape
        
        # (✘3) v をMoEレイヤーで計算
        self.v = self.moe_mlp(x) # (batch_size, n_seq, 1)
        v_squeezed = self.v.squeeze(-1) # (batch_size, n_seq)
        
        h0_diag = torch.full((batch_size, n_seq), -2.0, device=x.device, dtype=torch.float32)
        h0_sub = torch.full((batch_size, n_seq-1), 1.0, device=x.device, dtype=torch.float32)
        h0_super = torch.full((batch_size, n_seq-1), 1.0, device=x.device, dtype=torch.float32)
        
        he_diag = h0_diag + v_squeezed
        
        G_ii = self.vmapped_get_diag(
            he_diag, h0_super, h0_sub, self.z.to(x.device)
        )
        
        output_features = torch.stack(
            [G_ii.real, G_ii.imag], dim=-1
        ).to(torch.float32)
        
        final_output = self.output_proj(output_features)
        return final_output

# --- Toy Model for Benchmarking ---
class ToyClassifier(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.layer.output_proj.out_features, 1)
        
    def forward(self, x):
        x = self.layer(x)
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

# --- Toy Dataset ---
def create_dataset(num_samples, n_seq, d_model):
    x_data = torch.rand((num_samples, n_seq, d_model))
    y_data = (torch.mean(x_data, dim=(1,2)) > 0.5).float().unsqueeze(-1)
    return TensorDataset(x_data, y_data)

# --- (✘3) MoE Benchmark ---
def run_benchmark(n_seq, d_model, batch_size):
    print(f"--- Running Benchmark (Step 3: MoE PoC) ---")
    print(f"N_SEQ = {n_seq}, D_MODEL = {d_model}, BATCH_SIZE = {batch_size}")
    
    # (✘3) 比較対象: Step 1&2 の最終形態 (Dense) vs Step 3 (MoE)
    # (注: Analytic Gradは複雑すぎるため、一旦Autograd (BP) で比較する)
    models_to_run = {
        "ResNet-BK (Dense)": ToyClassifier(layer=ResNetBKLayer(d_model)),
        "ResNet-BK (MoE)": ToyClassifier(layer=MoEResNetBKLayer(d_model, num_experts=4, top_k=1)),
    }

    dataset = create_dataset(num_samples=500, n_seq=n_seq, d_model=d_model)
    loader = DataLoader(dataset, batch_size=batch_size)

    results = {}

    for name, model in models_to_run.items():
        print(f"\nBenchmarking: {name}")
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        
        start_time = time.time()
        num_batches = 0
        total_loss = 0
        correct_preds = 0
        total_preds = 0
        
        model.train()
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            
            if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
                results[name] = {"status": "FAILED (NaN/Inf)"}
                break
            
            loss = criterion(y_pred, y_batch)
            loss.backward()
            
            has_nan_grad = False
            for param in model.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_nan_grad = True
                        break
            if has_nan_grad:
                results[name] = {"status": "FAILED (NaN/Inf Grad)"}
                break

            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            preds = (torch.sigmoid(y_pred) > 0.5).float()
            correct_preds += (preds == y_batch).sum().item()
            total_preds += y_batch.size(0)

        if name not in results:
            end_time = time.time()
            total_time = end_time - start_time
            avg_loss = total_loss / num_batches
            accuracy = correct_preds / total_preds
            
            results[name] = {
                "status": "Success",
                "total_time_s": f"{total_time:.4f}",
                "avg_loss": f"{avg_loss:.4f}",
                "accuracy": f"{accuracy*100:.2f}%"
            }

    print("\n--- Benchmark Results ---")
    print(f"N_SEQ={n_seq}, D_MODEL={d_model}, BATCH_SIZE={batch_size}")
    header = "| Model               | Status          | Time (s) | Avg. Loss | Accuracy |"
    print(header)
    print("|-"* (len(header)//2) + "|")
    
    for name, res in results.items():
        if res['status'] == "Success":
            print(f"| {name:<19} | {res['status']:<15} | {res['total_time_s']:<8} | {res['avg_loss']:<9} | {res['accuracy']:<8} |")
        else:
            print(f"| {name:<19} | {res['status']:<15} | {'N/A':<8} | {'N/A':<9} | {'N/A':<8} |")
            
# --- Configuration ---
N_SEQ_TEST = 64
D_MODEL_TEST = 16
BATCH_SIZE_TEST = 32

if __name__ == "__main__":
    run_benchmark(N_SEQ_TEST, D_MODEL_TEST, BATCH_SIZE_TEST)