import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
from torch.utils.data import DataLoader, TensorDataset
import copy # 2つのモデルを比較するために使用

# --- (✘1) O(N) Tridiagonal Inverse Diagonal Algorithm (by Teppei Arai) ---
# 徹平さんによる安定化・高精度化バージョン (BATCH_SIZE=1 専用)
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


# --- (✘3) ResNet-BK Layer (BATCH_SIZE=1) ---
class ResNetBKLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # (✘2) このMLPの勾配を「解析的」に計算するのが目標
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        self.z = torch.tensor(1.0j, dtype=torch.complex64)
        # (✘2) 特徴量次元を d_model に戻す
        self.output_proj = nn.Linear(2, d_model) 
        
        # (✘2) forwardパスで計算した中間値を保存するフック
        self.v = None
        self.G_ii = None
        self.output_features = None

    def forward(self, x):
        # x shape: (1, n_seq, d_model)
        batch_size, n_seq, d_model = x.shape
        if batch_size != 1:
            raise ValueError("This layer is for BATCH_SIZE=1 (PoC for Analytic Grad)")
            
        x_squeezed = x.squeeze(0) # (n_seq, d_model)
        
        # (✘2) vを計算し、backwardのために保存
        self.v = self.mlp(x_squeezed) # (n_seq, 1)
        v_squeezed = self.v.squeeze(-1) # (n_seq)
        
        # H0
        h0_diag = torch.full((n_seq,), -2.0, device=x.device, dtype=torch.float32)
        h0_sub = torch.full((n_seq-1,), 1.0, device=x.device, dtype=torch.float32)
        h0_super = torch.full((n_seq-1,), 1.0, device=x.device, dtype=torch.float32)
        
        he_diag = h0_diag + v_squeezed
        
        # G_ii (y_i) を計算し、backwardのために保存
        self.G_ii = get_tridiagonal_inverse_diagonal(
            he_diag, h0_super, h0_sub, self.z.to(x.device)
        )
        # self.G_ii shape: (n_seq)
        
        # (✘2) G_ii (複素数) を実数の特徴量に変換
        self.output_features = torch.stack(
            [self.G_ii.real, self.G_ii.imag], dim=-1
        ).to(torch.float32)
        # self.output_features shape: (n_seq, 2)
        
        # 最終出力を計算し、バッチ次元を戻す
        final_output = self.output_proj(self.output_features)
        return final_output.unsqueeze(0) # (1, n_seq, d_model)

# --- Toy Model for Benchmarking ---
class ToyClassifier(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.pool = nn.AdaptiveAvgPool1d(1)
        # (✘2) 最終的な分類器
        self.fc = nn.Linear(layer.output_proj.out_features, 1) 
        
    def forward(self, x):
        # (✘2) 徹平さんの修正に合わせて、forwardの出力を layer(x)[0] ではなく layer(x) に
        x = self.layer(x) # (1, n_seq, d_model)
        x = x.permute(0, 2, 1) # (1, d_model, n_seq)
        x = self.pool(x).squeeze(-1) # (1, d_model)
        x = self.fc(x) # (1, 1)
        return x

# --- (✘3) Toy Dataset ---
def create_dataset(num_samples, n_seq, d_model):
    x_data = torch.rand((num_samples, n_seq, d_model))
    y_data = (torch.mean(x_data, dim=(1,2)) > 0.5).float().unsqueeze(-1)
    return TensorDataset(x_data, y_data)

# --- (✘2) Step 2: 2つの異なる学習ステップ ---

# 方法1：PyTorchの標準的なAutograd
def train_step_autograd(model, optimizer, criterion, x_batch, y_batch):
    optimizer.zero_grad()
    y_pred = model(x_batch)
    
    # NaN/Inf チェック
    if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
        return None, True # FAILED
        
    loss = criterion(y_pred, y_batch)
    
    # (✘2) ここでPyTorchが G_ii や v をすべて微分する
    loss.backward()
    
    # 勾配のNaN/Inf チェック
    for param in model.parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                return None, True # FAILED
                
    optimizer.step()
    return loss.item(), False

# 
# (✘2) --- 徹平さんによる完璧な実装 ---
#
# 方法2：我々の解析的勾配 (Analytic Gradient)
def train_step_analytic(model, optimizer, criterion, x_batch, y_batch):
    optimizer.zero_grad()
    
    # 1. Forward pass (中間値 G_ii, v, output_features が保存される)
    y_pred = model(x_batch)  # Shape: (1, 1)

    if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
        return None, True  # FAILED
    
    loss = criterion(y_pred, y_batch)

    # (✘2) --- ここからが「解析的」逆伝播 ---
    # A. loss -> fc までの勾配
    # BCEWithLogitsLoss の dL/d(logit)
    grad_y_pred = (torch.sigmoid(y_pred) - y_batch).squeeze()  # scalar

    # ResNetBKLayer で保存した特徴
    F = model.layer.output_features           # (n_seq, 2)
    
    # (✘2) 徹平さんの修正： P (output_projの出力) を正しく計算
    P = model.layer.output_proj(F)            # (n_seq, d_model)
    P_avg = P.mean(dim=0)                     # (d_model,)

    # --- fc の勾配 ---
    # fc.weight: (1, d_model)
    grad_fc_w = grad_y_pred * P_avg           # (d_model,)
    grad_fc_w = grad_fc_w.unsqueeze(0)        # (1, d_model)

    # fc.bias: (1,)
    grad_fc_b = grad_y_pred.view_as(model.fc.bias)  # (1,)

    # dL/dP_avg
    grad_P_avg = grad_y_pred * model.fc.weight.data.squeeze(0)  # (d_model,)

    # --- pool (平均) の逆伝播 ---
    n_seq = F.shape[0]
    # 各 token に同じ勾配が 1/n_seq で配分される
    grad_P = grad_P_avg.unsqueeze(0).expand(n_seq, -1) / n_seq  # (n_seq, d_model)

    # --- output_proj の勾配 ---
    # output_proj.weight: (d_model, 2)
    W = model.layer.output_proj.weight.data       # (d_model, 2)

    # y = F @ W^T + b なので、
    # dL/dW = grad_P^T @ F   (形状: (d_model, 2))
    grad_proj_w = grad_P.T @ F                    # (d_model, 2)

    # バイアスは token 方向に和
    grad_proj_b = grad_P.sum(dim=0)               # (d_model,)

    # 入力 F への勾配: dL/dF = grad_P @ W
    grad_F = grad_P @ W                           # (n_seq, 2)

    # --- G_ii への勾配 ---
    grad_G_ii_features = grad_F                   # (n_seq, 2)
    grad_G_ii_complex = torch.complex(
        grad_G_ii_features[:, 0], 
        grad_G_ii_features[:, 1]
    )                                             # (n_seq,)

    G_ii = model.layer.G_ii                       # (n_seq,)

    # d(G_ii)/d(v_i) = - (G_ii)**2 を用いて、
    # dL/dv_i = - Re( dL/dG_ii / (G_ii**2) )
    grad_v = - (grad_G_ii_complex / (G_ii**2 + 1e-18)).real
    grad_v = grad_v.unsqueeze(-1)                 # (n_seq, 1)

    # --- v -> mlp の勾配は autograd に任せる ---
    model.layer.v.backward(gradient=grad_v)

    # --- 手計算した勾配を各パラメータにセット ---
    model.fc.weight.grad = grad_fc_w
    model.fc.bias.grad = grad_fc_b
    model.layer.output_proj.weight.grad = grad_proj_w
    model.layer.output_proj.bias.grad = grad_proj_b
    
    # --- パラメータ更新 ---
    optimizer.step()
    
    return loss.item(), False

# --- (✘1, ✘2, ✘3) The Benchmark ---
def run_benchmark(n_seq, d_model):
    print(f"--- Running Benchmark ---")
    print(f"N_SEQ = {n_seq}, D_MODEL = {d_model}, BATCH_SIZE = 1")
    
    # (✘2) 2つのモデルを準備。重みを共有して公平に比較
    base_model = ToyClassifier(layer=ResNetBKLayer(d_model))
    
    models_to_run = {
        "Autograd (BP)": (copy.deepcopy(base_model), train_step_autograd),
        "Analytic Grad": (copy.deepcopy(base_model), train_step_analytic),
    }

    dataset = create_dataset(num_samples=500, n_seq=n_seq, d_model=d_model)
    loader = DataLoader(dataset, batch_size=1) # BATCH_SIZE=1 で検証

    results = {}

    for name, (model, train_step_fn) in models_to_run.items():
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
            loss, failed = train_step_fn(model, optimizer, criterion, x_batch, y_batch)
            
            if failed:
                print(f"ERROR: {name} FAILED STABILITY TEST.")
                results[name] = {"status": "FAILED (NaN/Inf)"}
                break
            
            total_loss += loss
            num_batches += 1
            
            # 精度検証 (推論モードで)
            model.eval()
            with torch.no_grad():
                y_pred_eval = model(x_batch)
                preds = (torch.sigmoid(y_pred_eval) > 0.5).float()
                correct_preds += (preds == y_batch).sum().item()
                total_preds += y_batch.size(0)
            model.train()

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
    print(f"N_SEQ={n_seq}, D_MODEL={d_model}, BATCH_SIZE=1")
    header = "| Model           | Status          | Time (s) | Avg. Loss | Accuracy |"
    print(header)
    print("|-"* (len(header)//2) + "|")
    
    for name, res in results.items():
        if res['status'] == "Success":
            print(f"| {name:<15} | {res['status']:<15} | {res['total_time_s']:<8} | {res['avg_loss']:<9} | {res['accuracy']:<8} |")
        else:
            print(f"| {name:<15} | {res['status']:<15} | {'N/A':<8} | {'N/A':<9} | {'N/A':<8} |")
            
# --- Configuration ---
N_SEQ_TEST = 64
D_MODEL_TEST = 16

if __name__ == "__main__":
    run_benchmark(N_SEQ_TEST, D_MODEL_TEST)