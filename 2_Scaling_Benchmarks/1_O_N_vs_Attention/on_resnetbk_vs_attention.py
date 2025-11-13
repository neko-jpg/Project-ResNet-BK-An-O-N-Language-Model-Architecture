import torch
import torch.nn as nn
import time
from torch.func import vmap # (✘1.6) VMAP: バッチ並列化の切り札

# --- (✘1) O(N) Tridiagonal Inverse Diagonal Algorithm (by Teppei Arai) ---
# 徹平さんによる安定化・高精度化バージョン
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

    # --- Forward sweep: theta[k] = det(1..k) ---
    theta = []
    theta.append(torch.ones((), dtype=torch.complex128, device=device))
    theta.append(a_shifted[0])
    for i in range(1, n):
        theta.append(a_shifted[i] * theta[i] - c_c[i-1] * b_c[i-1] * theta[i-1])
    theta = torch.stack(theta)
    det_T = theta[-1]

    # --- Backward sweep: phi[i] = det(i+1..n) ---
    phi = [None] * n
    phi[-1] = torch.ones((), dtype=torch.complex128, device=device)
    if n > 1:
        phi[-2] = a_shifted[-1]
        for i in range(n - 3, -1, -1):
            phi[i] = a_shifted[i+1] * phi[i+1] - c_c[i] * b_c[i] * phi[i+2]
    phi = torch.stack(phi)

    # --- Diagonal of inverse ---
    eps = torch.tensor(1e-18, dtype=torch.complex128, device=device)
    diag_inv = theta[:-1] * phi / (det_T + eps)

    return diag_inv.to(torch.complex64)


# --- (✘3) ResNet-BK Layer (Batched & Vmapped) ---
class ResNetBKLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        self.z = torch.tensor(1.0j, dtype=torch.complex64)
        self.vmapped_get_diag = vmap(
            get_tridiagonal_inverse_diagonal, 
            in_dims=(0, 0, 0, None), 
            out_dims=0
        )
        # (✘1) 出力次元をd_modelに合わせるための最終線形層
        self.output_proj = nn.Linear(2, d_model)


    def forward(self, x):
        batch_size, n_seq, d_model = x.shape
        v = self.mlp(x).squeeze(-1)
        
        h0_diag = torch.full((batch_size, n_seq), -2.0, device=x.device, dtype=torch.float32)
        h0_sub = torch.full((batch_size, n_seq-1), 1.0, device=x.device, dtype=torch.float32)
        h0_super = torch.full((batch_size, n_seq-1), 1.0, device=x.device, dtype=torch.float32)
        
        he_diag = h0_diag + v
        
        G_ii_batched = self.vmapped_get_diag(
            he_diag, h0_super, h0_sub, self.z.to(x.device)
        )
        
        # (✘2) Handle complex values
        output_features = torch.stack([G_ii_batched.real, G_ii_batched.imag], dim=-1).to(torch.float32)
        
        # (✘1) 性能をAttentionと公平に比較するため、出力次元をd_modelに戻す
        return self.output_proj(output_features)

# --- (✘3) Standard Attention Layer (for comparison) ---
class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_head=2): # Head数を増やして現実的に
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_head, batch_first=True)
    
    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return attn_output

# --- (✘1) スケーリング計測用の関数 ---
def time_model(model, x, num_runs=100):
    # (修正) time_model が device を認識できるように引数を追加
    device = x.device 
    
    # GPUのウォームアップ
    for _ in range(10):
        y = model(x)
        y.sum().backward()
        
    # (修正) GPUが利用可能な場合のみ synchronize を呼び出す
    if device.type == 'cuda':
        torch.cuda.synchronize() # GPUの実行完了を待つ
    
    start_time = time.time()
    for _ in range(num_runs):
        y = model(x)
        y.sum().backward() # backwardも含めた時間を計測
    
    # (修正) GPUが利用可能な場合のみ synchronize を呼び出す
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    
    return (end_time - start_time) / num_runs # 1回あたりの平均時間

# --- メインの実行ブロック ---
if __name__ == "__main__":
    # --- Configuration ---
    D_MODEL = 64     # より現実的なモデルサイズ
    BATCH_SIZE = 8   # 現実的なバッチサイズ
    N_SEQ_LIST = [64, 256, 1024, 2048] # (✘1) スケーリングテスト
    
    # GPUが使えるかチェック
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU")
        device = torch.device("cpu")

    # モデルの初期化
    resnet_bk = ResNetBKLayer(D_MODEL).to(device)
    attention = AttentionLayer(D_MODEL, n_head=4).to(device) # Head数を4に

    results = {"ResNet-BK": [], "Attention": []}

    print("\n--- (✘1) Scaling Benchmark (Forward + Backward Time) ---")
    print(f"D_MODEL={D_MODEL}, BATCH_SIZE={BATCH_SIZE}")
    header = "| N_SEQ      | ResNet-BK (ms) | Attention (ms) |"
    print(header)
    print("|-" * (len(header)//2) + "|")

    for n_seq in N_SEQ_LIST:
        # (修正) 変数をNoneで初期化
        t_bk, t_attn = None, None
        
        # ダミーデータ
        x = torch.rand((BATCH_SIZE, n_seq, D_MODEL), device=device)
        
        # ResNet-BK
        try:
            # (修正) time_model に device を渡す (x.device で自動的に渡される)
            t_bk = time_model(resnet_bk, x) * 1000 # ミリ秒に変換
            results["ResNet-BK"].append(t_bk)
        except Exception as e:
            print(f"ResNet-BK failed at N={n_seq}: {e}")
            results["ResNet-BK"].append(None)
            
        # Attention
        try:
            # (修正) time_model に device を渡す (x.device で自動的に渡される)
            t_attn = time_model(attention, x) * 1000 # ミリ秒に変換
            results["Attention"].append(t_attn)
        except Exception as e:
            print(f"Attention failed at N={n_seq}: {e}")
            results["Attention"].append(None)

        # (修正) 失敗した場合に備えて、Noneでないかチェック
        t_bk_str = f"{t_bk:<14.3f}" if t_bk is not None else "Failed"
        t_attn_str = f"{t_attn:<14.3f}" if t_attn is not None else "Failed"
        print(f"| {n_seq:<10} | {t_bk_str:<14} | {t_attn_str:<14} |")

    # --- (✘1) グラフプロット ---
    try:
        import matplotlib.pyplot as plt
        
        # (修正) 失敗した (None) データをプロットから除外
        plot_n_seq_bk = [n for n, t in zip(N_SEQ_LIST, results["ResNet-BK"]) if t is not None]
        plot_t_bk = [t for t in results["ResNet-BK"] if t is not None]
        plot_n_seq_attn = [n for n, t in zip(N_SEQ_LIST, results["Attention"]) if t is not None]
        plot_t_attn = [t for t in results["Attention"] if t is not None]

        plt.figure(figsize=(10, 6))
        plt.plot(plot_n_seq_bk, plot_t_bk, 'bo-', label="ResNet-BK (O(N)) - Teppei's Model")
        plt.plot(plot_n_seq_attn, plot_t_attn, 'ro-', label="Attention (O(N^2)) - Standard")
        
        plt.xlabel("Sequence Length (N)")
        plt.ylabel("Time per Batch (ms)")
        plt.title(f"Scaling Benchmark: ResNet-BK vs Attention (B={BATCH_SIZE}, D={D_MODEL})")
        plt.legend()
        plt.grid(True)
        
        # O(N^2) と O(N) の理論的な線を引く (オプション)
        # N^2
        if results["Attention"][0]:
            n_start = N_SEQ_LIST[0]
            t_start = results["Attention"][0]
            n_sq_theory = [t_start * (n/n_start)**2 for n in N_SEQ_LIST]
            plt.plot(N_SEQ_LIST, n_sq_theory, 'r:', label="Theory O(N^2) slope")
        # N
        if results["ResNet-BK"][0]:
            n_start = N_SEQ_LIST[0]
            t_start = results["ResNet-BK"][0]
            n_lin_theory = [t_start * (n/n_start) for n in N_SEQ_LIST]
            plt.plot(N_SEQ_LIST, n_lin_theory, 'b:', label="Theory O(N) slope")
        
        plt.legend()
        
        # スケーリングを見るために両対数グラフも表示
        plt.figure(figsize=(10, 6))
        plt.loglog(plot_n_seq_bk, plot_t_bk, 'bo-', label="ResNet-BK (O(N)) - Teppei's Model")
        plt.loglog(plot_n_seq_attn, plot_t_attn, 'ro-', label="Attention (O(N^2)) - Standard")
        
        plt.xlabel("Sequence Length (N) [log scale]")
        plt.ylabel("Time per Batch (ms) [log scale]")
        plt.title(f"Log-Log Scaling Plot: O(N) vs O(N^2)")
        plt.legend()
        plt.grid(True)

        plot_filename = "scaling_benchmark.png"
        plt.savefig(plot_filename)
        print(f"\nBenchmark graph saved to: {plot_filename}")

    except ImportError:
        print("\nMatplotlib not found. Please install it to see the graph:")
        print("pip install matplotlib")