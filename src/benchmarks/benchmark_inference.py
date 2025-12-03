
import time
import torch
import argparse
from src.inference.inference_engine import InferenceEngine, patch_model_with_kv_cache
from src.models.phase7.hyperbolic_attention import HyperbolicMultiHeadAttention
import torch.nn as nn

class MockModel(nn.Module):
    def __init__(self, d_model=256, num_heads=8):
        super().__init__()
        self.attn = HyperbolicMultiHeadAttention(d_model, num_heads, use_triton_kernel=False)
        self.fc = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        out, _ = self.attn(x)
        return self.fc(out)

def benchmark_inference(max_new_tokens=100, device='cuda'):
    print(f"\nBenchmarking Inference (Tokens={max_new_tokens})...")
    
    model = MockModel().to(device)
    engine = InferenceEngine(model, max_cache_size=1024)
    patch_model_with_kv_cache(model, engine)
    
    input_emb = torch.randn(1, 10, 256, device=device) # Start with 10 tokens
    
    # Warmup
    for _ in range(5):
        model(input_emb[:, -1:])
        
    torch.cuda.synchronize()
    start = time.time()
    
    # Simulate generation loop
    curr_input = input_emb
    for _ in range(max_new_tokens):
        # Step
        with torch.no_grad():
            _ = model(curr_input[:, -1:])
        # Dummy next token (just reuse input)
        curr_input = torch.cat([curr_input, torch.randn(1, 1, 256, device=device)], dim=1)
        
    torch.cuda.synchronize()
    duration = time.time() - start
    
    tps = max_new_tokens / duration
    print(f"Time: {duration:.2f} s")
    print(f"Tokens/sec: {tps:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
    else:
        benchmark_inference(device=args.device)
