import torch
import torch.nn as nn
from src.models.phase8.config import Phase8Config
from src.models.phase8.integrated_model import Phase8IntegratedModel
import time

def run_reproduction():
    print("=== Reproducing NaN Issue ===")

    # 1. Config for 10B (scaled down slightly for reproduction speed, but keeping logic)
    # Using small dimensions to run quickly, but structure is identical
    config = Phase8Config(
        vocab_size=1000,
        d_model=256,
        n_layers=2,
        htt_rank=16,
        quantized_htt=True,  # Crucial: This triggers the suspected bug
        use_bk_hyperbolic=True,
        use_ar_ssm_fusion=True
    )

    print(f"Initializing Phase 8 Model (Quantized HTT: {config.quantized_htt})...")
    model = Phase8IntegratedModel(config)

    # Check initialization of Quantized HTT
    htt = model.phase7_model.htt_embedding
    if hasattr(htt, 'core1_q'):
        print(f"Core1_q stats: Min={htt.core1_q.min()}, Max={htt.core1_q.max()}, Mean={htt.core1_q.float().mean()}")
        if htt.core1_q.abs().sum() == 0:
            print("CRITICAL: Core1_q is all ZEROS!")

    # 2. Mock Data
    # ResNetBK expects exact n_seq length
    input_ids = torch.randint(0, 1000, (4, 128))

    # 3. Training Step
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    print("\nRunning forward pass...")

    try:
        logits, diag = model(input_ids, return_diagnostics=True)
        print(f"Logits stats: Min={logits.min()}, Max={logits.max()}, NaN? {torch.isnan(logits).any()}")

        if torch.isnan(logits).any():
            print("FAILURE: NaNs detected in logits!")
            return

        # Mock targets
        targets = torch.randint(0, 1000, (4, 128))
        loss = loss_fn(logits.view(-1, 1000), targets.view(-1))
        print(f"Loss: {loss.item()}")

        print("Running backward pass...")
        loss.backward()

        # Check gradients
        has_nan_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"NaN gradient in {name}")
                    has_nan_grad = True
                    break

        if has_nan_grad:
            print("FAILURE: NaNs detected in gradients!")
        else:
            print("SUCCESS: Backward pass complete without NaNs.")

    except RuntimeError as e:
        print(f"CRASH: {e}")

if __name__ == "__main__":
    run_reproduction()
