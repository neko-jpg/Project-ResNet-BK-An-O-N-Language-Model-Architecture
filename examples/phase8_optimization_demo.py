import torch
from src.models.phase8.adaptive import AdaptiveComputation

def main():
    print("=== Adaptive Computation Demo ===")
    d_model = 16
    adaptive = AdaptiveComputation(d_model)

    # 1. Origin Token (Complex/Abstract -> High Exit Probability?)
    # Wait, my implementation logic was:
    # "Fewer layers for tokens near origin" -> High Exit Prob.
    x_origin = torch.zeros(1, 1, d_model)

    should_exit_o, prob_o = adaptive(x_origin, 0, 10)
    print(f"Origin Token Exit Prob: {prob_o.item():.4f}")

    # 2. Boundary Token (Specific/Stable -> Low Exit Probability? Or More Layers?)
    # Re-reading task: "Fewer layers for tokens near origin"
    # So Origin -> Exit Early -> High Prob.
    # Boundary -> Stay Late -> Low Prob.
    x_boundary = torch.randn(1, 1, d_model)
    x_boundary = x_boundary / x_boundary.norm() * 0.95

    should_exit_b, prob_b = adaptive(x_boundary, 0, 10)
    print(f"Boundary Token Exit Prob: {prob_b.item():.4f}")

    if prob_o > prob_b:
        print("SUCCESS: Origin tokens exit earlier (fewer layers).")
    else:
        print("FAIL: Probability logic inverted.")

if __name__ == "__main__":
    main()
