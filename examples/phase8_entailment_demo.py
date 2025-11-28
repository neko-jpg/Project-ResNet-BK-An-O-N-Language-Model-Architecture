import torch
from src.models.phase8.entailment import EntailmentCone

def main():
    print("=== Entailment Cone Demo ===")
    d_model = 16
    cone = EntailmentCone(d_model)

    # 1. Create Hierarchy
    # Origin (Animal) -> Far (Dog)
    root = torch.randn(1, d_model) * 0.01
    leaf = root + torch.randn(1, d_model) * 0.5

    # 2. Check Entailment (Root -> Leaf)
    # Expectation: Entailment Holds (Penalty ~ 0)
    penalty_rl, aperture = cone(root, leaf)
    print(f"Root -> Leaf Penalty: {penalty_rl.item():.6f} (Expected Low)")
    print(f"Aperture Angle: {aperture.item():.4f}")

    # 3. Check Entailment (Leaf -> Root)
    # Expectation: Violation (Penalty > 0) because leaf is far and direction is wrong?
    # Actually order violation (Leaf norm > Root norm) prevents Leaf -> Root if we enforce strict norm order.
    # Our EntailmentCone logic includes `order_violation`.
    penalty_lr, _ = cone(leaf, root)
    print(f"Leaf -> Root Penalty: {penalty_lr.item():.6f} (Expected High)")

    # 4. Check Disjoint
    other = torch.randn(1, d_model) * 0.5
    penalty_disjoint, _ = cone(leaf, other)
    print(f"Leaf -> Disjoint Penalty: {penalty_disjoint.item():.6f} (Expected High)")

if __name__ == "__main__":
    main()
