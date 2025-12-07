#!/bin/bash
cd /mnt/c/dev/Project-ResNet-BK-An-O-N-Language-Model-Architecture
source .venv_wsl/bin/activate
export CUDA_HOME=/usr/local/cuda-12.6

python3 << 'EOF'
import torch
import torch.nn as nn
import sys
sys.path.insert(0, ".")

from src.training.revolutionary_trainer import RevolutionaryConfig, RevolutionaryTrainer

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(100, 32)
        self.fc = nn.Linear(32, 100)
    def forward(self, x):
        return self.fc(self.emb(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = TinyModel().to(device)
data = torch.randint(0, 100, (2, 16), device=device)
targets = torch.randint(0, 100, (2, 16), device=device)
loss_fn = nn.CrossEntropyLoss()

# Test only fast algorithms (skip zeta, sheaf, closed_form - too slow)
config = RevolutionaryConfig(
    use_holographic=True,
    use_closed_form=False,
    use_topological=False,
    use_retrocausal=True,
    use_zeta=False,
    use_sheaf=False,
    use_diffractive=True,
    log_interval=100,
)

trainer = RevolutionaryTrainer(model, config, device)
print("Testing holographic, retrocausal, diffractive...")

for i in range(15):
    loss, metrics = trainer.train_step(data, targets, loss_fn)
    algo = metrics.get("algorithm", "?")
    time_ms = metrics.get("step_time_ms", 0)
    print(f"Step {i+1}: {algo} - {time_ms:.1f}ms")

print("\n✅ Integration test passed!")
EOF
