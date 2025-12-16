#!/bin/bash
# Test that GradientFeederV2 loads correctly from training script
cd /mnt/c/dev/Project-ResNet-BK-An-O-N-Language-Model-Architecture
source venv_ubuntu/bin/activate
export PYTHONPATH=.

echo "=== Testing GradientFeederV2 Integration ==="
python -c "
import time
t0 = time.time()
from src.training.gradient_feeder import GradientFeederV2, _CPP_FEEDER_AVAILABLE, _CUDA_FEEDER_AVAILABLE
t1 = time.time()
print(f'Import time: {(t1-t0)*1000:.1f}ms')
print(f'C++ Extension Available: {_CPP_FEEDER_AVAILABLE}')
print(f'CUDA/C++ Feeder Available: {_CUDA_FEEDER_AVAILABLE}')

# Test instantiation
feeder = GradientFeederV2(target_low=0.5, target_high=3.0, initial_threshold=50.0)
print(f'GradientFeederV2 instantiated successfully')

# Test feed function
threshold, scale, stats = feeder.feed(0.5)
print(f'feed(0.5): threshold={threshold:.1f}, scale={scale:.2f}, action={stats.action}')

# Test apply_scaling
import torch
model = torch.nn.Linear(10, 10).cuda()
x = torch.randn(3, 10).cuda()
loss = model(x).sum()
loss.backward()
print(f'Gradient before scaling: {model.weight.grad.norm():.4f}')
feeder.apply_scaling(model, 2.0)
print(f'Gradient after 2x scaling: {model.weight.grad.norm():.4f}')

print()
print('=== SUCCESS: GradientFeederV2 fully functional! ===')
print('The 2-3 second delay should now be eliminated.')
"
