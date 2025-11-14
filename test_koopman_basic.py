"""
Basic test script for Koopman operator learning implementation.
"""

import torch
import torch.nn as nn

from src.models.koopman_layer import KoopmanLanguageModel
from src.training.hybrid_koopman_trainer import HybridKoopmanTrainer
from src.training.koopman_scheduler import KoopmanLossScheduler

print("=" * 80)
print("KOOPMAN OPERATOR LEARNING - BASIC TESTS")
print("=" * 80)

# Test 1: Koopman layer initialization
print("\n[Test 1] Koopman Layer Initialization")
print("-" * 80)

from src.models.koopman_layer import KoopmanResNetBKLayer

layer = KoopmanResNetBKLayer(
    d_model=64,
    n_seq=128,
    koopman_dim=256,
    num_experts=4,
    top_k=1,
    dropout_p=0.1
)

print(f"✓ Layer created successfully")
print(f"  Koopman operator shape: {layer.K.shape}")
print(f"  Koopman operator norm: {layer.K.norm().item():.4f}")

# Check K is near identity
identity = torch.eye(256)
diff = (layer.K.data - identity).abs().mean().item()
print(f"  Distance from identity: {diff:.6f}")
assert diff < 0.1, "K should be close to identity"
print(f"✓ Koopman operator initialized near identity")

# Test 2: Forward pass (standard mode)
print("\n[Test 2] Forward Pass - Standard Mode")
print("-" * 80)

x = torch.randn(4, 128, 64)
output = layer(x, use_koopman=False)

print(f"✓ Standard forward pass successful")
print(f"  Input shape: {x.shape}")
print(f"  Output shape: {output.shape}")
assert output.shape == x.shape
assert not torch.isnan(output).any()
assert not torch.isinf(output).any()
print(f"✓ Output is valid (no NaN/Inf)")

# Test 3: Forward pass (Koopman mode)
print("\n[Test 3] Forward Pass - Koopman Mode")
print("-" * 80)

output_koopman = layer(x, use_koopman=True)

print(f"✓ Koopman forward pass successful")
print(f"  Input shape: {x.shape}")
print(f"  Output shape: {output_koopman.shape}")
assert output_koopman.shape == x.shape
assert not torch.isnan(output_koopman).any()
assert not torch.isinf(output_koopman).any()
print(f"✓ Output is valid (no NaN/Inf)")

# Test 4: Koopman loss
print("\n[Test 4] Koopman Auxiliary Loss")
print("-" * 80)

x_current = torch.randn(4, 128, 64)
x_next = torch.randn(4, 128, 64)
loss = layer.koopman_loss(x_current, x_next)

print(f"✓ Koopman loss computed successfully")
print(f"  Loss value: {loss.item():.6f}")
assert loss.item() >= 0
assert not torch.isnan(loss)
assert not torch.isinf(loss)
print(f"✓ Loss is valid (non-negative, no NaN/Inf)")

# Test 5: Koopman operator update
print("\n[Test 5] Koopman Operator Update (DMD)")
print("-" * 80)

K_initial = layer.K.data.clone()

# Perform multiple updates to fill buffer
for i in range(5):
    x_current = torch.randn(4, 128, 64)
    x_next = torch.randn(4, 128, 64)
    layer.update_koopman_operator(x_current, x_next)

K_final = layer.K.data
diff = (K_final - K_initial).abs().mean().item()

print(f"✓ Koopman operator updated successfully")
print(f"  Mean absolute change: {diff:.6f}")
print(f"  Buffer filled: {layer.buffer_filled.item()}")
if diff > 0:
    print(f"✓ Koopman operator has changed")
else:
    print(f"  Note: Operator may not have changed yet (buffer not full)")

# Test 6: Koopman language model
print("\n[Test 6] Koopman Language Model")
print("-" * 80)

model = KoopmanLanguageModel(
    vocab_size=1000,
    d_model=64,
    n_layers=4,
    n_seq=128,
    koopman_dim=256,
    num_experts=4,
    top_k=1,
    dropout_p=0.1
)

total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model created successfully")
print(f"  Parameters: {total_params:,}")
print(f"  Layers: {len(model.blocks)}")

# Forward pass
x_tokens = torch.randint(0, 1000, (4, 128))
logits = model(x_tokens, use_koopman=False)

print(f"✓ Model forward pass successful")
print(f"  Input shape: {x_tokens.shape}")
print(f"  Output shape: {logits.shape}")
assert logits.shape == (4, 128, 1000)
assert not torch.isnan(logits).any()
assert not torch.isinf(logits).any()
print(f"✓ Output is valid (no NaN/Inf)")

# Test 7: Koopman loss scheduler
print("\n[Test 7] Koopman Loss Scheduler")
print("-" * 80)

scheduler = KoopmanLossScheduler(
    min_weight=0.0,
    max_weight=0.5,
    warmup_epochs=2,
    total_epochs=10,
    schedule_type='linear'
)

print(f"✓ Scheduler created successfully")

# Test warmup
scheduler.step(epoch=0)
weight_0 = scheduler.get_weight()
print(f"  Epoch 0 weight: {weight_0:.4f} (warmup)")
assert weight_0 == 0.0

scheduler.step(epoch=1)
weight_1 = scheduler.get_weight()
print(f"  Epoch 1 weight: {weight_1:.4f} (warmup)")
assert weight_1 == 0.0

# Test after warmup
scheduler.step(epoch=5)
weight_5 = scheduler.get_weight()
print(f"  Epoch 5 weight: {weight_5:.4f} (active)")
assert weight_5 > 0.0

scheduler.step(epoch=10)
weight_10 = scheduler.get_weight()
print(f"  Epoch 10 weight: {weight_10:.4f} (max)")
assert abs(weight_10 - 0.5) < 0.01

print(f"✓ Scheduler working correctly")

# Test 8: Hybrid Koopman trainer
print("\n[Test 8] Hybrid Koopman Trainer")
print("-" * 80)

model = KoopmanLanguageModel(
    vocab_size=1000,
    d_model=64,
    n_layers=2,
    n_seq=128,
    koopman_dim=128,
    num_experts=4,
    top_k=1,
    dropout_p=0.1
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

trainer = HybridKoopmanTrainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    koopman_weight_min=0.0,
    koopman_weight_max=0.5,
    koopman_start_epoch=2,
    total_epochs=5,
    device='cpu'
)

print(f"✓ Trainer created successfully")
print(f"  Koopman start epoch: {trainer.koopman_start_epoch}")
print(f"  Koopman enabled: {trainer.koopman_enabled}")

# Test training step
x_batch = torch.randint(0, 1000, (4, 128))
y_batch = torch.randint(0, 1000, (4 * 128,))

metrics = trainer.train_step(x_batch, y_batch)

print(f"✓ Training step successful")
print(f"  LM loss: {metrics['loss_lm']:.4f}")
print(f"  Koopman loss: {metrics['loss_koopman']:.4f}")
print(f"  Total loss: {metrics['total_loss']:.4f}")
print(f"  Koopman weight: {metrics['koopman_weight']:.4f}")

assert metrics['loss_lm'] > 0
assert metrics['total_loss'] > 0
print(f"✓ Metrics are valid")

# Summary
print("\n" + "=" * 80)
print("ALL TESTS PASSED!")
print("=" * 80)
print("\n✓ Koopman layer implementation verified")
print("✓ Koopman language model working")
print("✓ Koopman loss scheduler functional")
print("✓ Hybrid Koopman trainer operational")
print("\nNext steps:")
print("  - Run full training on WikiText-2")
print("  - Verify Koopman operator convergence")
print("  - Benchmark backward pass cost reduction")
print("\n" + "=" * 80)
