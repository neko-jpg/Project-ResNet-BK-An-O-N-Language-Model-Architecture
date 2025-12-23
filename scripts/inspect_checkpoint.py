#!/usr/bin/env python3
"""Inspect checkpoint contents to debug training resume issues."""
import torch
import sys

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/phase8_300m_scaling/step_5000.pt"

print(f"Loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu")

print("\n=== Checkpoint Keys ===")
for k in ckpt.keys():
    v = ckpt[k]
    if isinstance(v, dict):
        print(f"  {k}: dict with {len(v)} keys")
    elif isinstance(v, (int, float, str)):
        print(f"  {k}: {v}")
    else:
        print(f"  {k}: {type(v).__name__}")

print(f"\n=== Critical Values ===")
print(f"  step: {ckpt.get('step', 'NOT FOUND')}")
print(f"  epoch: {ckpt.get('epoch', 'NOT FOUND')}")
print(f"  loss: {ckpt.get('loss', 'NOT FOUND')}")

# Check optimizer state
opt_state = ckpt.get('optimizer_state_dict', {})
print(f"\n=== Optimizer State ===")
print(f"  Has 'state': {'state' in opt_state}")
print(f"  Has 'param_groups': {'param_groups' in opt_state}")

if 'state' in opt_state:
    state = opt_state['state']
    print(f"  Number of parameter states: {len(state)}")
    if state:
        first_key = list(state.keys())[0]
        first_state = state[first_key]
        print(f"  First param state keys: {list(first_state.keys())}")
        if 'step' in first_state:
            print(f"  First param optimizer step: {first_state['step']}")

# Check scheduler state
sched_state = ckpt.get('scheduler_state_dict', {})
print(f"\n=== Scheduler State ===")
print(f"  current_step: {sched_state.get('current_step', 'NOT FOUND')}")
print(f"  warmup_steps: {sched_state.get('warmup_steps', 'NOT FOUND')}")
print(f"  total_steps: {sched_state.get('total_steps', 'NOT FOUND')}")
print(f"  peak_lr: {sched_state.get('peak_lr', 'NOT FOUND')}")

print("\n=== Analysis Complete ===")
