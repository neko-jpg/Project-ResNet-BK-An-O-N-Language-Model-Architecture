#!/usr/bin/env python3
"""Analyze training log to find patterns in loss increase."""
import json
import sys

log_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/phase8_300m_scaling/training_all_log.json"

with open(log_path, 'r') as f:
    data = json.load(f)

steps = data.get('steps', [])
print(f"Total logged steps: {len(steps)}")

# Find resumption point (optimizer_step resets)
prev_opt_step = None
resume_points = []
for s in steps:
    opt_step = s.get('optimizer_step', 0)
    if prev_opt_step is not None and opt_step < prev_opt_step:
        resume_points.append(s['step'])
    prev_opt_step = opt_step

print(f"\nResumption points detected: {resume_points}")

# Analyze loss trend in chunks
chunk_size = 50
print(f"\n=== Loss Trend (every {chunk_size} entries) ===")
print(f"{'Step Range':<20} {'Avg Loss':<10} {'Min Loss':<10} {'Max Loss':<10} {'Trend'}")
print("-" * 65)

for i in range(0, len(steps), chunk_size):
    chunk = steps[i:i+chunk_size]
    if not chunk:
        continue
    
    losses = [s.get('loss', 0) for s in chunk if s.get('loss')]
    if not losses:
        continue
    
    avg_loss = sum(losses) / len(losses)
    min_loss = min(losses)
    max_loss = max(losses)
    
    step_start = chunk[0].get('step', 0)
    step_end = chunk[-1].get('step', 0)
    
    # Determine trend
    if len(losses) >= 2:
        first_half = sum(losses[:len(losses)//2]) / (len(losses)//2)
        second_half = sum(losses[len(losses)//2:]) / (len(losses) - len(losses)//2)
        if second_half > first_half * 1.05:
            trend = "↗️ UP"
        elif second_half < first_half * 0.95:
            trend = "↘️ DOWN"
        else:
            trend = "→ FLAT"
    else:
        trend = "?"
    
    print(f"{step_start:>6} - {step_end:<9} {avg_loss:<10.4f} {min_loss:<10.4f} {max_loss:<10.4f} {trend}")

# Find when things went wrong
print(f"\n=== Key Observations ===")
if steps:
    first = steps[0]
    last = steps[-1]
    print(f"First entry: step={first.get('step')}, loss={first.get('loss'):.4f}, ppl={first.get('ppl'):.1f}")
    print(f"Last entry:  step={last.get('step')}, loss={last.get('loss'):.4f}, ppl={last.get('ppl'):.1f}")
    
    # Find minimum loss
    min_entry = min(steps, key=lambda x: x.get('loss', float('inf')))
    print(f"Min loss:    step={min_entry.get('step')}, loss={min_entry.get('loss'):.4f}, ppl={min_entry.get('ppl'):.1f}")
    
    # Check for anomalies (loss jumps > 0.3)
    anomalies = []
    for i in range(1, len(steps)):
        if steps[i].get('loss', 0) - steps[i-1].get('loss', 0) > 0.3:
            anomalies.append({
                'from_step': steps[i-1].get('step'),
                'to_step': steps[i].get('step'),
                'jump': steps[i].get('loss', 0) - steps[i-1].get('loss', 0)
            })
    
    if anomalies:
        print(f"\nLoss jumps > 0.3 detected: {len(anomalies)}")
        for a in anomalies[:5]:
            print(f"  Step {a['from_step']} → {a['to_step']}: +{a['jump']:.4f}")
