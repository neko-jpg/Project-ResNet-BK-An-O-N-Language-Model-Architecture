import json

d = json.load(open("checkpoints/phase8_10b_japanese/training_log.json"))
entries = d.get("entries", d.get("steps", []))

if entries:
    print(f"Initial Loss: {entries[0]['loss']:.4f}")
    print(f"Step 100 Loss: {entries[99]['loss']:.4f}")
    print(f"Step 200 Loss: {entries[199]['loss']:.4f}")
    print(f"Final Loss: {d['final_loss']:.4f}")
    print(f"---")
    print(f"Initial PPL: {entries[0]['ppl']:.2f}")
    print(f"Final PPL: {entries[-1]['ppl']:.2f}")
    print(f"---")
    print(f"Loss Change: {entries[0]['loss']:.4f} -> {d['final_loss']:.4f} ({(1 - d['final_loss']/entries[0]['loss'])*100:.1f}% decrease)")
else:
    print("No entries found")
    print(json.dumps(list(d.keys()), indent=2))
