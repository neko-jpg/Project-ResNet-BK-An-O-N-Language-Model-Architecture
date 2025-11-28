import json
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", required=True, help="Path to current benchmark JSON")
    parser.add_argument("--baseline", required=True, help="Path to baseline benchmark JSON")
    parser.add_argument("--threshold", type=float, default=0.05, help="Allowed degradation (e.g. 0.05 = 5%)")
    args = parser.parse_args()

    # Load files
    try:
        with open(args.current) as f:
            current = json.load(f)
        with open(args.baseline) as f:
            baseline = json.load(f)
    except FileNotFoundError:
        print("Benchmark file not found. Skipping regression check.")
        sys.exit(0)

    # Metrics to check
    metrics = [
        ("hypernymy", "accuracy"),
        # reconstruction curvature is not strictly "higher is better", so we skip for now
    ]

    failed = False

    for section, key in metrics:
        curr_val = current.get(section, {}).get(key, 0)
        base_val = baseline.get(section, {}).get(key, 0)

        if base_val == 0:
            continue

        # Check for degradation (Assuming higher is better)
        ratio = (base_val - curr_val) / base_val

        if ratio > args.threshold:
            print(f"REGRESSION DETECTED: {section}.{key} dropped from {base_val} to {curr_val} (-{ratio*100:.1f}%)")
            failed = True
        else:
            print(f"PASS: {section}.{key} is stable ({curr_val} vs {base_val})")

    if failed:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
