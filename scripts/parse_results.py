#!/usr/bin/env python3
"""Parse and display benchmark results."""
import json

with open("results/revolutionary_benchmark.json") as f:
    d = json.load(f)

print("=== REVOLUTIONARY ALGORITHMS BENCHMARK ===\n")
for name, result in d["results"].items():
    status = result.get("status", "?")
    emoji = "✅" if status == "PASS" else "❌" if status == "ERROR" else "⚠️"
    print(f"{emoji} {name}: {status}")
    if "kpi" in result:
        for kpi_name, kpi_val in result["kpi"].items():
            passed = kpi_val.get("passed", False)
            actual = kpi_val.get("actual", "?")
            threshold = kpi_val.get("pass_threshold", "?")
            mark = "✓" if passed else "✗"
            print(f"    {mark} {kpi_name}: {actual:.4f} (need {threshold})")
    print()

passed = d["summary"]["passed"]
total = d["summary"]["total"]
rate = d["summary"]["pass_rate"]
print(f"TOTAL: {passed}/{total} ({rate:.0f}%)")
