#!/bin/bash
# Run all Phase 4 tests and benchmarks

echo "--- Running Phase 4 Unit Tests ---"
export PYTHONPATH=$PYTHONPATH:.
python -m unittest discover tests -p "test_phase4_*.py"

if [ $? -ne 0 ]; then
    echo "Tests failed!"
    exit 1
fi

echo "--- Running Phase 4 Benchmark (Quick) ---"
python scripts/benchmark_phase4.py

echo "--- Running Examples ---"
python examples/phase4_integrated_demo.py

echo "Success!"
