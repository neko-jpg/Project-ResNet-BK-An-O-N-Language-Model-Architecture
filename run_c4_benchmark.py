#!/usr/bin/env python3
"""
Run C4 Benchmark

This script runs the comprehensive C4 benchmark for task 9.5:
- Train on 100M tokens from C4
- Measure perplexity across domains
- Compare to other datasets (WikiText-2, WikiText-103, Penn Treebank)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.benchmarks.c4_benchmark import main

if __name__ == '__main__':
    print("=" * 80)
    print("C4 Benchmark Runner")
    print("=" * 80)
    print("\nThis benchmark will:")
    print("  1. Load 100M tokens from C4 dataset")
    print("  2. Train ResNet-BK baseline and optimized models")
    print("  3. Measure perplexity across different domains")
    print("  4. Compare to WikiText-2, WikiText-103, and Penn Treebank")
    print("\nNote: This may take significant time due to large dataset size")
    print("      Ensure you have stable internet connection for dataset download")
    print("=" * 80 + "\n")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError running benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
