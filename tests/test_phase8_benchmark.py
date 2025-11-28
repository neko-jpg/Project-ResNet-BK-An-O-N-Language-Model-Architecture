import unittest
import json
import os
from benchmarks.phase8_hierarchical import HierarchicalBenchmark

class TestHierarchicalBenchmark(unittest.TestCase):
    def test_run_benchmark(self):
        """Test that benchmark runs and produces valid JSON."""
        bench = HierarchicalBenchmark(d_model=16)
        results = bench.run()

        # Check structure
        self.assertIn("hypernymy", results)
        self.assertIn("reconstruction", results)
        self.assertIn("accuracy", results["hypernymy"])

        # Check values are sane
        self.assertGreaterEqual(results["hypernymy"]["accuracy"], 0.0)
        self.assertLessEqual(results["hypernymy"]["accuracy"], 1.0)

    def test_tree_generation(self):
        bench = HierarchicalBenchmark(d_model=16)
        nodes = bench.generate_mock_tree(depth=2, branching=2)
        # Root (1) + Depth 1 (2) + Depth 2 (4) = 7 nodes
        self.assertEqual(len(nodes), 7)
        self.assertIn("root", nodes)
        self.assertIn("root_0", nodes)
        self.assertIn("root_0_0", nodes)

if __name__ == '__main__':
    unittest.main()
