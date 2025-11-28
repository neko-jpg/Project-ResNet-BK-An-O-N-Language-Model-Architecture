import torch
import unittest
from src.models.phase8.kv_cache import HyperbolicKVCache

class TestHyperbolicKVCache(unittest.TestCase):
    def setUp(self):
        self.d_model = 16
        self.cache = HyperbolicKVCache(self.d_model, max_cache_size=20)

    def test_eviction_policy(self):
        """Test that tokens near origin are kept over tokens near boundary."""
        # 1. Create Data
        # - 5 "Central" tokens (Norm ~ 0.1)
        # - 25 "Boundary" tokens (Norm ~ 0.9)
        # - 10 "Local" tokens (Recent) -> Will be kept by window policy (logic is hardcoded to 10)
        # Total = 40. Max Cache = 20.
        # Expect: 10 Local + 5 Central + 5 Boundary (closest ones of the boundary set) = 20
        # Actually, if we have 5 Central (0.1) and 25 Boundary (0.9),
        # and we need to keep 10 from history (20 max - 10 local),
        # we should keep the 5 Central and the 5 "least boundary-ish" boundary tokens.

        B, D = 1, self.d_model

        central = torch.randn(B, 5, D) * 0.01 # Norm ~ small
        boundary = torch.randn(B, 25, D)
        boundary = boundary / boundary.norm(dim=-1, keepdim=True) * 0.9 # Norm ~ 0.9

        recent = torch.randn(B, 10, D) # Norm random, doesn't matter, kept by window

        # Sequence: Central -> Boundary -> Recent
        # History (Pre-window) = Central + Boundary = 30 tokens.
        # We need to reduce History to 10 tokens.
        # It should keep all 5 Central (lowest norms) and 5 from Boundary.

        # Add sequentially
        self.cache.update(central, central) # Key/Val same for test
        self.cache.update(boundary, boundary)
        self.cache.update(recent, recent)

        k_final, _ = self.cache.get_view()

        self.assertEqual(k_final.shape[1], 20)

        # Verify content
        # The first 5 tokens of the sorted history (kept part) should be the Central ones
        # because we sort indices by time.
        # Wait, Central were added first (time 0-5).
        # So they should be at the beginning of the cache.

        first_5_norms = k_final[:, :5, :].norm(dim=-1)
        print(f"First 5 norms: {first_5_norms}")

        # Should be small (< 0.1)
        self.assertTrue(torch.all(first_5_norms < 0.2))

        # The recent window (last 10) should be preserved exactly at the end
        last_10 = k_final[:, -10:, :]
        self.assertTrue(torch.allclose(last_10, recent))

if __name__ == '__main__':
    unittest.main()
