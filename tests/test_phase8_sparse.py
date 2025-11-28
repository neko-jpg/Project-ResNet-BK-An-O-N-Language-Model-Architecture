import torch
import unittest
from src.models.phase8.sparse_attention import SparseHyperbolicAttention

class TestSparseAttention(unittest.TestCase):
    def setUp(self):
        self.d_model = 16
        self.block_size = 4
        self.top_k = 2
        self.sparse = SparseHyperbolicAttention(self.d_model, self.block_size, self.top_k)

    def test_sparsity_mask(self):
        """Test that the mask preserves approximately Top-K blocks."""
        B, N, D = 1, 16, self.d_model
        # 16 tokens / 4 block_size = 4 blocks
        # Top-K = 2 blocks per query block

        q = torch.randn(B, N, D)
        k = torch.randn(B, N, D)

        mask = self.sparse.create_sparse_mask(q, k)

        # Check shape
        self.assertEqual(mask.shape, (B, N, N))

        # Check sparsity ratio
        # Total elements = 16*16 = 256
        # Expected True: 4 query blocks * 2 key blocks * (4*4 elements) = 4 * 2 * 16 = 128
        # Wait, mask is block-wise.
        # Mask shape (4, 4) blocks.
        # For each row (query block), 2 cols (key blocks) are True.
        # So in full mask, fraction should be 2/4 = 0.5

        true_count = mask.sum().item()
        total_count = mask.numel()
        ratio = true_count / total_count

        self.assertAlmostEqual(ratio, 0.5, delta=0.01)

    def test_forward_runs(self):
        q = torch.randn(1, 16, self.d_model)
        k = torch.randn(1, 16, self.d_model)
        v = torch.randn(1, 16, self.d_model)
        out = self.sparse(q, k, v)
        self.assertEqual(out.shape, q.shape)

if __name__ == '__main__':
    unittest.main()
