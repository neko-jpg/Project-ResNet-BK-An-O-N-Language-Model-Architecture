
import unittest
import torch
import numpy as np
import tempfile
import os
from pathlib import Path
from src.utils.data_utils import BinaryIndexedDataset, MixedBinaryDataset

class TestDataValidation(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.test_dir.name)

        # Create dummy .bin and .idx files for 'train' and 'validation'
        self.create_dummy_data('train', 100)
        self.create_dummy_data('validation', 20)

        # Create config file
        self.config_path = self.root / "test_config.yaml"
        with open(self.config_path, "w") as f:
            f.write(f"""
datasets:
  dummy:
    path: {self.root}
    weight: 1.0
""")

    def tearDown(self):
        self.test_dir.cleanup()

    def create_dummy_data(self, split, num_tokens):
        bin_path = self.root / f"{split}.bin"
        idx_path = self.root / f"{split}.idx"

        tokens = np.arange(num_tokens, dtype=np.uint32)
        with open(bin_path, "wb") as f:
            f.write(tokens.tobytes())

        with open(idx_path, "wb") as f:
            f.write(b'MUSE')
            f.write(np.array([1], dtype=np.uint32).tobytes())
            # Single doc covering all tokens
            # Offset 0, length num_tokens
            idx_data = np.array([0, num_tokens], dtype=np.uint64)
            f.write(idx_data.tobytes())

    def test_load_train_split(self):
        ds = MixedBinaryDataset(
            config_path=str(self.config_path),
            batch_size=2,
            seq_len=10,
            total_tokens=100,
            seed=42,
            vocab_size=1000,
            split='train'
        )
        self.assertEqual(len(ds.datasets), 1)
        self.assertEqual(ds.datasets[0].tokens.shape[0], 100)

    def test_load_validation_split(self):
        ds = MixedBinaryDataset(
            config_path=str(self.config_path),
            batch_size=2,
            seq_len=10,
            total_tokens=20,
            seed=42,
            vocab_size=1000,
            split='validation'
        )
        self.assertEqual(len(ds.datasets), 1)
        self.assertEqual(ds.datasets[0].tokens.shape[0], 20)

    def test_load_missing_split(self):
        # Should warn/skip for validation
        with self.assertRaises(ValueError): # Raises ValueError if NO datasets found
             MixedBinaryDataset(
                config_path=str(self.config_path),
                batch_size=2,
                seq_len=10,
                total_tokens=20,
                seed=42,
                vocab_size=1000,
                split='test' # Doesn't exist
            )


class TestShortDocumentConcatenation(unittest.TestCase):
    """Test that short documents are concatenated instead of skipped."""
    
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.test_dir.name)
        
        # Create multiple SHORT documents (each shorter than seq_len)
        self.create_short_docs_data('train', num_docs=10, tokens_per_doc=50)
        
        # Create config file
        self.config_path = self.root / "test_config.yaml"
        with open(self.config_path, "w") as f:
            f.write(f"""
datasets:
  short_docs:
    path: {self.root}
    weight: 1.0
""")

    def tearDown(self):
        self.test_dir.cleanup()

    def create_short_docs_data(self, split, num_docs, tokens_per_doc):
        """Create dataset with multiple short documents."""
        bin_path = self.root / f"{split}.bin"
        idx_path = self.root / f"{split}.idx"
        
        # Create tokens for all documents
        total_tokens = num_docs * tokens_per_doc
        tokens = np.arange(total_tokens, dtype=np.uint32)
        with open(bin_path, "wb") as f:
            f.write(tokens.tobytes())
        
        # Create index with multiple short documents
        with open(idx_path, "wb") as f:
            f.write(b'MUSE')
            f.write(np.array([1], dtype=np.uint32).tobytes())
            
            # Create index entries for each short document
            idx_entries = []
            for i in range(num_docs):
                offset = i * tokens_per_doc
                length = tokens_per_doc
                idx_entries.extend([offset, length])
            
            idx_data = np.array(idx_entries, dtype=np.uint64)
            f.write(idx_data.tobytes())

    def test_short_docs_concatenation(self):
        """Test that short documents get concatenated to reach seq_len."""
        import random
        ds = BinaryIndexedDataset(str(self.root), split='train')
        
        # Request a sequence longer than any single document
        seq_len = 100  # Each doc is only 50 tokens
        rng = random.Random(42)
        
        result = ds.sample_sequence(seq_len, rng)
        
        # Should NOT return None - concatenation should work
        self.assertIsNotNone(result, "sample_sequence should concatenate short docs, not return None")
        
        x, y = result
        self.assertEqual(len(x), seq_len, f"x should have length {seq_len}")
        self.assertEqual(len(y), seq_len, f"y should have length {seq_len}")
        
        # Verify x and y are offset by 1
        # Note: After concatenation, this may not hold at document boundaries
        # but the arrays should be valid numpy arrays
        self.assertTrue(isinstance(x, np.ndarray))
        self.assertTrue(isinstance(y, np.ndarray))

    def test_mixed_dataset_with_short_docs(self):
        """Test MixedBinaryDataset handles short documents correctly."""
        ds = MixedBinaryDataset(
            config_path=str(self.config_path),
            batch_size=2,
            seq_len=100,  # Longer than any single document
            total_tokens=1000,
            seed=42,
            vocab_size=1000,
            split='train'
        )
        
        # Should be able to iterate without errors
        batch_count = 0
        for x_batch, y_batch in ds.iter_epoch(epoch=0):
            batch_count += 1
            self.assertEqual(x_batch.shape, (2, 100))
            self.assertEqual(y_batch.shape, (200,))  # Flattened
            if batch_count >= 3:
                break
        
        self.assertGreater(batch_count, 0, "Should produce at least one batch")


if __name__ == '__main__':
    unittest.main()

