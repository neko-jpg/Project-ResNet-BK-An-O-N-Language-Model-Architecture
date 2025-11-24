
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

if __name__ == '__main__':
    unittest.main()
