"""
Data Loading Utilities
"""

import random
import struct
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch
import yaml
from collections import Counter
from datasets import load_dataset


def get_data_loader(batch_size, n_seq, dataset_name='wikitext-2', data_limit=500000):
    """
    Create data loader for language modeling.
    
    Args:
        batch_size: batch size
        n_seq: sequence length
        dataset_name: dataset name (default: 'wikitext-2')
        data_limit: maximum number of tokens to use
    
    Returns:
        train_data: (seq_len_total, batch_size) LongTensor
        vocab: dict with stoi, itos, vocab_size
        get_batch: function (source, i) -> (data, target)
    """
    try:
        if dataset_name == 'wikitext-2':
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        else:
            dataset = load_dataset(dataset_name)
        train_texts = dataset["train"]["text"]
    except Exception as e:
        print(f"Failed to load dataset {dataset_name}: {e}")
        print("Please check network connection and dataset name.")
        return None, None, None

    # Simple word tokenization & vocab construction
    counter = Counter()
    for line in train_texts:
        tokens = line.strip().split()
        if tokens:
            counter.update(tokens)

    special_tokens = ["<unk>"]
    stoi = {}
    itos = []

    for sp in special_tokens:
        stoi[sp] = len(itos)
        itos.append(sp)

    # Limit vocabulary size
    VOCAB_LIMIT = 30000
    for tok, freq in counter.most_common(VOCAB_LIMIT - len(special_tokens)):
        if tok not in stoi:
            stoi[tok] = len(itos)
            itos.append(tok)

    vocab_size = len(itos)
    unk_id = stoi["<unk>"]

    def encode_texts(texts):
        ids = []
        for line in texts:
            for tok in line.strip().split():
                ids.append(stoi.get(tok, unk_id))
        return torch.tensor(ids, dtype=torch.long)

    train_ids = encode_texts(train_texts)

    # Limit tokens for memory constraints
    if train_ids.numel() > data_limit:
        train_ids = train_ids[:data_limit]

    def batchify(data, bsz):
        seq_len = data.size(0) // bsz
        data = data.narrow(0, 0, seq_len * bsz)
        data = data.view(bsz, seq_len).t().contiguous()  # (seq_len, batch_size)
        return data

    train_data = batchify(train_ids, batch_size)

    def get_batch(source, i):
        seq_len = min(n_seq, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].reshape(-1)
        return data, target

    vocab = {
        "stoi": stoi,
        "itos": itos,
        "vocab_size": vocab_size,
    }

    return train_data, vocab, get_batch



def get_wikitext2_dataloaders(batch_size=32, seq_len=128, num_workers=2, vocab_size_limit=30000):
    """
    Create PyTorch DataLoaders for WikiText-2 dataset.
    
    Args:
        batch_size: batch size for DataLoader
        seq_len: sequence length
        num_workers: number of workers for DataLoader
        vocab_size_limit: maximum vocabulary size
    
    Returns:
        train_loader: training DataLoader
        val_loader: validation DataLoader
        vocab_size: vocabulary size
    """
    from torch.utils.data import Dataset, DataLoader
    
    # Load dataset
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        train_texts = dataset["train"]["text"]
        val_texts = dataset["validation"]["text"]
    except Exception as e:
        print(f"Failed to load WikiText-2: {e}")
        print("Using dummy data for testing...")
        # Dummy data for testing
        train_texts = ["hello world"] * 1000
        val_texts = ["hello world"] * 100
    
    # Build vocabulary
    counter = Counter()
    for line in train_texts:
        tokens = line.strip().split()
        if tokens:
            counter.update(tokens)
    
    special_tokens = ["<pad>", "<unk>"]
    stoi = {}
    itos = []
    
    for sp in special_tokens:
        stoi[sp] = len(itos)
        itos.append(sp)
    
    # Limit vocabulary
    for tok, freq in counter.most_common(vocab_size_limit - len(special_tokens)):
        if tok not in stoi:
            stoi[tok] = len(itos)
            itos.append(tok)
    
    vocab_size = len(itos)
    unk_id = stoi["<unk>"]
    pad_id = stoi["<pad>"]
    
    def encode_texts(texts):
        """Encode texts to token IDs."""
        ids = []
        for line in texts:
            tokens = line.strip().split()
            if tokens:
                for tok in tokens:
                    ids.append(stoi.get(tok, unk_id))
        return ids
    
    # Encode datasets
    train_ids = encode_texts(train_texts)
    val_ids = encode_texts(val_texts)
    
    # Create Dataset class
    class TextDataset(Dataset):
        def __init__(self, token_ids, seq_len):
            self.token_ids = token_ids
            self.seq_len = seq_len
            # Calculate number of sequences
            self.num_sequences = len(token_ids) // (seq_len + 1)
        
        def __len__(self):
            return self.num_sequences
        
        def __getitem__(self, idx):
            start_idx = idx * (self.seq_len + 1)
            end_idx = start_idx + self.seq_len + 1
            
            # Get sequence
            sequence = self.token_ids[start_idx:end_idx]
            
            # Pad if necessary
            if len(sequence) < self.seq_len + 1:
                sequence = sequence + [pad_id] * (self.seq_len + 1 - len(sequence))
            
            # Input and target
            x = torch.tensor(sequence[:-1], dtype=torch.long)
            y = torch.tensor(sequence[1:], dtype=torch.long)
            
            return x, y
    
    # Create datasets
    train_dataset = TextDataset(train_ids, seq_len)
    val_dataset = TextDataset(val_ids, seq_len)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, vocab_size



class BinaryIndexedDataset:
    """
    Memory-mapped reader for .bin/.idx pairs produced by prepare_datasets.py.

    The .idx file layout:
      - 4 bytes: magic "MUSE"
      - 4 bytes: version (uint32, little-endian)
      - Followed by uint64 pairs: (offset, length) for each document
    The .bin file stores uint32 token ids concatenated for all documents.
    """

    def __init__(self, path: str, split: str = "train"):
        self.use_rust = False
        try:
            import rust_loader
            self.rust_loader = rust_loader.RustDataLoader(str(path), split)
            self.use_rust = True
            # We still need some properties if accessed directly, but let's rely on rust loader methods
        except (ImportError, AttributeError):
            # Fallback to Python implementation
            pass

        if not self.use_rust:
            root = Path(path)
            bin_path = root / f"{split}.bin"
            idx_path = root / f"{split}.idx"

            if not bin_path.exists() or not idx_path.exists():
                if split == "validation" and not bin_path.exists():
                     raise FileNotFoundError(f"Validation split not found at {bin_path}")
                raise FileNotFoundError(f"Missing bin/idx files under {root} for split '{split}'")

            with open(idx_path, "rb") as f:
                magic = f.read(4)
                if magic != b"MUSE":
                    raise ValueError(f"Invalid magic in {idx_path}: {magic}")
                _version = struct.unpack("<I", f.read(4))[0]
                idx_data = np.fromfile(f, dtype=np.uint64)

            if idx_data.size % 2 != 0:
                raise ValueError(f"Corrupted idx file: {idx_path}")

            self.index = idx_data.reshape(-1, 2)  # (num_docs, 2): offset, length
            self.tokens = np.memmap(bin_path, dtype=np.uint32, mode="r")

    @property
    def num_docs(self) -> int:
        if self.use_rust:
            return self.rust_loader.num_docs()
        return self.index.shape[0]

    def sample_sequence(self, seq_len: int, rng: random.Random) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Sample a (input, target) pair of length seq_len from a random doc."""
        if self.use_rust:
            # Use Rust implementation
            # Note: Rust implementation uses its own internal RNG for now.
            # If we want to sync RNG, we need to pass seed or state, but for performance internal is fine.
            res = self.rust_loader.sample_sequence(seq_len)
            if res is None:
                return None
            x, y = res
            # Rust returns numpy arrays (via pyo3-numpy)
            return x, y

        for _ in range(8):  # retry a few times for short docs
            doc_id = rng.randrange(self.num_docs)
            offset, length = self.index[doc_id]
            # Cast to Python int for downstream random ranges
            offset = int(offset)
            length = int(length)
            seq_len = int(seq_len)
            if length <= seq_len:
                continue
            start = rng.randrange(0, length - seq_len)
            end = start + seq_len + 1  # +1 for target shift
            slice_tokens = self.tokens[offset + start : offset + end]
            x = slice_tokens[:-1]
            y = slice_tokens[1:]
            return x, y
        return None


class MixedBinaryDataset:
    """
    Weighted mixture of multiple BinaryIndexedDataset datasets.
    """

    def __init__(
        self,
        config_path: str,
        batch_size: int,
        seq_len: int,
        total_tokens: int,
        seed: int,
        vocab_size: int,
        split: str = "train",
    ):
        self.config_path = Path(config_path)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.total_tokens = total_tokens
        self.seed = seed
        self.vocab_size = vocab_size
        self.split = split

        with open(self.config_path, "r") as f:
            cfg = yaml.safe_load(f)

        ds_cfg: Dict[str, Dict[str, float]] = cfg.get("datasets", {})
        if not ds_cfg:
            raise ValueError(f"No datasets defined in {self.config_path}")

        self.datasets: List[BinaryIndexedDataset] = []
        self.weights: List[float] = []
        self.dataset_names: List[str] = []

        for name, info in ds_cfg.items():
            path = info.get("path")
            weight = float(info.get("weight", 0.0))
            if not path:
                continue

            # Try to load the dataset for the requested split
            try:
                ds = BinaryIndexedDataset(path, split=split)
                # Keep even if weight is 0 initially, to allow dynamic enabling
                self.datasets.append(ds)
                self.weights.append(weight)
                self.dataset_names.append(name)
            except FileNotFoundError:
                if split == "train":
                    # Critical failure if training data is missing
                    raise
                else:
                    # Soft failure for validation: just warn and skip this dataset
                    print(f"[Dataset] Warning: Split '{split}' missing for {name}, skipping.")
                    continue

        if not self.datasets:
             if split == "validation":
                 print(f"[Dataset] Warning: No datasets found for split '{split}'. Validation will be empty.")
             else:
                 raise ValueError(f"All datasets missing in {self.config_path} for split {split}")

        # Normalize initial weights
        self._normalize_weights()

        tokens_per_step = batch_size * seq_len
        self.steps_per_epoch = max(1, total_tokens // tokens_per_step)

    def _normalize_weights(self):
        """Normalize weights to sum to 1.0."""
        weight_sum = sum(self.weights)
        if weight_sum <= 0:
            # Uniform fallback
            self.weights = [1.0 / len(self.weights)] * len(self.weights)
        else:
            self.weights = [w / weight_sum for w in self.weights]

    def update_weights_by_name(self, name_weight_map: Dict[str, float]):
        """Update dataset mixing weights dynamically."""
        changed = False
        for i, name in enumerate(self.dataset_names):
            if name in name_weight_map:
                self.weights[i] = max(0.0, float(name_weight_map[name]))
                changed = True

        if changed:
            self._normalize_weights()
            # Log update (optional, print for now)
            print(f"[Curriculum] Updated weights: {dict(zip(self.dataset_names, [f'{w:.2f}' for w in self.weights]))}")

    def iter_epoch(self, epoch: int, start_step: int = 0) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Yield batches for one epoch.
        
        Args:
            epoch: Current epoch number (used for RNG seed)
            start_step: Step to start from within this epoch (for resume support)
        
        Note:
            Instead of trying to replay RNG states (which is error-prone due to
            variable RNG consumption in sample_sequence), we use start_step as
            part of the seed to guarantee fresh, non-duplicate data on resume.
        """
        # Include start_step in seed to ensure different data sequence on resume
        # This prevents data duplication without requiring exact RNG replay
        rng = random.Random(self.seed + epoch * 100000 + start_step)
        choices = list(range(len(self.datasets)))
        
        # Yield batches from start_step to end of epoch
        for step in range(start_step, self.steps_per_epoch):
            x_list: List[np.ndarray] = []
            y_list: List[np.ndarray] = []
            while len(x_list) < self.batch_size:
                # Use current weights (support dynamic update)
                ds_idx = rng.choices(choices, weights=self.weights, k=1)[0]
                sample = self.datasets[ds_idx].sample_sequence(self.seq_len, rng)
                if sample is None:
                    continue
                x, y = sample
                x_list.append(x)
                y_list.append(y)

            x_batch = torch.from_numpy(np.stack(x_list).astype(np.int64, copy=False)).long()
            y_batch = torch.from_numpy(np.stack(y_list).astype(np.int64, copy=False)).long().reshape(-1)
            yield x_batch, y_batch

    def vocab(self) -> Dict[str, object]:
        return {"stoi": {}, "itos": [], "vocab_size": self.vocab_size}

    def num_tokens_per_epoch(self) -> int:
        return self.steps_per_epoch * self.batch_size * self.seq_len

    def max_token_id(self) -> int:
        """Return the maximum token id across all mixed datasets (uint32 memmaps)."""
        max_id = 0
        for name, ds in zip(self.dataset_names, self.datasets):
            try:
                if ds.tokens.size == 0:
                    continue
                ds_max = int(np.max(ds.tokens))
                max_id = max(max_id, ds_max)
            except Exception as e:
                print(f"[Dataset] Warning: failed to scan tokens for {name}: {e}")
        return max_id


def get_mixed_data_loader(
    config_path: str,
    batch_size: int,
    n_seq: int,
    total_tokens: int,
    seed: int,
    vocab_size: int,
    split: str = "train",
) -> Tuple[MixedBinaryDataset, Dict[str, object], int]:
    """
    Build a mixed dataset loader from .bin/.idx datasets defined in YAML.
    """
    mixed = MixedBinaryDataset(
        config_path=config_path,
        batch_size=batch_size,
        seq_len=n_seq,
        total_tokens=total_tokens,
        seed=seed,
        vocab_size=vocab_size,
        split=split,
    )

    # Only expand vocab based on training data to ensure consistency,
    # but we scan whatever we loaded.
    max_token_id = mixed.max_token_id()
    if max_token_id + 1 > mixed.vocab_size:
        mixed.vocab_size = max_token_id + 1
        print(f"[Dataset] Expanded vocab_size to {mixed.vocab_size} to fit token ids (max id={max_token_id}).")
    return mixed, mixed.vocab(), mixed.steps_per_epoch
