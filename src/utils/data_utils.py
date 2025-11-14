"""
Data Loading Utilities
"""

import torch
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
