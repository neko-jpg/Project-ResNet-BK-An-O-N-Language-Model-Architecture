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
