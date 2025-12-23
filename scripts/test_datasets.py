#!/usr/bin/env python3
"""Test which Japanese datasets are accessible without login."""
from datasets import load_dataset

datasets_to_test = [
    ('range3/wiki40b-ja', None),
    ('llm-book/livedoor-news-corpus', None),
    ('shunk031/JGLUE', 'JCommonsenseQA'),
]

for ds_name, subset in datasets_to_test:
    try:
        if subset:
            d = load_dataset(ds_name, subset, split='train', streaming=True)
        else:
            d = load_dataset(ds_name, split='train', streaming=True)
        sample = next(iter(d))
        print(f'✅ {ds_name}: OK')
        print(f'   Keys: {list(sample.keys())}')
    except Exception as e:
        print(f'❌ {ds_name}: {str(e)[:80]}')
