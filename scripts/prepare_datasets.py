"""
Dataset Preparation Script for Mamba-Killer ResNet-BK

Downloads and preprocesses WikiText-2, WikiText-103, C4, and The Pile datasets
into a standardized format for reproducible benchmarking.

Usage:
    python scripts/prepare_datasets.py --datasets wikitext2 wikitext103 c4 pile --output_dir ./data
    python scripts/prepare_datasets.py --all --output_dir ./data
"""

import argparse
import os
import json
from pathlib import Path
from typing import List, Dict, Optional
import logging

try:
    from datasets import load_dataset
    import torch
    from transformers import AutoTokenizer
except ImportError:
    print("Error: Required packages not installed.")
    print("Please install: pip install datasets transformers torch")
    exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetPreparator:
    """Handles downloading and preprocessing of benchmark datasets."""
    
    def __init__(self, output_dir: str = "./data", tokenizer_name: str = "gpt2"):
        """
        Initialize dataset preparator.
        
        Args:
            output_dir: Directory to save processed datasets
            tokenizer_name: Tokenizer to use for preprocessing
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.dataset_configs = {
            'wikitext2': {
                'hf_name': 'wikitext',
                'hf_config': 'wikitext-2-raw-v1',
                'splits': ['train', 'validation', 'test']
            },
            'wikitext103': {
                'hf_name': 'wikitext',
                'hf_config': 'wikitext-103-raw-v1',
                'splits': ['train', 'validation', 'test']
            },
            'c4': {
                'hf_name': 'c4',
                'hf_config': 'en',
                'splits': ['train', 'validation'],
                'streaming': True  # C4 is very large
            },
            'pile': {
                'hf_name': 'EleutherAI/pile',
                'hf_config': 'all',
                'splits': ['train', 'validation', 'test'],
                'streaming': True  # The Pile is very large
            }
        }
    
    def prepare_wikitext2(self) -> Dict[str, str]:
        """Download and preprocess WikiText-2."""
        logger.info("Preparing WikiText-2...")
        return self._prepare_wikitext('wikitext2')
    
    def prepare_wikitext103(self) -> Dict[str, str]:
        """Download and preprocess WikiText-103."""
        logger.info("Preparing WikiText-103...")
        return self._prepare_wikitext('wikitext103')
    
    def _prepare_wikitext(self, dataset_name: str) -> Dict[str, str]:
        """Generic WikiText preparation."""
        config = self.dataset_configs[dataset_name]
        dataset_dir = self.output_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        output_paths = {}
        
        try:
            # Load dataset from Hugging Face
            dataset = load_dataset(
                config['hf_name'],
                config['hf_config']
            )
            
            for split in config['splits']:
                logger.info(f"Processing {dataset_name} {split} split...")
                
                # Extract text
                texts = [item['text'] for item in dataset[split] if item['text'].strip()]
                
                # Tokenize
                logger.info(f"Tokenizing {len(texts)} documents...")
                tokenized = self.tokenizer(
                    texts,
                    truncation=False,
                    padding=False,
                    return_attention_mask=False
                )
                
                # Save tokenized data
                output_file = dataset_dir / f"{split}.pt"
                torch.save({
                    'input_ids': tokenized['input_ids'],
                    'num_documents': len(texts),
                    'total_tokens': sum(len(ids) for ids in tokenized['input_ids']),
                    'tokenizer': self.tokenizer.name_or_path
                }, output_file)
                
                output_paths[split] = str(output_file)
                logger.info(f"Saved {split} split to {output_file}")
            
            # Save metadata
            metadata = {
                'dataset': dataset_name,
                'tokenizer': self.tokenizer.name_or_path,
                'vocab_size': self.tokenizer.vocab_size,
                'splits': output_paths,
                'config': config
            }
            
            metadata_file = dataset_dir / 'metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✓ {dataset_name} preparation complete")
            return output_paths
            
        except Exception as e:
            logger.error(f"Error preparing {dataset_name}: {e}")
            raise
    
    def prepare_c4(self, max_samples: Optional[int] = 100000) -> Dict[str, str]:
        """
        Download and preprocess C4 dataset.
        
        Args:
            max_samples: Maximum samples per split (C4 is very large)
        """
        logger.info(f"Preparing C4 (max {max_samples} samples per split)...")
        config = self.dataset_configs['c4']
        dataset_dir = self.output_dir / 'c4'
        dataset_dir.mkdir(exist_ok=True)
        
        output_paths = {}
        
        try:
            for split in config['splits']:
                logger.info(f"Processing C4 {split} split...")
                
                # Load dataset in streaming mode
                dataset = load_dataset(
                    config['hf_name'],
                    config['hf_config'],
                    split=split,
                    streaming=True
                )
                
                # Collect samples
                texts = []
                for i, item in enumerate(dataset):
                    if max_samples and i >= max_samples:
                        break
                    if item['text'].strip():
                        texts.append(item['text'])
                    
                    if (i + 1) % 10000 == 0:
                        logger.info(f"Collected {i + 1} samples...")
                
                logger.info(f"Tokenizing {len(texts)} documents...")
                tokenized = self.tokenizer(
                    texts,
                    truncation=False,
                    padding=False,
                    return_attention_mask=False
                )
                
                # Save tokenized data
                output_file = dataset_dir / f"{split}.pt"
                torch.save({
                    'input_ids': tokenized['input_ids'],
                    'num_documents': len(texts),
                    'total_tokens': sum(len(ids) for ids in tokenized['input_ids']),
                    'tokenizer': self.tokenizer.name_or_path,
                    'max_samples': max_samples
                }, output_file)
                
                output_paths[split] = str(output_file)
                logger.info(f"Saved {split} split to {output_file}")
            
            # Save metadata
            metadata = {
                'dataset': 'c4',
                'tokenizer': self.tokenizer.name_or_path,
                'vocab_size': self.tokenizer.vocab_size,
                'splits': output_paths,
                'max_samples': max_samples,
                'config': config
            }
            
            metadata_file = dataset_dir / 'metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("✓ C4 preparation complete")
            return output_paths
            
        except Exception as e:
            logger.error(f"Error preparing C4: {e}")
            raise
    
    def prepare_pile(self, max_samples: Optional[int] = 50000) -> Dict[str, str]:
        """
        Download and preprocess The Pile dataset.
        
        Args:
            max_samples: Maximum samples per split (The Pile is very large)
        """
        logger.info(f"Preparing The Pile (max {max_samples} samples per split)...")
        config = self.dataset_configs['pile']
        dataset_dir = self.output_dir / 'pile'
        dataset_dir.mkdir(exist_ok=True)
        
        output_paths = {}
        
        try:
            for split in config['splits']:
                logger.info(f"Processing Pile {split} split...")
                
                # Load dataset in streaming mode
                dataset = load_dataset(
                    config['hf_name'],
                    split=split,
                    streaming=True
                )
                
                # Collect samples
                texts = []
                for i, item in enumerate(dataset):
                    if max_samples and i >= max_samples:
                        break
                    if item['text'].strip():
                        texts.append(item['text'])
                    
                    if (i + 1) % 5000 == 0:
                        logger.info(f"Collected {i + 1} samples...")
                
                logger.info(f"Tokenizing {len(texts)} documents...")
                tokenized = self.tokenizer(
                    texts,
                    truncation=False,
                    padding=False,
                    return_attention_mask=False
                )
                
                # Save tokenized data
                output_file = dataset_dir / f"{split}.pt"
                torch.save({
                    'input_ids': tokenized['input_ids'],
                    'num_documents': len(texts),
                    'total_tokens': sum(len(ids) for ids in tokenized['input_ids']),
                    'tokenizer': self.tokenizer.name_or_path,
                    'max_samples': max_samples
                }, output_file)
                
                output_paths[split] = str(output_file)
                logger.info(f"Saved {split} split to {output_file}")
            
            # Save metadata
            metadata = {
                'dataset': 'pile',
                'tokenizer': self.tokenizer.name_or_path,
                'vocab_size': self.tokenizer.vocab_size,
                'splits': output_paths,
                'max_samples': max_samples,
                'config': config
            }
            
            metadata_file = dataset_dir / 'metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("✓ The Pile preparation complete")
            return output_paths
            
        except Exception as e:
            logger.error(f"Error preparing The Pile: {e}")
            raise
    
    def prepare_all(self, c4_samples: int = 100000, pile_samples: int = 50000):
        """Prepare all datasets."""
        logger.info("=" * 60)
        logger.info("Preparing all datasets for Mamba-Killer benchmarks")
        logger.info("=" * 60)
        
        results = {}
        
        # WikiText-2
        try:
            results['wikitext2'] = self.prepare_wikitext2()
        except Exception as e:
            logger.error(f"Failed to prepare WikiText-2: {e}")
            results['wikitext2'] = None
        
        # WikiText-103
        try:
            results['wikitext103'] = self.prepare_wikitext103()
        except Exception as e:
            logger.error(f"Failed to prepare WikiText-103: {e}")
            results['wikitext103'] = None
        
        # C4
        try:
            results['c4'] = self.prepare_c4(max_samples=c4_samples)
        except Exception as e:
            logger.error(f"Failed to prepare C4: {e}")
            results['c4'] = None
        
        # The Pile
        try:
            results['pile'] = self.prepare_pile(max_samples=pile_samples)
        except Exception as e:
            logger.error(f"Failed to prepare The Pile: {e}")
            results['pile'] = None
        
        # Save summary
        summary_file = self.output_dir / 'preparation_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("=" * 60)
        logger.info(f"Dataset preparation complete. Summary saved to {summary_file}")
        logger.info("=" * 60)
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Prepare datasets for Mamba-Killer ResNet-BK benchmarks"
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=['wikitext2', 'wikitext103', 'c4', 'pile'],
        help='Datasets to prepare'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Prepare all datasets'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data',
        help='Output directory for processed datasets'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='gpt2',
        help='Tokenizer to use (default: gpt2)'
    )
    parser.add_argument(
        '--c4_samples',
        type=int,
        default=100000,
        help='Maximum samples for C4 dataset'
    )
    parser.add_argument(
        '--pile_samples',
        type=int,
        default=50000,
        help='Maximum samples for The Pile dataset'
    )
    
    args = parser.parse_args()
    
    if not args.all and not args.datasets:
        parser.error("Must specify either --all or --datasets")
    
    preparator = DatasetPreparator(
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer
    )
    
    if args.all:
        preparator.prepare_all(
            c4_samples=args.c4_samples,
            pile_samples=args.pile_samples
        )
    else:
        for dataset in args.datasets:
            if dataset == 'wikitext2':
                preparator.prepare_wikitext2()
            elif dataset == 'wikitext103':
                preparator.prepare_wikitext103()
            elif dataset == 'c4':
                preparator.prepare_c4(max_samples=args.c4_samples)
            elif dataset == 'pile':
                preparator.prepare_pile(max_samples=args.pile_samples)


if __name__ == '__main__':
    main()
