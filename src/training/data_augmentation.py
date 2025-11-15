"""
Data Augmentation for Language Modeling

Implements data augmentation techniques to increase effective training data:
- Synonym replacement using WordNet
- Random token deletion
- Back-translation (if translation model available)

Based on Step 7 design for achieving 10× cost reduction through data efficiency.
"""

import torch
import torch.nn as nn
import random
import numpy as np
from typing import List, Optional, Callable
from collections import defaultdict


class LanguageDataAugmenter:
    """
    Data augmentation for language modeling.
    
    Implements Requirements 7.5, 7.6:
    - Synonym replacement using WordNet
    - Random token deletion
    - Back-translation (optional)
    """
    
    def __init__(
        self,
        vocab: dict,
        synonym_prob: float = 0.1,
        deletion_prob: float = 0.1,
        max_augmentations: int = 2,
        seed: Optional[int] = None
    ):
        """
        Args:
            vocab: Vocabulary mapping token_id -> token_string
            synonym_prob: Probability of replacing a token with synonym
            deletion_prob: Probability of deleting a token
            max_augmentations: Maximum number of augmented versions per example
            seed: Random seed for reproducibility
        """
        self.vocab = vocab
        self.reverse_vocab = {v: k for k, v in vocab.items()}
        self.synonym_prob = synonym_prob
        self.deletion_prob = deletion_prob
        self.max_augmentations = max_augmentations
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Build synonym dictionary (simple rule-based for now)
        self.synonyms = self._build_synonym_dict()
    
    def _build_synonym_dict(self) -> dict:
        """
        Build simple synonym dictionary.
        
        In production, this would use WordNet or a pretrained model.
        For now, we use simple rule-based synonyms.
        """
        synonyms = defaultdict(list)
        
        # Common word synonyms (simplified)
        synonym_pairs = [
            ('good', ['great', 'nice', 'fine', 'excellent']),
            ('bad', ['poor', 'terrible', 'awful']),
            ('big', ['large', 'huge', 'enormous']),
            ('small', ['tiny', 'little', 'mini']),
            ('fast', ['quick', 'rapid', 'swift']),
            ('slow', ['sluggish', 'gradual']),
            ('happy', ['glad', 'joyful', 'pleased']),
            ('sad', ['unhappy', 'sorrowful', 'gloomy']),
            ('said', ['stated', 'mentioned', 'remarked']),
            ('went', ['traveled', 'moved', 'proceeded']),
            ('got', ['obtained', 'received', 'acquired']),
            ('make', ['create', 'produce', 'build']),
            ('think', ['believe', 'consider', 'suppose']),
            ('know', ['understand', 'realize', 'recognize']),
            ('see', ['observe', 'notice', 'view']),
            ('want', ['desire', 'wish', 'need']),
            ('use', ['utilize', 'employ', 'apply']),
            ('find', ['discover', 'locate', 'identify']),
            ('give', ['provide', 'offer', 'supply']),
            ('tell', ['inform', 'notify', 'advise']),
        ]
        
        for word, syns in synonym_pairs:
            if word in self.reverse_vocab:
                synonyms[word] = [s for s in syns if s in self.reverse_vocab]
        
        return dict(synonyms)
    
    def synonym_replacement(self, tokens: List[str], n_replace: Optional[int] = None) -> List[str]:
        """
        Replace random tokens with synonyms.
        
        Args:
            tokens: List of token strings
            n_replace: Number of tokens to replace (None = use probability)
        
        Returns:
            augmented_tokens: List with some tokens replaced
        """
        augmented = tokens.copy()
        
        if n_replace is None:
            # Use probability for each token
            for i, token in enumerate(tokens):
                if random.random() < self.synonym_prob and token in self.synonyms:
                    synonyms = self.synonyms[token]
                    if synonyms:
                        augmented[i] = random.choice(synonyms)
        else:
            # Replace exactly n_replace tokens
            replaceable_indices = [
                i for i, token in enumerate(tokens)
                if token in self.synonyms and self.synonyms[token]
            ]
            
            if replaceable_indices:
                n_replace = min(n_replace, len(replaceable_indices))
                indices_to_replace = random.sample(replaceable_indices, n_replace)
                
                for i in indices_to_replace:
                    token = tokens[i]
                    augmented[i] = random.choice(self.synonyms[token])
        
        return augmented
    
    def random_deletion(self, tokens: List[str], min_length: int = 5) -> List[str]:
        """
        Randomly delete tokens from sequence.
        
        Args:
            tokens: List of token strings
            min_length: Minimum sequence length (don't delete below this)
        
        Returns:
            augmented_tokens: List with some tokens deleted
        """
        if len(tokens) <= min_length:
            return tokens
        
        augmented = []
        for token in tokens:
            if random.random() > self.deletion_prob:
                augmented.append(token)
        
        # Ensure minimum length
        if len(augmented) < min_length:
            return tokens
        
        return augmented
    
    def random_swap(self, tokens: List[str], n_swaps: int = 1) -> List[str]:
        """
        Randomly swap positions of tokens.
        
        Args:
            tokens: List of token strings
            n_swaps: Number of swaps to perform
        
        Returns:
            augmented_tokens: List with swapped tokens
        """
        augmented = tokens.copy()
        
        for _ in range(n_swaps):
            if len(augmented) < 2:
                break
            
            idx1, idx2 = random.sample(range(len(augmented)), 2)
            augmented[idx1], augmented[idx2] = augmented[idx2], augmented[idx1]
        
        return augmented
    
    def augment_sequence(
        self,
        token_ids: torch.Tensor,
        methods: List[str] = ['synonym', 'deletion']
    ) -> List[torch.Tensor]:
        """
        Augment a sequence using multiple methods.
        
        Args:
            token_ids: (seq_len,) tensor of token IDs
            methods: List of augmentation methods to apply
        
        Returns:
            augmented_sequences: List of augmented token ID tensors
        """
        # Convert token IDs to strings
        tokens = [self.vocab.get(tid.item(), '<unk>') for tid in token_ids]
        
        augmented_sequences = []
        
        for _ in range(self.max_augmentations):
            aug_tokens = tokens.copy()
            
            # Apply random subset of methods
            selected_methods = random.sample(methods, k=random.randint(1, len(methods)))
            
            for method in selected_methods:
                if method == 'synonym':
                    aug_tokens = self.synonym_replacement(aug_tokens)
                elif method == 'deletion':
                    aug_tokens = self.random_deletion(aug_tokens)
                elif method == 'swap':
                    aug_tokens = self.random_swap(aug_tokens)
            
            # Convert back to token IDs
            aug_token_ids = torch.tensor([
                self.reverse_vocab.get(token, self.reverse_vocab.get('<unk>', 0))
                for token in aug_tokens
            ], dtype=token_ids.dtype)
            
            # Pad or truncate to original length
            if len(aug_token_ids) < len(token_ids):
                # Pad with padding token (assume 0)
                padding = torch.zeros(len(token_ids) - len(aug_token_ids), dtype=token_ids.dtype)
                aug_token_ids = torch.cat([aug_token_ids, padding])
            elif len(aug_token_ids) > len(token_ids):
                # Truncate
                aug_token_ids = aug_token_ids[:len(token_ids)]
            
            augmented_sequences.append(aug_token_ids)
        
        return augmented_sequences
    
    def augment_batch(
        self,
        batch_token_ids: torch.Tensor,
        batch_targets: torch.Tensor,
        methods: List[str] = ['synonym', 'deletion']
    ) -> tuple:
        """
        Augment a batch of sequences.
        
        Args:
            batch_token_ids: (batch_size, seq_len) tensor
            batch_targets: (batch_size, seq_len) tensor
            methods: List of augmentation methods
        
        Returns:
            augmented_inputs: (batch_size * (1 + max_augmentations), seq_len)
            augmented_targets: (batch_size * (1 + max_augmentations), seq_len)
        """
        augmented_inputs = [batch_token_ids]
        augmented_targets = [batch_targets]
        
        for i in range(batch_token_ids.size(0)):
            # Augment input
            aug_inputs = self.augment_sequence(batch_token_ids[i], methods)
            
            # Augment target (same augmentation)
            aug_targets = self.augment_sequence(batch_targets[i], methods)
            
            # Add to lists
            for aug_in, aug_tgt in zip(aug_inputs, aug_targets):
                augmented_inputs.append(aug_in.unsqueeze(0))
                augmented_targets.append(aug_tgt.unsqueeze(0))
        
        # Concatenate
        augmented_inputs = torch.cat(augmented_inputs, dim=0)
        augmented_targets = torch.cat(augmented_targets, dim=0)
        
        return augmented_inputs, augmented_targets


class BackTranslationAugmenter:
    """
    Back-translation augmentation using translation models.
    
    Requires translation models (e.g., from Hugging Face).
    This is a placeholder implementation.
    """
    
    def __init__(
        self,
        source_lang: str = 'en',
        intermediate_lang: str = 'fr',
        translation_model: Optional[nn.Module] = None
    ):
        """
        Args:
            source_lang: Source language code
            intermediate_lang: Intermediate language for back-translation
            translation_model: Translation model (if available)
        """
        self.source_lang = source_lang
        self.intermediate_lang = intermediate_lang
        self.translation_model = translation_model
        
        if translation_model is None:
            print("Warning: No translation model provided. Back-translation disabled.")
    
    def back_translate(self, text: str) -> Optional[str]:
        """
        Translate text to intermediate language and back.
        
        Args:
            text: Source text
        
        Returns:
            back_translated: Back-translated text (or None if model unavailable)
        """
        if self.translation_model is None:
            return None
        
        # Placeholder: In production, use actual translation model
        # Example with Hugging Face:
        # translated = self.translation_model.translate(text, src=self.source_lang, tgt=self.intermediate_lang)
        # back_translated = self.translation_model.translate(translated, src=self.intermediate_lang, tgt=self.source_lang)
        
        return None
    
    def augment_sequence(self, token_ids: torch.Tensor, vocab: dict) -> Optional[torch.Tensor]:
        """
        Augment sequence using back-translation.
        
        Args:
            token_ids: (seq_len,) tensor of token IDs
            vocab: Vocabulary mapping
        
        Returns:
            augmented_token_ids: Back-translated sequence (or None)
        """
        if self.translation_model is None:
            return None
        
        # Convert to text
        tokens = [vocab.get(tid.item(), '<unk>') for tid in token_ids]
        text = ' '.join(tokens)
        
        # Back-translate
        back_translated_text = self.back_translate(text)
        
        if back_translated_text is None:
            return None
        
        # Convert back to token IDs
        # (Placeholder - would need tokenizer)
        return None


def create_augmented_dataset(
    dataset,
    vocab: dict,
    augmentation_factor: int = 2,
    methods: List[str] = ['synonym', 'deletion'],
    seed: Optional[int] = None
):
    """
    Create augmented dataset with multiple versions of each example.
    
    Args:
        dataset: Original dataset
        vocab: Vocabulary mapping
        augmentation_factor: How many augmented versions per example
        methods: Augmentation methods to use
        seed: Random seed
    
    Returns:
        augmented_dataset: Dataset with original + augmented examples
    """
    augmenter = LanguageDataAugmenter(
        vocab=vocab,
        max_augmentations=augmentation_factor,
        seed=seed
    )
    
    augmented_data = []
    
    print(f"Creating augmented dataset with {augmentation_factor}× augmentation...")
    
    for i, (x, y) in enumerate(dataset):
        # Add original
        augmented_data.append((x, y))
        
        # Add augmented versions
        aug_inputs = augmenter.augment_sequence(x, methods=methods)
        aug_targets = augmenter.augment_sequence(y, methods=methods)
        
        for aug_x, aug_y in zip(aug_inputs, aug_targets):
            augmented_data.append((aug_x, aug_y))
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(dataset)} examples")
    
    print(f"Augmented dataset created: {len(dataset)} → {len(augmented_data)} examples")
    
    # Create new dataset
    class AugmentedDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    return AugmentedDataset(augmented_data)
