"""
Complex Tokenizer for LOGOS Architecture (Phase 4)

This module implements the "Sentiment Phase Shifting" logic.
It pre-processes text to detect emotional markers (punctuation, emojis)
and assigns an initial phase angle to the token sequence.

Logic:
    - '!' (Exclamation): +pi/4 (Emphasis/Intensity)
    - '?' (Question): +pi/2 (Uncertainty/Orthogonality)
    - Default: 0.0

The calculated phase is passed to the ComplexEmbedding layer to rotate
the initial meaning vector in the complex plane.
"""

import torch
import re
from typing import List, Tuple, Union, Dict

class ComplexTokenizer:
    """
    Tokenizer wrapper that adds physical phase information to tokens.
    """

    def __init__(self, base_tokenizer=None):
        """
        Args:
            base_tokenizer: Underlying tokenizer (e.g., HuggingFace tokenizer).
                            If None, assumes input is already tokenized or raw text handling is manual.
        """
        self.base_tokenizer = base_tokenizer

        # Phase shift constants
        self.PHASE_EMPHASIS = 3.14159 / 4.0  # pi/4
        self.PHASE_QUESTION = 3.14159 / 2.0  # pi/2
        self.PHASE_NEUTRAL = 0.0

    def get_phase_shift(self, text: str) -> float:
        """
        Determine the global phase shift for a text segment based on punctuation.

        Note: In a more advanced version, this could be per-token.
        For this implementation, we apply it to the whole segment or
        detect specific markers.
        """
        phase = self.PHASE_NEUTRAL

        # Check for exclamation marks (Emphasis)
        if '!' in text or '！' in text:
            phase += self.PHASE_EMPHASIS

        # Check for question marks (Uncertainty)
        if '?' in text or '？' in text:
            # Orthogonal rotation for questions
            phase += self.PHASE_QUESTION

        return phase

    def process_batch(self, texts: List[str], max_length: int, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """
        Process a batch of texts and return input_ids and initial_phases.

        Args:
            texts: List of input strings.
            max_length: Max sequence length.
            device: Torch device.

        Returns:
            Dict containing 'input_ids' and 'initial_phase'.
        """
        if self.base_tokenizer is None:
            raise ValueError("Base tokenizer not initialized.")

        encoded = self.base_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(device)
        batch_size, seq_len = input_ids.shape

        # Calculate phase for each text in batch
        # Currently applying one phase shift per sequence based on input text
        # (Broadcasting to all tokens in the sequence for the "Contextual Mood")
        phases = []
        for text in texts:
            p = self.get_phase_shift(text)
            phases.append(p)

        # Create tensor (B, N) - expanded to sequence length
        phase_tensor = torch.tensor(phases, device=device).unsqueeze(1).expand(batch_size, seq_len)

        # For more granular control (e.g. specific tokens), we would map tokens back to text spans.
        # But "Sentiment Phase Shifting" as described is often a global or sentence-level modulation.

        return {
            'input_ids': input_ids,
            'attention_mask': encoded.get('attention_mask').to(device) if 'attention_mask' in encoded else None,
            'initial_phase': phase_tensor
        }
