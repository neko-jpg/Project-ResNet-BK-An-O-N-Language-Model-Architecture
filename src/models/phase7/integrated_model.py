"""
Phase 7 Integrated Model with HTT Embedding and Hybrid Hyperbolic Attention

This model integrates all core Phase 7 components:
1.  HolographicTTEmbedding for extreme parameter compression.
2.  The core ResNet-BK architecture for the main body.
3.  HybridHyperbolicAttention enabled within the ResNet-BK layers for
    physically-grounded dynamic routing.
"""
import torch
import torch.nn as nn
from dataclasses import dataclass

from src.models.phase1.htt_embedding import HolographicTTEmbedding
from src.models.resnet_bk import LanguageModel, ResNetBKConfig

@dataclass
class Phase7Config(ResNetBKConfig):
    """
    Configuration for the Phase 7 Integrated Model.
    Inherits from ResNetBKConfig and adds Phase 7 specific parameters.
    """
    htt_rank: int = 16 # Rank for the HolographicTTEmbedding

class Phase7IntegratedModel(nn.Module):
    """
    The complete, integrated model for Phase 7.
    It uses the ResNetBK LanguageModel as its core but replaces the standard
    embedding with the highly efficient HolographicTTEmbedding.
    """
    def __init__(self, config: Phase7Config):
        super().__init__()
        self.config = config

        # 1. Instantiate the core LanguageModel, ensuring Hybrid Attention is enabled
        # We pass a copy of the config, converted to a dict, after forcing hybrid attention
        model_config = config.__dict__.copy()
        model_config['use_hybrid_attention'] = True

        # Pop htt_rank as it's not a parameter for the base LanguageModel
        model_config.pop('htt_rank', None)

        # Ensure the config object passed to LanguageModel is of the correct base type
        base_config = ResNetBKConfig(**model_config)
        self.model = LanguageModel(config=base_config)

        # 2. Instantiate the HTT Embedding
        self.htt_embedding = HolographicTTEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            rank=config.htt_rank
        )

        # 3. Replace the standard embedding layer with our HTT embedding
        self.model.token_embedding = self.htt_embedding

    def forward(self, input_ids: torch.Tensor, return_diagnostics: bool = False) -> torch.Tensor:
        """
        Forward pass of the model. Delegates to the core LanguageModel.
        """
        # The base LanguageModel does not have a `return_diagnostics` flag in its forward
        # but diagnostics can be fetched from layers after the forward pass.
        # This wrapper can be extended to provide that functionality if needed.
        return self.model(input_ids)

    def get_total_parameter_count(self):
        """
        Returns the total number of parameters in the model.
        """
        return sum(p.numel() for p in self.parameters())

    def get_embedding_parameter_count(self):
        """
        Returns the number of parameters in the HTT embedding layer.
        """
        return sum(p.numel() for p in self.htt_embedding.parameters())
