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

from src.models.phase1.htt_embedding import HolographicTTEmbedding, HTTDecoder
from src.models.resnet_bk import LanguageModel, ResNetBKConfig

@dataclass
class Phase7Config(ResNetBKConfig):
    """
    Configuration for the Phase 7 Integrated Model.
    Inherits from ResNetBKConfig and adds Phase 7 specific parameters.
    
    Phase 7 combines:
    - HTT Embedding for extreme parameter compression
    - Hybrid Hyperbolic Attention for hierarchical relationships
    - AR-SSM for efficient global context
    """
    htt_rank: int = 16  # Rank for the HolographicTTEmbedding
    
    # Override defaults for Phase 7
    use_hybrid_attention: bool = True
    hyperbolic_window_size: int = 64
    num_heads: int = 8
    use_triton_kernel: bool = True
    triton_kernel_version: str = 'fast'
    use_gradient_checkpointing: bool = True
    use_mixed_precision: bool = True

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
        # Use dataclasses.asdict() to properly get all inherited fields
        from dataclasses import asdict, fields
        
        # Get all fields from the config (including inherited ones)
        model_config = {}
        for f in fields(config):
            model_config[f.name] = getattr(config, f.name)
        
        model_config['use_hybrid_attention'] = True

        # Pop htt_rank as it's not a parameter for the base LanguageModel
        model_config.pop('htt_rank', None)

        # Filter to only include fields that exist in ResNetBKConfig
        resnet_bk_fields = {f.name for f in fields(ResNetBKConfig)}
        filtered_config = {k: v for k, v in model_config.items() if k in resnet_bk_fields}

        # Ensure the config object passed to LanguageModel is of the correct base type
        base_config = ResNetBKConfig(**filtered_config)
        self.model = LanguageModel(config=base_config)

        # 2. Instantiate the HTT Embedding
        # use_complex_phase=False for stability (complex exp(iθ) causes NaN in mixed precision)
        self.htt_embedding = HolographicTTEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            rank=config.htt_rank,
            use_complex_phase=False,  # cos(θ) approximation for stability
        )
        self.htt_embedding.use_triton_kernel = config.use_triton_kernel

        # 3. Replace the standard embedding layer with our HTT embedding
        self.model.token_embedding = self.htt_embedding

        # 4. Replace the standard lm_head with the HTT decoder
        self.model.lm_head = HTTDecoder(self.htt_embedding)

    def forward(self, input_ids: torch.Tensor, return_diagnostics: bool = False) -> torch.Tensor:
        """
        Forward pass of the model. Delegates to the core LanguageModel.
        
        Args:
            input_ids: Input token IDs. Shape: (batch, seq_len)
            return_diagnostics: If True, returns diagnostics dict (not yet implemented)
        
        Returns:
            logits: Output logits. Shape: (batch, seq_len, vocab_size)
        """
        # The base LanguageModel handles the forward pass
        # Diagnostics can be fetched from layers after the forward pass if needed
        logits = self.model(input_ids)
        
        if return_diagnostics:
            # Collect diagnostics from hybrid attention layers if available
            diagnostics = {}
            for i, block in enumerate(self.model.blocks):
                if hasattr(block, 'bk_layer') and hasattr(block.bk_layer, 'last_hybrid_diagnostics'):
                    diagnostics[f'block_{i}'] = block.bk_layer.last_hybrid_diagnostics
            return logits, diagnostics
        
        return logits

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
