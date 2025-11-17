"""
Hugging Face Transformers-compatible ResNet-BK Model

This module provides a transformers-compatible wrapper for ResNet-BK models,
enabling seamless integration with the Hugging Face ecosystem including:
- AutoModel/AutoConfig/AutoTokenizer
- Trainer API
- Model Hub upload/download
- Pipeline integration
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from dataclasses import dataclass
import os

try:
    from transformers import PreTrainedModel, PretrainedConfig
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.utils import ModelOutput
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    # Fallback for when transformers is not installed
    
    class PreTrainedModel(nn.Module):
        """Fallback PreTrainedModel when transformers is not available."""
        config_class = None
        base_model_prefix = "model"
        
        def __init__(self, config):
            super().__init__()
            self.config = config
        
        def save_pretrained(self, save_directory):
            """Save model and config."""
            os.makedirs(save_directory, exist_ok=True)
            
            # Save config
            self.config.save_pretrained(save_directory)
            
            # Save model weights
            model_path = os.path.join(save_directory, "pytorch_model.bin")
            torch.save(self.state_dict(), model_path)
        
        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
            """Load model from directory."""
            # Load config
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path)
            
            # Create model
            model = cls(config)
            
            # Load weights
            model_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            
            return model
        
        def post_init(self):
            """Post-initialization (for compatibility)."""
            pass
        
        @property
        def use_return_dict(self):
            """Whether to return dict (for compatibility)."""
            return True
    
    class PretrainedConfig:
        """Fallback config class when transformers is not available."""
        model_type = "resnet_bk"
        
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        def to_dict(self):
            return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        
        def save_pretrained(self, save_directory):
            import json
            os.makedirs(save_directory, exist_ok=True)
            with open(os.path.join(save_directory, "config.json"), "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        
        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
            import json
            config_file = os.path.join(pretrained_model_name_or_path, "config.json")
            with open(config_file, "r") as f:
                config_dict = json.load(f)
            return cls(**config_dict)
    
    @dataclass
    class CausalLMOutputWithPast:
        loss: Optional[torch.FloatTensor] = None
        logits: torch.FloatTensor = None
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
        hidden_states: Optional[Tuple[torch.FloatTensor]] = None
        attentions: Optional[Tuple[torch.FloatTensor]] = None

from .resnet_bk import LanguageModel


class ResNetBKConfig(PretrainedConfig):
    """
    Configuration class for ResNet-BK models.
    
    This class stores the configuration of a ResNet-BK model and is compatible
    with Hugging Face's configuration system.
    
    Args:
        vocab_size (int): Vocabulary size. Default: 30000
        d_model (int): Model dimension. Default: 256
        n_layers (int): Number of ResNet-BK blocks. Default: 8
        n_seq (int): Maximum sequence length. Default: 2048
        num_experts (int): Number of MoE experts. Default: 4
        top_k (int): Number of experts to route to. Default: 1
        dropout_p (float): Dropout probability. Default: 0.1
        use_scattering_router (bool): Use scattering-based routing. Default: False
        scattering_scale (float): Scale for scattering router. Default: 0.1
        use_birman_schwinger (bool): Use Birman-Schwinger core. Default: False
        epsilon (float): Regularization parameter for Birman-Schwinger. Default: 1.0
        use_mourre (bool): Enable Mourre estimate verification. Default: True
        use_lap (bool): Enable Limiting Absorption Principle. Default: True
        schatten_threshold (float): Threshold for Schatten norm clipping. Default: 100.0
        precision_upgrade_threshold (float): Condition number threshold for precision upgrade. Default: 1e6
        use_prime_bump (bool): Use Prime-Bump initialization. Default: False
        prime_bump_scale (float): Scale for Prime-Bump potential. Default: 0.02
        k_max (int): Maximum prime power for Prime-Bump. Default: 3
        use_semiseparable (bool): Use semiseparable matrix structure. Default: False
        low_rank (Optional[int]): Rank for semiseparable structure. Default: None (auto)
        use_act (bool): Use Adaptive Computation Time. Default: False
        act_halt_threshold (float): Halting threshold for ACT. Default: 0.2
        pad_token_id (int): Padding token ID. Default: 0
        bos_token_id (int): Beginning of sequence token ID. Default: 1
        eos_token_id (int): End of sequence token ID. Default: 2
    """
    
    model_type = "resnet_bk"
    
    def __init__(
        self,
        vocab_size: int = 30000,
        d_model: int = 256,
        n_layers: int = 8,
        n_seq: int = 2048,
        num_experts: int = 4,
        top_k: int = 1,
        dropout_p: float = 0.1,
        use_scattering_router: bool = False,
        scattering_scale: float = 0.1,
        scattering_scale_warmup_steps: int = 0,
        use_birman_schwinger: bool = False,
        epsilon: float = 1.0,
        use_mourre: bool = True,
        use_lap: bool = True,
        schatten_threshold: float = 100.0,
        precision_upgrade_threshold: float = 1e6,
        use_prime_bump: bool = False,
        prime_bump_scale: float = 0.02,
        k_max: int = 3,
        use_semiseparable: bool = False,
        low_rank: Optional[int] = None,
        use_act: bool = False,
        act_halt_threshold: float = 0.2,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_seq = n_seq
        self.num_experts = num_experts
        self.top_k = top_k
        self.dropout_p = dropout_p
        self.use_scattering_router = use_scattering_router
        self.scattering_scale = scattering_scale
        self.scattering_scale_warmup_steps = scattering_scale_warmup_steps
        self.use_birman_schwinger = use_birman_schwinger
        self.epsilon = epsilon
        self.use_mourre = use_mourre
        self.use_lap = use_lap
        self.schatten_threshold = schatten_threshold
        self.precision_upgrade_threshold = precision_upgrade_threshold
        self.use_prime_bump = use_prime_bump
        self.prime_bump_scale = prime_bump_scale
        self.k_max = k_max
        self.use_semiseparable = use_semiseparable
        self.low_rank = low_rank
        self.use_act = use_act
        self.act_halt_threshold = act_halt_threshold
        self.use_return_dict = True  # Always use return dict


class ResNetBKForCausalLM(PreTrainedModel):
    """
    ResNet-BK Model for Causal Language Modeling.
    
    This class wraps the ResNet-BK LanguageModel to be compatible with
    Hugging Face's PreTrainedModel interface.
    
    Example:
        ```python
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained("username/resnet-bk-1b")
        tokenizer = AutoTokenizer.from_pretrained("username/resnet-bk-1b")
        
        # Generate text
        inputs = tokenizer("The future of AI is", return_tensors="pt")
        outputs = model.generate(**inputs, max_length=50)
        print(tokenizer.decode(outputs[0]))
        ```
    """
    
    config_class = ResNetBKConfig
    base_model_prefix = "resnet_bk"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ResNetBKBlock"]
    
    def __init__(self, config: ResNetBKConfig):
        super().__init__(config)
        self.config = config
        
        # Create the underlying ResNet-BK model
        self.model = LanguageModel(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_seq=config.n_seq,
            num_experts=config.num_experts,
            top_k=config.top_k,
            dropout_p=config.dropout_p,
            use_scattering_router=config.use_scattering_router,
            scattering_scale=config.scattering_scale,
            scattering_scale_warmup_steps=config.scattering_scale_warmup_steps,
            use_birman_schwinger=config.use_birman_schwinger,
            epsilon=config.epsilon,
            use_mourre=config.use_mourre,
            use_lap=config.use_lap,
            schatten_threshold=config.schatten_threshold,
            precision_upgrade_threshold=config.precision_upgrade_threshold,
            prime_bump_init=config.use_prime_bump,  # Note: LanguageModel uses prime_bump_init
            prime_bump_scale=config.prime_bump_scale,
            k_max=config.k_max,
        )
        
        # Initialize weights
        self.post_init()
    
    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.model.token_embedding
    
    def set_input_embeddings(self, value):
        """Set input embeddings."""
        self.model.token_embedding = value
    
    def get_output_embeddings(self):
        """Get output embeddings (LM head)."""
        return self.model.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        """Set output embeddings (LM head)."""
        self.model.lm_head = new_embeddings
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass for causal language modeling.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length)
            attention_mask: Attention mask (currently not used by ResNet-BK)
            labels: Labels for language modeling loss
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return a ModelOutput object
            
        Returns:
            CausalLMOutputWithPast containing loss and logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Forward through the model
        logits = self.model(input_ids)
        
        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        **kwargs
    ):
        """
        Prepare inputs for generation.
        
        Args:
            input_ids: Input token IDs
            
        Returns:
            Dictionary of model inputs
        """
        return {
            "input_ids": input_ids,
        }
    
    @staticmethod
    def _reorder_cache(past, beam_idx):
        """
        Reorder cache for beam search (not applicable for ResNet-BK).
        """
        return past
    
    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None):
        """
        Resize token embeddings.
        
        Args:
            new_num_tokens: New vocabulary size
            
        Returns:
            New input embeddings
        """
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)
        
        # Also resize output embeddings (LM head)
        old_lm_head = self.get_output_embeddings()
        new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens)
        self.set_output_embeddings(new_lm_head)
        
        # Update config
        self.config.vocab_size = new_num_tokens
        
        return self.get_input_embeddings()
    
    def _get_resized_embeddings(
        self,
        old_embeddings: nn.Embedding,
        new_num_tokens: Optional[int] = None
    ) -> nn.Embedding:
        """
        Build a resized Embedding Module from a provided token Embedding Module.
        """
        if new_num_tokens is None:
            return old_embeddings
        
        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        
        if old_num_tokens == new_num_tokens:
            return old_embeddings
        
        # Create new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
        new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)
        
        # Copy old weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]
        
        return new_embeddings
    
    def _get_resized_lm_head(
        self,
        old_lm_head: nn.Linear,
        new_num_tokens: Optional[int] = None
    ) -> nn.Linear:
        """
        Build a resized Linear Module for LM head.
        """
        if new_num_tokens is None:
            return old_lm_head
        
        old_num_tokens, old_embedding_dim = old_lm_head.weight.size()
        
        if old_num_tokens == new_num_tokens:
            return old_lm_head
        
        # Create new LM head
        new_lm_head = nn.Linear(old_embedding_dim, new_num_tokens, bias=False)
        new_lm_head.to(old_lm_head.weight.device, dtype=old_lm_head.weight.dtype)
        
        # Copy old weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[:num_tokens_to_copy, :]
        
        return new_lm_head


# Register the model with transformers AutoModel
if HF_AVAILABLE:
    from transformers import AutoConfig, AutoModelForCausalLM
    
    AutoConfig.register("resnet_bk", ResNetBKConfig)
    AutoModelForCausalLM.register(ResNetBKConfig, ResNetBKForCausalLM)


def create_resnet_bk_for_hf(
    model_size: str = "1M",
    **config_kwargs
) -> ResNetBKForCausalLM:
    """
    Create a ResNet-BK model with predefined size configurations.
    
    Args:
        model_size: One of "1M", "10M", "100M", "1B", "10B"
        **config_kwargs: Additional configuration overrides
        
    Returns:
        ResNetBKForCausalLM model
        
    Example:
        ```python
        # Create a 1B parameter model
        model = create_resnet_bk_for_hf("1B", use_birman_schwinger=True)
        ```
    """
    # Predefined configurations for different model sizes
    size_configs = {
        "1M": {
            "d_model": 128,
            "n_layers": 4,
            "num_experts": 2,
            "n_seq": 512,
        },
        "10M": {
            "d_model": 256,
            "n_layers": 6,
            "num_experts": 4,
            "n_seq": 1024,
        },
        "100M": {
            "d_model": 512,
            "n_layers": 12,
            "num_experts": 8,
            "n_seq": 2048,
        },
        "1B": {
            "d_model": 1024,
            "n_layers": 24,
            "num_experts": 16,
            "n_seq": 4096,
        },
        "10B": {
            "d_model": 2048,
            "n_layers": 32,
            "num_experts": 32,
            "n_seq": 8192,
        },
    }
    
    if model_size not in size_configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(size_configs.keys())}")
    
    # Merge size config with user overrides
    config_dict = {**size_configs[model_size], **config_kwargs}
    
    # Create config and model
    config = ResNetBKConfig(**config_dict)
    model = ResNetBKForCausalLM(config)
    
    return model
