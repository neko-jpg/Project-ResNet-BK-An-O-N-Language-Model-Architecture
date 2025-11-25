"""
Configuration and Command-Line Argument Parsing
"""

import argparse
import sys
import yaml
from pathlib import Path
from src.models.configurable_resnet_bk import (
    ResNetBKConfig,
    BASELINE_CONFIG,
    STEP2_CONFIG,
    STEP4_CONFIG,
    STEP5_CONFIG,
    STEP6_CONFIG,
    FULL_CONFIG,
)


def parse_args():
    """Parse command-line arguments for ResNet-BK training."""
    parser = argparse.ArgumentParser(
        description="ResNet-BK: O(N) Language Model with 1B× Cost Reduction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration preset
    parser.add_argument(
        '--config-preset',
        type=str,
        default='baseline',
        choices=['baseline', 'step2', 'step4', 'step5', 'step6', 'full', 'custom'],
        help='Configuration preset to use'
    )

    # Config file
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to YAML configuration file (overrides defaults, overridden by CLI args)'
    )
    
    # Model architecture
    parser.add_argument('--vocab-size', type=int, default=30000, help='Vocabulary size')
    parser.add_argument('--d-model', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--n-layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--n-seq', type=int, default=128, help='Sequence length')
    parser.add_argument('--num-experts', type=int, default=4, help='Number of MoE experts')
    parser.add_argument('--top-k', type=int, default=1, help='Top-k experts to route to')
    parser.add_argument('--dropout-p', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--prime-bump-init', action='store_true', help='Enable prime-bump initialization for BK core embeddings')
    parser.add_argument('--prime-bump-scale', type=float, default=0.02, help='Std/offset scale used in prime-bump initialization')
    parser.add_argument('--use-scattering-router', action='store_true', help='Enable scattering-based router (token norm modulation)')
    parser.add_argument('--scattering-scale', type=float, default=0.1, help='Scaling factor for scattering-based router modulation')
    parser.add_argument('--scattering-scale-warmup-steps', type=int, default=0, help='Warmup steps to double scattering scale (0=off)')
    
    # Birman-Schwinger parameters
    parser.add_argument('--use-birman-schwinger', action='store_true', help='Enable Birman-Schwinger core with LAP stability')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Regularization parameter epsilon (0.5-1.0)')
    parser.add_argument('--use-mourre', action='store_true', default=True, help='Enable Mourre estimate verification')
    parser.add_argument('--use-lap', action='store_true', default=True, help='Enable Limiting Absorption Principle')
    parser.add_argument('--schatten-threshold', type=float, default=100.0, help='Threshold for automatic spectral clipping')
    parser.add_argument('--precision-upgrade-threshold', type=float, default=1e6, help='Condition number threshold for precision upgrade')
    parser.add_argument('--k-max', type=int, default=3, help='Maximum prime power for Prime-Bump potential')
    
    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=20, help='Batch size')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--grad-clip', type=float, default=0.5, help='Gradient clipping threshold')
    
    # Step 2: Learning algorithm
    parser.add_argument('--use-analytic-gradient', action='store_true', help='Enable analytic gradient')
    parser.add_argument('--grad-blend', type=float, default=0.5, help='Gradient blend factor')
    parser.add_argument('--use-koopman', action='store_true', help='Enable Koopman learning')
    parser.add_argument('--koopman-dim', type=int, default=256, help='Koopman space dimension')
    parser.add_argument('--use-physics-informed', action='store_true', help='Enable physics-informed learning')
    
    # Step 4: Compression
    parser.add_argument('--use-quantization', action='store_true', help='Enable quantization')
    parser.add_argument('--quantization-bits', type=int, default=8, choices=[4, 8, 16, 32], help='Quantization bits')
    parser.add_argument('--use-pruning', action='store_true', help='Enable pruning')
    parser.add_argument('--prune-threshold', type=float, default=0.05, help='Pruning threshold')
    parser.add_argument('--use-distillation', action='store_true', help='Enable distillation')
    
    # Step 5: Hardware
    parser.add_argument('--use-mixed-precision', action='store_true', help='Enable mixed precision')
    parser.add_argument('--use-custom-kernels', action='store_true', help='Enable custom CUDA kernels')
    parser.add_argument('--use-gradient-checkpointing', action='store_true', help='Enable gradient checkpointing')
    
    # Step 6: Algorithms
    parser.add_argument('--use-adaptive-computation', action='store_true', help='Enable adaptive computation')
    parser.add_argument('--use-multi-scale', action='store_true', help='Enable multi-scale processing')
    parser.add_argument('--use-learned-sparsity', action='store_true', help='Enable learned sparsity')
    
    # Step 7: System
    parser.add_argument('--use-curriculum-learning', action='store_true', help='Enable curriculum learning')
    parser.add_argument('--use-active-learning', action='store_true', help='Enable active learning')
    parser.add_argument('--use-gradient-caching', action='store_true', help='Enable gradient caching')
    
    # Numerical stability
    parser.add_argument('--v-max', type=float, default=3.0, help='Potential clipping range')
    parser.add_argument('--feature-clamp', type=float, default=10.0, help='Feature clipping range')
    
    # Data and logging
    parser.add_argument('--dataset', type=str, default='wikitext-2', help='Dataset name')
    parser.add_argument(
        '--dataset-mix-config',
        type=str,
        default='configs/dataset_mixing.yaml',
        help='YAML config for mixed binary datasets (prepared via prepare_datasets.py)',
    )
    parser.add_argument(
        '--use-mixed-datasets',
        action='store_true',
        help='Use mixed datasets defined in dataset-mix-config instead of a single dataset',
    )
    parser.add_argument('--data-limit', type=int, default=500000, help='Max tokens to use')
    parser.add_argument('--log-interval', type=int, default=50, help='Logging interval (steps)')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Checkpoint save directory')
    parser.add_argument('--resume-from', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Device
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], help='Device to use')
    
    return parser.parse_args()


def get_config_from_args(args):
    """
    Create a configuration object from command-line arguments.
    Dynamically returns a Phase 7 config if specified.
    
    Args:
        args: parsed arguments from parse_args()
    
    Returns:
        config: A configuration object (ResNetBKConfig or Namespace)
    """
    # Default to a standard config object unless overridden by YAML
    config = ResNetBKConfig()
    yaml_config = {}

    # ---------------------------------------------------------
    # Apply Configuration File (if provided)
    # ---------------------------------------------------------
    if args.config:
        try:
            with open(args.config, 'r') as f:
                yaml_config = yaml.safe_load(f)
            print(f"Loading configuration from {args.config}")
        except Exception as e:
            print(f"Warning: Failed to load config file {args.config}: {e}")

    # ---------------------------------------------------------
    # Decide Model Type and create appropriate config object
    # ---------------------------------------------------------
    model_type = yaml_config.get('model_type', 'phase3')

    if model_type == 'phase7':
        # For Phase 7, we use a flexible Namespace object that can hold any parameter.
        config = argparse.Namespace()
        config.model_type = 'phase7'
    else: # phase3 or default
        if args.config_preset == 'baseline':
            config = BASELINE_CONFIG
        elif args.config_preset == 'step2':
            config = STEP2_CONFIG
        elif args.config_preset == 'step4':
            config = STEP4_CONFIG
        elif args.config_preset == 'step5':
            config = STEP5_CONFIG
        elif args.config_preset == 'step6':
            config = STEP6_CONFIG
        elif args.config_preset == 'full':
            config = FULL_CONFIG
        else:  # custom
            config = ResNetBKConfig()

    # ---------------------------------------------------------
    # Update config from YAML and args
    # ---------------------------------------------------------
    def update_config_from_dict(cfg_obj, cfg_dict):
        for k, v in cfg_dict.items():
            # Set attribute on the config object (Namespace or ResNetBKConfig)
            setattr(cfg_obj, k, v)
            # Also update args so it reflects in logs and usage in train.py
            if hasattr(args, k):
                setattr(args, k, v)

    if yaml_config:
        update_config_from_dict(config, yaml_config)

    # ---------------------------------------------------------
    # Override with CLI arguments (Explicitly provided)
    # ---------------------------------------------------------
    # We check sys.argv to see if the user explicitly provided an argument.
    # If so, we ensure it overrides whatever came from the config file.
    # Note: Boolean flags are tricky because store_true/false.

    def is_arg_passed(arg_name):
        return any(arg == f'--{arg_name}' or arg.startswith(f'--{arg_name}=') for arg in sys.argv)

    # Manual mapping of args to config

    if is_arg_passed('vocab-size'): config.vocab_size = args.vocab_size
    if is_arg_passed('d-model'): config.d_model = args.d_model
    if is_arg_passed('n-layers'): config.n_layers = args.n_layers
    if is_arg_passed('n-seq'): config.n_seq = args.n_seq
    if is_arg_passed('num-experts'): config.num_experts = args.num_experts
    if is_arg_passed('top-k'): config.top_k = args.top_k
    if is_arg_passed('dropout-p'): config.dropout_p = args.dropout_p

    if is_arg_passed('prime-bump-init'): config.prime_bump_init = args.prime_bump_init
    if is_arg_passed('prime-bump-scale'): config.prime_bump_scale = args.prime_bump_scale
    if is_arg_passed('use-scattering-router'): config.use_scattering_router = args.use_scattering_router
    if is_arg_passed('scattering-scale'): config.scattering_scale = args.scattering_scale
    if is_arg_passed('scattering-scale-warmup-steps'): config.scattering_scale_warmup_steps = args.scattering_scale_warmup_steps
    
    # Step 2
    if is_arg_passed('use-analytic-gradient'): config.use_analytic_gradient = args.use_analytic_gradient
    if is_arg_passed('grad-blend'): config.grad_blend = args.grad_blend
    if is_arg_passed('use-koopman'): config.use_koopman = args.use_koopman
    if is_arg_passed('koopman-dim'): config.koopman_dim = args.koopman_dim
    if is_arg_passed('use-physics-informed'): config.use_physics_informed = args.use_physics_informed
    
    # Step 4
    if is_arg_passed('use-quantization'): config.use_quantization = args.use_quantization
    if is_arg_passed('quantization-bits'): config.quantization_bits = args.quantization_bits
    if is_arg_passed('use-pruning'): config.use_pruning = args.use_pruning
    if is_arg_passed('prune-threshold'): config.prune_threshold = args.prune_threshold
    if is_arg_passed('use-distillation'): config.use_distillation = args.use_distillation
    
    # Step 5
    if is_arg_passed('use-mixed-precision'): config.use_mixed_precision = args.use_mixed_precision
    if is_arg_passed('use-custom-kernels'): config.use_custom_kernels = args.use_custom_kernels
    if is_arg_passed('use-gradient-checkpointing'): config.use_gradient_checkpointing = args.use_gradient_checkpointing
    
    # Step 6
    if is_arg_passed('use-adaptive-computation'): config.use_adaptive_computation = args.use_adaptive_computation
    if is_arg_passed('use-multi-scale'): config.use_multi_scale = args.use_multi_scale
    if is_arg_passed('use-learned-sparsity'): config.use_learned_sparsity = args.use_learned_sparsity
    
    # Step 7
    if is_arg_passed('use-curriculum-learning'): config.use_curriculum_learning = args.use_curriculum_learning
    if is_arg_passed('use-active-learning'): config.use_active_learning = args.use_active_learning
    if is_arg_passed('use-gradient-caching'): config.use_gradient_caching = args.use_gradient_caching
    
    # Numerical stability
    if is_arg_passed('v-max'): config.v_max = args.v_max
    if is_arg_passed('feature-clamp'): config.feature_clamp = args.feature_clamp
    if is_arg_passed('grad-clip'): config.grad_clip = args.grad_clip
    
    return config
