"""
Configuration and Command-Line Argument Parsing
"""

import argparse
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
    parser.add_argument('--data-limit', type=int, default=500000, help='Max tokens to use')
    parser.add_argument('--log-interval', type=int, default=50, help='Logging interval (steps)')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Checkpoint save directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Device
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], help='Device to use')
    
    return parser.parse_args()


def get_config_from_args(args):
    """
    Create ResNetBKConfig from command-line arguments.
    
    Args:
        args: parsed arguments from parse_args()
    
    Returns:
        config: ResNetBKConfig instance
    """
    # Start with preset if not custom
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
    
    # Override with command-line arguments
    config.vocab_size = args.vocab_size
    config.d_model = args.d_model
    config.n_layers = args.n_layers
    config.n_seq = args.n_seq
    config.num_experts = args.num_experts
    config.top_k = args.top_k
    config.dropout_p = args.dropout_p
    config.prime_bump_init = args.prime_bump_init
    config.prime_bump_scale = args.prime_bump_scale
    
    # Step 2
    if args.use_analytic_gradient:
        config.use_analytic_gradient = True
    config.grad_blend = args.grad_blend
    if args.use_koopman:
        config.use_koopman = True
    config.koopman_dim = args.koopman_dim
    if args.use_physics_informed:
        config.use_physics_informed = True
    
    # Step 4
    if args.use_quantization:
        config.use_quantization = True
    config.quantization_bits = args.quantization_bits
    if args.use_pruning:
        config.use_pruning = True
    config.prune_threshold = args.prune_threshold
    if args.use_distillation:
        config.use_distillation = True
    
    # Step 5
    if args.use_mixed_precision:
        config.use_mixed_precision = True
    if args.use_custom_kernels:
        config.use_custom_kernels = True
    if args.use_gradient_checkpointing:
        config.use_gradient_checkpointing = True
    
    # Step 6
    if args.use_adaptive_computation:
        config.use_adaptive_computation = True
    if args.use_multi_scale:
        config.use_multi_scale = True
    if args.use_learned_sparsity:
        config.use_learned_sparsity = True
    
    # Step 7
    if args.use_curriculum_learning:
        config.use_curriculum_learning = True
    if args.use_active_learning:
        config.use_active_learning = True
    if args.use_gradient_caching:
        config.use_gradient_caching = True
    
    # Numerical stability
    config.v_max = args.v_max
    config.feature_clamp = args.feature_clamp
    config.grad_clip = args.grad_clip
    
    return config
