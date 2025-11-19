"""
Phase 1 Integration Demo

Phase 1統合レイヤーの使用例を示すデモスクリプト。
既存モデルをPhase 1に変換し、効率化機能を有効化する方法を説明します。

Requirements:
    - Task 9: Create integration layer and factory functions

Author: Project MUSE Team
"""

import torch
import torch.nn as nn

from src.models.phase1 import (
    # Configuration
    Phase1Config,
    get_preset_8gb,
    get_preset_10gb,
    get_preset_inference,
    list_presets,
    print_preset_comparison,
    
    # Factory functions
    create_phase1_model,
    Phase1IntegratedModel,
    
    # Conversion utilities
    convert_all_embeddings_to_htt,
    initialize_htt_from_embedding,
    get_conversion_summary,
    print_conversion_summary,
    verify_conversion,
)


def demo_simple_model_conversion():
    """シンプルなモデルをPhase 1に変換するデモ"""
    print("=" * 80)
    print("Demo 1: Simple Model Conversion")
    print("=" * 80)
    print()
    
    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self, vocab_size=10000, d_model=512):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.linear1 = nn.Linear(d_model, d_model)
            self.linear2 = nn.Linear(d_model, d_model)
            self.output = nn.Linear(d_model, vocab_size)
        
        def forward(self, input_ids):
            x = self.embedding(input_ids)
            x = torch.relu(self.linear1(x))
            x = torch.relu(self.linear2(x))
            return self.output(x)
    
    model = SimpleModel()
    print(f"Original model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Convert to Phase 1
    config = get_preset_8gb()
    converted_model, info = convert_all_embeddings_to_htt(
        model,
        config,
        initialize_from_weights=False,
    )
    
    print()
    print(f"Converted {info['num_converted']} embeddings")
    print(f"Parameter reduction: {info['overall_compression_percentage']:.1f}%")
    print()
    
    # Test forward pass
    input_ids = torch.randint(0, 10000, (2, 32))
    output = converted_model(input_ids)
    print(f"Output shape: {output.shape}")
    print()


def demo_configuration_presets():
    """設定プリセットのデモ"""
    print("=" * 80)
    print("Demo 2: Configuration Presets")
    print("=" * 80)
    print()
    
    # List available presets
    print("Available presets:")
    presets = list_presets()
    for name, desc in presets.items():
        print(f"  - {name}: {desc}")
    print()
    
    # Print comparison table
    print_preset_comparison()
    print()
    
    # Use specific presets
    config_8gb = get_preset_8gb()
    print(f"8GB preset: AR-SSM rank={config_8gb.ar_ssm_max_rank}, "
          f"HTT rank={config_8gb.htt_rank}, "
          f"compression={config_8gb.htt_compression_target:.1%}")
    
    config_inference = get_preset_inference()
    print(f"Inference preset: LNS enabled={config_inference.lns_enabled}, "
          f"checkpointing={config_inference.use_gradient_checkpointing}")
    print()


def demo_phase1_integrated_model():
    """Phase1IntegratedModelのデモ"""
    print("=" * 80)
    print("Demo 3: Phase1IntegratedModel")
    print("=" * 80)
    print()
    
    # Create a simple base model
    class SimpleModel(nn.Module):
        def __init__(self, vocab_size=1000, d_model=128):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.linear = nn.Linear(d_model, d_model)
            self.output = nn.Linear(d_model, vocab_size)
        
        def forward(self, input_ids):
            x = self.embedding(input_ids)
            x = torch.relu(self.linear(x))
            return self.output(x)
    
    base_model = SimpleModel()
    
    # Wrap with Phase1IntegratedModel
    config = Phase1Config(
        ar_ssm_enabled=True,
        ar_ssm_max_rank=16,
        htt_enabled=True,
        htt_rank=8,
        stability_monitoring_enabled=True,
    )
    
    integrated_model = Phase1IntegratedModel(
        base_model=base_model,
        config=config,
        replace_embeddings=True,
        enable_stability_monitoring=True,
        enable_gradient_monitoring=True,
    )
    
    print(integrated_model.get_phase1_summary())
    print()
    
    # Forward pass with diagnostics
    input_ids = torch.randint(0, 1000, (2, 16))
    output, diagnostics = integrated_model(input_ids, return_diagnostics=True)
    
    print("Diagnostics:")
    print(f"  HTT compression ratio: {diagnostics.htt_compression_ratio:.2%}")
    print(f"  Forward time: {diagnostics.forward_time_ms:.2f} ms")
    print(f"  Peak VRAM: {diagnostics.peak_vram_mb:.1f} MB")
    print()


def demo_create_phase1_model_from_scratch():
    """Phase 1モデルをゼロから作成するデモ"""
    print("=" * 80)
    print("Demo 4: Create Phase 1 Model from Scratch")
    print("=" * 80)
    print()
    
    # Check if LanguageModel is available
    try:
        from src.models.resnet_bk import LanguageModel
    except ImportError:
        print("LanguageModel not available. Skipping this demo.")
        print()
        return
    
    # Create Phase 1 model with preset
    config = get_preset_10gb()
    
    model = create_phase1_model(
        vocab_size=10000,
        d_model=512,
        n_layers=6,
        n_seq=128,
        config=config,
        model_type="resnet_bk",
    )
    
    print()
    
    # Test forward pass
    input_ids = torch.randint(0, 10000, (2, 128))
    output, diagnostics = model(input_ids, return_diagnostics=True)
    
    print(f"Output shape: {output.shape}")
    print(f"HTT compression: {diagnostics.htt_compression_ratio:.2%}")
    print()


def demo_model_conversion_with_initialization():
    """重み初期化付きモデル変換のデモ"""
    print("=" * 80)
    print("Demo 5: Model Conversion with Weight Initialization")
    print("=" * 80)
    print()
    
    # Create and train a simple model
    class SimpleModel(nn.Module):
        def __init__(self, vocab_size=1000, d_model=128):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.linear = nn.Linear(d_model, vocab_size)
        
        def forward(self, input_ids):
            x = self.embedding(input_ids)
            return self.linear(x)
    
    model = SimpleModel()
    
    # Simulate training
    print("Training original model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for i in range(10):
        input_ids = torch.randint(0, 1000, (4, 16))
        output = model(input_ids)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % 5 == 0:
            print(f"  Step {i}: loss={loss.item():.4f}")
    print()
    
    # Convert with SVD initialization
    config = Phase1Config(htt_rank=8)
    converted_model, info = convert_all_embeddings_to_htt(
        model,
        config,
        initialize_from_weights=True,
        initialization_method="svd",
    )
    
    print()
    print("Conversion complete!")
    print(f"Compression: {info['overall_compression_percentage']:.1f}%")
    print()
    
    # Verify conversion
    test_input = torch.randint(0, 1000, (2, 16))
    result = verify_conversion(model, converted_model, test_input, tolerance=1e-1)
    
    if result['outputs_close']:
        print("✓ Conversion verified successfully!")
    else:
        print(f"⚠ Outputs differ: max_diff={result['max_diff']:.6f}")
    print()


def demo_conversion_summary():
    """変換サマリーのデモ"""
    print("=" * 80)
    print("Demo 6: Conversion Summary")
    print("=" * 80)
    print()
    
    # Create a model with multiple embeddings
    class MultiEmbeddingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_embedding = nn.Embedding(10000, 512)
            self.position_embedding = nn.Embedding(1024, 512)
            self.linear = nn.Linear(512, 10000)
        
        def forward(self, input_ids):
            token_emb = self.token_embedding(input_ids)
            pos_ids = torch.arange(input_ids.size(1), device=input_ids.device)
            pos_emb = self.position_embedding(pos_ids)
            x = token_emb + pos_emb
            return self.linear(x)
    
    model = MultiEmbeddingModel()
    
    print("Before conversion:")
    print_conversion_summary(model)
    print()
    
    # Convert
    config = Phase1Config(htt_rank=16)
    converted_model, _ = convert_all_embeddings_to_htt(
        model,
        config,
        initialize_from_weights=False,
    )
    
    print()
    print("After conversion:")
    print_conversion_summary(converted_model)
    print()


def main():
    """すべてのデモを実行"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "Phase 1 Integration Demo" + " " * 34 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    demos = [
        demo_simple_model_conversion,
        demo_configuration_presets,
        demo_phase1_integrated_model,
        demo_create_phase1_model_from_scratch,
        demo_model_conversion_with_initialization,
        demo_conversion_summary,
    ]
    
    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"Error in {demo.__name__}: {e}")
            print()
    
    print("=" * 80)
    print("All demos completed!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Try different configuration presets for your hardware")
    print("  2. Convert your existing models to Phase 1")
    print("  3. Benchmark memory usage and throughput")
    print("  4. Fine-tune hyperparameters for your use case")
    print()


if __name__ == "__main__":
    main()
