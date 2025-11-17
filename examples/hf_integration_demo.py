"""
Hugging Face Integration Demo

This script demonstrates how to use ResNet-BK models with Hugging Face
transformers, PyTorch Hub, and export to ONNX/TensorRT.
"""

import torch
import argparse
import os


def demo_huggingface_integration():
    """Demonstrate Hugging Face transformers integration."""
    print("\n" + "="*60)
    print("Demo 1: Hugging Face Transformers Integration")
    print("="*60)
    
    try:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from src.models.hf_resnet_bk import ResNetBKConfig, ResNetBKForCausalLM
    except ImportError as e:
        print(f"Skipping HF demo: {e}")
        return
    
    # Create a small model for demo
    print("\n1. Creating ResNet-BK model with Hugging Face API...")
    config = ResNetBKConfig(
        vocab_size=1000,
        d_model=128,
        n_layers=4,
        n_seq=256,
        num_experts=2,
    )
    model = ResNetBKForCausalLM(config)
    print(f"   Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    input_ids = torch.randint(0, 1000, (2, 128))
    outputs = model(input_ids)
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Output logits shape: {outputs.logits.shape}")
    
    # Test with labels (training mode)
    print("\n3. Testing with labels (computing loss)...")
    labels = torch.randint(0, 1000, (2, 128))
    outputs = model(input_ids, labels=labels)
    print(f"   Loss: {outputs.loss.item():.4f}")
    
    # Test generation
    print("\n4. Testing text generation...")
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            input_ids[:1, :10],
            max_length=50,
            do_sample=False,
        )
    print(f"   Generated sequence shape: {generated.shape}")
    
    # Save and load
    print("\n5. Testing save and load...")
    save_dir = "tmp_hf_model"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    print(f"   Model saved to {save_dir}")
    
    loaded_model = ResNetBKForCausalLM.from_pretrained(save_dir)
    print(f"   Model loaded successfully")
    
    # Cleanup
    import shutil
    shutil.rmtree(save_dir)
    
    print("\n✓ Hugging Face integration demo completed successfully!")


def demo_pytorch_hub():
    """Demonstrate PyTorch Hub integration."""
    print("\n" + "="*60)
    print("Demo 2: PyTorch Hub Integration")
    print("="*60)
    
    # Note: This would work after publishing to GitHub
    print("\n1. Loading model via PyTorch Hub...")
    print("   Example usage (after publishing):")
    print("   >>> model = torch.hub.load('username/resnet-bk', 'resnet_bk_1m')")
    print("   >>> model = torch.hub.load('username/resnet-bk', 'resnet_bk_1b', pretrained=True)")
    
    # Test local hubconf
    print("\n2. Testing local hubconf...")
    try:
        import sys
        sys.path.insert(0, '.')
        import hubconf
        
        model = hubconf.resnet_bk_1m(pretrained=False)
        print(f"   ✓ Created 1M model with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
        
        # Test custom model
        custom_model = hubconf.resnet_bk_custom(
            d_model=256,
            n_layers=6,
            use_birman_schwinger=True,
        )
        print(f"   ✓ Created custom model with {sum(p.numel() for p in custom_model.parameters())/1e6:.2f}M parameters")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n✓ PyTorch Hub demo completed!")


def demo_onnx_export():
    """Demonstrate ONNX export."""
    print("\n" + "="*60)
    print("Demo 3: ONNX Export")
    print("="*60)
    
    try:
        from src.models.hf_resnet_bk import create_resnet_bk_for_hf
        from src.models.onnx_export import export_to_onnx, verify_onnx_model
    except ImportError as e:
        print(f"Skipping ONNX demo: {e}")
        return
    
    # Create a small model
    print("\n1. Creating model for export...")
    model = create_resnet_bk_for_hf("1M")
    print(f"   Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Export to ONNX
    print("\n2. Exporting to ONNX...")
    output_path = "tmp_model.onnx"
    try:
        success = export_to_onnx(
            model,
            output_path,
            batch_size=1,
            seq_length=128,
            verify=True,
            tolerance=1e-5,
        )
        
        if success:
            print(f"   ✓ Model exported successfully to {output_path}")
            
            # Check file size
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"   File size: {file_size:.2f} MB")
            
            # Cleanup
            os.remove(output_path)
        else:
            print("   ✗ Export failed")
            
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n✓ ONNX export demo completed!")


def demo_tensorrt_export():
    """Demonstrate TensorRT export."""
    print("\n" + "="*60)
    print("Demo 4: TensorRT Export")
    print("="*60)
    
    try:
        import tensorrt as trt
        from src.models.hf_resnet_bk import create_resnet_bk_for_hf
        from src.models.onnx_export import export_to_onnx, export_to_tensorrt
    except ImportError as e:
        print(f"Skipping TensorRT demo: {e}")
        print("Note: TensorRT requires NVIDIA GPU and TensorRT installation")
        return
    
    # Create model
    print("\n1. Creating model...")
    model = create_resnet_bk_for_hf("1M")
    
    # Export to ONNX first
    print("\n2. Exporting to ONNX...")
    onnx_path = "tmp_model.onnx"
    export_to_onnx(model, onnx_path, verify=False)
    
    # Convert to TensorRT
    print("\n3. Converting to TensorRT...")
    trt_path = "tmp_model.trt"
    try:
        success = export_to_tensorrt(
            onnx_path,
            trt_path,
            fp16=True,
            max_batch_size=4,
            max_seq_length=512,
        )
        
        if success:
            print(f"   ✓ TensorRT engine created successfully")
            
            # Cleanup
            os.remove(trt_path)
        
    except Exception as e:
        print(f"   Error: {e}")
    
    # Cleanup
    if os.path.exists(onnx_path):
        os.remove(onnx_path)
    
    print("\n✓ TensorRT export demo completed!")


def demo_full_deployment_pipeline():
    """Demonstrate full deployment pipeline."""
    print("\n" + "="*60)
    print("Demo 5: Full Deployment Pipeline")
    print("="*60)
    
    try:
        from src.models.hf_resnet_bk import create_resnet_bk_for_hf
        from src.models.onnx_export import export_model_for_deployment
    except ImportError as e:
        print(f"Skipping deployment demo: {e}")
        return
    
    # Create model
    print("\n1. Creating model...")
    model = create_resnet_bk_for_hf("1M", use_birman_schwinger=True)
    print(f"   Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Export in all formats
    print("\n2. Exporting in multiple formats...")
    output_dir = "tmp_exports"
    try:
        exported_paths = export_model_for_deployment(
            model,
            output_dir,
            model_name="resnet_bk_1m",
            export_onnx=True,
            export_tensorrt=False,  # Skip TensorRT for demo
            optimize_onnx=True,
            verify=True,
        )
        
        print("\n3. Exported files:")
        for format_name, path in exported_paths.items():
            if os.path.exists(path):
                size = os.path.getsize(path) / (1024 * 1024)
                print(f"   {format_name}: {path} ({size:.2f} MB)")
        
        # Cleanup
        import shutil
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n✓ Deployment pipeline demo completed!")


def main():
    parser = argparse.ArgumentParser(description="ResNet-BK Hugging Face Integration Demo")
    parser.add_argument(
        "--demo",
        type=str,
        choices=["all", "hf", "hub", "onnx", "tensorrt", "deployment"],
        default="all",
        help="Which demo to run"
    )
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("ResNet-BK Hugging Face Integration Demo")
    print("="*60)
    
    if args.demo in ["all", "hf"]:
        demo_huggingface_integration()
    
    if args.demo in ["all", "hub"]:
        demo_pytorch_hub()
    
    if args.demo in ["all", "onnx"]:
        demo_onnx_export()
    
    if args.demo in ["all", "tensorrt"]:
        demo_tensorrt_export()
    
    if args.demo in ["all", "deployment"]:
        demo_full_deployment_pipeline()
    
    print("\n" + "="*60)
    print("All demos completed!")
    print("="*60)


if __name__ == "__main__":
    main()
