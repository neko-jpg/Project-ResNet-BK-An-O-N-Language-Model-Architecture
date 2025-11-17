"""
ONNX and TensorRT Export for ResNet-BK Models

This module provides utilities for exporting ResNet-BK models to ONNX format
and optimizing them with TensorRT for deployment.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Tuple, List
import os
import warnings


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    batch_size: int = 1,
    seq_length: int = 512,
    opset_version: int = 14,
    dynamic_axes: bool = True,
    verify: bool = True,
    tolerance: float = 1e-5,
) -> bool:
    """
    Export ResNet-BK model to ONNX format.
    
    Args:
        model: ResNet-BK model to export
        output_path: Path to save ONNX model
        batch_size: Batch size for dummy input
        seq_length: Sequence length for dummy input
        opset_version: ONNX opset version
        dynamic_axes: Enable dynamic batch and sequence dimensions
        verify: Verify numerical equivalence after export
        tolerance: Maximum allowed error for verification
        
    Returns:
        True if export successful, False otherwise
        
    Example:
        ```python
        from src.models.hf_resnet_bk import create_resnet_bk_for_hf
        from src.models.onnx_export import export_to_onnx
        
        model = create_resnet_bk_for_hf("100M")
        export_to_onnx(model, "resnet_bk_100m.onnx")
        ```
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        raise ImportError(
            "ONNX export requires onnx and onnxruntime. "
            "Install with: pip install onnx onnxruntime"
        )
    
    # Set model to eval mode
    model.eval()
    device = next(model.parameters()).device
    
    # Create dummy input
    dummy_input = torch.randint(0, 1000, (batch_size, seq_length), dtype=torch.long).to(device)
    
    # Define dynamic axes if requested
    dynamic_axes_dict = None
    if dynamic_axes:
        dynamic_axes_dict = {
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size', 1: 'sequence_length'}
        }
    
    # Export to ONNX
    print(f"Exporting model to ONNX format...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input_ids'],
            output_names=['logits'],
            dynamic_axes=dynamic_axes_dict,
        )
        print(f"Model exported to {output_path}")
    except Exception as e:
        print(f"Error during ONNX export: {e}")
        return False
    
    # Verify the exported model
    if verify:
        print("Verifying ONNX model...")
        try:
            # Load ONNX model
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX model is valid")
            
            # Verify numerical equivalence
            is_equivalent = verify_onnx_model(
                model, output_path, dummy_input, tolerance=tolerance
            )
            
            if is_equivalent:
                print(f"✓ Numerical verification passed (max error < {tolerance})")
                return True
            else:
                print(f"✗ Numerical verification failed (max error >= {tolerance})")
                return False
                
        except Exception as e:
            print(f"Error during verification: {e}")
            return False
    
    return True


def verify_onnx_model(
    pytorch_model: nn.Module,
    onnx_path: str,
    test_input: torch.Tensor,
    tolerance: float = 1e-5,
) -> bool:
    """
    Verify that ONNX model produces same outputs as PyTorch model.
    
    Args:
        pytorch_model: Original PyTorch model
        onnx_path: Path to ONNX model
        test_input: Test input tensor
        tolerance: Maximum allowed error
        
    Returns:
        True if models are equivalent within tolerance
    """
    try:
        import onnxruntime as ort
    except ImportError:
        warnings.warn("onnxruntime not available, skipping verification")
        return True
    
    # Get PyTorch output
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input)
        if hasattr(pytorch_output, 'logits'):
            pytorch_output = pytorch_output.logits
        pytorch_output = pytorch_output.cpu().numpy()
    
    # Get ONNX output
    ort_session = ort.InferenceSession(onnx_path)
    onnx_input = {ort_session.get_inputs()[0].name: test_input.cpu().numpy()}
    onnx_output = ort_session.run(None, onnx_input)[0]
    
    # Compare outputs
    max_error = np.abs(pytorch_output - onnx_output).max()
    mean_error = np.abs(pytorch_output - onnx_output).mean()
    
    print(f"Max error: {max_error:.2e}")
    print(f"Mean error: {mean_error:.2e}")
    
    return max_error < tolerance


def optimize_onnx_model(
    onnx_path: str,
    output_path: Optional[str] = None,
    optimization_level: str = "all",
) -> str:
    """
    Optimize ONNX model for inference.
    
    Args:
        onnx_path: Path to input ONNX model
        output_path: Path to save optimized model (default: add _optimized suffix)
        optimization_level: Optimization level ("basic", "extended", "all")
        
    Returns:
        Path to optimized model
    """
    try:
        from onnxruntime.transformers import optimizer
    except ImportError:
        warnings.warn("onnxruntime.transformers not available, skipping optimization")
        return onnx_path
    
    if output_path is None:
        base, ext = os.path.splitext(onnx_path)
        output_path = f"{base}_optimized{ext}"
    
    print(f"Optimizing ONNX model with level: {optimization_level}")
    
    # Map optimization level to numeric value
    opt_level_map = {
        "basic": 1,
        "extended": 2,
        "all": 99,
    }
    opt_level = opt_level_map.get(optimization_level, 99)
    
    # Optimize model
    try:
        optimized_model = optimizer.optimize_model(
            onnx_path,
            model_type='bert',  # Use generic transformer optimizations
            num_heads=0,  # Auto-detect
            hidden_size=0,  # Auto-detect
            optimization_options=None,
        )
        optimized_model.save_model_to_file(output_path)
        print(f"Optimized model saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Optimization failed: {e}")
        return onnx_path


def export_to_tensorrt(
    onnx_path: str,
    output_path: str,
    fp16: bool = True,
    int8: bool = False,
    max_batch_size: int = 8,
    max_seq_length: int = 2048,
    workspace_size: int = 4,
) -> bool:
    """
    Convert ONNX model to TensorRT engine for optimized inference.
    
    Args:
        onnx_path: Path to ONNX model
        output_path: Path to save TensorRT engine
        fp16: Enable FP16 precision
        int8: Enable INT8 precision (requires calibration)
        max_batch_size: Maximum batch size
        max_seq_length: Maximum sequence length
        workspace_size: Workspace size in GB
        
    Returns:
        True if conversion successful
        
    Example:
        ```python
        # First export to ONNX
        export_to_onnx(model, "model.onnx")
        
        # Then convert to TensorRT
        export_to_tensorrt("model.onnx", "model.trt", fp16=True)
        ```
    """
    try:
        import tensorrt as trt
    except ImportError:
        raise ImportError(
            "TensorRT export requires tensorrt. "
            "Install from: https://developer.nvidia.com/tensorrt"
        )
    
    print(f"Converting ONNX model to TensorRT engine...")
    print(f"  FP16: {fp16}")
    print(f"  INT8: {int8}")
    print(f"  Max batch size: {max_batch_size}")
    print(f"  Max sequence length: {max_seq_length}")
    
    # Create TensorRT logger
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    # Create builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX model
    with open(onnx_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            print("Failed to parse ONNX model")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
    
    # Configure builder
    config = builder.create_builder_config()
    config.max_workspace_size = workspace_size * (1 << 30)  # Convert GB to bytes
    
    # Enable precision modes
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 mode enabled")
    
    if int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        print("INT8 mode enabled (requires calibration)")
    
    # Set optimization profile for dynamic shapes
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "input_ids",
        min=(1, 1),
        opt=(max_batch_size // 2, max_seq_length // 2),
        max=(max_batch_size, max_seq_length)
    )
    config.add_optimization_profile(profile)
    
    # Build engine
    print("Building TensorRT engine (this may take several minutes)...")
    try:
        engine = builder.build_engine(network, config)
        if engine is None:
            print("Failed to build TensorRT engine")
            return False
        
        # Serialize and save engine
        with open(output_path, 'wb') as f:
            f.write(engine.serialize())
        
        print(f"TensorRT engine saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error building TensorRT engine: {e}")
        return False


def benchmark_tensorrt_speedup(
    pytorch_model: nn.Module,
    tensorrt_engine_path: str,
    batch_size: int = 1,
    seq_length: int = 512,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
) -> Dict[str, float]:
    """
    Benchmark TensorRT speedup compared to PyTorch.
    
    Args:
        pytorch_model: Original PyTorch model
        tensorrt_engine_path: Path to TensorRT engine
        batch_size: Batch size for benchmarking
        seq_length: Sequence length for benchmarking
        num_iterations: Number of iterations for timing
        warmup_iterations: Number of warmup iterations
        
    Returns:
        Dictionary with timing results and speedup
    """
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
    except ImportError:
        raise ImportError("Benchmarking requires tensorrt and pycuda")
    
    import time
    
    # Benchmark PyTorch
    print("Benchmarking PyTorch model...")
    pytorch_model.eval()
    device = next(pytorch_model.parameters()).device
    dummy_input = torch.randint(0, 1000, (batch_size, seq_length), dtype=torch.long).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = pytorch_model(dummy_input)
    
    # Time PyTorch
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = pytorch_model(dummy_input)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / num_iterations
    
    # Benchmark TensorRT
    print("Benchmarking TensorRT engine...")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    with open(tensorrt_engine_path, 'rb') as f:
        engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    
    # Allocate buffers
    input_shape = (batch_size, seq_length)
    output_shape = (batch_size, seq_length, pytorch_model.config.vocab_size)
    
    h_input = cuda.pagelocked_empty(np.prod(input_shape), dtype=np.int64)
    h_output = cuda.pagelocked_empty(np.prod(output_shape), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    
    stream = cuda.Stream()
    
    # Warmup
    for _ in range(warmup_iterations):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
    
    # Time TensorRT
    start = time.time()
    for _ in range(num_iterations):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
    tensorrt_time = (time.time() - start) / num_iterations
    
    # Calculate speedup
    speedup = pytorch_time / tensorrt_time
    
    results = {
        'pytorch_time_ms': pytorch_time * 1000,
        'tensorrt_time_ms': tensorrt_time * 1000,
        'speedup': speedup,
    }
    
    print(f"\nBenchmark Results:")
    print(f"  PyTorch: {results['pytorch_time_ms']:.2f} ms")
    print(f"  TensorRT: {results['tensorrt_time_ms']:.2f} ms")
    print(f"  Speedup: {results['speedup']:.2f}x")
    
    return results


def export_model_for_deployment(
    model: nn.Module,
    output_dir: str,
    model_name: str = "resnet_bk",
    export_onnx: bool = True,
    export_tensorrt: bool = True,
    optimize_onnx: bool = True,
    verify: bool = True,
) -> Dict[str, str]:
    """
    Export model in multiple formats for deployment.
    
    Args:
        model: ResNet-BK model to export
        output_dir: Directory to save exported models
        model_name: Base name for exported files
        export_onnx: Export to ONNX format
        export_tensorrt: Export to TensorRT engine
        optimize_onnx: Optimize ONNX model
        verify: Verify numerical equivalence
        
    Returns:
        Dictionary mapping format to file path
        
    Example:
        ```python
        from src.models.hf_resnet_bk import create_resnet_bk_for_hf
        from src.models.onnx_export import export_model_for_deployment
        
        model = create_resnet_bk_for_hf("100M")
        paths = export_model_for_deployment(model, "exports/")
        ```
    """
    os.makedirs(output_dir, exist_ok=True)
    
    exported_paths = {}
    
    # Export to ONNX
    if export_onnx:
        onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
        success = export_to_onnx(model, onnx_path, verify=verify)
        if success:
            exported_paths['onnx'] = onnx_path
            
            # Optimize ONNX
            if optimize_onnx:
                optimized_path = optimize_onnx_model(onnx_path)
                exported_paths['onnx_optimized'] = optimized_path
    
    # Export to TensorRT
    if export_tensorrt and 'onnx' in exported_paths:
        trt_path = os.path.join(output_dir, f"{model_name}.trt")
        onnx_source = exported_paths.get('onnx_optimized', exported_paths['onnx'])
        success = export_to_tensorrt(onnx_source, trt_path)
        if success:
            exported_paths['tensorrt'] = trt_path
    
    print(f"\nExported models:")
    for format_name, path in exported_paths.items():
        print(f"  {format_name}: {path}")
    
    return exported_paths
