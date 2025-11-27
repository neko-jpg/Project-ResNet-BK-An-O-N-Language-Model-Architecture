#!/usr/bin/env python3
"""
Phase 7 訓練環境チェックスクリプト

このスクリプトは、Phase 7 Chat AI訓練を開始する前に
必要な環境がすべて整っているかを確認します。

Usage:
    python scripts/verify_phase7_ready.py
    make verify-phase7
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def check_python():
    """Pythonバージョンチェック"""
    print("1. Python Version Check")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"   ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ✗ Python {version.major}.{version.minor} (3.10+ required)")
        return False


def check_cuda():
    """CUDAチェック"""
    print("\n2. CUDA Check")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"   ✓ CUDA available")
            print(f"   ✓ GPU: {gpu_name}")
            print(f"   ✓ VRAM: {vram_gb:.1f} GB")
            
            if vram_gb < 8:
                print(f"   ⚠ Warning: VRAM < 8GB. May need to reduce model size.")
            return True
        else:
            print("   ✗ CUDA not available")
            return False
    except ImportError:
        print("   ✗ PyTorch not installed")
        return False


def check_triton():
    """Tritonチェック"""
    print("\n3. Triton Check")
    try:
        import triton
        version = getattr(triton, '__version__', 'unknown')
        print(f"   ✓ Triton {version} installed")
        
        # 簡単なカーネルテスト
        try:
            import triton.language as tl
            print("   ✓ Triton language module available")
            return True
        except Exception as e:
            print(f"   ⚠ Triton language import warning: {e}")
            return True  # Tritonはあるが警告
    except ImportError:
        print("   ⚠ Triton not installed (PyTorch fallback will be used)")
        return False


def check_datasets():
    """データセットチェック"""
    print("\n4. Dataset Check")
    
    # dataset_mixing.yaml
    config_path = Path("configs/dataset_mixing.yaml")
    if not config_path.exists():
        print("   ✗ configs/dataset_mixing.yaml not found")
        print("   → Run 'make recipe' to configure datasets")
        return False
    
    print(f"   ✓ {config_path} exists")
    
    # データセットフォルダ
    import yaml
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    datasets = cfg.get('datasets', {})
    found = 0
    missing = []
    
    for name, info in datasets.items():
        path = Path(info.get('path', ''))
        bin_file = path / 'train.bin'
        idx_file = path / 'train.idx'
        
        if bin_file.exists() and idx_file.exists():
            found += 1
            print(f"   ✓ {name}: {path}")
        else:
            missing.append(name)
    
    if missing:
        print(f"   ⚠ Missing datasets: {', '.join(missing)}")
        print("   → Run 'make data-lite' or 'make data' to download")
    
    if found > 0:
        print(f"   ✓ {found}/{len(datasets)} datasets ready")
        return True
    else:
        print("   ✗ No datasets found")
        return False


def check_model_config():
    """モデル設定チェック"""
    print("\n5. Model Config Check")
    
    config_path = Path("configs/phase7_max_push.yaml")
    if not config_path.exists():
        print(f"   ✗ {config_path} not found")
        return False
    
    import yaml
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    print(f"   ✓ {config_path} exists")
    print(f"   - d_model: {cfg.get('d_model', 'N/A')}")
    print(f"   - n_layers: {cfg.get('n_layers', 'N/A')}")
    print(f"   - n_seq: {cfg.get('n_seq', 'N/A')}")
    print(f"   - batch_size: {cfg.get('batch_size', 'N/A')}")
    print(f"   - gradient_accumulation: {cfg.get('gradient_accumulation_steps', 'N/A')}")
    
    return True


def check_imports():
    """必要なモジュールのインポートチェック"""
    print("\n6. Module Import Check")
    
    modules = [
        ("torch", "PyTorch"),
        ("yaml", "PyYAML"),
        ("numpy", "NumPy"),
    ]
    
    all_ok = True
    for module, name in modules:
        try:
            __import__(module)
            print(f"   ✓ {name}")
        except ImportError:
            print(f"   ✗ {name} not installed")
            all_ok = False
    
    # プロジェクトモジュール
    try:
        from src.utils.data_utils import get_mixed_data_loader
        print("   ✓ src.utils.data_utils")
    except ImportError as e:
        print(f"   ⚠ src.utils.data_utils: {e}")
    
    try:
        from src.models.phase7.integrated_model import Phase7IntegratedModel
        print("   ✓ src.models.phase7.integrated_model")
    except ImportError as e:
        print(f"   ⚠ src.models.phase7: {e}")
    
    return all_ok


def test_model_creation():
    """モデル作成テスト"""
    print("\n7. Model Creation Test")
    
    try:
        import torch
        import gc
        
        # メモリクリア
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 小さいモデルでテスト
        from scripts.train_phase7_max import Phase7MaxModel
        
        model = Phase7MaxModel(
            vocab_size=1000,
            d_model=256,
            n_layers=2,
            n_seq=64,
            num_heads=4,
            embed_rank=32,
            ffn_rank=32,
            head_rank=32,
            use_checkpoint=True,
        )
        
        if torch.cuda.is_available():
            model = model.cuda().half()
            
            # Forward pass test
            x = torch.randint(0, 1000, (1, 64), device='cuda')
            with torch.cuda.amp.autocast():
                out = model(x)
            
            print(f"   ✓ Model created successfully")
            print(f"   ✓ Forward pass OK (output shape: {out.shape})")
            
            # Backward pass test
            loss = out.mean()
            loss.backward()
            print("   ✓ Backward pass OK")
            
            del model, x, out, loss
            gc.collect()
            torch.cuda.empty_cache()
            
            return True
        else:
            print("   ⚠ CUDA not available, skipping GPU test")
            return True
            
    except Exception as e:
        print(f"   ✗ Model test failed: {e}")
        return False


def main():
    print("=" * 60)
    print("Phase 7 Chat AI Training Environment Check")
    print("=" * 60)
    
    results = {
        "Python": check_python(),
        "CUDA": check_cuda(),
        "Triton": check_triton(),
        "Datasets": check_datasets(),
        "Config": check_model_config(),
        "Imports": check_imports(),
        "Model": test_model_creation(),
    }
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_pass = True
    critical_fail = False
    
    for name, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
        if not passed:
            all_pass = False
            if name in ["Python", "CUDA", "Model"]:
                critical_fail = True
    
    print()
    
    if all_pass:
        print("🎉 All checks passed! Ready to train.")
        print("\nStart training with:")
        print("  make train-chat")
        print("\nOr test with dummy data:")
        print("  make train-chat-test")
        return 0
    elif critical_fail:
        print("❌ Critical checks failed. Please fix before training.")
        return 1
    else:
        print("⚠ Some checks failed, but training may still work.")
        print("\nTry:")
        print("  make train-chat-test  # Test with dummy data first")
        return 0


if __name__ == "__main__":
    sys.exit(main())
