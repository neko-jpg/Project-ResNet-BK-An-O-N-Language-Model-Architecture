"""
Complex Number Support Demo

Phase 2準備: 複素数サポートインフラストラクチャのデモンストレーション

このスクリプトは、Phase 1で実装された複素数サポート機能を実演します。
Phase 2では、これらの機能を使用して非エルミート演算子による忘却機構を実装します。

Requirements: 11.1, 11.2, 11.3, 11.4, 11.5

Usage:
    python examples/complex_number_support_demo.py
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.phase1 import (
    # Type checking
    is_complex_tensor,
    
    # Conversion functions
    real_to_complex,
    complex_to_real,
    ensure_complex,
    ensure_real,
    
    # Complex operations
    complex_phase_rotation,
    check_dtype_compatibility,
    safe_complex_operation,
    
    # Complex layers
    ComplexLinear,
    
    # Documentation
    document_complex_support,
    get_complex_conversion_guide,
    
    # Phase 1 components
    AdaptiveRankSemiseparableLayer,
    HolographicTTEmbedding,
)


def demo_type_checking():
    """デモ: 複素数型のチェック"""
    print("=" * 80)
    print("Demo 1: Type Checking")
    print("=" * 80)
    
    # Real tensor
    x_real = torch.randn(10, 20)
    print(f"Real tensor: dtype={x_real.dtype}, is_complex={is_complex_tensor(x_real)}")
    
    # Complex tensor
    x_complex = torch.complex(torch.randn(10, 20), torch.randn(10, 20))
    print(f"Complex tensor: dtype={x_complex.dtype}, is_complex={is_complex_tensor(x_complex)}")
    
    print()


def demo_conversion():
    """デモ: 実数↔複素数変換"""
    print("=" * 80)
    print("Demo 2: Real ↔ Complex Conversion")
    print("=" * 80)
    
    # Real to complex
    x_real = torch.randn(5, 10)
    x_complex = real_to_complex(x_real)
    print(f"Real to complex: {x_real.shape} ({x_real.dtype}) → {x_complex.shape} ({x_complex.dtype})")
    
    # Complex to real (concat mode)
    x_concat = complex_to_real(x_complex, mode='concat')
    print(f"Complex to real (concat): {x_complex.shape} → {x_concat.shape}")
    
    # Complex to real (separate mode)
    real_part, imag_part = complex_to_real(x_complex, mode='separate')
    print(f"Complex to real (separate): {x_complex.shape} → ({real_part.shape}, {imag_part.shape})")
    
    # Complex to real (magnitude mode)
    magnitude = complex_to_real(x_complex, mode='magnitude')
    print(f"Complex to real (magnitude): {x_complex.shape} → {magnitude.shape}")
    
    # Ensure complex
    x_ensured = ensure_complex(x_real)
    print(f"Ensure complex: {x_real.dtype} → {x_ensured.dtype}")
    
    # Ensure real
    x_ensured_real = ensure_real(x_complex)
    print(f"Ensure real: {x_complex.dtype} → {x_ensured_real.dtype}")
    
    print()


def demo_phase_rotation():
    """デモ: 複素位相回転"""
    print("=" * 80)
    print("Demo 3: Complex Phase Rotation")
    print("=" * 80)
    
    x = torch.randn(4, 20, 128)
    phase = torch.randn(128)
    
    # Phase 1: Real approximation (cos(θ))
    x_rotated_real = complex_phase_rotation(x, phase, use_full_complex=False)
    print(f"Phase 1 (cos(θ)): {x.shape} ({x.dtype}) → {x_rotated_real.shape} ({x_rotated_real.dtype})")
    
    # Phase 2: Full complex rotation (exp(iθ))
    x_rotated_complex = complex_phase_rotation(x, phase, use_full_complex=True)
    print(f"Phase 2 (exp(iθ)): {x.shape} ({x.dtype}) → {x_rotated_complex.shape} ({x_rotated_complex.dtype})")
    
    print()


def demo_safe_operations():
    """デモ: 安全な混合演算"""
    print("=" * 80)
    print("Demo 4: Safe Mixed Real/Complex Operations")
    print("=" * 80)
    
    x_real = torch.randn(10, 20)
    y_complex = torch.complex(torch.randn(10, 20), torch.randn(10, 20))
    
    # Safe addition with auto-conversion
    result_add = safe_complex_operation(x_real, y_complex, operation='add', auto_convert=True)
    print(f"Safe add: real + complex → {result_add.dtype}")
    
    # Safe multiplication
    result_mul = safe_complex_operation(x_real, y_complex, operation='mul', auto_convert=True)
    print(f"Safe mul: real × complex → {result_mul.dtype}")
    
    # Dtype compatibility check (should raise error)
    try:
        check_dtype_compatibility(x_real, y_complex, operation="addition")
        print("Dtype check: PASSED (unexpected)")
    except TypeError as e:
        print(f"Dtype check: FAILED (expected) - {str(e)[:60]}...")
    
    print()


def demo_complex_linear():
    """デモ: 複素数線形層"""
    print("=" * 80)
    print("Demo 5: Complex Linear Layer")
    print("=" * 80)
    
    layer = ComplexLinear(in_features=128, out_features=256)
    
    # Complex input
    x_complex = torch.complex(torch.randn(10, 128), torch.randn(10, 128))
    y_complex = layer(x_complex)
    print(f"Complex input: {x_complex.shape} ({x_complex.dtype}) → {y_complex.shape} ({y_complex.dtype})")
    
    # Real input (auto-converted)
    x_real = torch.randn(10, 128)
    y_from_real = layer(x_real)
    print(f"Real input: {x_real.shape} ({x_real.dtype}) → {y_from_real.shape} ({y_from_real.dtype})")
    
    print()


def demo_ar_ssm_complex_input():
    """デモ: AR-SSMの複素数入力処理"""
    print("=" * 80)
    print("Demo 6: AR-SSM with Complex Input (Phase 1)")
    print("=" * 80)
    
    layer = AdaptiveRankSemiseparableLayer(d_model=128, max_rank=32)
    
    # Real input (recommended in Phase 1)
    x_real = torch.randn(4, 100, 128)
    y_real, diagnostics_real = layer(x_real)
    print(f"Real input: {x_real.shape} → {y_real.shape}")
    print(f"  input_was_complex: {diagnostics_real['input_was_complex']}")
    
    # Complex input (converts to real with warning in Phase 1)
    x_complex = torch.complex(torch.randn(4, 100, 128), torch.randn(4, 100, 128))
    print(f"\nComplex input: {x_complex.shape} ({x_complex.dtype})")
    print("  (Warning will be issued...)")
    
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        y_complex, diagnostics_complex = layer(x_complex)
        if len(w) > 0:
            print(f"  Warning: {w[0].message}")
    
    print(f"  Output: {y_complex.shape} ({y_complex.dtype})")
    print(f"  input_was_complex: {diagnostics_complex['input_was_complex']}")
    
    print()


def demo_htt_complex_forward():
    """デモ: HTTの複素数forward"""
    print("=" * 80)
    print("Demo 7: HTT Embedding Complex Forward (Phase 2 Preparation)")
    print("=" * 80)
    
    embedding = HolographicTTEmbedding(
        vocab_size=1000,
        d_model=128,
        rank=8,
        phase_encoding=True
    )
    
    input_ids = torch.randint(0, 1000, (4, 50))
    
    # Phase 1: Real output (cos(θ) phase rotation)
    output_real = embedding(input_ids)
    print(f"Phase 1 forward: {input_ids.shape} → {output_real.shape} ({output_real.dtype})")
    
    # Phase 2: Complex output (exp(iθ) phase rotation)
    output_complex = embedding.forward_complex(input_ids)
    print(f"Phase 2 forward_complex: {input_ids.shape} → {output_complex.shape} ({output_complex.dtype})")
    
    print()


def demo_component_support_status():
    """デモ: コンポーネントのサポート状況"""
    print("=" * 80)
    print("Demo 8: Component Complex Support Status")
    print("=" * 80)
    
    support_status = document_complex_support()
    
    for component, info in support_status.items():
        print(f"\n{component}:")
        print(f"  Complex Input Support: {info['complex_input_support']}")
        print(f"  Status: {info['status']}")
        print(f"  Phase 2 Ready: {info['phase2_ready']}")
        print(f"  Notes: {info['notes'][:80]}...")
    
    print()


def demo_migration_guide():
    """デモ: Phase 2移行ガイド"""
    print("=" * 80)
    print("Demo 9: Phase 2 Migration Guide")
    print("=" * 80)
    
    guide = get_complex_conversion_guide()
    print(guide)


def main():
    """メインデモ実行"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "Complex Number Support Demo" + " " * 31 + "║")
    print("║" + " " * 20 + "Phase 2 Preparation Features" + " " * 30 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")
    
    # Run all demos
    demo_type_checking()
    demo_conversion()
    demo_phase_rotation()
    demo_safe_operations()
    demo_complex_linear()
    demo_ar_ssm_complex_input()
    demo_htt_complex_forward()
    demo_component_support_status()
    demo_migration_guide()
    
    print("=" * 80)
    print("All demos completed successfully!")
    print("=" * 80)
    print("\nNext Steps:")
    print("  1. Review documentation: docs/implementation/COMPLEX_NUMBER_SUPPORT.md")
    print("  2. Run tests: pytest tests/test_phase2_compatibility.py")
    print("  3. Prepare for Phase 2 integration")
    print()


if __name__ == "__main__":
    main()
