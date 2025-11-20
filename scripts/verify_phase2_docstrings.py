#!/usr/bin/env python3
"""
Phase 2 Docstring Verification Script

このスクリプトは、Phase 2の全モジュールにdocstringが適切に記載されているかを検証します。

検証項目:
1. モジュールレベルdocstringの存在
2. クラスdocstringの存在
3. 関数/メソッドdocstringの存在
4. 物理的直観の記載（主要クラス）
5. 数式の記載（アルゴリズム）
6. 型ヒントの存在

Usage:
    python scripts/verify_phase2_docstrings.py
"""

import ast
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import json


class DocstringVerifier(ast.NodeVisitor):
    """Docstring検証用のASTビジター"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.module_docstring = None
        self.classes = []
        self.functions = []
        self.methods = []
        
    def visit_Module(self, node):
        """モジュールレベルdocstringを取得"""
        self.module_docstring = ast.get_docstring(node)
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        """クラスdocstringを取得"""
        docstring = ast.get_docstring(node)
        self.classes.append({
            'name': node.name,
            'docstring': docstring,
            'has_docstring': docstring is not None,
            'lineno': node.lineno,
        })
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        """関数/メソッドdocstringを取得"""
        docstring = ast.get_docstring(node)
        
        # 型ヒントの確認
        has_type_hints = False
        if node.returns is not None:
            has_type_hints = True
        for arg in node.args.args:
            if arg.annotation is not None:
                has_type_hints = True
                break
        
        func_info = {
            'name': node.name,
            'docstring': docstring,
            'has_docstring': docstring is not None,
            'has_type_hints': has_type_hints,
            'lineno': node.lineno,
        }
        
        # メソッドか関数かを判定
        if isinstance(node, ast.FunctionDef):
            # 親がクラスならメソッド
            parent = getattr(node, 'parent', None)
            if parent and isinstance(parent, ast.ClassDef):
                self.methods.append(func_info)
            else:
                self.functions.append(func_info)
        
        self.generic_visit(node)


def verify_file(filepath: Path) -> Dict:
    """ファイルのdocstringを検証"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        
        tree = ast.parse(source)
        verifier = DocstringVerifier(str(filepath))
        verifier.visit(tree)
        
        # 物理的直観と数式のキーワード検索
        has_physical_intuition = any([
            '物理的' in source,
            'Physical' in source,
            '直観' in source,
            'Intuition' in source,
        ])
        
        has_math_formulas = any([
            '数式' in source,
            'Formula' in source,
            '=' in source and ('Args:' in source or 'Returns:' in source),
        ])
        
        return {
            'filepath': str(filepath),
            'module_docstring': verifier.module_docstring,
            'has_module_docstring': verifier.module_docstring is not None,
            'classes': verifier.classes,
            'functions': verifier.functions,
            'methods': verifier.methods,
            'has_physical_intuition': has_physical_intuition,
            'has_math_formulas': has_math_formulas,
            'total_classes': len(verifier.classes),
            'total_functions': len(verifier.functions),
            'total_methods': len(verifier.methods),
            'classes_with_docstring': sum(1 for c in verifier.classes if c['has_docstring']),
            'functions_with_docstring': sum(1 for f in verifier.functions if f['has_docstring']),
            'methods_with_docstring': sum(1 for m in verifier.methods if m['has_docstring']),
        }
    
    except Exception as e:
        return {
            'filepath': str(filepath),
            'error': str(e),
        }


def main():
    """メイン処理"""
    print("=" * 80)
    print("Phase 2 Docstring Verification")
    print("=" * 80)
    print()
    
    # Phase 2モジュールのパス
    phase2_dir = Path('src/models/phase2')
    kernel_dir = Path('src/kernels')
    
    # 検証対象ファイル
    files_to_verify = [
        # Kernels
        kernel_dir / 'bk_scan.py',
        
        # Phase 2 Models
        phase2_dir / 'non_hermitian.py',
        phase2_dir / 'gradient_safety.py',
        phase2_dir / 'dissipative_hebbian.py',
        phase2_dir / 'memory_selection.py',
        phase2_dir / 'memory_resonance.py',
        phase2_dir / 'zeta_init.py',
        phase2_dir / 'integrated_model.py',
        phase2_dir / 'factory.py',
        phase2_dir / '__init__.py',
    ]
    
    # 検証実行
    results = []
    for filepath in files_to_verify:
        if not filepath.exists():
            print(f"⚠️  File not found: {filepath}")
            continue
        
        print(f"Verifying: {filepath}")
        result = verify_file(filepath)
        results.append(result)
    
    print()
    print("=" * 80)
    print("Verification Results")
    print("=" * 80)
    print()
    
    # 統計情報
    total_files = len(results)
    files_with_module_docstring = sum(1 for r in results if r.get('has_module_docstring', False))
    total_classes = sum(r.get('total_classes', 0) for r in results)
    classes_with_docstring = sum(r.get('classes_with_docstring', 0) for r in results)
    total_functions = sum(r.get('total_functions', 0) for r in results)
    functions_with_docstring = sum(r.get('functions_with_docstring', 0) for r in results)
    total_methods = sum(r.get('total_methods', 0) for r in results)
    methods_with_docstring = sum(r.get('methods_with_docstring', 0) for r in results)
    files_with_physical = sum(1 for r in results if r.get('has_physical_intuition', False))
    files_with_math = sum(1 for r in results if r.get('has_math_formulas', False))
    
    # 結果表示
    print(f"📁 Total Files: {total_files}")
    print(f"📄 Module Docstrings: {files_with_module_docstring}/{total_files} ({files_with_module_docstring/total_files*100:.1f}%)")
    print()
    
    print(f"🏛️  Classes: {total_classes}")
    print(f"   With Docstring: {classes_with_docstring}/{total_classes} ({classes_with_docstring/total_classes*100:.1f}%)" if total_classes > 0 else "   With Docstring: N/A")
    print()
    
    print(f"🔧 Functions: {total_functions}")
    print(f"   With Docstring: {functions_with_docstring}/{total_functions} ({functions_with_docstring/total_functions*100:.1f}%)" if total_functions > 0 else "   With Docstring: N/A")
    print()
    
    print(f"⚙️  Methods: {total_methods}")
    print(f"   With Docstring: {methods_with_docstring}/{total_methods} ({methods_with_docstring/total_methods*100:.1f}%)" if total_methods > 0 else "   With Docstring: N/A")
    print()
    
    print(f"🔬 Physical Intuition: {files_with_physical}/{total_files} files ({files_with_physical/total_files*100:.1f}%)")
    print(f"📐 Math Formulas: {files_with_math}/{total_files} files ({files_with_math/total_files*100:.1f}%)")
    print()
    
    # 詳細結果
    print("=" * 80)
    print("Detailed Results")
    print("=" * 80)
    print()
    
    for result in results:
        if 'error' in result:
            print(f"❌ {result['filepath']}: ERROR - {result['error']}")
            continue
        
        filepath = Path(result['filepath']).name
        status = "✅" if result['has_module_docstring'] else "❌"
        
        print(f"{status} {filepath}")
        print(f"   Module Docstring: {'Yes' if result['has_module_docstring'] else 'No'}")
        print(f"   Classes: {result['classes_with_docstring']}/{result['total_classes']}")
        print(f"   Functions: {result['functions_with_docstring']}/{result['total_functions']}")
        print(f"   Methods: {result['methods_with_docstring']}/{result['total_methods']}")
        print(f"   Physical Intuition: {'Yes' if result['has_physical_intuition'] else 'No'}")
        print(f"   Math Formulas: {'Yes' if result['has_math_formulas'] else 'No'}")
        print()
    
    # JSON出力
    output_file = Path('results/benchmarks/phase2_docstring_verification.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total_files': total_files,
                'files_with_module_docstring': files_with_module_docstring,
                'total_classes': total_classes,
                'classes_with_docstring': classes_with_docstring,
                'total_functions': total_functions,
                'functions_with_docstring': functions_with_docstring,
                'total_methods': total_methods,
                'methods_with_docstring': methods_with_docstring,
                'files_with_physical': files_with_physical,
                'files_with_math': files_with_math,
            },
            'details': results,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"📊 Results saved to: {output_file}")
    print()
    
    # 合否判定
    print("=" * 80)
    print("Final Verdict")
    print("=" * 80)
    print()
    
    # 合格基準
    module_docstring_threshold = 0.9  # 90%
    class_docstring_threshold = 0.9   # 90%
    function_docstring_threshold = 0.8  # 80%
    method_docstring_threshold = 0.8   # 80%
    physical_threshold = 0.7  # 70%
    math_threshold = 0.7      # 70%
    
    checks = []
    
    # モジュールdocstring
    module_pass = files_with_module_docstring / total_files >= module_docstring_threshold
    checks.append(('Module Docstrings', module_pass, f"{files_with_module_docstring}/{total_files}"))
    
    # クラスdocstring
    if total_classes > 0:
        class_pass = classes_with_docstring / total_classes >= class_docstring_threshold
        checks.append(('Class Docstrings', class_pass, f"{classes_with_docstring}/{total_classes}"))
    
    # 関数docstring
    if total_functions > 0:
        function_pass = functions_with_docstring / total_functions >= function_docstring_threshold
        checks.append(('Function Docstrings', function_pass, f"{functions_with_docstring}/{total_functions}"))
    
    # メソッドdocstring
    if total_methods > 0:
        method_pass = methods_with_docstring / total_methods >= method_docstring_threshold
        checks.append(('Method Docstrings', method_pass, f"{methods_with_docstring}/{total_methods}"))
    
    # 物理的直観
    physical_pass = files_with_physical / total_files >= physical_threshold
    checks.append(('Physical Intuition', physical_pass, f"{files_with_physical}/{total_files}"))
    
    # 数式
    math_pass = files_with_math / total_files >= math_threshold
    checks.append(('Math Formulas', math_pass, f"{files_with_math}/{total_files}"))
    
    # 結果表示
    all_passed = all(check[1] for check in checks)
    
    for name, passed, count in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name} ({count})")
    
    print()
    
    if all_passed:
        print("🎉 All checks passed! Phase 2 docstrings are complete.")
        return 0
    else:
        print("⚠️  Some checks failed. Please review the docstrings.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
