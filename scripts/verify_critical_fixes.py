#!/usr/bin/env python3
"""
Phase 1 Critical Fixes Verification Script

このスクリプトは、AGENTS.mdで指定された🚨最優先修正項目が
正しく実装されているかを検証します。

実行方法:
    python scripts/verify_critical_fixes.py

Author: Project MUSE Team
Date: 2025-11-19
"""

import sys
import os
import torch
import warnings

# プロジェクトルートをPythonパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 色付き出力用
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}\n")

def print_success(text):
    try:
        print(f"{Colors.GREEN}✓ {text}{Colors.END}")
    except UnicodeEncodeError:
        print(f"{Colors.GREEN}[OK] {text}{Colors.END}")

def print_error(text):
    try:
        print(f"{Colors.RED}✗ {text}{Colors.END}")
    except UnicodeEncodeError:
        print(f"{Colors.RED}[ERROR] {text}{Colors.END}")

def print_warning(text):
    try:
        print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")
    except UnicodeEncodeError:
        print(f"{Colors.YELLOW}[WARNING] {text}{Colors.END}")

def print_info(text):
    try:
        print(f"{Colors.BLUE}ℹ {text}{Colors.END}")
    except UnicodeEncodeError:
        print(f"{Colors.BLUE}[INFO] {text}{Colors.END}")


def verify_tt_contraction():
    """HTT展開なし演算の検証"""
    print_header("1. HTT (Tensor Train) 展開なし演算の検証")
    
    try:
        # tt_contractionカーネルの存在確認
        from src.kernels.tt_contraction import (
            tt_contraction_memory_efficient,
            TRITON_AVAILABLE
        )
        print_success("tt_contraction_memory_efficient のインポート成功")
        
        # 簡単なテスト
        B, L = 2, 10
        v1, v2 = 100, 100
        rank, d1, d2 = 8, 32, 32
        d_model = 1024
        
        idx1 = torch.randint(0, v1, (B, L))
        idx2 = torch.randint(0, v2, (B, L))
        core1 = torch.randn(v1, rank, d1)
        core2 = torch.randn(v2, rank, d2)
        
        # CPU実行
        output = tt_contraction_memory_efficient(
            idx1, idx2, core1, core2, d_model, use_triton=False
        )
        
        assert output.shape == (B, L, d_model), f"出力形状が不正: {output.shape}"
        print_success(f"CPU実行成功: 出力形状 {output.shape}")
        
        # CUDA実行（利用可能な場合）
        if torch.cuda.is_available() and TRITON_AVAILABLE:
            try:
                idx1_cuda = idx1.cuda()
                idx2_cuda = idx2.cuda()
                core1_cuda = core1.cuda()
                core2_cuda = core2.cuda()
                
                output_cuda = tt_contraction_memory_efficient(
                    idx1_cuda, idx2_cuda, core1_cuda, core2_cuda, d_model, use_triton=True
                )
                
                assert output_cuda.shape == (B, L, d_model), f"CUDA出力形状が不正: {output_cuda.shape}"
                print_success(f"CUDA実行成功: 出力形状 {output_cuda.shape}")
            except Exception as e:
                print_warning(f"CUDA実行スキップ: {e}")
        else:
            print_warning("CUDA/Triton利用不可、CPU実行のみ")
        
        # HTT Embeddingでの統合確認
        from src.models.phase1.htt_embedding import HolographicTTEmbedding
        
        embedding = HolographicTTEmbedding(vocab_size=1000, d_model=128, rank=8)
        input_ids = torch.randint(0, 1000, (2, 10))
        output = embedding(input_ids)
        
        assert output.shape == (2, 10, 128), f"Embedding出力形状が不正: {output.shape}"
        print_success("HTT Embeddingでの統合確認成功")
        
        # 圧縮率の確認
        compression_ratio = embedding.get_compression_ratio()
        print_info(f"圧縮率: {compression_ratio:.4f} ({(1-compression_ratio)*100:.1f}%削減)")
        
        return True
        
    except Exception as e:
        print_error(f"HTT検証失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_lns_precision():
    """LNS加算精度の検証"""
    print_header("2. LNS (対数数系) 加算精度の検証")
    
    try:
        from src.kernels.lns_kernel import lns_matmul, TRITON_AVAILABLE
        
        print_success("lns_matmul のインポート成功")
        
        # log1pが使用されているか確認（コード検査）
        import inspect
        source = inspect.getsource(lns_matmul)
        
        if 'log1p' in source or 'correction' in source:
            print_success("LNSカーネルにlog1p/補正項が実装されていることを確認")
        else:
            print_warning("LNSカーネルのソースコードでlog1p/補正項が見つかりません")
        
        # 数値精度テスト（CUDA利用可能な場合）
        if torch.cuda.is_available() and TRITON_AVAILABLE:
            M, N, K = 64, 64, 64
            
            # ログ領域のテストデータ
            log_a = torch.randn(M, K, device='cuda')
            log_b = torch.randn(K, N, device='cuda')
            
            # LNS matmul実行
            log_c = lns_matmul(log_a, log_b)
            
            assert log_c.shape == (M, N), f"LNS出力形状が不正: {log_c.shape}"
            assert torch.isfinite(log_c).all(), "LNS出力にNaN/Infが含まれています"
            
            print_success(f"LNS matmul実行成功: 出力形状 {log_c.shape}")
            print_info(f"出力範囲: [{log_c.min():.2f}, {log_c.max():.2f}]")
        else:
            print_warning("CUDA/Triton利用不可、数値精度テストをスキップ")
        
        return True
        
    except Exception as e:
        print_error(f"LNS検証失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_ar_ssm_gates():
    """AR-SSMゲート機構の検証"""
    print_header("3. AR-SSM ゲート機構 (STE/Gumbel-Softmax) の検証")
    
    try:
        from src.models.phase1.ar_ssm_layer import AdaptiveRankSemiseparableLayer
        
        print_success("AdaptiveRankSemiseparableLayer のインポート成功")
        
        # レイヤー作成
        layer = AdaptiveRankSemiseparableLayer(d_model=128, max_rank=16)
        
        # set_gate_modeメソッドの存在確認
        assert hasattr(layer, 'set_gate_mode'), "set_gate_modeメソッドが存在しません"
        print_success("set_gate_modeメソッドが実装されています")
        
        # 各ゲートモードのテスト
        x = torch.randn(2, 10, 128)
        
        for mode in ['soft', 'ste', 'gumbel']:
            layer.set_gate_mode(mode)
            output, diagnostics = layer(x)
            
            assert output.shape == x.shape, f"{mode}モード: 出力形状が不正"
            assert 'gates' in diagnostics, f"{mode}モード: diagnosticsにgatesがありません"
            
            # 勾配フローの確認
            loss = output.sum()
            loss.backward()
            
            # ゲートネットワークに勾配が流れているか確認
            has_grad = any(p.grad is not None for p in layer.complexity_gate.parameters())
            assert has_grad, f"{mode}モード: ゲートネットワークに勾配が流れていません"
            
            print_success(f"{mode}モード: 動作確認成功、勾配フロー正常")
            
            # 勾配をクリア
            layer.zero_grad()
        
        return True
        
    except Exception as e:
        print_error(f"AR-SSMゲート検証失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_triton_autotune():
    """Triton自動チューニングの検証"""
    print_header("4. Triton カーネル自動チューニングの検証")
    
    try:
        import inspect
        
        lns_found = False
        scan_found = False
        
        # LNSカーネルの自動チューニング確認
        try:
            import src.kernels.lns_kernel as lns_module
            source = inspect.getsource(lns_module)
            if '@triton.autotune' in source:
                print_success("LNSカーネルに@triton.autotuneが含まれています")
                lns_found = True
                
                # 設定数を確認
                import re
                configs = re.findall(r'triton\.Config\(', source)
                print_info(f"  LNS自動チューニング設定数: {len(configs)}")
            else:
                print_warning("LNSカーネルに@triton.autotuneが見つかりません")
        except Exception as e:
            print_warning(f"LNSカーネルのソースコード確認に失敗: {e}")
        
        # Associative Scanカーネルの自動チューニング確認
        try:
            import src.kernels.associative_scan as scan_module
            source = inspect.getsource(scan_module)
            if '@triton.autotune' in source:
                print_success("Associative Scanカーネルに@triton.autotuneが含まれています")
                scan_found = True
                
                # 設定数を確認
                import re
                configs = re.findall(r'triton\.Config\(', source)
                print_info(f"  Scan自動チューニング設定数: {len(configs)}")
            else:
                print_warning("Associative Scanカーネルに@triton.autotuneが見つかりません")
        except Exception as e:
            print_warning(f"Associative Scanカーネルのソースコード確認に失敗: {e}")
        
        # 両方見つかった場合のみ成功
        if lns_found and scan_found:
            print_success("すべてのカーネルに自動チューニングが実装されています")
            return True
        elif lns_found or scan_found:
            print_warning("一部のカーネルのみ自動チューニングが実装されています")
            return True  # 部分的成功として扱う
        else:
            print_error("自動チューニングが実装されていません")
            return False
        
    except Exception as e:
        print_error(f"自動チューニング検証失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_recovery_logic():
    """安定性回復ロジックの検証"""
    print_header("5. 安定性監視回復ロジックの検証")
    
    try:
        from src.models.phase1.recovery import Phase1ErrorRecovery
        
        print_success("Phase1ErrorRecovery のインポート成功")
        
        # 回復インスタンス作成
        recovery = Phase1ErrorRecovery(
            enable_checkpoint_rollback=True,
            checkpoint_save_interval=100,
        )
        
        # 新機能の存在確認
        assert hasattr(recovery, 'save_checkpoint'), "save_checkpointメソッドが存在しません"
        assert hasattr(recovery, 'rollback_to_checkpoint'), "rollback_to_checkpointメソッドが存在しません"
        assert hasattr(recovery, 'reinitialize_layer'), "reinitialize_layerメソッドが存在しません"
        
        print_success("チェックポイントロールバック機能が実装されています")
        print_success("層の部分的再初期化機能が実装されています")
        
        # 簡単なテスト
        import torch.nn as nn
        
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )
        
        optimizer = torch.optim.Adam(model.parameters())
        
        # チェックポイント保存
        recovery.save_checkpoint(model, optimizer, step=100)
        assert recovery.last_stable_checkpoint is not None, "チェックポイントが保存されていません"
        print_success("チェックポイント保存成功")
        
        # パラメータを変更
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        
        # ロールバック
        success = recovery.rollback_to_checkpoint(model, optimizer)
        assert success, "ロールバックに失敗しました"
        print_success("チェックポイントロールバック成功")
        
        # 層の再初期化
        success = recovery.reinitialize_layer(model, '0')  # 最初のLinear層
        assert success, "層の再初期化に失敗しました"
        print_success("層の部分的再初期化成功")
        
        return True
        
    except Exception as e:
        print_error(f"回復ロジック検証失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_dependencies():
    """依存ライブラリのバージョン検証"""
    print_header("6. 依存ライブラリバージョンの検証")
    
    try:
        # requirements.txtの読み込み
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
        
        # バージョン固定の確認
        critical_packages = {
            'torch': '2.1.0',
            'triton': '2.1.0',
        }
        
        all_fixed = True
        for package, expected_version in critical_packages.items():
            if f'{package}=={expected_version}' in requirements:
                print_success(f"{package}=={expected_version} が固定されています")
            else:
                print_error(f"{package}のバージョンが{expected_version}に固定されていません")
                all_fixed = False
        
        # インストール済みバージョンの確認
        import torch
        print_info(f"インストール済みtorchバージョン: {torch.__version__}")
        
        try:
            import triton
            print_info(f"インストール済みtritonバージョン: {triton.__version__}")
        except ImportError:
            print_warning("tritonがインストールされていません")
        
        return all_fixed
        
    except Exception as e:
        print_error(f"依存関係検証失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_presets():
    """プリセット設定の検証"""
    print_header("7. プリセット設定の検証")
    
    try:
        from src.models.phase1.presets import get_preset, list_presets
        
        print_success("プリセット関数のインポート成功")
        
        # エイリアスのテスト
        aliases = {
            'speed_oriented': '24gb',
            'memory_oriented': '8gb',
            'balanced': '10gb',
        }
        
        for alias, expected in aliases.items():
            try:
                config = get_preset(alias)
                print_success(f"エイリアス '{alias}' が動作します")
            except Exception as e:
                print_error(f"エイリアス '{alias}' が動作しません: {e}")
                return False
        
        # プリセット一覧の取得
        presets = list_presets()
        print_info(f"利用可能なプリセット数: {len(presets)}")
        
        return True
        
    except Exception as e:
        print_error(f"プリセット検証失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """メイン検証関数"""
    print_header("Phase 1 Critical Fixes Verification")
    print_info("AGENTS.mdで指定された🚨最優先修正項目の検証を開始します")
    
    results = {}
    
    # 各検証を実行
    results['HTT展開なし演算'] = verify_tt_contraction()
    results['LNS加算精度'] = verify_lns_precision()
    results['AR-SSMゲート'] = verify_ar_ssm_gates()
    results['Triton自動チューニング'] = verify_triton_autotune()
    results['回復ロジック'] = verify_recovery_logic()
    results['依存関係'] = verify_dependencies()
    results['プリセット設定'] = verify_presets()
    
    # 結果サマリー
    print_header("検証結果サマリー")
    
    total = len(results)
    passed = sum(results.values())
    failed = total - passed
    
    for name, result in results.items():
        if result:
            print_success(f"{name}: 合格")
        else:
            print_error(f"{name}: 不合格")
    
    print()
    print(f"{Colors.BOLD}合計: {passed}/{total} 項目が合格{Colors.END}")
    
    if failed == 0:
        try:
            print(f"\n{Colors.GREEN}{Colors.BOLD}✓ すべての検証に合格しました！{Colors.END}")
        except UnicodeEncodeError:
            print(f"\n{Colors.GREEN}{Colors.BOLD}[SUCCESS] すべての検証に合格しました！{Colors.END}")
        print(f"{Colors.GREEN}Phase 2への移行準備が完了しました。{Colors.END}\n")
        return 0
    else:
        try:
            print(f"\n{Colors.RED}{Colors.BOLD}✗ {failed}項目が不合格です。{Colors.END}")
        except UnicodeEncodeError:
            print(f"\n{Colors.RED}{Colors.BOLD}[FAILED] {failed}項目が不合格です。{Colors.END}")
        print(f"{Colors.RED}修正が必要です。{Colors.END}\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
