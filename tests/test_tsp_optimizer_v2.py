#!/usr/bin/env python3
"""
TSP Path Optimizer v2 テスト

テスト内容:
1. 基本機能（都市遷移、評価）
2. 日本語LLMプリセット
3. 適応的ε減衰
4. プラトー検出
5. 状態保存/復元
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.tsp_path_optimizer import (
    TSPPathOptimizer,
    create_tsp_optimizer,
    City,
    DEFAULT_CITIES,
    JAPANESE_LLM_CITIES,
    CITY_PRESETS,
    WindowMetrics,
    TransitionEvent,
)
import math
import random


class DummyOptimizer:
    """テスト用ダミーオプティマイザ"""
    def __init__(self, lr: float = 0.05):
        self.param_groups = [{"lr": lr}]


def test_basic_creation():
    """基本的なTSP作成テスト"""
    print("=" * 60)
    print("Test 1: Basic Creation")
    print("=" * 60)
    
    tsp = create_tsp_optimizer(base_lr=0.05)
    
    assert tsp is not None, "TSP should be created"
    assert tsp.current_city is not None, "Should have initial city"
    assert len(tsp.cities) == len(DEFAULT_CITIES), "Should have default cities"
    
    print(f"  ✓ TSP created with {len(tsp.cities)} cities")
    print(f"  ✓ Initial city: {tsp.current_city.name}")
    print(f"  ✓ Base LR: {tsp.base_lr}")
    print()


def test_japanese_llm_preset():
    """日本語LLM都市プリセットテスト"""
    print("=" * 60)
    print("Test 2: Japanese LLM Preset")
    print("=" * 60)
    
    # プリセットの確認
    assert "japanese_llm" in CITY_PRESETS, "Should have japanese_llm preset"
    assert "default" in CITY_PRESETS, "Should have default preset"
    
    # 日本語LLMプリセットで作成
    tsp = create_tsp_optimizer(base_lr=0.05, city_preset="japanese_llm")
    
    assert len(tsp.cities) == len(JAPANESE_LLM_CITIES), "Should have 5 Japanese LLM cities"
    
    city_names = [c.name for c in tsp.cities]
    assert all(name.startswith("J_") for name in city_names), "All cities should start with J_"
    
    print(f"  ✓ Japanese LLM cities: {city_names}")
    print(f"  ✓ City count: {len(tsp.cities)}")
    
    # 都市の安定度を確認
    for city in tsp.cities:
        print(f"    - {city.name}: stability={city.stability:.2f}, lr_scale={city.lr_scale}")
    print()


def test_adaptive_epsilon():
    """適応的ε減衰テスト"""
    print("=" * 60)
    print("Test 3: Adaptive Epsilon Decay")
    print("=" * 60)
    
    tsp = create_tsp_optimizer(
        base_lr=0.05,
        use_adaptive_epsilon=True,
        epsilon_start=0.30,
        epsilon_end=0.05,
        epsilon_decay_steps=1000,
    )
    
    # 初期ε
    eps_0 = tsp.get_effective_epsilon()
    assert abs(eps_0 - 0.30) < 0.01, f"Initial epsilon should be ~0.30, got {eps_0}"
    print(f"  ✓ Initial ε: {eps_0:.3f}")
    
    # 500ステップ後（半分）
    for _ in range(500):
        tsp.record(loss=5.0, grad_norm=1.0)
    eps_500 = tsp.get_effective_epsilon()
    expected_500 = 0.30 + (0.05 - 0.30) * 0.5  # 0.175
    assert abs(eps_500 - expected_500) < 0.01, f"ε at 500 steps should be ~{expected_500}, got {eps_500}"
    print(f"  ✓ ε at 500 steps: {eps_500:.3f}")
    
    # 1000ステップ後（完了）
    for _ in range(500):
        tsp.record(loss=5.0, grad_norm=1.0)
    eps_1000 = tsp.get_effective_epsilon()
    assert abs(eps_1000 - 0.05) < 0.01, f"ε at 1000 steps should be ~0.05, got {eps_1000}"
    print(f"  ✓ ε at 1000 steps: {eps_1000:.3f}")
    
    # 減衰が完了しても下限を下回らない
    for _ in range(500):
        tsp.record(loss=5.0, grad_norm=1.0)
    eps_1500 = tsp.get_effective_epsilon()
    assert eps_1500 >= 0.05 - 1e-9, f"ε should not go below 0.05, got {eps_1500}"  # tolerance for float precision
    print(f"  ✓ ε at 1500 steps (clamped): {eps_1500:.3f}")
    print()


def test_plateau_detection():
    """プラトー検出テスト"""
    print("=" * 60)
    print("Test 4: Plateau Detection")
    print("=" * 60)
    
    tsp = create_tsp_optimizer(
        base_lr=0.05,
        window_size=10,
        eval_interval=10,
        city_preset="japanese_llm",
    )
    tsp.plateau_window_count = 3  # 3回連続停滞でプラトー
    tsp.min_dwell_steps = 5  # テスト用に短く
    
    optimizer = DummyOptimizer(lr=0.05)
    
    # 停滞するLoss（ほぼ一定）
    plateau_loss = 5.0
    
    print(f"  Initial city: {tsp.current_city.name}")
    
    # 停滞データを記録
    transition_count = 0
    for i in range(100):
        loss = plateau_loss + random.gauss(0, 0.001)  # 非常に小さい変動
        grad = 1.0 + random.gauss(0, 0.01)
        
        evt = tsp.step(i + 1, loss, grad, optimizer)
        if evt is not None:
            transition_count += 1
            print(f"    Step {i+1}: Transition {evt.from_city} → {evt.to_city}")
    
    print(f"  ✓ Plateau counter: {tsp._plateau_counter}")
    print(f"  ✓ Total transitions: {transition_count}")
    print()


def test_state_persistence():
    """状態保存/復元テスト"""
    print("=" * 60)
    print("Test 5: State Persistence")
    print("=" * 60)
    
    # 元のTSP
    tsp1 = create_tsp_optimizer(
        base_lr=0.05,
        city_preset="japanese_llm",
        use_adaptive_epsilon=True,
    )
    
    optimizer = DummyOptimizer(lr=0.05)
    
    # いくつかのステップを実行して状態を変更
    for i in range(50):
        loss = 5.0 - 0.01 * i + random.gauss(0, 0.1)
        grad = 1.0 + random.gauss(0, 0.1)
        tsp1.step(i + 1, loss, grad, optimizer)
    
    print(f"  Original TSP state:")
    print(f"    - City: {tsp1.current_city.name}")
    print(f"    - Steps in city: {tsp1.steps_in_city}")
    print(f"    - Total steps: {tsp1.total_steps}")
    print(f"    - Transitions: {tsp1.transition_count}")
    print(f"    - ε: {tsp1.get_effective_epsilon():.3f}")
    
    # 状態を保存
    state = tsp1.state_dict()
    
    # 新しいTSPを作成して状態を復元
    tsp2 = create_tsp_optimizer(
        base_lr=0.05,
        city_preset="japanese_llm",
        use_adaptive_epsilon=True,
    )
    tsp2.load_state_dict(state)
    
    print(f"  Restored TSP state:")
    print(f"    - City: {tsp2.current_city.name}")
    print(f"    - Steps in city: {tsp2.steps_in_city}")
    print(f"    - Total steps: {tsp2.total_steps}")
    print(f"    - Transitions: {tsp2.transition_count}")
    print(f"    - ε: {tsp2.get_effective_epsilon():.3f}")
    
    # 検証
    assert tsp1.current_city.name == tsp2.current_city.name, "City should match"
    assert tsp1.steps_in_city == tsp2.steps_in_city, "Steps in city should match"
    assert tsp1.total_steps == tsp2.total_steps, "Total steps should match"
    assert tsp1.transition_count == tsp2.transition_count, "Transition count should match"
    
    print(f"  ✓ All state fields restored correctly!")
    print()


def test_metrics_summary():
    """メトリクスサマリーテスト"""
    print("=" * 60)
    print("Test 6: Metrics Summary")
    print("=" * 60)
    
    tsp = create_tsp_optimizer(base_lr=0.05, city_preset="japanese_llm")
    
    # データを記録
    for i in range(20):
        tsp.record(loss=5.0 - 0.1 * i, grad_norm=1.0)
    
    summary = tsp.get_metrics_summary()
    
    print(f"  Metrics summary:")
    for key, value in summary.items():
        print(f"    - {key}: {value}")
    
    assert "tsp_city" in summary, "Should have tsp_city"
    assert "tsp_epsilon" in summary, "Should have tsp_epsilon"
    assert "tsp_total_steps" in summary, "Should have tsp_total_steps"
    
    print(f"  ✓ All expected metrics present!")
    print()


def test_emergency_recovery():
    """緊急退避テスト"""
    print("=" * 60)
    print("Test 7: Emergency Recovery")
    print("=" * 60)
    
    tsp = create_tsp_optimizer(
        base_lr=0.05,
        window_size=10,
        eval_interval=10,
        city_preset="japanese_llm",
    )
    tsp.min_dwell_steps = 5  # テスト用に短く
    
    optimizer = DummyOptimizer(lr=0.05)
    
    print(f"  Initial city: {tsp.current_city.name}")
    
    # 正常なデータ
    for i in range(20):
        tsp.step(i + 1, loss=5.0 + random.gauss(0, 0.1), grad_norm=1.0, optimizer=optimizer)
    
    print(f"  After normal data: {tsp.current_city.name}")
    
    # 高変動データ（緊急条件をトリガー）
    for i in range(20):
        # cv_loss > emergency_loss_cv (0.05) になるような高変動
        loss = 5.0 + random.gauss(0, 1.0)  # 高いstd
        evt = tsp.step(i + 21, loss, grad_norm=5.0 + random.gauss(0, 2.0), optimizer=optimizer)
        if evt is not None:
            print(f"    Step {i+21}: Emergency transition → {evt.to_city}")
    
    print(f"  ✓ Emergency cooldown remaining: {tsp._emergency_cooldown_remaining}")
    print()


def main():
    """全テストを実行"""
    print("\n" + "=" * 60)
    print("TSP Path Optimizer v2 Test Suite")
    print("=" * 60 + "\n")
    
    tests = [
        test_basic_creation,
        test_japanese_llm_preset,
        test_adaptive_epsilon,
        test_plateau_detection,
        test_state_persistence,
        test_metrics_summary,
        test_emergency_recovery,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n🎉 All tests passed! TSP v2 is working correctly.\n")
    else:
        print(f"\n⚠️ {failed} test(s) failed.\n")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
