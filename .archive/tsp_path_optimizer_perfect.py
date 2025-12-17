#!/usr/bin/env python3
"""
TSP Path Optimizer - 巡回セールスマン的学習経路最適化（改良版）

徹平さんの元コード（ε-greedy + 1軸「安定化強度」都市）を、実運用向けに強化しました。

主な改善点（要約）
- 入力バリデーション（window_size / epsilon / city stability など）
- 評価窓(window)と評価間隔(interval)の分離（“100ステップごと評価”を柔軟化）
- 変動のスケール不変判定（CV = std/mean を loss/grad 両方に適用）
- 「望ましい安定度(desired_stability)」を連続値で推定 → 都市を距離最小で選択
- ヒステリシス + 最低滞在ステップ(min_dwell_steps)で“パタパタ遷移”を抑制
- 緊急退避（NaN/Inf / 極端な振動・勾配）で最安定都市へ
- OptimizerはPyTorchに依存しない Protocol で受ける（param_groups だけあればOK）
- clip/feeder/ghost は callback で外部適用できるように（lr以外も一括で管理可能）

使い方（学習ループ内の例）
    tsp.record(loss, grad_norm)
    evt = tsp.evaluate_and_transition(step, optimizer, apply_extras=my_apply)
    if evt:
        print(evt)

またはワンショット:
    evt = tsp.step(step, loss, grad_norm, optimizer, apply_extras=my_apply)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Protocol
from collections import deque
import math
import random


# -----------------------------
# Interfaces
# -----------------------------
class OptimizerLike(Protocol):
    """PyTorch optimizer互換: param_groupsに lr を持つ辞書のリストがあればOK"""
    param_groups: List[Dict[str, float]]


ApplyExtrasFn = Callable[["City"], None]


# -----------------------------
# Data classes
# -----------------------------
@dataclass(frozen=True)
class City:
    """
    学習設定 = 都市（安定化強度1軸）

    stability: 0.0 (探索寄り) → 1.0 (最安定)
    """
    name: str
    stability: float
    lr_scale: float
    clip_value: float
    feeder_enabled: bool
    ghost_enabled: bool

    def __post_init__(self) -> None:
        if not (0.0 <= self.stability <= 1.0):
            raise ValueError(f"City.stability must be in [0, 1]. got {self.stability!r}")
        if self.lr_scale <= 0.0:
            raise ValueError(f"City.lr_scale must be > 0. got {self.lr_scale!r}")
        if self.clip_value <= 0.0:
            raise ValueError(f"City.clip_value must be > 0. got {self.clip_value!r}")

    def apply_lr(self, optimizer: OptimizerLike, base_lr: float) -> float:
        """この都市のLR設定をoptimizerへ適用し、適用後LRを返す"""
        lr = base_lr * self.lr_scale
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr


DEFAULT_CITIES: List[City] = [
    City("A_explore",   stability=0.00, lr_scale=1.0, clip_value=1.0, feeder_enabled=True,  ghost_enabled=False),
    City("B_stabilize", stability=0.35, lr_scale=0.5, clip_value=0.8, feeder_enabled=True,  ghost_enabled=False),
    City("C_converge",  stability=0.70, lr_scale=0.2, clip_value=0.6, feeder_enabled=True,  ghost_enabled=True),
    City("D_final",     stability=1.00, lr_scale=0.1, clip_value=0.6, feeder_enabled=False, ghost_enabled=True),
]


@dataclass(frozen=True)
class WindowMetrics:
    """window内で計測した統計値"""
    mean_loss: float
    std_loss: float
    cv_loss: float
    loss_slope: float  # 1 stepあたりの傾き（負が良い）

    mean_grad: float
    std_grad: float
    cv_grad: float
    grad_slope: float  # 1 stepあたりの傾き（目安）

    score: float       # 参考：小さいほど良い（従来Jを踏襲）


@dataclass(frozen=True)
class TransitionEvent:
    """遷移イベント"""
    step: int
    from_city: str
    to_city: str
    desired_stability: float
    metrics: WindowMetrics
    steps_in_city: int
    effective_lr: float


# -----------------------------
# Helpers (math/stat)
# -----------------------------
def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    """母標準偏差（len=1でも0に）"""
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / n
    return math.sqrt(var)


def _linear_slope(y: List[float]) -> float:
    """
    最小二乗の直線回帰 y = a*x + b の a。
    係数aは “1 stepあたりの変化量” になる（負が良い）。
    """
    n = len(y)
    if n < 2:
        return 0.0

    x_mean = (n - 1) / 2.0
    y_mean = _mean(y)

    num = 0.0
    den = 0.0
    for i, yi in enumerate(y):
        dx = i - x_mean
        num += dx * (yi - y_mean)
        den += dx * dx

    if den <= 1e-12:
        return 0.0
    return num / den


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _rescale01(x: float, lo: float, hi: float) -> float:
    """xを[lo,hi]区間で0..1へ線形変換（外はclamp）"""
    if hi <= lo:
        return 0.0
    return _clamp((x - lo) / (hi - lo), 0.0, 1.0)


def _is_finite(x: float) -> bool:
    return math.isfinite(x)


# -----------------------------
# Optimizer core
# -----------------------------
@dataclass
class TSPPathOptimizer:
    """
    巡回セールスマン的メタ最適化器（実運用強化版）

    - 都市は stability(0..1) で一直線上に並ぶ前提。
    - 直近windowから「望ましい安定度 desired_stability」を推定し、
      そこに最も近い都市へ遷移（ε-greedy探索も可）。
    """
    cities: List[City] = field(default_factory=lambda: list(DEFAULT_CITIES))
    base_lr: float = 0.05

    # 評価
    window_size: int = 100
    eval_interval: Optional[int] = None  # Noneならwindow_sizeと同じ

    # 探索
    epsilon: float = 0.10

    # “パタパタ防止”
    min_dwell_steps: int = 200              # 最低滞在（評価間隔より大きめ推奨）
    hysteresis_delta: float = 0.10          # desired_stability変化がこれ未満なら据え置き
    desired_ema: float = 0.80               # desired_stability のEMA（0.0で平滑なし）

    # CV閾値（loss/grad）
    loss_cv_low: float = 0.010
    loss_cv_high: float = 0.030
    grad_cv_low: float = 0.050
    grad_cv_high: float = 0.150

    # 停滞判定（loss_slope / mean_loss の閾値）
    stagnation_norm_slope: float = -1e-4    # これより上（例: 0や+）だと停滞扱い

    # 緊急退避（このどれかを満たすと最安定都市へ）
    emergency_loss_cv: float = 0.050
    emergency_grad_cv: float = 0.250

    # スコア（参考：従来Jを踏襲）
    alpha: float = 1.0  # loss振動
    beta: float = 0.5   # grad振動
    gamma: float = 2.0  # loss_slope（負が良い）
    lambda_stay: float = 0.001  # 滞在コスト

    # 履歴
    max_history: int = 2000
    loss_history: Deque[float] = field(default_factory=lambda: deque(maxlen=2000))
    grad_history: Deque[float] = field(default_factory=lambda: deque(maxlen=2000))

    # 状態
    current_city: Optional[City] = None
    steps_in_city: int = 0
    transition_log: List[Dict[str, float]] = field(default_factory=list)

    # RNG（再現性）
    seed: Optional[int] = None
    _rng: random.Random = field(init=False, repr=False)

    # desired_stabilityの状態
    _desired_prev: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # validate
        if self.base_lr <= 0.0:
            raise ValueError("base_lr must be > 0")
        if self.window_size < 2:
            raise ValueError("window_size must be >= 2")
        if self.eval_interval is None:
            self.eval_interval = self.window_size
        if self.eval_interval < 1:
            raise ValueError("eval_interval must be >= 1")
        if not (0.0 <= self.epsilon <= 1.0):
            raise ValueError("epsilon must be in [0, 1]")
        if self.min_dwell_steps < 0:
            raise ValueError("min_dwell_steps must be >= 0")
        if not (0.0 <= self.desired_ema <= 1.0):
            raise ValueError("desired_ema must be in [0, 1]")
        if self.max_history < self.window_size:
            raise ValueError("max_history must be >= window_size")

        # ensure deques maxlen matches max_history
        if self.loss_history.maxlen != self.max_history:
            self.loss_history = deque(self.loss_history, maxlen=self.max_history)
        if self.grad_history.maxlen != self.max_history:
            self.grad_history = deque(self.grad_history, maxlen=self.max_history)

        # cities
        if not self.cities:
            raise ValueError("cities must not be empty")
        # sort by stability (1D axis)
        self.cities = sorted(self.cities, key=lambda c: c.stability)

        # current city default: least stable (探索寄り)
        if self.current_city is None:
            self.current_city = self.cities[0]

        self._rng = random.Random(self.seed)

    # ----- public API -----
    def record(self, loss: float, grad_norm: float) -> None:
        """毎ステップの指標を記録"""
        self.loss_history.append(float(loss))
        self.grad_history.append(float(grad_norm))
        self.steps_in_city += 1

    def step(
        self,
        step: int,
        loss: float,
        grad_norm: float,
        optimizer: OptimizerLike,
        apply_extras: Optional[ApplyExtrasFn] = None,
    ) -> Optional[TransitionEvent]:
        """record + evaluate_and_transition を1回で行う便利メソッド"""
        self.record(loss, grad_norm)
        return self.evaluate_and_transition(step, optimizer, apply_extras=apply_extras)

    def should_evaluate(self, step: int) -> bool:
        return (
            step > 0
            and (step % int(self.eval_interval) == 0)
            and (len(self.loss_history) >= self.window_size)
            and (len(self.grad_history) >= self.window_size)
        )

    def evaluate_and_transition(
        self,
        step: int,
        optimizer: OptimizerLike,
        apply_extras: Optional[ApplyExtrasFn] = None,
    ) -> Optional[TransitionEvent]:
        """
        直近windowを評価し、必要なら都市遷移する。

        - lr はここでoptimizerへ適用
        - clip/feeder/ghost等は apply_extras(city) で外部適用可能
        """
        if not self.should_evaluate(step):
            return None

        metrics = self._evaluate_window()
        desired = self._desired_stability(metrics)

        # ヒステリシス（desiredがほぼ同じなら据え置き）
        if self._desired_prev is not None and abs(desired - self._desired_prev) < self.hysteresis_delta:
            desired = self._desired_prev
        self._desired_prev = desired

        next_city = self._choose_city(desired, metrics)

        # 最低滞在ステップ（ただし緊急退避は優先）
        if (not self._is_emergency(metrics)) and (self.steps_in_city < self.min_dwell_steps):
            next_city = self.current_city

        if next_city == self.current_city:
            # 据え置きでもログは残しておく（デバッグが楽）
            self._append_log(step, self.current_city, next_city, desired, metrics)
            return None

        # 遷移実行
        from_city = self.current_city
        self.current_city = next_city
        self.steps_in_city = 0

        effective_lr = next_city.apply_lr(optimizer, self.base_lr)
        if apply_extras is not None:
            apply_extras(next_city)

        self._append_log(step, from_city, next_city, desired, metrics, effective_lr=effective_lr)

        return TransitionEvent(
            step=step,
            from_city=from_city.name if from_city else "None",
            to_city=next_city.name,
            desired_stability=desired,
            metrics=metrics,
            steps_in_city=self.steps_in_city,
            effective_lr=effective_lr,
        )

    def get_current_config(self) -> Dict[str, object]:
        """現在の都市設定（外部適用用）"""
        c = self.current_city
        if c is None:
            return {}
        return {
            "city": c.name,
            "stability": c.stability,
            "lr_scale": c.lr_scale,
            "clip_value": c.clip_value,
            "feeder_enabled": c.feeder_enabled,
            "ghost_enabled": c.ghost_enabled,
            "effective_lr": self.base_lr * c.lr_scale,
        }

    # ----- internals -----
    def _evaluate_window(self) -> WindowMetrics:
        recent_loss = list(self.loss_history)[-self.window_size:]
        recent_grad = list(self.grad_history)[-self.window_size:]

        mean_loss = _mean(recent_loss)
        std_loss = _std(recent_loss)
        cv_loss = std_loss / max(abs(mean_loss), 1e-12)
        loss_slope = _linear_slope(recent_loss)

        mean_grad = _mean(recent_grad)
        std_grad = _std(recent_grad)
        cv_grad = std_grad / max(abs(mean_grad), 1e-12)
        grad_slope = _linear_slope(recent_grad)

        # 参考スコア: 元コードのJを踏襲（progress=loss_slope）
        stay_cost = self.lambda_stay * float(self.steps_in_city)
        score = self.alpha * std_loss + self.beta * std_grad + self.gamma * loss_slope + stay_cost

        return WindowMetrics(
            mean_loss=mean_loss,
            std_loss=std_loss,
            cv_loss=cv_loss,
            loss_slope=loss_slope,
            mean_grad=mean_grad,
            std_grad=std_grad,
            cv_grad=cv_grad,
            grad_slope=grad_slope,
            score=score,
        )

    def _desired_stability(self, m: WindowMetrics) -> float:
        """
        直近windowから “欲しい安定度(0..1)” を推定。
        - CVが高いほど安定度↑
        - CVが低くて停滞しているなら安定度↓（探索寄り）
        """
        # NaN/Inf なら最安定
        if not all(_is_finite(v) for v in [m.mean_loss, m.std_loss, m.cv_loss, m.loss_slope, m.mean_grad, m.std_grad, m.cv_grad]):
            return 1.0

        press_loss = _rescale01(m.cv_loss, self.loss_cv_low, self.loss_cv_high)
        press_grad = _rescale01(m.cv_grad, self.grad_cv_low, self.grad_cv_high)
        pressure = max(press_loss, press_grad)

        # 停滞判定（スケール不変化）: slope/mean
        norm_slope = m.loss_slope / max(abs(m.mean_loss), 1e-12)
        is_stagnating = norm_slope >= self.stagnation_norm_slope

        # 「振動も小さいのに進まない」→探索へ戻す
        if is_stagnating and (pressure < 0.25):
            desired = 0.0
        else:
            desired = pressure

        # EMA平滑
        if self._desired_prev is not None and self.desired_ema > 0.0:
            desired = self.desired_ema * self._desired_prev + (1.0 - self.desired_ema) * desired

        return _clamp(desired, 0.0, 1.0)

    def _choose_city(self, desired: float, m: WindowMetrics) -> City:
        """
        ε-greedy:
          - ε でランダム探索（ただし“緊急”時は探索しない）
          - それ以外は desired に最も近い都市（+ 遷移コスト）
        """
        assert self.current_city is not None

        # 緊急時は最安定都市へ
        if self._is_emergency(m):
            return self.cities[-1]

        if self.epsilon > 0.0 and (self._rng.random() < self.epsilon):
            # Exploration: 現都市以外からランダム
            others = [c for c in self.cities if c != self.current_city]
            return self._rng.choice(others) if others else self.current_city

        # Greedy: desiredに近い都市を選ぶ（距離 + 遷移ペナルティ）
        cur = self.current_city
        best = cur
        best_cost = float("inf")
        transition_penalty = 0.35  # 大ジャンプ抑制（0..1 くらいで調整）

        for c in self.cities:
            # 「目的地との差」 + 「現在地からの移動距離」
            cost = abs(c.stability - desired) + transition_penalty * abs(c.stability - cur.stability)
            if cost < best_cost:
                best_cost = cost
                best = c

        return best

    def _is_emergency(self, m: WindowMetrics) -> bool:
        """緊急退避判定"""
        if not all(_is_finite(v) for v in [m.mean_loss, m.std_loss, m.cv_loss, m.loss_slope, m.mean_grad, m.std_grad, m.cv_grad]):
            return True
        if m.cv_loss >= self.emergency_loss_cv:
            return True
        if m.cv_grad >= self.emergency_grad_cv:
            return True
        return False

    def _append_log(
        self,
        step: int,
        from_city: Optional[City],
        to_city: Optional[City],
        desired: float,
        m: WindowMetrics,
        effective_lr: Optional[float] = None,
    ) -> None:
        self.transition_log.append({
            "step": float(step),
            "from": 0.0 if from_city is None else float(from_city.stability),
            "to": 0.0 if to_city is None else float(to_city.stability),
            "from_name": (from_city.name if from_city else "None"),
            "to_name": (to_city.name if to_city else "None"),
            "desired": float(desired),
            "mean_loss": float(m.mean_loss),
            "std_loss": float(m.std_loss),
            "cv_loss": float(m.cv_loss),
            "loss_slope": float(m.loss_slope),
            "mean_grad": float(m.mean_grad),
            "std_grad": float(m.std_grad),
            "cv_grad": float(m.cv_grad),
            "grad_slope": float(m.grad_slope),
            "score": float(m.score),
            "steps_in_city": float(self.steps_in_city),
            "effective_lr": float(effective_lr) if effective_lr is not None else float("nan"),
        })


def create_tsp_optimizer(
    base_lr: float = 0.05,
    window_size: int = 100,
    eval_interval: Optional[int] = None,
    epsilon: float = 0.1,
    cities: Optional[List[City]] = None,
    seed: Optional[int] = None,
) -> TSPPathOptimizer:
    """TSP Path Optimizer を作成（ユーティリティ）"""
    return TSPPathOptimizer(
        cities=list(cities) if cities is not None else list(DEFAULT_CITIES),
        base_lr=base_lr,
        window_size=window_size,
        eval_interval=eval_interval,
        epsilon=epsilon,
        seed=seed,
    )


# -----------------------------
# Quick self-test (no torch required)
# -----------------------------
class _DummyOptimizer:
    def __init__(self, lr: float):
        self.param_groups = [{"lr": lr}]


if __name__ == "__main__":
    print("=== TSP Path Optimizer (improved) Test ===\n")

    optimizer = _DummyOptimizer(lr=0.05)
    tsp = create_tsp_optimizer(base_lr=0.05, window_size=100, eval_interval=100, epsilon=0.10, seed=42)

    print("Initial:", tsp.get_current_config())
    print()

    # 振動するloss/gradのシミュレーション
    for step in range(1, 501):
        loss = 9.0 + 0.5 * math.sin(step * 0.07) + random.gauss(0, 0.08)
        grad = 2.0 + 0.5 * math.cos(step * 0.06) + random.gauss(0, 0.15)

        evt = tsp.step(step, loss, grad, optimizer)
        if evt:
            print(f"Step {evt.step}: {evt.from_city} → {evt.to_city} | desired={evt.desired_stability:.2f} | lr={evt.effective_lr:.4f}")
            print(f"  cv_loss={evt.metrics.cv_loss:.3f} cv_grad={evt.metrics.cv_grad:.3f} slope={evt.metrics.loss_slope:.5f} score={evt.metrics.score:.3f}")

    print("\nFinal:", tsp.get_current_config())
    print("\n✅ Test completed.")
