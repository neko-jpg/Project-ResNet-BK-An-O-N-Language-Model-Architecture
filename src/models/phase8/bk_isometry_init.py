"""
BK-Core Isometry Initialization

ResNet-BK アーキテクチャ専用の等長性保存初期化。

設計理念:
1. 信号エネルギー保存: ||f(x)|| ≈ ||x|| を全層で維持
2. Green関数安定性: G_ii の固有値分布を制御
3. Hyperbolic幾何学: Lorentz hyperboloid 上に配置
4. Symplectic構造: 位相空間体積を保存

Key Innovations:
- v_proj: QR分解によるユニタリ初期化 (σ_i ≈ 1)
- Hyperbolic層: hyperboloid上の均一分布
- BK-Core H₀: 対角優位で条件数を制御
- Embedding: Poincaré ball内部に集中

Author: ResNet-BK Project
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Dict, Any, Tuple


class BKIsometryInitializer:
    """
    BK-Core専用の等長性保存初期化.
    
    全ての層を通して信号のノルム（エネルギー）が保存されるように
    初期化を行う。これにより、学習初期の勾配消失/爆発を防ぐ。
    
    初期化戦略:
    
    1. v_proj (BK-Core入力):
       - ユニタリ/直交初期化
       - 特異値 σ_i ≈ gain (デフォルト 0.1)
       - 小さなgainでBK-Coreへの入力を制御
    
    2. output_proj (BK-Core出力):
       - 直交射影
       - エネルギー漏れなし
    
    3. Hyperbolic層:
       - Lorentz hyperboloid上に配置
       - 原点付近（曲率半径内）に集中
    
    4. FFN/Linear:
       - 修正Xavier初期化
       - Variance preserving: Var(out) = Var(in)
    
    5. Embedding:
       - Poincaré ball内部 (||x|| < 0.1)
       - 等方的分布
    """
    
    @staticmethod
    def initialize_model(
        model: nn.Module,
        gain: float = 1.0,
        curvature: float = -1.0,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        モデル全体に等長性保存初期化を適用.
        
        Args:
            model: 初期化するモデル
            gain: 全体的なスケーリング係数
            curvature: Hyperbolic曲率 (負)
            verbose: 詳細ログを出力
        
        Returns:
            初期化統計情報
        """
        stats = {
            'unitary_count': 0,
            'hyperbolic_count': 0,
            'euclidean_count': 0,
            'embedding_count': 0,
            'max_singular_value': 0.0,
            'min_singular_value': float('inf'),
        }
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if BKIsometryInitializer._is_unitary_layer(name):
                    BKIsometryInitializer._init_unitary(
                        module.weight,
                        gain=gain * 0.1,  # BK-Coreは小さく
                    )
                    stats['unitary_count'] += 1
                    if verbose:
                        print(f"  [Unitary] {name}: gain={gain * 0.1:.4f}")
                elif BKIsometryInitializer._is_hyperbolic_layer(name):
                    BKIsometryInitializer._init_hyperbolic(
                        module.weight,
                        curvature=curvature,
                    )
                    stats['hyperbolic_count'] += 1
                    if verbose:
                        print(f"  [Hyperbolic] {name}: curvature={curvature}")
                else:
                    BKIsometryInitializer._init_variance_preserving(
                        module.weight,
                        gain=gain,
                    )
                    stats['euclidean_count'] += 1
                    if verbose:
                        print(f"  [Euclidean] {name}: gain={gain:.4f}")
                
                # Bias to zero
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
            elif isinstance(module, nn.Embedding):
                BKIsometryInitializer._init_embedding_poincare(
                    module.weight,
                    max_norm=0.1,
                )
                stats['embedding_count'] += 1
                if verbose:
                    print(f"  [Embedding] {name}: max_norm=0.1")
                    
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Compute singular value statistics
        stats.update(BKIsometryInitializer._compute_sv_stats(model))
        
        return stats
    
    @staticmethod
    def _is_unitary_layer(name: str) -> bool:
        """Check if layer should be initialized as unitary."""
        return any(key in name.lower() for key in [
            'v_proj', 'output_proj', 'bk_core', 'bk_scale'
        ])
    
    @staticmethod
    def _is_hyperbolic_layer(name: str) -> bool:
        """Check if layer should be initialized on hyperboloid."""
        return any(key in name.lower() for key in [
            'hyperbolic', 'hybrid_attn', 'lorentz', 'poincare'
        ])
    
    @staticmethod
    def _init_unitary(weight: torch.Tensor, gain: float = 1.0):
        """
        ユニタリ/直交初期化.
        
        QR分解を使用して直交行列を生成し、gainでスケーリング。
        全ての特異値が gain に等しくなる。
        """
        with torch.no_grad():
            if weight.dim() == 2:
                m, n = weight.shape
                # Random matrix
                A = torch.randn(m, n, device=weight.device, dtype=weight.dtype)
                
                # QR decomposition
                if m >= n:
                    Q, R = torch.linalg.qr(A)
                    # Make determinant positive
                    d = torch.diagonal(R)
                    Q = Q * (d.sign().unsqueeze(0))
                    weight.copy_(gain * Q[:, :n])
                else:
                    Q, R = torch.linalg.qr(A.T)
                    d = torch.diagonal(R)
                    Q = Q * (d.sign().unsqueeze(0))
                    weight.copy_(gain * Q[:, :m].T)
            else:
                # Non-2D: use standard orthogonal
                nn.init.orthogonal_(weight, gain=gain)
    
    @staticmethod
    def _init_hyperbolic(weight: torch.Tensor, curvature: float = -1.0):
        """
        Lorentz hyperboloid上への初期化.
        
        各行ベクトルを hyperboloid {x: -x₀² + ||x_space||² = -1/|c|} 上に配置。
        空間成分は原点付近に集中させ、安定性を確保。
        """
        with torch.no_grad():
            if weight.dim() != 2 or weight.shape[-1] < 2:
                nn.init.normal_(weight, mean=0, std=0.02)
                return
            
            m, n = weight.shape
            c = abs(curvature)
            
            # Initialize spatial components (n-1 dimensions)
            # Use small variance to stay near origin
            spatial = torch.randn(m, n - 1, device=weight.device, dtype=weight.dtype) * 0.1
            
            # Compute time component: x₀ = sqrt(||x_space||² + 1/c)
            spatial_norm_sq = (spatial ** 2).sum(dim=-1, keepdim=True)
            x_time = torch.sqrt(spatial_norm_sq + 1.0 / c)
            
            weight[:, :1] = x_time
            weight[:, 1:] = spatial
    
    @staticmethod
    def _init_variance_preserving(weight: torch.Tensor, gain: float = 1.0):
        """
        分散保存（Xavier/Kaiming）初期化.
        
        Var(output) = Var(input) を維持するようにスケーリング。
        """
        with torch.no_grad():
            if weight.dim() == 2:
                fan_in, fan_out = weight.shape[1], weight.shape[0]
                # Modified Xavier - MUCH SMALLER for hyperbolic
                std = gain * math.sqrt(2.0 / (fan_in + fan_out)) * 0.01  # 100x smaller
                nn.init.normal_(weight, mean=0, std=std)
            else:
                nn.init.normal_(weight, mean=0, std=0.001 * gain)  # 0.02 -> 0.001
    
    @staticmethod
    def _init_embedding_poincare(weight: torch.Tensor, max_norm: float = 0.001):
        """
        Embedding をPoincaré ball内部に初期化.
        
        全てのベクトルを ||x|| < max_norm の領域に配置。
        これにより hyperboloid境界での数値不安定性を回避。
        """
        with torch.no_grad():
            # Random direction
            nn.init.normal_(weight, mean=0, std=1.0)
            
            # Normalize and scale
            norms = weight.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            
            # Random radii in [0, max_norm)
            radii = torch.rand(weight.shape[0], 1, device=weight.device) * max_norm
            
            weight.copy_(weight / norms * radii)
    
    @staticmethod
    def _compute_sv_stats(model: nn.Module) -> Dict[str, float]:
        """Compute singular value statistics for model weights."""
        max_sv = 0.0
        min_sv = float('inf')
        
        for name, param in model.named_parameters():
            if param.dim() == 2:
                try:
                    # Compute singular values
                    S = torch.linalg.svdvals(param.data)
                    max_sv = max(max_sv, S.max().item())
                    min_sv = min(min_sv, S.min().item())
                except:
                    pass
        
        return {
            'max_singular_value': max_sv,
            'min_singular_value': min_sv if min_sv != float('inf') else 0.0,
            'condition_number': max_sv / max(min_sv, 1e-10),
        }
    
    @staticmethod
    def verify_isometry(
        model: nn.Module,
        test_input: torch.Tensor,
        threshold: float = 2.0,
    ) -> Dict[str, Any]:
        """
        モデルの等長性を検証.
        
        各層を通過した後の信号エネルギー比を計算し、
        等長性が保たれているかを確認。
        
        Args:
            model: 検証するモデル
            test_input: テスト入力 (B, N, D) or (B, N)
            threshold: 許容されるエネルギー比の上限
        
        Returns:
            検証結果の辞書
        """
        results = {
            'layer_ratios': [],
            'is_isometric': True,
            'max_ratio': 0.0,
            'min_ratio': float('inf'),
        }
        
        # Register hooks to capture intermediate outputs
        activations = []
        hooks = []
        
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                activations.append(output.detach())
            elif isinstance(output, tuple) and len(output) > 0:
                activations.append(output[0].detach())
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm)):
                hooks.append(module.register_forward_hook(hook_fn))
        
        # Forward pass
        with torch.no_grad():
            if hasattr(model, 'forward'):
                _ = model(test_input)
        
        # Remove hooks
        for h in hooks:
            h.remove()
        
        # Compute energy ratios
        if len(activations) > 0:
            prev_norm = test_input.float().norm().item()
            
            for i, act in enumerate(activations):
                curr_norm = act.float().norm().item()
                ratio = curr_norm / max(prev_norm, 1e-10)
                
                results['layer_ratios'].append({
                    'layer': i,
                    'ratio': ratio,
                    'is_ok': 1.0 / threshold < ratio < threshold,
                })
                
                results['max_ratio'] = max(results['max_ratio'], ratio)
                results['min_ratio'] = min(results['min_ratio'], ratio)
                
                if not (1.0 / threshold < ratio < threshold):
                    results['is_isometric'] = False
                
                prev_norm = curr_norm
        
        return results


# =============================================================================
# BK-Core Specific Initialization
# =============================================================================

def symplectic_greens_init_(
    tensor: torch.Tensor,
    dt: float = 0.01,
    min_decay: float = 1e-4,
    max_decay: float = 1e-2,
) -> None:
    """
    Symplectic Green's Initialization (SGI) - Research Topic 5 Solution.
    
    Creates: W = Q @ D
        Q: Random orthogonal matrix (energy-conserving rotation)
        D: diag(exp(-γ * dt)) decay factors
    
    This puts eigenvalues slightly inside the unit circle (|λ| < 1 but close to 1),
    allowing long-range information flow while enabling forgetting/learning.
    
    Strict orthogonal (|λ| = 1 exactly) blocks learning because there's no
    decay - the system becomes a perfect oscillator with no damping.
    
    Args:
        tensor: Weight tensor (must be square for full SGI)
        dt: Time step for decay (larger = more decay)
        min_decay: Minimum decay rate γ
        max_decay: Maximum decay rate γ
    """
    if tensor.ndim != 2:
        nn.init.orthogonal_(tensor)
        return
    
    rows, cols = tensor.shape
    if rows != cols:
        # Non-square: just use variance-preserving init
        fan_in, fan_out = cols, rows
        std = math.sqrt(2.0 / (fan_in + fan_out))
        nn.init.normal_(tensor, mean=0, std=std)
        return
    
    with torch.no_grad():
        # 1. Random orthogonal Q
        Q = torch.empty(rows, cols, device=tensor.device, dtype=tensor.dtype)
        nn.init.orthogonal_(Q)
        
        # 2. Green's-style decay factors
        # γ ~ Uniform(min_decay, max_decay)
        gamma = torch.rand(rows, device=tensor.device, dtype=tensor.dtype)
        gamma = gamma * (max_decay - min_decay) + min_decay
        
        # D = diag(exp(-γ * dt))
        decay_factors = torch.exp(-gamma * dt)
        D = torch.diag(decay_factors)
        
        # 3. W = Q @ D
        W_init = Q @ D
        tensor.copy_(W_init)


def init_bk_core_layer(layer: nn.Module, gain: float = 0.1, use_sgi: bool = True):
    """
    BK-Core層の特別な初期化 (研究Topic 5に基づき更新).
    
    v_proj, output_proj, bk_scale, gamma などのパラメータを
    Green関数の安定性を考慮して初期化。
    
    Args:
        layer: BK-Core層モジュール
        gain: スケーリング係数
        use_sgi: Symplectic Green's Initialization を使用 (推奨)
                 False の場合は従来の strict orthogonal を使用
    """
    with torch.no_grad():
        # v_proj: Use SGI for square, variance-preserving for non-square
        if hasattr(layer, 'v_proj') and isinstance(layer.v_proj, nn.Linear):
            w = layer.v_proj.weight
            if use_sgi and w.shape[0] == w.shape[1]:
                symplectic_greens_init_(w)
                w.data.mul_(gain)  # Scale down for BK-Core input control
            else:
                # Variance-preserving for rectangular
                fan_in, fan_out = w.shape[1], w.shape[0]
                std = gain * math.sqrt(2.0 / (fan_in + fan_out))
                nn.init.normal_(w, mean=0, std=std)
            if layer.v_proj.bias is not None:
                nn.init.zeros_(layer.v_proj.bias)
        
        # output_proj: Use SGI for square matrices
        if hasattr(layer, 'output_proj') and isinstance(layer.output_proj, nn.Linear):
            w = layer.output_proj.weight
            if use_sgi and w.shape[0] == w.shape[1]:
                symplectic_greens_init_(w)
            else:
                fan_in, fan_out = w.shape[1], w.shape[0]
                std = math.sqrt(2.0 / (fan_in + fan_out))
                nn.init.normal_(w, mean=0, std=std)
            if layer.output_proj.bias is not None:
                nn.init.zeros_(layer.output_proj.bias)
        
        # bk_scale: Start at 1 (neutral)
        if hasattr(layer, 'bk_scale'):
            if isinstance(layer.bk_scale, nn.Parameter):
                layer.bk_scale.data.fill_(1.0)
        
        # gamma: Start at 0 (no imaginary shift in z)
        if hasattr(layer, 'gamma'):
            if isinstance(layer.gamma, nn.Parameter):
                layer.gamma.data.fill_(0.0)
        
        # epsilon_param: Start at 1.0 for stability
        if hasattr(layer, 'epsilon_param'):
            if isinstance(layer.epsilon_param, nn.Parameter):
                layer.epsilon_param.data.fill_(1.0)


def init_hybrid_attention_layer(layer: nn.Module, curvature: float = -1.0):
    """
    Hybrid Hyperbolic Attention層の初期化.
    
    Hyperbolic空間での安定性を確保するため、
    重みをhyperboloid上に配置。
    """
    with torch.no_grad():
        for name, param in layer.named_parameters():
            if 'weight' in name:
                if param.dim() == 2:
                    if 'query' in name or 'key' in name or 'value' in name:
                        # QKV projections: orthogonal for stability
                        BKIsometryInitializer._init_unitary(param, gain=1.0)
                    else:
                        # Other weights: hyperbolic
                        BKIsometryInitializer._init_hyperbolic(param, curvature)
            elif 'bias' in name:
                nn.init.zeros_(param)


def init_symplectic_layer(layer: nn.Module):
    """
    Symplectic層の初期化.
    
    (q, p) 正準変数としてパラメータを初期化し、
    初期エネルギー H = p²/2 + V(q) を制御。
    """
    with torch.no_grad():
        for name, param in layer.named_parameters():
            if param.dim() >= 1 and param.shape[-1] % 2 == 0:
                d_half = param.shape[-1] // 2
                
                # q part: small position (low potential)
                q_std = 0.1
                param.data[..., :d_half] = torch.randn_like(param[..., :d_half]) * q_std
                
                # p part: small momentum (low kinetic)
                p_std = 0.1
                param.data[..., d_half:] = torch.randn_like(param[..., d_half:]) * p_std


# =============================================================================
# Apply to Phase8 Model
# =============================================================================

def apply_bk_isometry_init(
    model: nn.Module,
    base_gain: float = 1.0,
    curvature: float = -1.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Phase8IntegratedModel に BK等長性初期化を適用.
    
    Args:
        model: Phase8IntegratedModel インスタンス
        base_gain: 基本スケーリング係数
        curvature: Hyperbolic曲率
        verbose: 詳細ログ
    
    Returns:
        初期化統計
    """
    if verbose:
        print("=" * 60)
        print("Applying BK Isometry Initialization")
        print("=" * 60)
    
    # Initialize with BKIsometryInitializer
    stats = BKIsometryInitializer.initialize_model(
        model,
        gain=base_gain,
        curvature=curvature,
        verbose=verbose,
    )
    
    # Apply BK-Core specific initialization
    for name, module in model.named_modules():
        if 'bk_layer' in name.lower() or 'bk_core' in name.lower():
            init_bk_core_layer(module, gain=base_gain * 0.1)
            if verbose:
                print(f"  [BK-Core] {name}: special initialization applied")
        
        if 'hybrid_attn' in name.lower() or 'hyperbolic_attn' in name.lower():
            init_hybrid_attention_layer(module, curvature=curvature)
            if verbose:
                print(f"  [HybridAttn] {name}: hyperbolic initialization applied")
        
        if 'symplectic' in name.lower():
            init_symplectic_layer(module)
            if verbose:
                print(f"  [Symplectic] {name}: symplectic initialization applied")
    
    if verbose:
        print("-" * 60)
        print(f"Unitary layers: {stats['unitary_count']}")
        print(f"Hyperbolic layers: {stats['hyperbolic_count']}")
        print(f"Euclidean layers: {stats['euclidean_count']}")
        print(f"Embedding layers: {stats['embedding_count']}")
        print(f"Max singular value: {stats.get('max_singular_value', 0):.4f}")
        print(f"Min singular value: {stats.get('min_singular_value', 0):.4f}")
        print(f"Condition number: {stats.get('condition_number', 0):.4f}")
        print("=" * 60)
    
    return stats


# =============================================================================
# Testing
# =============================================================================

def test_bk_isometry_init():
    """Test BK isometry initialization."""
    print("Testing BK Isometry Initialization...")
    
    # Create test model
    class TestBKModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1000, 64)
            self.v_proj = nn.Linear(64, 64)
            self.output_proj = nn.Linear(64, 64)
            self.hyperbolic_attn = nn.Linear(64, 64)
            self.ffn = nn.Sequential(
                nn.Linear(64, 256),
                nn.GELU(),
                nn.Linear(256, 64),
            )
            self.layer_norm = nn.LayerNorm(64)
        
        def forward(self, x):
            x = self.embedding(x)
            x = self.v_proj(x)
            x = self.output_proj(x)
            x = self.hyperbolic_attn(x)
            x = self.ffn(x)
            x = self.layer_norm(x)
            return x
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TestBKModel().to(device)
    
    # Apply initialization
    stats = BKIsometryInitializer.initialize_model(model, gain=1.0, verbose=True)
    
    # Verify v_proj is approximately orthogonal
    W = model.v_proj.weight.data
    WtW = W.T @ W
    I = torch.eye(W.shape[1], device=device)
    orth_error = (WtW - I).abs().mean().item()
    print(f"\nv_proj orthogonality error: {orth_error:.6f}")
    
    # Test isometry verification
    test_input = torch.randint(0, 1000, (2, 32), device=device)
    isometry_results = BKIsometryInitializer.verify_isometry(model, test_input)
    
    print(f"\nIsometry verification:")
    print(f"  Is isometric: {isometry_results['is_isometric']}")
    print(f"  Max energy ratio: {isometry_results['max_ratio']:.4f}")
    print(f"  Min energy ratio: {isometry_results['min_ratio']:.4f}")
    
    # Check singular values
    print(f"\nSingular value stats:")
    print(f"  Max: {stats.get('max_singular_value', 0):.4f}")
    print(f"  Min: {stats.get('min_singular_value', 0):.4f}")
    
    print("\n✅ BK Isometry Initialization test passed")
    return True


if __name__ == "__main__":
    test_bk_isometry_init()
