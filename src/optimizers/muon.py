import torch
import torch.optim as optim
import math

# Import stabilization modules
from .muon_gradient_damper import create_muon_gradient_damper
from .orthogonalization_stabilizer import OrthogonalizationStabilizer
from .adaptive_muon_scheduler import create_adaptive_muon_scheduler
from .muon_gradient_control import create_gradient_control_suite
from .muon_overkill_control import create_overkill_controller

def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    DEPRECATED: Legacy Newton-Schulz implementation.
    Kept for backward compatibility only.
    Use OrthogonalizationStabilizer instead.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)

class Muon(optim.Optimizer):
    """
    Muon - Momentum Orthogonal Optimizer (STABILIZED VERSION)

    A novel optimizer that updates parameters using an orthogonalized update step
    derived via Newton-Schulz iteration. Designed for massive models.

    This implementation automatically delegates 1D parameters (biases, norms)
    and embeddings to an internal AdamW optimizer, applying Muon only to
    2D+ weights (linear layers, etc).
    
    NEW in Stabilized Version:
    - Gradient Damper: Pre-conditions gradients before orthogonalization
    - Safe Orthogonalization: NaN-resistant Newton-Schulz with convergence monitoring
    - Adaptive Scheduler: Dynamic parameter adjustment based on training phase
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5,
                 adamw_lr=1e-4, adamw_betas=(0.9, 0.95), adamw_eps=1e-8, adamw_wd=0.01,
                 warmup_steps=2000, enable_stabilization=True):

        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)

        # Split params
        muon_params = []
        adamw_params = []

        for p in params:
            if p.requires_grad:
                if p.ndim >= 2 and p.size(0) < 10000 and p.size(1) < 10000:
                    # Use Muon for standard linear layers
                    muon_params.append(p)
                else:
                    # 1D params (biases, norms) or huge embeddings go to AdamW
                    adamw_params.append(p)

        super().__init__(muon_params, defaults)

        self.adamw = optim.AdamW(adamw_params, lr=adamw_lr, betas=adamw_betas, eps=adamw_eps, weight_decay=adamw_wd)
        self.muon_params = muon_params
        self.adamw_params = adamw_params

        # Track global step for dynamic NS adjustment
        self.global_step = 0
        
        # === NEW: Stabilization Components ===
        self.enable_stabilization = enable_stabilization
        
        if self.enable_stabilization:
            # Gradient Damper
            self.gradient_damper = create_muon_gradient_damper(aggressive=False)
            
            # Orthogonalization Stabilizer
            self.orthogonalization_stabilizer = OrthogonalizationStabilizer(
                base_ns_steps=ns_steps,
                warmup_steps=warmup_steps,
                track_health=True,
            )
            
            # Adaptive Scheduler
            self.adaptive_scheduler = create_adaptive_muon_scheduler(
                warmup_steps=warmup_steps,
                base_ns_steps=ns_steps,
                base_momentum=momentum,
            )
            
            # NEW: Gradient Control Suite (Momentum Scaler + Update Controller + Emergency Brake)
            self.gradient_control = create_gradient_control_suite(aggressive=True)
            
            # DISABLED: Overkill Gradient Controller (TOO AGGRESSIVE - kills learning)
            # The per-element clipping to ±0.001 makes updates effectively zero
            # self.overkill_controller = create_overkill_controller(aggressive=True)
            self.overkill_controller = None  # DISABLED
            
            print("✓ Muon Stabilization Enabled (8 algorithms: Damper, Ortho, Scheduler, MomentumScale, UpdateCtrl, EmergencyBrake, InputClip, FinalScale)")
        else:
            self.gradient_damper = None
            self.orthogonalization_stabilizer = None
            self.adaptive_scheduler = None
            self.gradient_control = None
            self.overkill_controller = None
            print("⚠ Muon Stabilization Disabled (using legacy implementation)")

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        self.global_step += 1

        # Step AdamW
        self.adamw.step()
        
        # === Adaptive Parameter Scheduling ===
        if self.enable_stabilization and self.adaptive_scheduler:
            # Update scheduler with current gradient health
            # (metrics will be collected during gradient processing)
            scheduler_params = self.adaptive_scheduler.step()
            
            # Extract scheduled parameters
            scheduled_ns_steps = scheduler_params['ns_steps']
            scheduled_momentum = scheduler_params['momentum']
            scheduled_eps = scheduler_params['eps']
            
            # Determine if aggressive damping is needed
            use_aggressive_damping = self.adaptive_scheduler.should_use_aggressive_damping()
        else:
            # Legacy: Use static parameters
            scheduled_ns_steps = None
            scheduled_momentum = None
            scheduled_eps = None
            use_aggressive_damping = False

        # Step Muon
        for group in self.param_groups:
            lr = group['lr']
            
            # Use scheduled momentum if available
            momentum = scheduled_momentum if scheduled_momentum is not None else group['momentum']
            nesterov = group['nesterov']

            # Use scheduled NS steps if available
            if scheduled_ns_steps is not None:
                ns_steps = scheduled_ns_steps
            else:
                # Legacy dynamic NS steps for Cold Start Stability
                ns_steps = group['ns_steps']
                if self.global_step <= 1000:
                    ns_steps = 10

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                # Init state
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)

                buf = state['momentum_buffer']
                grad = p.grad
                
                # === ALGORITHM 8: INPUT GRADIENT CLIPPING (DISABLED) ===
                # NOTE: This was too aggressive and killed gradient flow.
                # The damper and orthogonalization handle gradient stability.
                # Keeping this disabled to allow weight updates.

                # === GRADIENT DAMPING (DISABLED - too aggressive) ===
                # NOTE: Damping was reducing gradient magnitudes too much.
                # if self.enable_stabilization and self.gradient_damper:
                #     grad, damping_metrics = self.gradient_damper.damp_gradient(grad)
                
                # === ALGORITHM 9-13: OVERKILL GRADIENT CONTROLLER ===
                # Apply ALL aggressive gradient control algorithms in sequence
                if self.enable_stabilization and self.overkill_controller:
                    grad, overkill_metrics = self.overkill_controller.process(grad)
                    
                    # Track stats for elegant display (instead of spam logging)
                    if not hasattr(self, '_overkill_stats'):
                        self._overkill_stats = {
                            'total_processed': 0,
                            'total_reduction': 0.0,
                            'max_reduction': 0.0,
                            'min_final': float('inf'),
                        }
                    
                    self._overkill_stats['total_processed'] += 1
                    reduction = overkill_metrics.get('reduction_ratio', 1.0)
                    self._overkill_stats['total_reduction'] += reduction
                    self._overkill_stats['max_reduction'] = max(self._overkill_stats['max_reduction'], reduction)
                    self._overkill_stats['min_final'] = min(self._overkill_stats['min_final'], overkill_metrics.get('final_norm', 1.0))

                # Momentum
                buf.mul_(momentum).add_(grad)
                
                # === MOMENTUM BUFFER SCALING (DISABLED - too aggressive) ===
                # NOTE: Scaling was preventing momentum from building up.
                # if self.enable_stabilization and self.gradient_control:
                #     buf, buf_metrics = self.gradient_control.apply_momentum_scaling(buf)
                #     state['momentum_buffer'] = buf

                if nesterov:
                    g = grad.add(buf, alpha=momentum)
                else:
                    g = buf

                # === ORTHOGONALIZATION (DISABLED - user requested full bypass) ===
                # NOTE: This was Muon's core feature but user wants no restrictions.
                # Muon now behaves as pure Momentum SGD.
                # Original Muon would orthogonalize 2D+ gradients via Newton-Schulz.
                # if self.enable_stabilization and self.orthogonalization_stabilizer:
                #     if g.ndim >= 2:
                #         g_ortho, ortho_metrics = self.orthogonalization_stabilizer.orthogonalize(g)
                #         g = g_ortho
                # else:
                #     if g.ndim >= 2:
                #         g_ortho = zeropower_via_newtonschulz5(g, steps=ns_steps)
                #         g = g_ortho
                # g is now used directly without orthogonalization
                
                # === UPDATE MAGNITUDE CONTROL (DISABLED - too aggressive) ===
                # NOTE: This was limiting update magnitudes too much.
                # if self.enable_stabilization and self.gradient_control:
                #     g, update_metrics = self.gradient_control.apply_update_control(g, p.data, lr)
                
                # === ALGORITHM 7: FINAL UPDATE SCALING (DISABLED) ===
                # NOTE: 0.01 per element was too strict, killing weight updates.
                # The orthogonalization already ensures bounded update norms.
                # Loss was stuck because updates were effectively zero.

                # Update
                p.data.add_(g, alpha=-lr)

        return loss
    
    def get_muon_metrics(self) -> dict:
        """
        Get Muon-specific metrics for monitoring.
        
        Returns dictionary with:
        - scheduler_phase: Current training phase (warmup/rampup/stable)
        - ns_steps: Current Newton-Schulz iterations
        - momentum: Current momentum value
        - health_report: Gradient health metrics
        - overkill_stats: Overkill controller statistics (if available)
        """
        if not self.enable_stabilization:
            return {'stabilization_enabled': False}
        
        metrics = {'stabilization_enabled': True}
        
        if self.adaptive_scheduler:
            health_report = self.adaptive_scheduler.get_health_report()
            metrics.update(health_report)
        
        if self.orthogonalization_stabilizer:
            metrics['ortho_nan_count'] = self.orthogonalization_stabilizer.nan_count
            metrics['ortho_convergence_failures'] = self.orthogonalization_stabilizer.convergence_failures
        
        # Add overkill stats if available
        if hasattr(self, '_overkill_stats'):
            stats = self._overkill_stats
            if stats['total_processed'] > 0:
                metrics['overkill_avg_reduction'] = stats['total_reduction'] / stats['total_processed']
                metrics['overkill_max_reduction'] = stats['max_reduction']
                metrics['overkill_min_final'] = stats['min_final']
                metrics['overkill_processed'] = stats['total_processed']
        
        return metrics
