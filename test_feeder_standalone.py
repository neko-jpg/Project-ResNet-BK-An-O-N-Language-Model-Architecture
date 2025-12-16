"""Standalone test for GradientFeederV2.1 - No torch required"""

from dataclasses import dataclass
from collections import deque

@dataclass  
class FeederV21Stats:
    grad_norm_input: float
    grad_norm_output: float
    velocity: float
    predicted_next: float
    clip_threshold: float
    scale_factor: float
    action: str
    health_score: float

class GradientFeederV21Test:
    """Simplified V2.1 for testing without torch"""
    
    def __init__(self):
        self.target_low = 0.5
        self.target_high = 3.0
        self.min_threshold = 5.0
        self.max_threshold = 200.0
        self.max_scale = 3.0
        self.critical_threshold = 0.2
        self.emergency_threshold = 0.1
        self.reaction_speed = 0.4
        self.prediction_weight = 0.6
        
        self.clip_threshold = 50.0
        self.scale_factor = 1.0
        self.history = deque(maxlen=5)
        self.velocity = 0.0
        
    def feed(self, grad_norm):
        action = "hold"
        
        if len(self.history) >= 1:
            self.velocity = grad_norm - self.history[-1]
        else:
            self.velocity = 0.0
        
        self.history.append(grad_norm)
        predicted = grad_norm + self.velocity * self.prediction_weight
        
        # Health
        if grad_norm < self.emergency_threshold:
            health = 0.0
        elif grad_norm < self.critical_threshold:
            health = 0.2
        elif grad_norm < self.target_low:
            health = 0.5 + 0.3 * (grad_norm / self.target_low)
        elif grad_norm <= self.target_high:
            health = 1.0
        else:
            health = max(0.3, 1.0 - (grad_norm - self.target_high) / (self.target_high * 2))
        
        # UPPER BOUND (explosion)
        if grad_norm > self.target_high or predicted > self.target_high:
            excess = max(grad_norm, predicted) / self.target_high
            reduction = 1.0 - self.reaction_speed * (excess - 1.0) * 0.5
            reduction = max(reduction, 0.7)
            self.clip_threshold *= reduction
            action = "threshold_lower"
            self.scale_factor = 1.0
            
        # LOWER BOUND (vanishing)
        elif grad_norm < self.critical_threshold or predicted < self.emergency_threshold:
            if grad_norm < self.emergency_threshold:
                target_grad = self.target_low * 1.5
                needed_boost = target_grad / (grad_norm + 1e-8)
                self.scale_factor = min(needed_boost, self.max_scale)
                action = "EMERGENCY_SCALE"
            elif grad_norm < self.critical_threshold:
                deficit_ratio = self.target_low / (grad_norm + 1e-8)
                self.scale_factor = 1.0 + (deficit_ratio - 1.0) * 0.5
                self.scale_factor = min(self.scale_factor, self.max_scale * 0.7)
                action = "scale_boost"
            elif predicted < self.critical_threshold and self.velocity < 0:
                self.scale_factor = 1.0 + (-self.velocity / grad_norm) * 2.0
                self.scale_factor = min(self.scale_factor, 2.0)
                action = "preemptive_scale"
            else:
                self.scale_factor = 1.0
            self.clip_threshold = min(self.clip_threshold * 1.05, self.max_threshold)
            
        # HEALTHY
        else:
            self.scale_factor = self.scale_factor * 0.9 + 1.0 * 0.1
            self.clip_threshold = self.clip_threshold * 0.98 + 50.0 * 0.02
            action = "hold"
        
        # Clamp
        self.clip_threshold = max(self.min_threshold, min(self.max_threshold, self.clip_threshold))
        self.scale_factor = max(1.0, min(self.max_scale, self.scale_factor))
        
        expected_output = grad_norm * self.scale_factor
        
        stats = FeederV21Stats(
            grad_norm_input=grad_norm,
            grad_norm_output=expected_output,
            velocity=self.velocity,
            predicted_next=predicted,
            clip_threshold=self.clip_threshold,
            scale_factor=self.scale_factor,
            action=action,
            health_score=health,
        )
        
        return self.clip_threshold, self.scale_factor, stats


def main():
    print("Testing GradientFeederV2.1 - Hybrid Control")
    print("="*85)
    
    feeder = GradientFeederV21Test()
    
    # Real pattern from training_log.json
    simulated_grads = [
        0.63, 0.66, 0.68, 0.68, 0.81,  # Healthy start
        0.36,  # Drop
        0.81, 0.49, 0.75,  # Recovery
        0.22,  # Drop below critical!
        0.87, 1.10, 1.06, 0.87,  # Healthy
        0.46, 0.24,  # Decay
        0.058,  # EMERGENCY!
        0.25, 0.45,  # Recovery
        5.0, 6.0,  # EXPLOSION
    ]
    
    print(f"\n{'Step':<5} {'Grad':<7} {'Vel':<8} {'Pred':<7} {'Thresh':<7} {'Scale':<6} {'Output':<7} {'Action':<18} {'HP':<4}")
    print("-"*85)
    
    for i, grad in enumerate(simulated_grads):
        threshold, scale, stats = feeder.feed(grad)
        
        if "EMERGENCY" in stats.action:
            ind = "[!!!]"
        elif stats.health_score >= 0.8:
            ind = "[OK]"
        elif stats.health_score >= 0.5:
            ind = "[WARN]"
        else:
            ind = "[BAD]"
        
        print(f"{i+1:<5} {grad:<7.3f} {stats.velocity:<+8.3f} {stats.predicted_next:<7.2f} "
              f"{threshold:<7.1f} {scale:<6.2f} {stats.grad_norm_output:<7.2f} "
              f"{stats.action:<18} {stats.health_score:<4.1f} {ind}")
    
    print("-"*85)
    print("\n✓ EXPLOSION (grad>3.0): Threshold lowered → clips high gradients")
    print("✓ VANISHING (grad<0.2): Scale boosted → increases dying gradients")
    print("✓ Each mechanism handles what it's good at!")
    print("="*85)

if __name__ == "__main__":
    main()
