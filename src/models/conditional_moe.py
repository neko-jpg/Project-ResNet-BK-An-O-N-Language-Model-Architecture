"""
Conditional MoE Layer
Dynamically adjusts number of experts based on input difficulty.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalMoELayer(nn.Module):
    """
    Conditional Mixture of Experts that dynamically adjusts num_experts based on input difficulty.
    
    Uses entropy of input distribution as a proxy for difficulty:
    - Low entropy (easy): route to 1 expert
    - High entropy (hard): route to multiple experts (up to max_experts)
    
    Args:
        d_model: hidden dimension
        max_experts: maximum number of experts (default: 4)
        min_experts: minimum number of experts (default: 1)
        dropout_p: dropout probability
        entropy_threshold_low: entropy below this uses min_experts (default: 0.5)
        entropy_threshold_high: entropy above this uses max_experts (default: 2.0)
    """
    
    def __init__(
        self, 
        d_model, 
        max_experts=4, 
        min_experts=1,
        dropout_p=0.1,
        entropy_threshold_low=0.5,
        entropy_threshold_high=2.0
    ):
        super().__init__()
        self.d_model = d_model
        self.max_experts = max_experts
        self.min_experts = min_experts
        self.entropy_threshold_low = entropy_threshold_low
        self.entropy_threshold_high = entropy_threshold_high
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(d_model * 2, d_model),
            )
            for _ in range(max_experts)
        ])
        
        # Gating network
        self.gating_network = nn.Linear(d_model, max_experts)
        
        # Difficulty predictor (predicts entropy from input)
        self.difficulty_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Softplus()  # Ensure positive output (entropy is non-negative)
        )
        
        # Statistics tracking
        self.register_buffer('avg_num_experts_used', torch.tensor(0.0))
        self.register_buffer('num_forward_calls', torch.tensor(0))
        
    def compute_input_entropy(self, x):
        """
        Compute entropy of input distribution as a measure of difficulty.
        
        Uses the difficulty predictor to estimate entropy from input features.
        
        Args:
            x: (B, N, D) input tensor
            
        Returns:
            entropy: (B, N) entropy values
        """
        # Use learned difficulty predictor
        entropy = self.difficulty_predictor(x).squeeze(-1)  # (B, N)
        return entropy
    
    def determine_num_experts(self, entropy):
        """
        Determine number of experts to use based on entropy.
        
        Linear interpolation between min_experts and max_experts based on entropy:
        - entropy <= threshold_low: min_experts
        - entropy >= threshold_high: max_experts
        - in between: linear interpolation
        
        Args:
            entropy: (B, N) entropy values
            
        Returns:
            num_experts: (B, N) number of experts to use (integer)
        """
        # Normalize entropy to [0, 1] range
        normalized_entropy = (entropy - self.entropy_threshold_low) / (
            self.entropy_threshold_high - self.entropy_threshold_low
        )
        normalized_entropy = torch.clamp(normalized_entropy, 0.0, 1.0)
        
        # Linear interpolation
        num_experts_float = (
            self.min_experts + 
            normalized_entropy * (self.max_experts - self.min_experts)
        )
        
        # Round to nearest integer
        num_experts = torch.round(num_experts_float).long()
        
        # Clamp to valid range
        num_experts = torch.clamp(num_experts, self.min_experts, self.max_experts)
        
        return num_experts
    
    def forward(self, x):
        """
        Forward pass with conditional expert routing.
        
        Args:
            x: (B, N, D) input tensor
        
        Returns:
            output: (B, N, D) routed through experts
            stats: dict with routing statistics
        """
        B, N, D = x.shape
        x_flat = x.reshape(B * N, D)  # (T, D), T = B*N
        
        # Compute input difficulty (entropy)
        entropy = self.compute_input_entropy(x)  # (B, N)
        entropy_flat = entropy.reshape(B * N)  # (T,)
        
        # Determine number of experts for each token
        num_experts_per_token = self.determine_num_experts(entropy)  # (B, N)
        num_experts_flat = num_experts_per_token.reshape(B * N)  # (T,)
        
        # Compute router logits
        router_logits = self.gating_network(x_flat)  # (T, max_experts)
        
        # Initialize output
        out_flat = torch.zeros_like(x_flat)  # (T, D)
        
        # Route each token based on its difficulty
        for k in range(self.min_experts, self.max_experts + 1):
            # Find tokens that should use k experts
            mask = (num_experts_flat == k)
            
            if not mask.any():
                continue
            
            # Get subset of tokens
            sub_x = x_flat[mask]  # (T_k, D)
            sub_logits = router_logits[mask]  # (T_k, max_experts)
            
            if k == 1:
                # Single expert: use argmax
                indices = sub_logits.argmax(dim=-1)  # (T_k,)
                
                for e in range(self.max_experts):
                    expert_mask = (indices == e)
                    if expert_mask.any():
                        expert_input = sub_x[expert_mask]
                        expert_output = self.experts[e](expert_input)
                        
                        # Map back to original positions
                        original_indices = torch.where(mask)[0][expert_mask]
                        out_flat[original_indices] = expert_output.to(out_flat.dtype)
            else:
                # Multiple experts: use top-k with softmax weighting
                top_k = min(k, self.max_experts)
                
                # Get top-k experts
                top_k_logits, top_k_indices = torch.topk(sub_logits, top_k, dim=-1)  # (T_k, k)
                top_k_weights = F.softmax(top_k_logits, dim=-1)  # (T_k, k)
                
                # Compute weighted sum of expert outputs
                sub_output = torch.zeros_like(sub_x)  # (T_k, D)
                
                for i in range(top_k):
                    # Get expert indices for this position
                    expert_indices = top_k_indices[:, i]  # (T_k,)
                    weights = top_k_weights[:, i].unsqueeze(-1)  # (T_k, 1)
                    
                    # Process each expert
                    for e in range(self.max_experts):
                        expert_mask = (expert_indices == e)
                        if expert_mask.any():
                            expert_input = sub_x[expert_mask]
                            expert_output = self.experts[e](expert_input)
                            
                            # Add weighted contribution
                            sub_output[expert_mask] += expert_output * weights[expert_mask]
                
                # Map back to original positions
                original_indices = torch.where(mask)[0]
                out_flat[original_indices] = sub_output.to(out_flat.dtype)
        
        # Update statistics
        with torch.no_grad():
            avg_experts = num_experts_flat.float().mean()
            self.avg_num_experts_used = (
                self.avg_num_experts_used * self.num_forward_calls + avg_experts
            ) / (self.num_forward_calls + 1)
            self.num_forward_calls += 1
        
        # Reshape output
        output = out_flat.view(B, N, D)
        
        # Collect statistics
        stats = {
            'avg_entropy': entropy.mean().item(),
            'avg_num_experts': num_experts_flat.float().mean().item(),
            'min_num_experts': num_experts_flat.min().item(),
            'max_num_experts': num_experts_flat.max().item(),
            'entropy_std': entropy.std().item(),
        }
        
        return output, stats
    
    def get_routing_statistics(self):
        """
        Get cumulative routing statistics.
        
        Returns:
            dict with statistics
        """
        return {
            'avg_num_experts_used': self.avg_num_experts_used.item(),
            'num_forward_calls': self.num_forward_calls.item(),
        }
    
    def reset_statistics(self):
        """Reset routing statistics."""
        self.avg_num_experts_used.zero_()
        self.num_forward_calls.zero_()


class ConditionalMoEWithLoadBalancing(ConditionalMoELayer):
    """
    Conditional MoE with load balancing loss to encourage uniform expert usage.
    
    Adds auxiliary loss to prevent expert collapse (all tokens routed to same expert).
    """
    
    def __init__(
        self, 
        d_model, 
        max_experts=4, 
        min_experts=1,
        dropout_p=0.1,
        entropy_threshold_low=0.5,
        entropy_threshold_high=2.0,
        load_balance_weight=0.01
    ):
        super().__init__(
            d_model=d_model,
            max_experts=max_experts,
            min_experts=min_experts,
            dropout_p=dropout_p,
            entropy_threshold_low=entropy_threshold_low,
            entropy_threshold_high=entropy_threshold_high
        )
        self.load_balance_weight = load_balance_weight
        
        # Track expert usage for load balancing
        self.register_buffer('expert_usage_count', torch.zeros(max_experts))
    
    def compute_load_balance_loss(self, router_logits, num_experts_per_token):
        """
        Compute load balancing loss to encourage uniform expert usage.
        
        L_balance = coefficient_of_variation(expert_usage)
        
        Args:
            router_logits: (T, max_experts) router logits
            num_experts_per_token: (T,) number of experts per token
            
        Returns:
            loss: scalar load balance loss
        """
        # Compute expert assignment probabilities
        router_probs = F.softmax(router_logits, dim=-1)  # (T, max_experts)
        
        # Weight by number of experts used (tokens using more experts contribute more)
        weights = num_experts_per_token.float().unsqueeze(-1) / self.max_experts  # (T, 1)
        weighted_probs = router_probs * weights  # (T, max_experts)
        
        # Compute average usage per expert
        expert_usage = weighted_probs.sum(dim=0)  # (max_experts,)
        
        # Coefficient of variation: std / mean
        # Lower is better (more uniform distribution)
        mean_usage = expert_usage.mean()
        std_usage = expert_usage.std()
        
        # Avoid division by zero
        cv = std_usage / (mean_usage + 1e-6)
        
        return cv
    
    def forward(self, x):
        """
        Forward pass with load balancing.
        
        Args:
            x: (B, N, D) input tensor
        
        Returns:
            output: (B, N, D) routed through experts
            stats: dict with routing statistics including load balance loss
        """
        B, N, D = x.shape
        x_flat = x.reshape(B * N, D)
        
        # Compute input difficulty
        entropy = self.compute_input_entropy(x)
        entropy_flat = entropy.reshape(B * N)
        
        # Determine number of experts
        num_experts_per_token = self.determine_num_experts(entropy)
        num_experts_flat = num_experts_per_token.reshape(B * N)
        
        # Compute router logits
        router_logits = self.gating_network(x_flat)
        
        # Compute load balance loss
        load_balance_loss = self.compute_load_balance_loss(router_logits, num_experts_flat)
        
        # Standard forward pass (reuse parent implementation)
        # We need to manually implement routing here since we can't call super().forward()
        out_flat = torch.zeros_like(x_flat)
        
        for k in range(self.min_experts, self.max_experts + 1):
            mask = (num_experts_flat == k)
            
            if not mask.any():
                continue
            
            sub_x = x_flat[mask]
            sub_logits = router_logits[mask]
            
            if k == 1:
                indices = sub_logits.argmax(dim=-1)
                
                for e in range(self.max_experts):
                    expert_mask = (indices == e)
                    if expert_mask.any():
                        expert_input = sub_x[expert_mask]
                        expert_output = self.experts[e](expert_input)
                        
                        original_indices = torch.where(mask)[0][expert_mask]
                        out_flat[original_indices] = expert_output.to(out_flat.dtype)
                        
                        # Update usage count
                        with torch.no_grad():
                            self.expert_usage_count[e] += expert_mask.sum().item()
            else:
                top_k = min(k, self.max_experts)
                top_k_logits, top_k_indices = torch.topk(sub_logits, top_k, dim=-1)
                top_k_weights = F.softmax(top_k_logits, dim=-1)
                
                sub_output = torch.zeros_like(sub_x)
                
                for i in range(top_k):
                    expert_indices = top_k_indices[:, i]
                    weights = top_k_weights[:, i].unsqueeze(-1)
                    
                    for e in range(self.max_experts):
                        expert_mask = (expert_indices == e)
                        if expert_mask.any():
                            expert_input = sub_x[expert_mask]
                            expert_output = self.experts[e](expert_input)
                            
                            sub_output[expert_mask] += expert_output * weights[expert_mask]
                            
                            # Update usage count
                            with torch.no_grad():
                                self.expert_usage_count[e] += expert_mask.sum().item()
                
                original_indices = torch.where(mask)[0]
                out_flat[original_indices] = sub_output.to(out_flat.dtype)
        
        # Update statistics
        with torch.no_grad():
            avg_experts = num_experts_flat.float().mean()
            self.avg_num_experts_used = (
                self.avg_num_experts_used * self.num_forward_calls + avg_experts
            ) / (self.num_forward_calls + 1)
            self.num_forward_calls += 1
        
        output = out_flat.view(B, N, D)
        
        stats = {
            'avg_entropy': entropy.mean().item(),
            'avg_num_experts': num_experts_flat.float().mean().item(),
            'min_num_experts': num_experts_flat.min().item(),
            'max_num_experts': num_experts_flat.max().item(),
            'entropy_std': entropy.std().item(),
            'load_balance_loss': load_balance_loss.item(),
        }
        
        return output, stats
    
    def get_expert_usage_distribution(self):
        """
        Get expert usage distribution.
        
        Returns:
            usage_distribution: (max_experts,) normalized usage counts
        """
        total_usage = self.expert_usage_count.sum()
        if total_usage > 0:
            return self.expert_usage_count / total_usage
        else:
            return torch.zeros_like(self.expert_usage_count)
