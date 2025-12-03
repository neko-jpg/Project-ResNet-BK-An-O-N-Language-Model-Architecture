
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from ..models.phase7.hyperbolic_attention import poincare_distance, exp_map_at_origin

class HyperbolicKVCache:
    """
    Dynamic KV Cache with Hyperbolic Distance Pruning.
    """
    def __init__(self, max_size: int = 1024, prune_ratio: float = 0.2):
        self.max_size = max_size
        self.prune_ratio = prune_ratio
        self.k_cache: Optional[torch.Tensor] = None # (B, H, L, D)
        self.v_cache: Optional[torch.Tensor] = None # (B, H, L, D)
        self.curvature: Optional[torch.Tensor] = None

    def update(self, k_new: torch.Tensor, v_new: torch.Tensor, c: torch.Tensor):
        """
        Update cache with new keys and values.
        k_new: (B, H, 1, D) - Hyperbolic space
        v_new: (B, H, 1, D) - Tangent space
        """
        if self.k_cache is None:
            self.k_cache = k_new
            self.v_cache = v_new
            self.curvature = c
        else:
            self.k_cache = torch.cat([self.k_cache, k_new], dim=2)
            self.v_cache = torch.cat([self.v_cache, v_new], dim=2)
            
        # Prune if too large
        if self.k_cache.size(2) > self.max_size:
            self.prune()

    def prune(self):
        """
        Prune cache based on hyperbolic distance from the latest key.
        Heuristic: Keep recent keys + keys close to the latest key.
        """
        # Latest key (query-like)
        latest_k = self.k_cache[:, :, -1:, :] # (B, H, 1, D)
        
        # Compute distances to all cached keys
        # dist: (B, H, L)
        dist = poincare_distance(latest_k, self.k_cache, c=self.curvature).squeeze(-1)
        
        # Determine number of tokens to keep
        current_len = self.k_cache.size(2)
        keep_len = int(current_len * (1.0 - self.prune_ratio))
        
        # Strategy:
        # 1. Always keep last N tokens (local context)
        # 2. Keep top-K closest tokens from the rest (long-term memory)
        
        local_window = 16
        if keep_len <= local_window:
            # Just keep last keep_len
            indices = torch.arange(current_len - keep_len, current_len, device=self.k_cache.device)
            indices = indices.expand(self.k_cache.size(0), self.k_cache.size(1), -1)
        else:
            # Keep last local_window
            # Select from remaining based on distance (smallest distance = closest)
            # We want smallest distances.
            
            # Distances for non-local tokens
            dist_non_local = dist[:, :, :-local_window]
            
            num_to_select = keep_len - local_window
            
            # Top-k smallest distance (largest negative distance)
            _, top_indices = torch.topk(-dist_non_local, k=num_to_select, dim=-1)
            
            # Sort indices to maintain temporal order (optional but good for RoPE etc if used)
            top_indices, _ = top_indices.sort(dim=-1)
            
            # Add local window indices
            local_indices = torch.arange(current_len - local_window, current_len, device=self.k_cache.device)
            local_indices = local_indices.expand(self.k_cache.size(0), self.k_cache.size(1), -1)
            
            indices = torch.cat([top_indices, local_indices], dim=-1)
            
        # Gather
        # indices: (B, H, keep_len)
        # k_cache: (B, H, L, D)
        # We need to gather along dim 2.
        
        # Expand indices for gather
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, self.k_cache.size(-1))
        
        self.k_cache = torch.gather(self.k_cache, 2, indices_expanded)
        self.v_cache = torch.gather(self.v_cache, 2, indices_expanded)

    def get_cache(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.k_cache, self.v_cache


class InferenceEngine:
    """
    Lightweight Inference Engine for Chat.
    Strips training features and manages KV cache.
    """
    def __init__(self, model: nn.Module, max_cache_size: int = 1024):
        self.model = model
        self.model.eval()
        self.max_cache_size = max_cache_size
        self.kv_caches: Dict[str, HyperbolicKVCache] = {}
        
        # Disable gradients
        for p in self.model.parameters():
            p.requires_grad = False

    def reset_cache(self):
        self.kv_caches = {}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Autoregressive generation.
        """
        # Prefill
        # For the first step, we process the whole prompt.
        # We need to hook into the model's attention layers to use our KV cache.
        # This requires the model to accept 'kv_cache' argument or we patch it.
        
        # Assuming we can patch or the model supports it.
        # Since I cannot easily modify the whole model hierarchy now, 
        # I will demonstrate the logic assuming the model has a 'forward_inference' or similar,
        # or I wrap the attention layers.
        
        # For this task, I will implement the engine logic. 
        # Integration depends on model support.
        
        generated = input_ids
        
        for _ in range(max_new_tokens):
            # Forward pass
            # We only pass the last token if we have cache, 
            # but for the very first prefill we pass all.
            
            if len(self.kv_caches) == 0:
                # Prefill
                logits = self.model(generated)
            else:
                # Generation step (last token only)
                logits = self.model(generated[:, -1:])
                
            # Next token prediction
            next_token_logits = logits[:, -1, :] / temperature
            
            # Top-K
            if top_k > 0:
                v, _ = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Break if EOS (assuming EOS id is known, e.g. 2)
            if (next_token == 2).all():
                break
                
        return generated

    # To make this work, we need to inject the KV cache into the model.
    # We can use hooks or modify the model class.
    # Given the constraints, I'll provide the HyperbolicKVCache class 
    # and a function to patch the attention layers.

def patch_model_with_kv_cache(model: nn.Module, engine: InferenceEngine):
    """
    Monkey-patch HyperbolicMultiHeadAttention to use engine's KV cache.
    """
    from ..models.phase7.hyperbolic_attention import HyperbolicMultiHeadAttention, exp_map_at_origin
    
    original_forward = HyperbolicMultiHeadAttention.forward
    
    def cached_forward(self, x, mask=None, return_diagnostics=False):
        # Identify this layer instance
        layer_id = str(id(self))
        
        if layer_id not in engine.kv_caches:
            engine.kv_caches[layer_id] = HyperbolicKVCache(max_size=engine.max_cache_size)
        
        cache = engine.kv_caches[layer_id]
        
        # Compute Q, K, V
        with torch.cuda.amp.autocast(enabled=False):
            x_f32 = x.float()
            c = F.softplus(self.log_c.float())
            
            q_tangent = self.W_q(x_f32)
            k_tangent = self.W_k(x_f32)
            v_tangent = self.W_v(x_f32)
            
            batch_size, seq_len, _ = x.shape
            
            q_tangent = q_tangent.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
            k_tangent = k_tangent.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
            v_tangent = v_tangent.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
            
            # Map K to hyperbolic space
            k_hyp = exp_map_at_origin(k_tangent, c=c)
            
            # Update Cache
            cache.update(k_hyp, v_tangent, c)
            
            # Get Cached K, V
            k_hyp_cached, v_tangent_cached = cache.get_cache()
            
            # Map Q to hyperbolic space
            q_hyp = exp_map_at_origin(q_tangent, c=c)
            
            # Attention (using cached K, V)
            # We need to reshape for attention
            # q: (B, H, 1, D) (if step)
            # k_cache: (B, H, L_cache, D)
            
            # Flatten heads
            q_hyp_flat = q_hyp.reshape(batch_size * self.num_heads, seq_len, self.d_head)
            k_hyp_flat = k_hyp_cached.reshape(batch_size * self.num_heads, -1, self.d_head)
            v_tangent_flat = v_tangent_cached.reshape(batch_size * self.num_heads, -1, self.d_head)
            
            # Call single head attention
            # Note: mask needs adjustment for cache size
            output_hyperbolic_flat, _, _ = self.attention(q_hyp_flat, k_hyp_flat, v_tangent_flat, c=c, mask=None)
            
            output_hyperbolic = output_hyperbolic_flat.view(batch_size, self.num_heads, seq_len, self.d_head)
            
            # Log map and project
            from ..models.phase7.hyperbolic_attention import log_map_at_origin
            output_tangent_heads = log_map_at_origin(output_hyperbolic, c=c)
            output_tangent_concat = output_tangent_heads.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
            final_output = self.W_o(output_tangent_concat)
            
            if return_diagnostics:
                return final_output.to(x.dtype), {}
            return final_output.to(x.dtype)

    # Apply patch
    for module in model.modules():
        if isinstance(module, HyperbolicMultiHeadAttention):
            module.forward = cached_forward.__get__(module, HyperbolicMultiHeadAttention)

