"""
Koopman Operator Learning for ResNet-BK
Implements Koopman theory-based learning to reduce gradient computation cost.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KoopmanResNetBKLayer(nn.Module):
    """
    ResNet-BK layer with Koopman operator learning.
    
    Koopman theory: Represent nonlinear dynamics as linear operators in lifted space.
    - State: x_t = hidden_state at layer t
    - Dynamics: x_{t+1} = f(x_t)  [nonlinear ResNet-BK layer]
    - Lifted space: z_t = φ(x_t)  [embedding into Koopman space]
    - Linear dynamics: z_{t+1} = K * z_t  [Koopman operator K]
    
    Architecture:
        Standard mode: x -> ResNetBKLayer -> x_next
        Koopman mode: x -> φ (lift) -> K (linear) -> ψ (inverse lift) -> x_next
    """
    
    def __init__(self, d_model, n_seq, koopman_dim=256, num_experts=4, top_k=1, dropout_p=0.1):
        """
        Initialize Koopman ResNet-BK layer.
        
        Args:
            d_model: hidden dimension
            n_seq: sequence length
            koopman_dim: dimension of Koopman lifted space
            num_experts: number of MoE experts
            top_k: number of experts to route to
            dropout_p: dropout probability
        """
        super().__init__()
        self.d_model = d_model
        self.n_seq = n_seq
        self.koopman_dim = koopman_dim
        
        # Standard ResNet-BK components
        from .resnet_bk import MoEResNetBKLayer
        self.bk_layer = MoEResNetBKLayer(d_model, n_seq, num_experts, top_k, dropout_p)
        
        # Koopman lifting function φ: R^d_model → R^koopman_dim
        # Use bounded activation (Tanh) for numerical stability
        self.phi = nn.Sequential(
            nn.Linear(d_model, koopman_dim),
            nn.Tanh(),
            nn.Linear(koopman_dim, koopman_dim)
        )
        
        # Koopman operator K: R^koopman_dim → R^koopman_dim
        # Initialize as identity + small perturbation for stability
        K_init = torch.eye(koopman_dim) + 0.01 * torch.randn(koopman_dim, koopman_dim)
        self.K = nn.Parameter(K_init)
        
        # Inverse lifting ψ: R^koopman_dim → R^d_model
        self.psi = nn.Sequential(
            nn.Linear(koopman_dim, koopman_dim),
            nn.Tanh(),
            nn.Linear(koopman_dim, d_model)
        )
        
        # Buffers for DMD computation (will be implemented in subtask 3.2)
        # Store last 500 state pairs for streaming DMD (increased from 100 for better estimation)
        buffer_size = 500
        self.register_buffer('Z_current', torch.zeros(koopman_dim, buffer_size))
        self.register_buffer('Z_next', torch.zeros(koopman_dim, buffer_size))
        self.register_buffer('buffer_idx', torch.tensor(0, dtype=torch.long))
        self.register_buffer('buffer_filled', torch.tensor(False, dtype=torch.bool))
    
    def forward(self, x, use_koopman=False):
        """
        Forward pass with optional Koopman prediction.
        
        Args:
            x: (B, N, D) input tensor
            use_koopman: if True, use Koopman prediction; else use standard forward
        
        Returns:
            output: (B, N, D) output tensor
        """
        if not use_koopman:
            # Standard forward pass through ResNet-BK layer
            return self.bk_layer(x)
        else:
            # Koopman-based forward pass
            B, N, D = x.shape
            
            # Lift to Koopman space: z = φ(x)
            z = self.phi(x)  # (B, N, koopman_dim)
            
            # Apply Koopman operator: z_next = K * z
            # Use einsum for batched matrix multiplication
            z_next = torch.einsum('bnk,kl->bnl', z, self.K)  # (B, N, koopman_dim)
            
            # Project back to state space: x_next = ψ(z_next)
            x_next = self.psi(z_next)  # (B, N, D)
            
            return x_next
    
    def get_koopman_features(self, x):
        """
        Get Koopman lifted features for analysis.
        
        Args:
            x: (B, N, D) input tensor
        
        Returns:
            z: (B, N, koopman_dim) lifted features
        """
        return self.phi(x)
    
    def predict_next_state(self, x):
        """
        Predict next state using Koopman operator.
        
        Args:
            x: (B, N, D) current state
        
        Returns:
            x_next_pred: (B, N, D) predicted next state
        """
        with torch.no_grad():
            return self.forward(x, use_koopman=True)
    
    def update_koopman_operator(self, x_current, x_next):
        """
        Update Koopman operator K using streaming DMD (Dynamic Mode Decomposition).
        
        DMD algorithm:
        1. Collect state pairs: (z_current, z_next) where z = φ(x)
        2. Compute K = Z_next @ Z_current^+ (pseudoinverse)
        3. Use SVD for numerical stability: K = Z_next @ V @ S^{-1} @ U^T
        4. Apply exponential moving average for smooth updates
        
        Args:
            x_current: (B, N, D) - current states
            x_next: (B, N, D) - next states (from standard forward pass)
        """
        with torch.no_grad():
            # Lift to Koopman space
            z_current = self.phi(x_current)  # (B, N, koopman_dim)
            z_next = self.phi(x_next)  # (B, N, koopman_dim)
            
            # Flatten batch and sequence dimensions
            z_current_flat = z_current.reshape(-1, self.koopman_dim).T  # (koopman_dim, B*N)
            z_next_flat = z_next.reshape(-1, self.koopman_dim).T  # (koopman_dim, B*N)
            
            # Update circular buffer
            buffer_size = self.Z_current.shape[1]
            batch_size = z_current_flat.shape[1]
            
            if batch_size <= buffer_size:
                # Simple case: batch fits in buffer
                start_idx = self.buffer_idx.item()
                end_idx = (start_idx + batch_size) % buffer_size
                
                if end_idx > start_idx:
                    # No wrap-around
                    self.Z_current[:, start_idx:end_idx] = z_current_flat
                    self.Z_next[:, start_idx:end_idx] = z_next_flat
                else:
                    # Wrap around
                    first_chunk_size = buffer_size - start_idx
                    self.Z_current[:, start_idx:] = z_current_flat[:, :first_chunk_size]
                    self.Z_current[:, :end_idx] = z_current_flat[:, first_chunk_size:]
                    self.Z_next[:, start_idx:] = z_next_flat[:, :first_chunk_size]
                    self.Z_next[:, :end_idx] = z_next_flat[:, first_chunk_size:]
                
                self.buffer_idx.copy_(torch.tensor(end_idx, dtype=torch.long))
                
                # Mark buffer as filled once we've wrapped around
                if end_idx < start_idx or (start_idx + batch_size >= buffer_size):
                    self.buffer_filled.copy_(torch.tensor(True, dtype=torch.bool))
            
            # Only update K if we have enough data
            if not self.buffer_filled:
                return
            
            # Compute Koopman operator using DMD with SVD
            # K = Z_next @ Z_current^+ (pseudoinverse)
            try:
                # SVD: Z_current = U @ S @ V^T
                U, S, Vt = torch.svd(self.Z_current)
                
                # Singular value thresholding for numerical stability
                threshold = 1e-6
                S_inv = torch.where(
                    S > threshold,
                    1.0 / S,
                    torch.zeros_like(S)
                )
                
                # K = Z_next @ V @ S^{-1} @ U^T
                # Compute in steps to avoid large intermediate matrices
                temp1 = Vt.T @ torch.diag(S_inv)  # (koopman_dim, koopman_dim)
                temp2 = temp1 @ U.T  # (koopman_dim, koopman_dim)
                K_new = self.Z_next @ temp2  # (koopman_dim, koopman_dim)
                
                # Exponential moving average update
                # Increased from 0.1 to 0.3 for faster adaptation
                alpha = 0.3  # Learning rate for Koopman operator
                self.K.data = (1 - alpha) * self.K.data + alpha * K_new
                
            except RuntimeError as e:
                # SVD may fail for ill-conditioned matrices
                # In this case, skip the update
                print(f"Warning: Koopman operator update failed: {e}")
                pass
    
    def koopman_loss(self, x_current, x_next):
        """
        Auxiliary loss to enforce linear dynamics in Koopman space.
        
        L_koopman = ||φ(x_next) - K * φ(x_current)||^2
        
        This loss encourages the Koopman operator to accurately predict
        the next state in the lifted space, enforcing linear dynamics.
        
        Args:
            x_current: (B, N, D) - current states
            x_next: (B, N, D) - next states (from standard forward pass)
        
        Returns:
            loss: scalar tensor - Koopman prediction loss
        """
        # Lift to Koopman space
        z_current = self.phi(x_current)  # (B, N, koopman_dim)
        z_next_true = self.phi(x_next)  # (B, N, koopman_dim)
        
        # Predict next state in Koopman space using operator K
        z_next_pred = torch.einsum('bnk,kl->bnl', z_current, self.K)  # (B, N, koopman_dim)
        
        # MSE loss between predicted and true next state
        loss = F.mse_loss(z_next_pred, z_next_true)
        
        return loss


class KoopmanResNetBKBlock(nn.Module):
    """
    ResNet-BK Block with Koopman operator learning.
    
    Architecture:
        Input -> LayerNorm -> KoopmanResNetBKLayer -> Add(Input) -> Output
    """
    
    def __init__(self, d_model, n_seq, koopman_dim=256, num_experts=4, top_k=1, dropout_p=0.1):
        """
        Initialize Koopman ResNet-BK block.
        
        Args:
            d_model: hidden dimension
            n_seq: sequence length
            koopman_dim: dimension of Koopman lifted space
            num_experts: number of MoE experts
            top_k: number of experts to route to
            dropout_p: dropout probability
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.bk_layer = KoopmanResNetBKLayer(
            d_model, n_seq, koopman_dim, num_experts, top_k, dropout_p
        )
    
    def forward(self, x, use_koopman=False):
        """
        Forward pass with pre-norm residual structure.
        
        Args:
            x: (B, N, D) input tensor
            use_koopman: if True, use Koopman prediction; else use standard forward
        
        Returns:
            output: (B, N, D) output tensor
        """
        return x + self.bk_layer(self.layer_norm(x), use_koopman=use_koopman)


class KoopmanLanguageModel(nn.Module):
    """
    ResNet-BK Language Model with Koopman operator learning.
    
    Architecture:
        Token Embedding + Position Embedding
        -> KoopmanResNetBKBlock × n_layers
        -> LayerNorm
        -> LM Head
    """
    
    def __init__(
        self,
        vocab_size,
        d_model=64,
        n_layers=4,
        n_seq=128,
        koopman_dim=256,
        num_experts=4,
        top_k=1,
        dropout_p=0.1,
    ):
        """
        Initialize Koopman language model.
        
        Args:
            vocab_size: vocabulary size
            d_model: hidden dimension
            n_layers: number of ResNet-BK blocks
            n_seq: sequence length
            koopman_dim: dimension of Koopman lifted space
            num_experts: number of MoE experts
            top_k: number of experts to route to
            dropout_p: dropout probability
        """
        super().__init__()
        self.d_model = d_model
        self.n_seq = n_seq
        self.n_layers = n_layers
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(n_seq, d_model)
        
        self.blocks = nn.ModuleList([
            KoopmanResNetBKBlock(
                d_model=d_model,
                n_seq=n_seq,
                koopman_dim=koopman_dim,
                num_experts=num_experts,
                top_k=top_k,
                dropout_p=dropout_p,
            )
            for _ in range(n_layers)
        ])
        
        self.layer_norm_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, use_koopman=False):
        """
        Forward pass.
        
        Args:
            x: (batch_size, n_seq) token indices
            use_koopman: if True, use Koopman prediction; else use standard forward
        
        Returns:
            logits: (batch_size, n_seq, vocab_size)
        """
        batch_size, n_seq = x.shape
        assert n_seq == self.n_seq, f"n_seq mismatch: expected {self.n_seq}, got {n_seq}"
        
        # Embeddings
        tok_emb = self.token_embedding(x)  # (B, N, D)
        pos = torch.arange(0, n_seq, dtype=torch.long, device=x.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos)  # (1, N, D)
        h = tok_emb + pos_emb
        
        # Forward through blocks
        for block in self.blocks:
            h = block(h, use_koopman=use_koopman)
        
        # Output
        h = self.layer_norm_final(h)
        logits = self.lm_head(h)  # (B, N, vocab_size)
        return logits
    
    def get_hidden_states(self, x):
        """
        Get hidden states at each layer for Koopman operator updates.
        
        Args:
            x: (batch_size, n_seq) token indices
        
        Returns:
            hidden_states: list of (B, N, D) tensors, one per layer
        """
        batch_size, n_seq = x.shape
        
        # Embeddings
        tok_emb = self.token_embedding(x)
        pos = torch.arange(0, n_seq, dtype=torch.long, device=x.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos)
        h = tok_emb + pos_emb
        
        hidden_states = []
        for block in self.blocks:
            h = block(h, use_koopman=False)
            hidden_states.append(h)
        
        return hidden_states
