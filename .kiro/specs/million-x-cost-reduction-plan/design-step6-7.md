# Design Document (Continued) - Steps 6 & 7

## Step 6: Algorithmic Innovations

**Objective**: Achieve 10× cost reduction through adaptive computation, multi-scale processing, and learned sparsity.

### Adaptive Computation Time (ACT)

**Design**:

```python
class AdaptiveResNetBKBlock(nn.Module):
    """
    ResNet-BK block with adaptive computation time.
    Each token decides whether to continue processing.
    """
    
    def __init__(self, d_model, n_seq, threshold=0.99):
        super().__init__()
        self.bk_layer = MoEResNetBKLayer(d_model, n_seq)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Halting unit: predicts whether to stop processing
        self.halting_unit = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        self.threshold = threshold
        self.register_buffer('ponder_cost', torch.tensor(0.0))
    
    def forward(self, x, halting_prob_cumsum=None, still_running=None):
        """
        Args:
            x: (B, N, D)
            halting_prob_cumsum: (B, N) - cumulative halting probability
            still_running: (B, N) - mask of tokens still processing
        
        Returns:
            output: (B, N, D)
            halting_prob_cumsum: updated
            still_running: updated
        """
        B, N, D = x.shape
        
        if halting_prob_cumsum is None:
            halting_prob_cumsum = torch.zeros(B, N, device=x.device)
            still_running = torch.ones(B, N, dtype=torch.bool, device=x.device)
        
        # Process layer
        x_processed = x + self.bk_layer(self.layer_norm(x))
        
        # Compute halting probability
        p_halt = self.halting_unit(x_processed).squeeze(-1)  # (B, N)
        
        # Update cumulative halting probability
        halting_prob_cumsum_new = halting_prob_cumsum + p_halt * still_running.float()
        
        # Determine which tokens should halt
        should_halt = halting_prob_cumsum_new >= self.threshold
        
        # Update running mask
        still_running_new = still_running & (~should_halt)
        
        # Weighted output (for tokens that halted this step)
        just_halted = should_halt & still_running
        weight = torch.where(
            just_halted,
            1.0 - halting_prob_cumsum,  # Remainder probability
            p_halt * still_running.float()
        )
        
        # Accumulate ponder cost (for loss)
        self.ponder_cost += weight.sum()
        
        return x_processed, halting_prob_cumsum_new, still_running_new, weight


class ACTLanguageModel(nn.Module):
    """
    Language model with adaptive computation time.
    """
    
    def __init__(self, vocab_size, d_model=64, n_layers=4, n_seq=128, act_lambda=0.01):
        super().__init__()
        self.d_model = d_model
        self.n_seq = n_seq
        self.act_lambda = act_lambda
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(n_seq, d_model)
        
        self.blocks = nn.ModuleList([
            AdaptiveResNetBKBlock(d_model, n_seq) for _ in range(n_layers)
        ])
        
        self.layer_norm_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        B, N = x.shape
        
        # Embeddings
        tok_emb = self.token_embedding(x)
        pos = torch.arange(0, N, dtype=torch.long, device=x.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos)
        h = tok_emb + pos_emb
        
        # Adaptive processing
        halting_prob_cumsum = None
        still_running = None
        output_accumulator = torch.zeros_like(h)
        
        for block in self.blocks:
            h, halting_prob_cumsum, still_running, weight = block(
                h, halting_prob_cumsum, still_running
            )
            
            # Accumulate weighted outputs
            output_accumulator += h * weight.unsqueeze(-1)
            
            # Early exit if all tokens halted
            if not still_running.any():
                break
        
        # Final output
        h_final = self.layer_norm_final(output_accumulator)
        logits = self.lm_head(h_final)
        
        # Compute ponder cost
        ponder_cost = sum(block.ponder_cost for block in self.blocks) / (B * N)
        
        return logits, ponder_cost
    
    def compute_loss(self, logits, targets, ponder_cost):
        """
        Loss = CE_loss + λ * ponder_cost
        """
        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets)
        total_loss = ce_loss + self.act_lambda * ponder_cost
        return total_loss, ce_loss, ponder_cost
```

### Multi-Scale Sequence Processing

**Design**:

```python
class MultiScaleResNetBKLayer(nn.Module):
    """
    Process sequence at multiple resolutions.
    
    Architecture:
      Input (N) → Downsample (N/2) → Process → Upsample (N) → Output
    """
    
    def __init__(self, d_model, n_seq):
        super().__init__()
        self.d_model = d_model
        self.n_seq = n_seq
        
        # Learned downsampling: N → N/2
        self.downsample = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=2, stride=2),
            nn.LayerNorm([d_model, n_seq // 2])
        )
        
        # Process at lower resolution
        self.bk_layer_low_res = MoEResNetBKLayer(d_model, n_seq // 2)
        
        # Learned upsampling: N/2 → N
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(d_model, d_model, kernel_size=2, stride=2),
            nn.LayerNorm([d_model, n_seq])
        )
        
        # Refinement at full resolution
        self.bk_layer_full_res = MoEResNetBKLayer(d_model, n_seq)
    
    def forward(self, x):
        """
        x: (B, N, D)
        """
        B, N, D = x.shape
        
        # Downsample
        x_down = self.downsample(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, N/2, D)
        
        # Process at low resolution
        x_low_res = self.bk_layer_low_res(x_down)  # (B, N/2, D)
        
        # Upsample
        x_up = self.upsample(x_low_res.permute(0, 2, 1)).permute(0, 2, 1)  # (B, N, D)
        
        # Refine at full resolution
        x_refined = self.bk_layer_full_res(x + x_up)  # Residual connection
        
        return x_refined
```

### Learned Sparsity in BK-Core

**Design**:

```python
class SparseBKCore(nn.Module):
    """
    BK-Core with learned sparsity: predict which G_ii elements to compute.
    """
    
    def __init__(self, d_model, n_seq, target_sparsity=0.5):
        super().__init__()
        self.d_model = d_model
        self.n_seq = n_seq
        self.target_sparsity = target_sparsity
        
        # Importance predictor: which positions are important
        self.importance_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        
        # Standard BK-Core
        self.bk_core = BKCoreFunction.apply
        
        # Interpolation network: fill in masked positions
        self.interpolator = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 2, kernel_size=3, padding=1)
        )
    
    def forward(self, x, v):
        """
        Args:
            x: (B, N, D) - input features
            v: (B, N) - potential
        
        Returns:
            features: (B, N, 2) - [real(G_ii), imag(G_ii)]
            mask: (B, N) - binary mask of computed positions
        """
        B, N = v.shape
        
        # Predict importance scores
        importance_scores = self.importance_predictor(x).squeeze(-1)  # (B, N)
        
        # Gumbel-Sigmoid for differentiable binary mask
        mask = F.gumbel_softmax(
            torch.stack([importance_scores, -importance_scores], dim=-1),
            hard=True,
            tau=1.0
        )[:, :, 0]  # (B, N)
        
        # Compute G_ii only for masked positions
        features_sparse = torch.zeros(B, N, 2, device=x.device)
        
        for b in range(B):
            mask_b = mask[b]  # (N,)
            if mask_b.sum() == 0:
                continue
            
            # Extract masked positions
            v_masked = v[b][mask_b]  # (num_masked,)
            
            # Compute BK-Core for masked positions
            # (Simplified: compute full, then mask. Optimized version would skip computation)
            features_full = self.bk_core(
                v[b].unsqueeze(0),
                torch.full((1, N-1), 1.0, device=x.device),
                torch.full((1, N-1), 1.0, device=x.device),
                torch.tensor(1.0j, device=x.device)
            )  # (1, N, 2)
            
            features_sparse[b] = features_full[0] * mask_b.unsqueeze(-1)
        
        # Interpolate missing positions
        features_interpolated = self.interpolator(features_sparse.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Combine: use computed values where available, interpolated otherwise
        features_final = torch.where(
            mask.unsqueeze(-1).expand(-1, -1, 2) > 0.5,
            features_sparse,
            features_interpolated
        )
        
        return features_final, mask
    
    def sparsity_loss(self, mask):
        """
        Encourage target sparsity level.
        """
        current_sparsity = mask.mean()
        return (current_sparsity - self.target_sparsity) ** 2
```

**Expected Speedup**:
- ACT: 30% reduction in average layers executed → 1.4× speedup
- Multi-scale: 2× speedup for middle layers (N/4 resolution)
- Learned sparsity: 50% sparsity → 1.8× speedup
- **Total: 1.4 × 2 × 1.8 ≈ 5× (conservative estimate, targeting 10×)**

## Step 7: System Integration and Data Efficiency

**Objective**: Achieve 10× cost reduction through curriculum learning, data efficiency, and system optimizations.

### Curriculum Learning

**Design**:

```python
class CurriculumLearningScheduler:
    """
    Order training examples by difficulty, gradually increase difficulty.
    """
    
    def __init__(self, dataset, model, difficulty_metric='perplexity'):
        self.dataset = dataset
        self.model = model
        self.difficulty_metric = difficulty_metric
        self.difficulties = None
    
    def compute_difficulties(self):
        """
        Compute difficulty score for each example using pretrained model.
        """
        self.model.eval()
        difficulties = []
        
        with torch.no_grad():
            for x, y in self.dataset:
                logits = self.model(x.unsqueeze(0))
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.unsqueeze(0).view(-1))
                difficulties.append(loss.item())
        
        self.difficulties = torch.tensor(difficulties)
        return self.difficulties
    
    def get_curriculum_dataloader(self, epoch, total_epochs, batch_size):
        """
        Return dataloader with examples ordered by difficulty.
        
        Early epochs: easy examples
        Later epochs: gradually add harder examples
        """
        # Compute difficulty percentile threshold
        progress = epoch / total_epochs
        percentile = progress * 100  # 0% → 100%
        
        threshold = torch.quantile(self.difficulties, percentile / 100.0)
        
        # Filter examples below threshold
        indices = (self.difficulties <= threshold).nonzero(as_tuple=True)[0]
        
        # Create subset
        subset = torch.utils.data.Subset(self.dataset, indices)
        
        # Create dataloader
        dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        
        return dataloader
```

### Active Learning

**Design**:

```python
class ActiveLearningSelector:
    """
    Select most informative examples for training.
    """
    
    def __init__(self, model, selection_strategy='uncertainty'):
        self.model = model
        self.selection_strategy = selection_strategy
    
    def compute_uncertainty(self, x):
        """
        Compute model uncertainty for example x.
        
        Uncertainty = entropy of output distribution
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x.unsqueeze(0))
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            return entropy.mean().item()
    
    def select_examples(self, unlabeled_pool, num_select):
        """
        Select num_select most informative examples from pool.
        """
        uncertainties = []
        
        for x, _ in unlabeled_pool:
            uncertainty = self.compute_uncertainty(x)
            uncertainties.append(uncertainty)
        
        # Select top-k most uncertain
        uncertainties = torch.tensor(uncertainties)
        _, indices = torch.topk(uncertainties, num_select)
        
        selected = [unlabeled_pool[i] for i in indices]
        return selected
```

### Gradient Caching

**Design**:

```python
class GradientCachingTrainer:
    """
    Reuse gradients from similar examples to reduce backward pass frequency.
    """
    
    def __init__(self, model, cache_size=100, similarity_threshold=0.9):
        self.model = model
        self.cache_size = cache_size
        self.similarity_threshold = similarity_threshold
        
        # Cache: store (example_embedding, gradients)
        self.gradient_cache = []
    
    def compute_similarity(self, emb1, emb2):
        """
        Cosine similarity between embeddings.
        """
        return F.cosine_similarity(emb1, emb2, dim=0)
    
    def find_similar_cached(self, example_embedding):
        """
        Find cached gradients for similar example.
        """
        for cached_emb, cached_grads in self.gradient_cache:
            similarity = self.compute_similarity(example_embedding, cached_emb)
            if similarity > self.similarity_threshold:
                return cached_grads
        return None
    
    def train_step(self, x_batch, y_batch, optimizer, criterion):
        """
        Training step with gradient caching.
        """
        # Compute example embedding (mean of token embeddings)
        with torch.no_grad():
            example_emb = self.model.token_embedding(x_batch).mean(dim=(0, 1))
        
        # Check cache
        cached_grads = self.find_similar_cached(example_emb)
        
        if cached_grads is not None:
            # Use cached gradients
            optimizer.zero_grad()
            for param, cached_grad in zip(self.model.parameters(), cached_grads):
                param.grad = cached_grad.clone()
            optimizer.step()
            
            # Compute loss for monitoring (no backward)
            with torch.no_grad():
                logits = self.model(x_batch)
                loss = criterion(logits.view(-1, logits.size(-1)), y_batch)
            
            return loss.item(), True  # Used cache
        
        else:
            # Standard training step
            optimizer.zero_grad()
            logits = self.model(x_batch)
            loss = criterion(logits.view(-1, logits.size(-1)), y_batch)
            loss.backward()
            
            # Cache gradients
            grads = [param.grad.clone() for param in self.model.parameters()]
            self.gradient_cache.append((example_emb, grads))
            
            # Limit cache size
            if len(self.gradient_cache) > self.cache_size:
                self.gradient_cache.pop(0)
            
            optimizer.step()
            
            return loss.item(), False  # Computed gradients
```

**Expected Cost Reduction**:
- Curriculum learning: 30% fewer training steps → 1.4× speedup
- Active learning: 50% of data → 2× speedup
- Gradient caching: 20% cache hit rate → 1.25× speedup
- Transfer learning: 5× fewer epochs on target task → 5× speedup
- **Total: 1.4 × 2 × 1.25 × 5 ≈ 17× (exceeds 10× target)**

