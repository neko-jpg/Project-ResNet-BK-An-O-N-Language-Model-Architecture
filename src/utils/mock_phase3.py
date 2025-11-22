
import torch
import torch.nn as nn

class MockPhase3Model(nn.Module):
    """
    Mock Phase 3 Model for UI demonstration purposes.
    Simulates the interface of the real Phase3IntegratedModel.
    """
    def __init__(self, d_model=64, vocab_size=50257):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.config = type('Config', (), {'max_seq_len': 128, 'vocab_size': vocab_size})()

        # Dialectic Loop (Simulated as a layer for hooks)
        self.dialectic = nn.Linear(d_model, d_model)

        # Head
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, labels=None, return_diagnostics=False):
        B, N = input_ids.shape
        device = input_ids.device

        # Generate random hidden states
        hidden = torch.randn(B, N, self.d_model, device=device)

        # Pass through "dialectic" layer to trigger hooks
        hidden = self.dialectic(hidden)

        # Generate logits
        logits = self.head(hidden)

        return {
            'logits': logits,
            'loss': torch.tensor(0.0) if labels is not None else None,
            'diagnostics': {'phase3_metric': 0.99}
        }
