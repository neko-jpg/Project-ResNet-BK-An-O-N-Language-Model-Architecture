# src/training/losses.py
# This file contains custom loss functions for Project MUSE.

import torch
import torch.nn as nn
import torch.nn.functional as F

class FreeEnergyLoss(nn.Module):
    """
    Implements the Free Energy Principle (FEP) loss, which is composed of
    an Accuracy term and a Complexity term.

    L = Complexity - Accuracy

    This is equivalent to minimizing:
    L = Accuracy_Loss + Complexity_Loss

    Where:
    - Accuracy_Loss is the standard Cross-Entropy loss (Negative Log-Likelihood).
    - Complexity_Loss is the KL divergence between the posterior Q(z|x) and a prior P(z).
    """

    def __init__(self, hidden_dim: int, kl_weight: float = 1.0):
        """
        Args:
            hidden_dim (int): The dimensionality of the hidden state vectors (z).
            kl_weight (float): A weighting factor for the KL divergence term.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.kl_weight = kl_weight

        # This small linear layer acts as the "Uncertainty Head", predicting the
        # log variance of the posterior distribution Q(z|x) from the hidden state.
        self.uncertainty_head = nn.Linear(hidden_dim, hidden_dim)

        # The Accuracy term is the standard cross-entropy loss.
        self.accuracy_loss = nn.CrossEntropyLoss()

    def _calculate_kl_divergence(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Calculates the KL divergence between a diagonal Gaussian Q(z|x) = N(mu, exp(log_var))
        and a standard Gaussian prior P(z) = N(0, I).

        The formula is: 0.5 * sum(exp(log_var) + mu^2 - 1 - log_var)

        Args:
            mu (torch.Tensor): The mean of the posterior distribution.
            log_var (torch.Tensor): The log variance of the posterior distribution.

        Returns:
            torch.Tensor: The KL divergence loss.
        """
        # Calculate KL divergence per element
        kl_div_per_element = 0.5 * (torch.exp(log_var) + mu.pow(2) - 1 - log_var)

        # Sum over the feature dimension and average over the batch and sequence length
        # This gives a single scalar value for the complexity loss.
        kl_div_loss = kl_div_per_element.sum(dim=-1).mean()

        return kl_div_loss

    def forward(self, logits: torch.Tensor, hidden_states: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the Free Energy Loss.

        Args:
            logits (torch.Tensor): The raw output from the language model head.
                                   Shape: (batch_size, seq_len, vocab_size)
            hidden_states (torch.Tensor): The final hidden states from the model backbone.
                                          Shape: (batch_size, seq_len, hidden_dim)
            targets (torch.Tensor): The target token IDs.
                                    Shape: (batch_size, seq_len)

        Returns:
            torch.Tensor: The total scalar loss.
        """
        # Reshape logits and targets for CrossEntropyLoss
        # (batch, seq_len, vocab_size) -> (batch * seq_len, vocab_size)
        # (batch, seq_len) -> (batch * seq_len)
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(batch_size * seq_len, vocab_size)
        targets_flat = targets.view(batch_size * seq_len)

        # 1. Accuracy Term
        acc_loss = self.accuracy_loss(logits_flat, targets_flat)

        # 2. Complexity Term
        # The hidden state is treated as the mean (mu) of the posterior distribution Q(z|x)
        mu = hidden_states

        # Predict the log variance from the hidden state using the uncertainty head
        log_var = self.uncertainty_head(mu)

        # Calculate the KL divergence
        comp_loss = self._calculate_kl_divergence(mu, log_var)

        # 3. Total Loss
        # The final loss is a weighted sum of the two terms.
        total_loss = acc_loss + self.kl_weight * comp_loss

        return total_loss
