"""
Phase 3.6: Entropic Data Selection

Phase 2モデル（安定化済み）を使用して、データセットの「熱力学的サプライズ」を計測し、
学習効率の高いデータのみを選別する。

Metric:
    Free Energy F = E - T*S
    Surprise = || Gradient || (magnitude of weight update required)
    or simple Loss value.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

class EntropicSelector:
    def __init__(self, model, criterion, device='cuda'):
        self.model = model
        self.criterion = criterion
        self.device = device
        
    def score_dataset(self, dataset, batch_size=32):
        """
        Score each sample in dataset by Loss/Energy.
        High Loss = High Surprise (Keep)
        Low Loss = Boring (Discard)
        """
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        scores = []
        
        self.model.eval()
        self.model.to(self.device)
        
        with torch.no_grad():
            for batch in loader:
                inputs = batch['input_ids'].to(self.device)
                targets = batch['labels'].to(self.device)
                
                outputs = self.model(inputs)
                # Compute loss per sample (reduction='none')
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                loss = loss.view(inputs.size(0), -1).mean(dim=1)
                
                scores.extend(loss.cpu().tolist())
                
        return torch.tensor(scores)

    def filter_dataset(self, dataset, keep_ratio=0.3):
        """
        Returns a Subset of the top-k surprising samples.
        """
        print("Calculating entropic scores...")
        scores = self.score_dataset(dataset)
        
        num_keep = int(len(dataset) * keep_ratio)
        
        # Select indices with highest loss
        _, indices = torch.topk(scores, num_keep)
        
        print(f"Entropic Selection: Keeping {num_keep}/{len(dataset)} samples.")
        return Subset(dataset, indices)