"""
Curriculum Scheduler for MUSE (Curriculum Evolution)
Dynamically adjusts dataset mixing ratios based on training loss stagnation.
"""

import torch

class CurriculumScheduler:
    """
    Monitors loss and adjusts dataset weights in MixedBinaryDataset.
    """
    def __init__(self, mixed_loader, window_size=100, threshold=0.01, patience=10):
        """
        Args:
            mixed_loader: Instance of MixedBinaryDataset
            window_size: Number of steps to calculate moving average
            threshold: Minimum relative improvement to be considered "not stagnant"
            patience: Number of checks (steps) to wait before switching mode
        """
        self.loader = mixed_loader
        self.window_size = window_size
        self.threshold = threshold
        self.patience = patience

        self.loss_history = []
        self.stagnation_counter = 0
        self.mode = "normal"  # "normal" or "easy"
        self.cooldown = 0

        # Capture initial weights (Normal Mode)
        # Ensure we have the names and current weights
        if hasattr(mixed_loader, 'dataset_names') and hasattr(mixed_loader, 'weights'):
            self.normal_weights = dict(zip(mixed_loader.dataset_names, mixed_loader.weights))
        else:
            # Fallback or Empty if not ready
            self.normal_weights = {}

    def _calculate_easy_weights(self):
        """Heuristic to generate easier curriculum based on dataset names."""
        new_weights = {}
        for name, w in self.normal_weights.items():
            name_lower = name.lower()
            # Hard datasets -> Reduce
            if any(k in name_lower for k in ["code", "math", "arxiv", "complex", "advanced"]):
                new_weights[name] = w * 0.2
            # Easy/Explanatory datasets -> Boost
            elif any(k in name_lower for k in ["cosmopedia", "textbook", "wiki", "basic", "dialogue", "chat"]):
                new_weights[name] = w * 2.0
            else:
                new_weights[name] = w
        return new_weights

    def step(self, current_loss):
        """Call this every training step with the current loss."""
        if self.cooldown > 0:
            self.cooldown -= 1
            return

        self.loss_history.append(current_loss)
        if len(self.loss_history) > self.window_size:
            self.loss_history.pop(0)

            # Check for stagnation
            # Compare average of first half vs second half of window
            half = len(self.loss_history) // 2
            old_avg = sum(self.loss_history[:half]) / half
            new_avg = sum(self.loss_history[half:]) / half

            improvement = (old_avg - new_avg) / old_avg

            if improvement < self.threshold:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
                # If we are in easy mode and improving well, maybe go back?
                if self.mode == "easy" and improvement > self.threshold * 2:
                     self.switch_to_normal()

            # Trigger Switch
            if self.stagnation_counter > self.patience:
                if self.mode == "normal":
                    self.switch_to_easy()
                self.stagnation_counter = 0

    def switch_to_easy(self):
        print(f"\n[Curriculum] 📉 Loss Stagnation detected. Switching to EASY mode (Textbook focus).")
        easy_weights = self._calculate_easy_weights()
        if easy_weights:
            self.loader.update_weights_by_name(easy_weights)
        self.mode = "easy"
        self.cooldown = self.window_size  # Reset window effectively

    def switch_to_normal(self):
        print(f"\n[Curriculum] 📈 Improvement detected. Switching back to NORMAL mode.")
        if self.normal_weights:
            self.loader.update_weights_by_name(self.normal_weights)
        self.mode = "normal"
        self.cooldown = self.window_size
