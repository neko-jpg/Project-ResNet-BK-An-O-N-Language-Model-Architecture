"""
Example: Phase 4 Emotion Demo

Focuses on the Resonance Emotion Detector.
Demonstrates how prediction errors generate emotional states.
"""

import torch
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.phase4.emotion_core.resonance_detector import ResonanceEmotionDetector

def main():
    print("Initializing Resonance Emotion Detector...")
    d_model = 64
    n_seq = 128
    vocab_size = 100

    detector = ResonanceEmotionDetector(d_model, n_seq)

    # Simulate a sequence of predictions
    print("Simulating emotional response sequence...")

    resonance_history = []
    dissonance_history = []

    # 1. Perfect Prediction (Calm)
    print("\nScenario 1: Coherence (Low Error)")
    pred = torch.randn(1, n_seq, vocab_size) * 10 # High confidence
    target = pred.argmax(dim=-1)
    hidden = torch.randn(1, n_seq, d_model)

    info = detector(pred, target, hidden)
    print(f"Resonance: {info['resonance_score'].mean():.4f}")
    print(f"Dissonance: {info['dissonance_score'].mean():.4f}")

    # 2. High Error (Dissonance)
    print("\nScenario 2: Conflict (High Error)")
    pred = torch.randn(1, n_seq, vocab_size)
    target = torch.randint(0, vocab_size, (1, n_seq)) # Random targets

    info = detector(pred, target, hidden)
    print(f"Resonance: {info['resonance_score'].mean():.4f}")
    print(f"Dissonance: {info['dissonance_score'].mean():.4f}")

    print("\nDemo Complete.")

if __name__ == "__main__":
    main()
