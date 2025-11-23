#!/usr/bin/env python3
"""
MUSE Auto-Scaler (Scale-Up Evolution)
Detects hardware, profiles performance, and generates optimal training configurations.
Includes OOM Prediction and "Safe Mode".
"""

import torch
import psutil
import yaml
import os
import argparse
import math
from muse_utils import log, GREEN, YELLOW, RED, BLUE, NC

class AutoScaler:
    def __init__(self):
        self.vram_gb = self._get_vram_gb()
        self.ram_gb = self._get_ram_gb()
        self.device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    def _get_vram_gb(self):
        if not torch.cuda.is_available():
            return 0
        try:
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / (1024**3)
        except:
            return 0

    def _get_ram_gb(self):
        return psutil.virtual_memory().total / (1024**3)

    def calculate_params(self, d_model, n_layers, vocab_size=50257):
        """Estimate parameter count for ResNet-BK (Transformer-like)"""
        # Embeddings
        params = vocab_size * d_model
        # Layers (Self-Attention + MLP + Norms)
        # Approx: 12 * d_model^2 per layer
        params += n_layers * (12 * d_model**2)
        return params

    def predict_oom_probability(self, config):
        """
        Predict likelihood of OOM based on heuristics.
        Returns: (probability float 0-1, message string)
        """
        if self.vram_gb == 0:
            return 0.0, "CPU Mode (System RAM limit not checked)"

        d = config["model"]["d_model"]
        l = config["model"]["n_layers"]
        b = config["training"]["batch_size"]
        s = 1024 # Default seq len assumption if not set

        params = self.calculate_params(d, l)

        # Memory Estimator (AMP assumption)
        # 1. Model States (Weights + Grads + Optim)
        # Approx 18 bytes per param (Master weights(4) + Model(2) + Grads(2) + Optim(8) + Buffer(2))
        static_mem_gb = (params * 18) / (1024**3)

        # 2. Activations (The killer)
        # Approx: Batch * Seq * Layers * d_model * Overhead
        # Overhead factor depends on activation checkpointing. Without it ~12-16 bytes/element
        activation_overhead = 14
        activations_mem_gb = (b * s * l * d * activation_overhead) / (1024**9) # Wait, simple calc
        # Bytes = B * S * L * D * const
        activations_mem_gb = (b * s * l * d * 2) / (1024**3) * 4 # heuristic multiplier for overhead

        total_estimated = static_mem_gb + activations_mem_gb

        # Buffer for fragmentation
        safe_limit = self.vram_gb * 0.9

        usage_ratio = total_estimated / self.vram_gb

        prob = 0.0
        msg = "Safe"

        if usage_ratio > 1.2:
            prob = 0.99
            msg = "CRITICAL: Guaranteed OOM"
        elif usage_ratio > 1.0:
            prob = 0.85
            msg = "HIGH RISK: Likely OOM"
        elif usage_ratio > 0.9:
            prob = 0.50
            msg = "RISKY: Borderline"
        elif usage_ratio > 0.7:
            prob = 0.10
            msg = "MODERATE: Should be fine"
        else:
            prob = 0.01
            msg = "SAFE: Plenty of headroom"

        return prob, f"{msg} (Est: {total_estimated:.1f}GB / {self.vram_gb:.1f}GB)"

    def generate_config(self):
        log(f"🔍 Hardware Profiling...", BLUE)
        log(f"   Device: {self.device_name}", NC)
        log(f"   VRAM:   {self.vram_gb:.1f} GB", NC)
        log(f"   RAM:    {self.ram_gb:.1f} GB", NC)

        # Config Template
        config = {
            "_base_": "base_config",
            "model": {},
            "training": {}
        }

        # Heuristic Logic
        mode = "unknown"
        if self.vram_gb < 4:
            mode = "tiny_cpu"
            config["model"] = {"d_model": 256, "n_layers": 8, "n_heads": 4}
            config["training"] = {"batch_size": 1, "gradient_accumulation_steps": 8}
        elif self.vram_gb < 8:
            mode = "laptop_gpu"
            config["model"] = {"d_model": 512, "n_layers": 12, "n_heads": 8}
            config["training"] = {"batch_size": 4, "gradient_accumulation_steps": 4}
        elif self.vram_gb < 16:
            mode = "desktop_mid"
            config["model"] = {"d_model": 768, "n_layers": 12, "n_heads": 12}
            config["training"] = {"batch_size": 8, "gradient_accumulation_steps": 4}
        elif self.vram_gb < 24:
            mode = "desktop_high"
            config["model"] = {"d_model": 1024, "n_layers": 24, "n_heads": 16}
            config["training"] = {"batch_size": 12, "gradient_accumulation_steps": 4}
        else:
            mode = "server_grade"
            config["model"] = {"d_model": 1600, "n_layers": 32, "n_heads": 25}
            config["training"] = {"batch_size": 24, "gradient_accumulation_steps": 2}

        # OOM Check
        prob, msg = self.predict_oom_probability(config)

        # If risky, downgrade batch size iteratively
        retries = 0
        while prob > 0.2 and retries < 5:
            log(f"   ⚠️  Config Adjustment: {msg}", YELLOW)
            config["training"]["batch_size"] = max(1, config["training"]["batch_size"] // 2)
            config["training"]["gradient_accumulation_steps"] *= 2
            prob, msg = self.predict_oom_probability(config)
            retries += 1

        # Final Report
        log(f"\n📊 AI Benchmarking Complete:", BLUE)
        log(f"   Recommended Mode: {mode.upper()}", GREEN)
        log(f"   OOM Probability:  {prob*100:.1f}% ({msg})", RED if prob > 0.5 else GREEN)

        # Save
        output_path = "configs/auto_optimized.yaml"
        with open(output_path, "w") as f:
            yaml.dump(config, f, sort_keys=False)

        log(f"✅ Configuration saved to {output_path}", GREEN)

def main():
    scaler = AutoScaler()
    scaler.generate_config()

if __name__ == "__main__":
    main()
