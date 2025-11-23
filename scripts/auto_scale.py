#!/usr/bin/env python3
"""
MUSE Auto-Scaler (Scale-Up Evolution)
Detects hardware and generates optimal training configuration.
"""

import torch
import psutil
import yaml
import os
from muse_utils import log, GREEN, YELLOW, RED, BLUE

def get_vram_gb():
    if not torch.cuda.is_available():
        return 0
    try:
        # Use index 0 for now
        props = torch.cuda.get_device_properties(0)
        return props.total_memory / (1024**3)
    except:
        return 0

def get_ram_gb():
    return psutil.virtual_memory().total / (1024**3)

def generate_config(vram, ram):
    log(f"🔍 Hardware Detected: VRAM={vram:.1f}GB, RAM={ram:.1f}GB", BLUE)

    # Base templates
    config = {
        "_base_": "base_config",
        "model": {},
        "training": {}
    }

    model_size = "small"

    if vram < 4:
        log("   -> Ultra Low Resource Mode (CPU/Tiny GPU)", YELLOW)
        config["model"]["d_model"] = 256
        config["model"]["n_layers"] = 8
        config["model"]["n_heads"] = 4
        config["training"]["batch_size"] = 1
        config["training"]["gradient_accumulation_steps"] = 8
        model_size = "tiny"

    elif vram < 8:
        log("   -> Low Resource Mode (Laptop GPU)", YELLOW)
        config["model"]["d_model"] = 512
        config["model"]["n_layers"] = 12
        config["model"]["n_heads"] = 8
        config["training"]["batch_size"] = 4
        config["training"]["gradient_accumulation_steps"] = 4
        model_size = "small"

    elif vram < 12:
        log("   -> Mid Resource Mode (RTX 3080 Target)", GREEN)
        config["model"]["d_model"] = 768
        config["model"]["n_layers"] = 12 # ResNet-BK is deep-efficient
        config["model"]["n_heads"] = 12
        config["training"]["batch_size"] = 8
        config["training"]["gradient_accumulation_steps"] = 4
        model_size = "base"

    elif vram < 24:
        log("   -> High Resource Mode (3090/4090)", GREEN)
        config["model"]["d_model"] = 1024
        config["model"]["n_layers"] = 24
        config["model"]["n_heads"] = 16
        config["training"]["batch_size"] = 16
        config["training"]["gradient_accumulation_steps"] = 2
        model_size = "large"

    else:
        log("   -> Ultra Resource Mode (A100/H100)", GREEN)
        config["model"]["d_model"] = 1600
        config["model"]["n_layers"] = 32
        config["model"]["n_heads"] = 25
        config["training"]["batch_size"] = 32
        config["training"]["gradient_accumulation_steps"] = 1
        model_size = "xl"

    # Adjust for System RAM if bottleneck
    if ram < 16 and model_size in ["large", "xl"]:
        log("⚠️  System RAM is low, reducing batch size...", YELLOW)
        config["training"]["batch_size"] = max(1, config["training"]["batch_size"] // 2)

    # Write config
    output_path = "configs/auto_optimized.yaml"
    with open(output_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)

    log(f"✅ Optimized config written to {output_path}", GREEN)
    log(f"   Model Size: {model_size.upper()}", BLUE)
    log("\nTo start training, run:", NC)
    log(f"   make train-user CONFIG={output_path}", GREEN)

if __name__ == "__main__":
    vram = get_vram_gb()
    ram = get_ram_gb()
    generate_config(vram, ram)
