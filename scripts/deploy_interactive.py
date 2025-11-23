#!/usr/bin/env python3
"""
Interactive Deploy Script for MUSE
"""

import os
import sys
import glob
import subprocess
from setup_hf_token import setup_token
from muse_utils import log, GREEN, YELLOW, BLUE, RED

def get_checkpoints():
    # Find all checkpoints
    files = []
    if os.path.exists("checkpoints"):
        files = glob.glob("checkpoints/*.pt")
    files.sort(key=os.path.getmtime, reverse=True)
    return files

def main():
    log("🚀 MUSE Interactive Deploy", BLUE)

    # 1. Check Token
    if not os.path.exists(".env") or "HF_TOKEN" not in open(".env").read():
        log("⚠️  HF Token not found.", YELLOW)
        setup_token()

    # Load env (simple)
    with open(".env") as f:
        for line in f:
            if line.startswith("HF_TOKEN="):
                os.environ["HF_TOKEN"] = line.split("=")[1].strip()
                break

    # 2. Select Model
    checkpoints = get_checkpoints()
    if not checkpoints:
        log("❌ No checkpoints found in checkpoints/.", RED)
        return

    print("\nAvailable Checkpoints:")
    for i, f in enumerate(checkpoints[:5]):
        print(f"  [{i+1}] {f}")

    choice = input(f"\nSelect model [1]: ").strip()
    if not choice: choice = "1"

    try:
        model_path = checkpoints[int(choice)-1]
    except:
        log("❌ Invalid selection.", RED)
        return

    log(f"Selected: {model_path}", GREEN)

    # 3. Model Size & Repo
    # Try to guess size from filename
    size_guess = "100M"
    if "1b" in model_path.lower(): size_guess = "1B"
    if "10m" in model_path.lower(): size_guess = "10M"

    model_size = input(f"Model Size (10M, 100M, 1B) [{size_guess}]: ").strip() or size_guess

    # Repo ID
    default_repo = "username/muse-v1"
    repo_id = input(f"Hugging Face Repo ID [{default_repo}]: ").strip() or default_repo

    if "/" not in repo_id:
        log("❌ Invalid Repo ID (must be user/repo).", RED)
        return

    # 4. Upload
    cmd = [
        sys.executable, "scripts/upload_to_hf_hub.py",
        "--model_path", model_path,
        "--repo_id", repo_id,
        "--model_size", model_size
    ]

    log("\n📤 Starting Upload...", BLUE)
    try:
        subprocess.run(cmd, check=True)
        log("\n✅ Deployment Complete!", GREEN)
    except subprocess.CalledProcessError:
        log("\n❌ Deployment Failed.", RED)

if __name__ == "__main__":
    main()
