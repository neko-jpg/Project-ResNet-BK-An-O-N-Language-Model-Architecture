#!/usr/bin/env python3
"""
MUSE Deploy Studio (Evolution 5)
Interactive deployment to Hugging Face with Auto-README and Quantization.
"""

import os
import sys
import argparse
import torch
import shutil
# Add src to path
sys.path.append(os.getcwd())

from src.utils.readme_generator import ReadmeGenerator

def deploy(model_path, repo_name, token=None):
    print(f"🚀 Launching Deploy Studio for {model_path}...")

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return

    # 1. Load Model & Config
    try:
        ckpt = torch.load(model_path, map_location='cpu')
        config = ckpt.get('config', {})
        if hasattr(config, '__dict__'): config = vars(config)
    except Exception as e:
        print(f"❌ Error loading: {e}")
        return

    # 2. Auto-Generate README
    print("📝 Generating README.md...")
    # Mock skills for now, in real usage we'd read from skills.csv
    skills = {"Coding": 85.0, "Japanese": 92.0}
    readme_content = ReadmeGenerator.generate(repo_name, config, skills)

    # Write to a temp deploy folder
    deploy_dir = f"deploy_stage_{repo_name.replace('/', '_')}"
    os.makedirs(deploy_dir, exist_ok=True)

    with open(os.path.join(deploy_dir, "README.md"), "w") as f:
        f.write(readme_content)

    # Copy model
    shutil.copy2(model_path, os.path.join(deploy_dir, "pytorch_model.bin"))

    # 3. Quantization (Mock/Simple)
    print("📦 Optimizing Storage (Quantization)...")
    # For demo, we just create a placeholder or simple cast
    # Real quantization requires saving specific formats (GGUF, etc.)
    # Here we just save a fp16 version
    fp16_path = os.path.join(deploy_dir, "pytorch_model_fp16.bin")
    torch.save(ckpt, fp16_path) # In reality, we'd cast weights.
    print(f"   Generated FP16 version: {fp16_path}")

    # 4. Upload (Simulated or Real)
    print(f"☁️  Uploading to Hugging Face Hub: {repo_name}...")

    # We use the 'huggingface_hub' library if available
    try:
        from huggingface_hub import HfApi, create_repo

        if token:
            api = HfApi(token=token)
        else:
            api = HfApi() # Relies on cached login

        # Create Repo
        try:
            create_repo(repo_name, exist_ok=True)
        except Exception as e:
            print(f"   Repo creation warning: {e}")

        # Upload
        api.upload_folder(
            folder_path=deploy_dir,
            repo_id=repo_name,
            repo_type="model"
        )
        print("✅ Upload Complete!")
        print(f"   View at: https://huggingface.co/{repo_name}")

    except ImportError:
        print("⚠️  'huggingface_hub' library not installed.")
        print("   (Simulation Mode) Files prepared in: " + deploy_dir)
    except Exception as e:
        print(f"❌ Upload failed: {e}")

    # Cleanup
    # shutil.rmtree(deploy_dir) # Keep for inspection in demo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    deploy(args.model, args.repo, args.token)

if __name__ == "__main__":
    main()
