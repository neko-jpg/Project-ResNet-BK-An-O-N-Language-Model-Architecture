#!/usr/bin/env python3
"""
Setup Hugging Face Token for MUSE Deploy
"""

import os
import sys

def setup_token():
    print("\n=== Hugging Face Token Setup ===")
    print("To deploy your model, you need a Hugging Face Write Token.")
    print("Get it here: https://huggingface.co/settings/tokens\n")

    token = input("Paste your HF Token: ").strip()

    if not token.startswith("hf_"):
        print("Warning: Token usually starts with 'hf_'. Proceeding anyway.")

    env_path = ".env"

    # Read existing .env
    lines = []
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            lines = f.readlines()

    # Update or append
    new_lines = []
    found = False
    for line in lines:
        if line.startswith("HF_TOKEN="):
            new_lines.append(f"HF_TOKEN={token}\n")
            found = True
        else:
            new_lines.append(line)

    if not found:
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"
        new_lines.append(f"HF_TOKEN={token}\n")

    # Write back
    with open(env_path, "w") as f:
        f.writelines(new_lines)

    print(f"✅ Token saved to {env_path}")

    # Update .gitignore
    gitignore_path = ".gitignore"
    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r") as f:
            content = f.read()
        if ".env" not in content:
            with open(gitignore_path, "a") as f:
                f.write("\n.env\n")
            print("✅ Added .env to .gitignore")

if __name__ == "__main__":
    setup_token()
