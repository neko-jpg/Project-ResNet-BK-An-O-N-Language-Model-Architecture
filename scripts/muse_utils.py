#!/usr/bin/env python3
"""
MUSE Utility Scripts (Utility Evolution)
Implements: CleanSafe, RestorePoint, Pack, VersionGuardian, Notifier
"""

import os
import sys
import shutil
import glob
import argparse
import datetime
import zipfile
import subprocess
import time
from pathlib import Path

# --- Colors ---
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
BLUE = '\033[0;34m'
NC = '\033[0m'

def log(msg, color=NC):
    print(f"{color}{msg}{NC}")

# --- 1. CleanSafe ---
def clean_safe(args):
    log("🧹 Starting CleanSafe (Locker Room Cleanup)...", BLUE)

    # 1. Remove __pycache__
    log("   Removing __pycache__...", YELLOW)
    for root, dirs, files in os.walk("."):
        for d in dirs:
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d))

    # 2. Remove temp files
    temp_patterns = ["*.tmp", "*.log", ".DS_Store"]
    for pattern in temp_patterns:
        for f in glob.glob(f"**/{pattern}", recursive=True):
            if "logs/" in f and not args.force_logs:
                continue # Skip logs unless forced
            try:
                os.remove(f)
            except OSError:
                pass

    # 3. Clean Checkpoints (Keep Final & Latest Step)
    if os.path.exists("checkpoints"):
        log("   Organizing Checkpoints...", YELLOW)
        # Group by prefix? For now, just simple logic
        # Keep all *_final.pt
        # For step_*.pt, keep only max step
        files = os.listdir("checkpoints")
        step_files = []
        for f in files:
            if f.startswith("checkpoint_step_") and f.endswith(".pt"):
                step_files.append(f)

        if step_files:
            # Sort by step number
            try:
                step_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                # Keep last
                latest = step_files[-1]
                for f in step_files[:-1]:
                    os.remove(os.path.join("checkpoints", f))
                    log(f"   Deleted old checkpoint: {f}", RED)
                log(f"   Kept latest checkpoint: {latest}", GREEN)
            except ValueError:
                pass # Format mismatch, skip

    log("✨ CleanSafe Complete!", GREEN)

# --- 2. RestorePoint ---
def restore_point(args):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    target_dir = os.path.join("restored_points", timestamp)

    log(f"⏳ Creating RestorePoint at {target_dir}...", BLUE)

    os.makedirs(target_dir, exist_ok=True)

    # Directories to backup
    backup_dirs = ["checkpoints", "configs", "logs"]
    for d in backup_dirs:
        if os.path.exists(d):
            shutil.copytree(d, os.path.join(target_dir, d))
            log(f"   Backed up: {d}", YELLOW)

    # Files to backup
    backup_files = [".muse_config", ".env"]
    for f in backup_files:
        if os.path.exists(f):
            shutil.copy2(f, target_dir)
            log(f"   Backed up: {f}", YELLOW)

    log(f"✅ RestorePoint Created: {timestamp}", GREEN)

# --- 3. Pack ---
def pack(args):
    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    zip_name = f"muse_pack_{timestamp}.zip"

    log(f"📦 Packing MUSE into {zip_name}...", BLUE)

    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Source code
        for root, dirs, files in os.walk("src"):
            if "__pycache__" in root: continue
            for file in files:
                if file.endswith(".pyc"): continue
                zipf.write(os.path.join(root, file))

        # Configs
        for root, dirs, files in os.walk("configs"):
            for file in files:
                zipf.write(os.path.join(root, file))

        # Scripts
        for root, dirs, files in os.walk("scripts"):
            if "__pycache__" in root: continue
            for file in files:
                zipf.write(os.path.join(root, file))

        # Checkpoints (Only final by default to save space)
        if os.path.exists("checkpoints"):
            for f in os.listdir("checkpoints"):
                if "final" in f and f.endswith(".pt"):
                    zipf.write(os.path.join("checkpoints", f))
                    log(f"   Included: {f}", YELLOW)

        # Meta
        files_to_include = ["README.md", "requirements.txt", "Makefile", "AGENTS.md"]
        for f in files_to_include:
            if os.path.exists(f):
                zipf.write(f)

    log(f"🎒 Pack Complete! Size: {os.path.getsize(zip_name) / 1024 / 1024:.2f} MB", GREEN)

# --- 4. VersionGuardian ---
def version_guardian(args):
    log("🛡️  Version Guardian checking...", BLUE)
    try:
        subprocess.run(["git", "fetch"], check=True, stdout=subprocess.DEVNULL)
        status = subprocess.run(["git", "status", "-uno"], capture_output=True, text=True).stdout

        if "behind" in status:
            log("⚠️  Update Available! (Your branch is behind)", RED)
            log("   Run 'git pull' to update.", YELLOW)
        else:
            log("✅ System Up-to-Date.", GREEN)
    except Exception as e:
        log(f"   Git check failed: {e}", RED)

# --- 5. Notifier ---
def notifier(args):
    message = args.message or "Task Complete!"
    log(f"🔔 {message}", GREEN)
    # Bell sound
    print('\a')

    # Mac Notification (if available)
    if sys.platform == 'darwin':
        try:
            subprocess.run(['osascript', '-e', f'display notification "{message}" with title "MUSE"'])
        except:
            pass
    # Linux Notification (notify-send)
    elif sys.platform == 'linux':
        if shutil.which('notify-send'):
            subprocess.run(['notify-send', 'MUSE', message])

def main():
    parser = argparse.ArgumentParser(description="MUSE Utility Toolbelt")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # CleanSafe
    p_clean = subparsers.add_parser("clean-safe", help="Clean garbage files")
    p_clean.add_argument("--force-logs", action="store_true", help="Delete logs too")

    # RestorePoint
    p_restore = subparsers.add_parser("restore-point", help="Create backup")

    # Pack
    p_pack = subparsers.add_parser("pack", help="Zip model for transport")

    # VersionGuardian
    p_guard = subparsers.add_parser("version-guardian", help="Check git status")

    # Notifier
    p_notify = subparsers.add_parser("notify", help="Send notification")
    p_notify.add_argument("--message", type=str, default=None)

    args = parser.parse_args()

    if args.command == "clean-safe":
        clean_safe(args)
    elif args.command == "restore-point":
        restore_point(args)
    elif args.command == "pack":
        pack(args)
    elif args.command == "version-guardian":
        version_guardian(args)
    elif args.command == "notify":
        notifier(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
