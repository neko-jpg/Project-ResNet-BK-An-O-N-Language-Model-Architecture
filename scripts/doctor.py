#!/usr/bin/env python3
"""
MUSE Doctor - System Diagnostic Tool
MUSE ドクター - システム診断ツール
"""
import os
import sys
import shutil
import platform
import psutil
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# --- Language Configuration ---
LANG = "1"
CONFIG_FILE = Path(".muse_config")
if CONFIG_FILE.exists():
    try:
        with open(CONFIG_FILE) as f:
            for line in f:
                if "MUSE_LANG" in line:
                    val = line.strip().split("=")[1]
                    LANG = val.strip("'\"")
    except:
        pass

IS_JP = (LANG == "2")

console = Console()

def t(en, jp):
    return jp if IS_JP else en

def check_system():
    table = Table(title=t("System Health Check", "システム健全性チェック"))
    table.add_column(t("Component", "項目"), style="cyan")
    table.add_column(t("Status", "状態"), style="magenta")
    table.add_column(t("Details", "詳細"), style="green")

    # Python
    py_ver = sys.version.split()[0]
    status = "OK" if sys.version_info >= (3, 9) else "FAIL"
    table.add_row("Python", status, f"Version {py_ver}")

    # OS
    os_info = f"{platform.system()} {platform.release()}"
    table.add_row("OS", "INFO", os_info)

    # RAM
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024**3)
    status = "OK" if total_gb >= 8 else "WARN"
    table.add_row("RAM", status, f"{total_gb:.1f} GB (Available: {mem.available / (1024**3):.1f} GB)")

    # CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            status = "OK"
            details = f"{gpu_name} ({vram:.1f} GB VRAM)"
        else:
            status = "WARN"
            details = t("CUDA not available (CPU Mode)", "CUDA利用不可 (CPUモード)")
    except ImportError:
        status = "FAIL"
        details = "PyTorch not installed"

    table.add_row("GPU (CUDA)", status, details)

    # Disk Space
    disk = psutil.disk_usage('.')
    free_gb = disk.free / (1024**3)
    status = "OK" if free_gb > 10 else "WARN"
    table.add_row("Disk Space", status, f"Free: {free_gb:.1f} GB")

    console.print(table)

def check_project():
    table = Table(title=t("Project Configuration", "プロジェクト設定状況"))
    table.add_column(t("Component", "項目"), style="cyan")
    table.add_column(t("Status", "状態"), style="magenta")
    table.add_column(t("Details", "詳細"), style="green")

    # Virtual Env
    venv_path = Path("venv_ubuntu")
    if venv_path.exists():
        table.add_row("Virtual Env", "OK", "venv_ubuntu detected")
    else:
        table.add_row("Virtual Env", "WARN", t("Not found (run 'make setup')", "未検出 ('make setup'を実行してください)"))

    # Data Directory
    data_path = Path("data")
    if data_path.exists():
        files = list(data_path.glob("*"))
        count = len(files)
        table.add_row("Data Dir", "OK", f"{count} items in ./data")
    else:
        table.add_row("Data Dir", "WARN", t("Not initialized", "未初期化"))

    # Import Directory
    import_path = Path("data/import")
    if import_path.exists():
         table.add_row("Import Dir", "OK", t("Ready to drop files", "ファイル投入可能"))
    else:
         table.add_row("Import Dir", "INFO", t("Will be created on usage", "使用時に作成されます"))

    console.print(table)

def main():
    console.print(Panel.fit(t("MUSE Doctor - Diagnostics", "MUSE ドクター - システム診断"), style="bold blue"))

    check_system()
    print()
    check_project()

    print()
    console.print(t("[bold green]Diagnostics Complete![/bold green]", "[bold green]診断完了！[/bold green]"))
    console.print(t("If you see any FAIL status, please run [bold yellow]make setup[/bold yellow] again.", "FAILなどのエラーがある場合は、もう一度 [bold yellow]make setup[/bold yellow] を試してください。"))

if __name__ == "__main__":
    main()
