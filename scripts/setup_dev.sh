#!/bin/bash
# ==============================================================================
# ResNet-BK Development Environment Setup Script
# ==============================================================================
#
# This script automates the setup of a local development environment for the
# ResNet-BK project. It performs the following steps:
#
# 1. Checks for the required Python version (3.9+).
# 2. Creates a Python virtual environment in `./.venv`.
# 3. Upgrades pip to the latest version.
# 4. Installs core dependencies from `requirements.txt`.
# 5. Installs development dependencies specified in `pyproject.toml`.
# 6. Installs the project in editable mode.
#
# Usage:
#   bash scripts/setup_dev.sh
#
# After running, activate the virtual environment with:
#   source .venv/bin/activate
#
# ==============================================================================

set -e  # Exit immediately if a command exits with a non-zero status.
set -o pipefail # Return the exit code of the last command in a pipeline to fail
set -u # Treat unset variables as an error

# --- Configuration ---
PYTHON_MAJOR=3
PYTHON_MINOR=9
VENV_DIR=".venv"
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

# --- Helper Functions ---
print_info() {
    echo -e "\033[34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[32m[SUCCESS]\033[0m $1"
}

print_error() {
    echo -e "\033[31m[ERROR]\033[0m $1" >&2
}

command_exists() {
    command -v "$1" &> /dev/null
}

# --- Main Logic ---

# 1. Check Python Version
print_info "Checking Python version..."
if ! command_exists python3; then
    print_error "Python 3 is not installed. Please install Python ${PYTHON_MAJOR}.${PYTHON_MINOR} or higher."
    exit 1
fi

PY_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
PY_MAJOR_VER=$(echo "$PY_VERSION" | cut -d'.' -f1)
PY_MINOR_VER=$(echo "$PY_VERSION" | cut -d'.' -f2)

if [ "$PY_MAJOR_VER" -lt "$PYTHON_MAJOR" ] || ([ "$PY_MAJOR_VER" -eq "$PYTHON_MAJOR" ] && [ "$PY_MINOR_VER" -lt "$PYTHON_MINOR" ]); then
    print_error "Python version ${PY_VERSION} is not sufficient. Please use Python ${PYTHON_MAJOR}.${PYTHON_MINOR} or higher."
    exit 1
fi
print_success "Python version ${PY_VERSION} is compatible."

# Change to project root directory
cd "$PROJECT_ROOT"

# 2. Create Virtual Environment
if [ -d "$VENV_DIR" ]; then
    print_info "Virtual environment '$VENV_DIR' already exists. Skipping creation."
else
    print_info "Creating Python virtual environment in '$VENV_DIR'..."
    python3 -m venv "$VENV_DIR"
    print_success "Virtual environment created."
fi

# Activate the virtual environment for the rest of the script
source "${VENV_DIR}/bin/activate"

# 3. Upgrade Pip
print_info "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel
print_success "Pip upgraded."

# 4. Install Core Dependencies
if [ -f "requirements.txt" ]; then
    print_info "Installing core dependencies from requirements.txt..."
    pip install -r requirements.txt
    print_success "Core dependencies installed."
else
    print_error "requirements.txt not found in project root."
    exit 1
fi

# 5. Install Development and All Optional Dependencies
print_info "Installing all optional dependencies (dev, jupyter, etc.) from pyproject.toml..."
pip install -e ".[all]"
print_success "All optional dependencies installed."

# 6. Install Project in Editable Mode
print_info "Installing the project in editable mode..."
pip install -e .
print_success "Project installed in editable mode."

# --- Final Instructions ---
echo
print_success "Development environment setup is complete!"
print_info "To activate the virtual environment in your shell, run:"
echo -e "  \033[1msource ${VENV_DIR}/bin/activate\033[0m"
echo
print_info "You can now run tests with 'pytest' and format code with 'black' and 'isort'."
