#!/usr/bin/env bash
# Setup script for project virtual environment
# Creates .venv, installs requirements and common plotting libs, and registers an ipykernel

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"
PY_SYSTEM="$(command -v python3 || command -v python)"

echo "Project root: $PROJECT_ROOT"

if [ -d "$VENV_DIR" ]; then
  echo "Using existing venv at $VENV_DIR"
else
  echo "Creating virtual environment at $VENV_DIR"
  "$PY_SYSTEM" -m venv "$VENV_DIR"
fi

VENV_PY="$VENV_DIR/bin/python"

echo "Upgrading pip, setuptools and wheel in venv..."
"$VENV_PY" -m pip install --upgrade pip setuptools wheel

REQ_FILE="$PROJECT_ROOT/requirements.txt"
if [ -f "$REQ_FILE" ]; then
  echo "Installing packages from requirements.txt"
  # Install only non-comment lines via pip -r (requirements already formatted)
  "$VENV_PY" -m pip install -r "$REQ_FILE"
else
  echo "No requirements.txt found at $REQ_FILE. Skipping."
fi

echo "Installing plotting and notebook helpers (matplotlib, seaborn, ipykernel)..."
"$VENV_PY" -m pip install "matplotlib>=3.8.0" "seaborn>=0.13.0" ipykernel

KERNEL_NAME="uab-wifi-venv"
KERNEL_DISPLAY_NAME="UAB WiFi (.venv)"
echo "Registering ipykernel kernel name=$KERNEL_NAME display_name=\"$KERNEL_DISPLAY_NAME\""
"$VENV_PY" -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY_NAME" || true

echo
echo "Done. To use the environment in a shell session run:"
echo "  source $VENV_DIR/bin/activate"
echo
echo "Or in VS Code / Jupyter choose the kernel named: $KERNEL_DISPLAY_NAME"

exit 0
