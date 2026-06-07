#!/usr/bin/env bash
#
# install.sh - Set up Ollama-Transcriber in a Python virtual environment on Linux.
#
# Usage:  ./install.sh          (CPU-only PyTorch)
#         ./install.sh --cuda   (PyTorch with CUDA support)
#
set -euo pipefail

VENV_DIR="venv"
CUDA_FLAG=false

for arg in "$@"; do
    case "$arg" in
        --cuda) CUDA_FLAG=true ;;
        *) echo "Unknown option: $arg"; echo "Usage: $0 [--cuda]"; exit 1 ;;
    esac
done

# ---------- helper ----------
check_command() {
    if ! command -v "$1" &>/dev/null; then
        echo "ERROR: '$1' is not installed."
        echo "$2"
        exit 1
    fi
}

# ---------- prerequisites ----------
echo "=== Checking prerequisites ==="

check_command python3 \
    "Install Python 3.8+:  sudo apt install python3 python3-venv python3-pip"

# Verify Python version >= 3.8
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 8 ]; }; then
    echo "ERROR: Python 3.8+ is required (found $PY_VER)."
    exit 1
fi
echo "  Python $PY_VER OK"

check_command ffmpeg \
    "Install FFmpeg:  sudo apt install ffmpeg"
echo "  ffmpeg OK"

# Check that python3-venv is available
if ! python3 -m venv --help &>/dev/null; then
    echo "ERROR: python3-venv is not installed."
    echo "Install it:  sudo apt install python3-venv"
    exit 1
fi
echo "  python3-venv OK"

# ---------- virtual environment ----------
echo ""
echo "=== Creating virtual environment ==="

if [ -d "$VENV_DIR" ]; then
    echo "  Existing venv found at ./$VENV_DIR — reusing it."
else
    python3 -m venv "$VENV_DIR"
    echo "  Created ./$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "  Activated venv ($(python --version))"

# ---------- pip upgrade ----------
echo ""
echo "=== Upgrading pip ==="
pip install --upgrade pip

# ---------- PyTorch ----------
echo ""
echo "=== Installing PyTorch ==="

if [ "$CUDA_FLAG" = true ]; then
    echo "  Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio
else
    echo "  Installing PyTorch (CPU-only)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# ---------- project requirements ----------
echo ""
echo "=== Installing project requirements ==="
pip install -r requirements.txt

# ---------- verify ----------
echo ""
echo "=== Verifying installation ==="
python pytorch_verify.py

echo ""
echo "=== Setup complete ==="
echo ""
echo "To use Ollama-Transcriber:"
echo "  1. Activate the environment:  source $VENV_DIR/bin/activate"
echo "  2. Start Ollama:              ollama serve"
echo "  3. Run the tool:              python main.py --help"
