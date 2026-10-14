#!/usr/bin/env bash
# eNegmatic launcher for macOS / Linux - sets up a local Python environment on first run.
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    echo "Setting up Python environment (first run only)..."
    python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
fi
.venv/bin/python -c "import rawpy, cv2, PySide6" 2>/dev/null || .venv/bin/python -m pip install -r requirements.txt
exec .venv/bin/python main.py "$@"
