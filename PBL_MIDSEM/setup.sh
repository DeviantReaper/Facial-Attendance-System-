#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  NEURO-SCAN · Facial Attendance System
#  macOS M-series (Apple Silicon) setup script
# ─────────────────────────────────────────────────────────────

set -e

echo ""
echo "  ╔══════════════════════════════════════════════╗"
echo "  ║   NEURO-SCAN  ·  Environment Setup           ║"
echo "  ╚══════════════════════════════════════════════╝"
echo ""

# 1. Create & activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

echo "  [1/4] Virtual environment activated."

# 2. Upgrade pip silently
pip install --upgrade pip -q

echo "  [2/4] pip upgraded."

# 3. Install dependencies
pip install flask opencv-python mediapipe numpy -q

echo "  [3/4] Dependencies installed: flask, opencv-python, mediapipe, numpy"

# 4. Done
echo "  [4/4] Setup complete!"
echo ""
echo "  To run the app:"
echo "    source .venv/bin/activate"
echo "    python app.py"
echo ""
echo "  Then open: http://127.0.0.1:5000"
echo ""
