#!/usr/bin/env bash
# One-shot CPU dev environment setup.
set -euo pipefail
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
emocapnet smoke --out runs/smoke --epochs 1
