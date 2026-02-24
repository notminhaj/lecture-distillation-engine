#!/usr/bin/env python
"""
Standalone script to run the pipeline.

Usage:
    python scripts/run_pipeline.py --input lecture.mp4 --domain khutba
    python scripts/run_pipeline.py --input https://www.youtube.com/watch?v=... --clips 3

For the CLI (after `pip install -e .`):
    distill run --input lecture.mp4 --domain khutba
"""

import sys
from pathlib import Path

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from distillation.cli import app

if __name__ == "__main__":
    app()
