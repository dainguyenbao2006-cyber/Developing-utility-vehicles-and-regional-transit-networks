"""Thin CLI wrapper script for running end-to-end ADAS inference."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from cli import main  # noqa: E402

if __name__ == "__main__":
    main()
