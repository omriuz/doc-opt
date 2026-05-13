#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / "src"
if SRC.as_posix() not in sys.path:
    sys.path.insert(0, SRC.as_posix())

from doc_opt.report_plot import main_steps


if __name__ == "__main__":
    raise SystemExit(main_steps())
