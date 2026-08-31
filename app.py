"""Ponto de entrada compatível para a interface web SPECTRA."""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spectra import web as _web


if __name__ == "__main__":
    _web.run()
else:
    sys.modules[__name__] = _web
