"""Ponto de entrada compatível para o pipeline SPECTRA.

O código da aplicação vive em ``src/spectra``. Este arquivo oferece o comando
direto ``python main.py``.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spectra import pipeline as _pipeline


if __name__ == "__main__":
    _pipeline.cli()
else:
    sys.modules[__name__] = _pipeline
