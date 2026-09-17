"""
Deja el paquete `shiwilu` importable sin necesidad de instalarlo.

Ver 1_objetivo1_corpus/pipeline/_ruta_raiz.py — mismo patron.
"""

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
