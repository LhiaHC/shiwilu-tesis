"""
Deja el paquete `shiwilu` importable sin necesidad de instalarlo.

Ver 1_objetivo1_corpus/pipeline/_ruta_raiz.py — mismo patron. Esta carpeta
esta dos niveles bajo la raiz del repo (3_baselines_y_aumento_datos/tecnicas_aumento/),
por eso usa parents[2].
"""

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]

if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
