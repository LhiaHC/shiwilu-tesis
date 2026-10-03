"""
Deja el paquete `shiwilu` importable sin necesidad de instalarlo.

Los scripts del pipeline hacen `import _ruta_raiz` antes de importar `shiwilu`.
La raiz se resuelve desde la ubicacion de este archivo, no desde el directorio
de trabajo, asi que el pipeline corre igual desde cualquier carpeta.

Si el proyecto se instala con `pip install -e .`, este modulo no hace nada.
"""

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]

if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
