"""
Nucleo compartido del proyecto: lo que cruza la frontera entre los objetivos.

OE1 (`oe1_corpus/`) lo usa para construir el corpus; OE2
(`oe2_aumento_de_datos/analisis_intrinseco/`) lo usa para analizarlo. Manteniendo la
taxonomia y las reglas de anotacion en un unico lugar se evita que ambos trabajen
sobre definiciones divergentes.

`shiwilu.excel` y `shiwilu.clasificacion` NO se importan aqui a proposito: el primero
depende de openpyxl (solo OE1) y el segundo de torch/scikit-learn (solo OE2-OE3).
Los scripts que los requieren hacen `from shiwilu.excel import ...` o
`from shiwilu.clasificacion import ...`.
Los scripts que lo requieren hacen `from shiwilu.excel import ...`.
"""

from shiwilu.anotacion import anotar_dominio, anotar_intencion
from shiwilu.dominios import COLOR_DOM, DOMINIOS
from shiwilu.rutas import (
    CORPUS_CSV,
    CORPUS_DIR,
    DATOS,
    FIGURAS,
    FLASHCARDS,
    INTERMEDIOS,
    LOGS,
    ORACIONES_TRAD,
    PARES_ANOTADOS,
    PDF,
    RAIZ,
    RESULTADOS,
    SALIDA,
    TABLAS,
    VOCABULARIO,
    preparar_directorios,
)
from shiwilu.taxonomia import (
    COLOR_INT,
    DESC_INT,
    INTENCIONES,
    NORMALIZAR_INT,
    TIPO_SEARLE,
)

__all__ = [
    "anotar_dominio", "anotar_intencion",
    "COLOR_DOM", "DOMINIOS",
    "CORPUS_CSV", "CORPUS_DIR", "DATOS", "FIGURAS", "FLASHCARDS", "INTERMEDIOS",
    "LOGS", "ORACIONES_TRAD", "PARES_ANOTADOS", "PDF", "RAIZ", "RESULTADOS",
    "SALIDA", "TABLAS", "VOCABULARIO", "preparar_directorios",
    "COLOR_INT", "DESC_INT", "INTENCIONES", "NORMALIZAR_INT", "TIPO_SEARLE",
]

__version__ = "1.0.0"
