"""
Nucleo compartido del proyecto: lo que cruza la frontera entre las dos fases.

La Fase 1 (`1_objetivo1_corpus/`) lo usa para construir el corpus; la Fase 2
(`3_baselines_y_aumento_datos/analisis_intrinseco/`) lo usa para analizarlo. Manteniendo la taxonomia y las
reglas de anotacion en un unico lugar se evita que ambas fases trabajen sobre
definiciones divergentes.

`shiwilu.excel` NO se importa aqui a proposito: depende de openpyxl, que solo
necesita la Fase 1. Asi la Fase 2 puede usar el paquete sin instalarlo.
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
