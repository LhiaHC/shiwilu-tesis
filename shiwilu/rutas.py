"""
Rutas del proyecto, ancladas a la raiz del repositorio.

Todas las rutas se derivan de la ubicacion de este archivo, no del directorio
de trabajo. Eso permite ejecutar los scripts del pipeline y los notebooks desde
cualquier carpeta sin que se rompan las rutas relativas.
"""

from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

# --- Fase 1: construccion del corpus -------------------------------------
FASE1        = RAIZ / "1_construccion_corpus"
DATOS        = FASE1 / "datos"
FLASHCARDS   = DATOS / "flashcards2.csv"
PDF          = DATOS / "II_TEXTOS_SHIWILU.pdf"

INTERMEDIOS  = FASE1 / "intermedios"
LOGS         = INTERMEDIOS / "logs"

PARES_ANOTADOS = INTERMEDIOS / "1_corpus_pares_anotados.xlsx"
ORACIONES_TRAD = INTERMEDIOS / "2_oraciones_traducidas.xlsx"
VOCABULARIO    = INTERMEDIOS / "vocabulario_dominios.json"

# --- Frontera entre fases: producto de la Fase 1, insumo de la Fase 2 ----
CORPUS_DIR = RAIZ / "corpus"
CORPUS_CSV = CORPUS_DIR / "corpus_shiwilu_final.csv"

# --- Fase 2: analisis del corpus -----------------------------------------
FASE2      = RAIZ / "2_analisis_corpus"
RESULTADOS = FASE2 / "resultados"
TABLAS     = RESULTADOS / "tablas"
FIGURAS    = RESULTADOS / "figuras"

# Alias historico: las etapas 0-2 escriben sus productos intermedios aqui.
SALIDA = INTERMEDIOS

# --- Fase 3: baseline de clasificacion (OE2/OE4, R4) ---------------------
FASE3               = RAIZ / "3_baseline_clasificacion"
BASELINE_RESULTADOS = FASE3 / "resultados"


def preparar_directorios() -> None:
    """Crea los directorios de salida que el pipeline necesita."""
    for d in (INTERMEDIOS, LOGS, CORPUS_DIR, BASELINE_RESULTADOS):
        d.mkdir(parents=True, exist_ok=True)
