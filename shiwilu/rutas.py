"""
Rutas del proyecto, ancladas a la raiz del repositorio.

Todas las rutas se derivan de la ubicacion de este archivo, no del directorio
de trabajo. Eso permite ejecutar los scripts del pipeline y los notebooks desde
cualquier carpeta sin que se rompan las rutas relativas.

Organizacion por objetivo especifico (ver Entregable 1):
  1_objetivo1_corpus/            OE1 (R1-R3): construccion del corpus
  2_baselines/                   OE2 (R4): baselines de clasificacion
  3_baselines_y_aumento_datos/   OE2 (R5-R6): analisis intrinseco + tecnicas
                                  de aumento, aplicadas sobre los baselines
"""

from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

# --- Objetivo 1: construccion del corpus (R1-R3) --------------------------
FASE1        = RAIZ / "1_objetivo1_corpus"
DATOS        = FASE1 / "datos"
FLASHCARDS   = DATOS / "flashcards2.csv"
PDF          = DATOS / "II_TEXTOS_SHIWILU.pdf"

INTERMEDIOS  = FASE1 / "intermedios"
LOGS         = INTERMEDIOS / "logs"

PARES_ANOTADOS = INTERMEDIOS / "1_corpus_pares_anotados.xlsx"
ORACIONES_TRAD = INTERMEDIOS / "2_oraciones_traducidas.xlsx"
VOCABULARIO    = INTERMEDIOS / "vocabulario_dominios.json"

# --- Frontera entre fases: producto de OE1, insumo del resto --------------
CORPUS_DIR = RAIZ / "corpus"
CORPUS_CSV = CORPUS_DIR / "corpus_shiwilu_final.csv"

# --- Objetivo 2: baselines de clasificacion (R4) --------------------------
FASE3               = RAIZ / "2_baselines"
BASELINE_RESULTADOS = FASE3 / "resultados"

# --- Objetivo 2: analisis intrinseco + aumento de datos (R5-R6) -----------
# Agrupa el analisis intrinseco del corpus (antes "2_analisis_corpus") y las
# tecnicas de aumento (antes "4_aumento_datos"), porque las tecnicas se
# aplican sobre los baselines usando insumos de ese analisis (marcadores,
# patrones candidatos).
FASE_AUMENTO = RAIZ / "3_baselines_y_aumento_datos"

FASE2      = FASE_AUMENTO / "analisis_intrinseco"
RESULTADOS = FASE2 / "resultados"
TABLAS     = RESULTADOS / "tablas"
FIGURAS    = RESULTADOS / "figuras"

# Alias historico: las etapas 0-2 (OE1) escriben sus productos intermedios aqui.
SALIDA = INTERMEDIOS

FASE4          = FASE_AUMENTO / "tecnicas_aumento"
AUMENTO_SALIDA = FASE4 / "salidas"

# Repo externo (F. Prado) clonado localmente para la tecnica de retrotraduccion
# — no se vendoriza, ver 3_baselines_y_aumento_datos/tecnicas_aumento/README.md
NMT_REPO_EXTERNO = FASE4 / "tesis_spa_jeb"

MARCADORES_CSV              = TABLAS / "analisis_marcadores_documentados.csv"
PALABRAS_CARACTERISTICAS_CSV = TABLAS / "analisis_palabras_caracteristicas.csv"
SECUENCIAS_INICIALES_CSV    = TABLAS / "analisis_secuencias_iniciales.csv"
SECUENCIAS_FINALES_CSV      = TABLAS / "analisis_secuencias_finales.csv"


def preparar_directorios() -> None:
    """Crea los directorios de salida que el pipeline necesita."""
    for d in (INTERMEDIOS, LOGS, CORPUS_DIR, BASELINE_RESULTADOS, AUMENTO_SALIDA):
        d.mkdir(parents=True, exist_ok=True)
