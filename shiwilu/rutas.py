"""
Rutas del proyecto, ancladas a la raiz del repositorio.

Todas las rutas se derivan de la ubicacion de este archivo, no del directorio
de trabajo. Eso permite ejecutar los scripts del pipeline y los notebooks desde
cualquier carpeta sin que se rompan las rutas relativas.

El repositorio esta organizado por objetivo especifico (OE) de la tesis; el
numero de cada carpeta `oeN_*` es el del objetivo:

  oe1_corpus/                       OE1 (R1-R3): construccion del corpus
  corpus/                           frontera: salida de OE1, entrada del resto
  oe2_aumento_de_datos/             OE2 (R4-R6): baselines, analisis intrinseco,
                                    tecnicas de aumento y su evaluacion
  oe3_caracterizacion_embeddings/   OE3 (R7-R8)
  oe4_sintesis/                     OE4 (R9)
"""

from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

# --- Objetivo 1: construccion del corpus (R1-R3) --------------------------
OE1          = RAIZ / "oe1_corpus"
DATOS        = OE1 / "datos"
FLASHCARDS   = DATOS / "flashcards2.csv"
PDF          = DATOS / "II_TEXTOS_SHIWILU.pdf"

INTERMEDIOS  = OE1 / "intermedios"
LOGS         = INTERMEDIOS / "logs"

PARES_ANOTADOS = INTERMEDIOS / "1_corpus_pares_anotados.xlsx"
ORACIONES_TRAD = INTERMEDIOS / "2_oraciones_traducidas.xlsx"
VOCABULARIO    = INTERMEDIOS / "vocabulario_dominios.json"

# Alias historico: las etapas 0-2 (OE1) escriben sus productos intermedios aqui.
SALIDA = INTERMEDIOS

# --- Frontera entre objetivos: producto de OE1, insumo del resto ----------
CORPUS_DIR = RAIZ / "corpus"
CORPUS_CSV = CORPUS_DIR / "corpus_shiwilu_final.csv"

# --- Objetivo 2: aumento de datos (R4-R6) ---------------------------------
OE2 = RAIZ / "oe2_aumento_de_datos"

# Particiones congeladas de la validacion cruzada (NO recalcular: ver
# shiwilu/clasificacion.py > cargar_folds).
PARTICIONES            = OE2 / "particiones"
FOLDS_FIJOS_CSV        = PARTICIONES / "folds_fijos.csv"
CORPUS_NORMALIZADO_CSV = PARTICIONES / "corpus_normalizado.csv"

# R5: analisis intrinseco del corpus
ANALISIS_INTRINSECO = OE2 / "analisis_intrinseco"
RESULTADOS          = ANALISIS_INTRINSECO / "resultados"
TABLAS              = RESULTADOS / "tablas"
FIGURAS             = RESULTADOS / "figuras"

MARCADORES_CSV               = TABLAS / "analisis_marcadores_documentados.csv"
PALABRAS_CARACTERISTICAS_CSV = TABLAS / "analisis_palabras_caracteristicas.csv"
SECUENCIAS_INICIALES_CSV     = TABLAS / "analisis_secuencias_iniciales.csv"
SECUENCIAS_FINALES_CSV       = TABLAS / "analisis_secuencias_finales.csv"

# R6: generadores de datos sinteticos y sus salidas
TECNICAS_AUMENTO = OE2 / "tecnicas_aumento"
AUMENTO_SALIDA   = TECNICAS_AUMENTO / "salidas"

# Repo externo (F. Prado) clonado localmente para la tecnica de retrotraduccion
# - no se vendoriza, ver oe2_aumento_de_datos/tecnicas_aumento/README.md
NMT_REPO_EXTERNO = TECNICAS_AUMENTO / "tesis_spa_jeb"

# R4 + R6: evaluacion vigente (validacion cruzada, comparaciones, curvas ROC)
EVALUACION            = OE2 / "evaluacion"
EVALUACION_RESULTADOS = EVALUACION / "resultados"

# Material historico (ya no se corre; se conserva como evidencia)
HISTORICO              = OE2 / "historico"
SPLIT_UNICO            = HISTORICO / "split_unico"      # split 70/15/15, texto crudo
SPLIT_FIJO_CSV         = SPLIT_UNICO / "split_fijo.csv"
BASELINE_RESULTADOS    = SPLIT_UNICO / "resultados_baselines"
TECNICAS_RESULTADOS    = SPLIT_UNICO / "resultados_tecnicas"
ATAJOS_TEXTO_CRUDO     = HISTORICO / "atajos_texto_crudo"  # CV con puntuacion / mayusculas

# --- Objetivo 3: caracterizacion de embeddings (R7-R8) --------------------
OE3 = RAIZ / "oe3_caracterizacion_embeddings"
CARACTERIZACION_RESULTADOS = OE3 / "resultados"

# --- Objetivo 4: sintesis comparativa final (R9) ---------------------------
# Cruza lo extrinseco (R4-R6: F1 de clasificacion) con lo intrinseco (R7-R8:
# calidad de agrupamiento) para identificar la "configuracion optima".
OE4 = RAIZ / "oe4_sintesis"
SINTESIS_RESULTADOS = OE4 / "resultados"


def preparar_directorios() -> None:
    """Crea los directorios de salida que el pipeline necesita."""
    for d in (INTERMEDIOS, LOGS, CORPUS_DIR, AUMENTO_SALIDA, EVALUACION_RESULTADOS):
        d.mkdir(parents=True, exist_ok=True)
