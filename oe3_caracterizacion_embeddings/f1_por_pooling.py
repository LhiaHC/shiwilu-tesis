"""
F1 de clasificacion para las 36 combinaciones de los experimentos intrinsecos: 3 modelos x 3 corpus (sin aumento, Retrotraduccion, GtR)
x 4 estrategias de pooling (CLS, mean, max, combinacion de capas).

Los experimentos intrinsecos de OE3 no usan clasificador; aqui se entrena el MISMO clasificador del protocolo B de OE2 (Regresion Logistica,
C elegido por CV interna K=5, 5 folds congelados, texto normalizado, sinteticos generados solo con el entrenamiento de cada particion) sobre los
embeddings de cada estrategia, para poder comparar la silueta con el F1.

Nota: en LaBSE el "mean pooling" de aqui es el del modelo crudo (AutoModel), distinto de la extraccion "de fabrica" de sentence-transformers que usan
los 12 experimentos de OE2; en mBERT y XLM-R el mean pooling es el mismo. Mixup queda fuera (no produce texto).

Salida: evaluacion/resultados/cv_interna_pooling/ (por corrida) y oe3_caracterizacion_embeddings/resultados/f1_por_pooling.csv
Uso (desde la raiz del repositorio; ~1-2 h en CPU):  python oe3_caracterizacion_embeddings/f1_por_pooling.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ)); sys.path.insert(0, str(RAIZ / "oe2_aumento_de_datos" / "evaluacion")); sys.path.insert(0, str(Path(__file__).resolve().parent))

import caracterizacion as car  # noqa: E402
import validacion_cruzada_cv_interna as cv  # noqa: E402
from shiwilu.rutas import CARACTERIZACION_RESULTADOS, EVALUACION_RESULTADOS  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

# --- el modelo se carga una sola vez (car.extraer_todas_las_estrategias lo recarga en cada llamada)
_m, _t = {}, {}
_from_m, _from_t = AutoModel.from_pretrained, AutoTokenizer.from_pretrained
AutoModel.from_pretrained = staticmethod(lambda i, **k: _m.setdefault(i, _from_m(i, **k)))
AutoTokenizer.from_pretrained = staticmethod(lambda i, **k: _t.setdefault(i, _from_t(i, **k)))

# --- cache de embeddings por (modelo, oracion): un forward produce las 4 estrategias
CACHE, ESTRATEGIA = {}, [None]


def extraer(oraciones, modelo, *a, **k):
    faltan = [o for o in dict.fromkeys(oraciones) if (modelo, o) not in CACHE]
    if faltan:
        d = car.extraer_todas_las_estrategias(faltan, modelo)
        for i, o in enumerate(faltan):
            CACHE[(modelo, o)] = {e: d[e][i] for e in d}
    return np.stack([CACHE[(modelo, o)][ESTRATEGIA[0]] for o in oraciones])


cv.extraer_embeddings = extraer
cv.CV_RESULTADOS_BASE = EVALUACION_RESULTADOS / "cv_interna_pooling"

CORRIDAS = [("sin_aumento", "sin aumento", []), ("retrotraduccion", "Retrotraduccion", ["--checkpoint", "x"]),
            ("generate_then_refine", "GtR 120", ["--cantidad", "120"])]
filas = []
for estrategia in car.ESTRATEGIAS:
    ESTRATEGIA[0] = estrategia
    for tecnica, nombre, extra in CORRIDAS:
        print(f"\n######## {estrategia} / {nombre} ########", flush=True)
        sys.argv = ["cv", "--tecnica", tecnica, "--etiqueta", estrategia] + extra
        cv.main()
        r = pd.read_csv(cv.CV_RESULTADOS_BASE / f"validacion_cruzada_resumen_sin_puntuacion_cv_interna_{tecnica}_{estrategia}.csv")
        for x in r.itertuples():
            filas.append({"modelo": x.modelo, "corpus": nombre, "estrategia": estrategia, "f1_macro": x.f1_macro_agrupado,
                          "ic95_bajo": x.ic95_bajo, "ic95_alto": x.ic95_alto})
        pd.DataFrame(filas).to_csv(CARACTERIZACION_RESULTADOS / "f1_por_pooling.csv", index=False)
print("listo")
