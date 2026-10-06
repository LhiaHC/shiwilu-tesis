"""
Separa, en los corpus aumentados de OE3, la silueta (mean pooling, coseno) de las oraciones REALES y de las SINTETICAS.

La silueta global de un corpus aumentado mezcla ambos tipos de punto. Si las oraciones sinteticas quedan bien agrupadas por categoria
(porque son "tipicas"), suben la silueta global aunque las oraciones reales no esten mejor organizadas. Este script lo comprueba.

Salida: oe3_caracterizacion_embeddings/resultados/silueta_real_vs_sintetico.csv
Uso (desde la raiz del repositorio): python oe3_caracterizacion_embeddings/silueta_real_vs_sintetico.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import caracterizacion as c  # noqa: E402
from sklearn.metrics import silhouette_samples  # noqa: E402
from shiwilu.clasificacion import quitar_puntuacion  # noqa: E402

filas = []
for variante in ["generate_then_refine", "retrotraduccion"]:
    df = c.cargar_corpus_variante(variante)
    es_real = np.arange(len(df)) < 700   # las 700 reales van primero
    oraciones = [quitar_puntuacion(t, quitar_tildes=True) for t in df["shiwilu"]]
    y = df["intencion"].to_numpy()
    for modelo in ["labse", "mbert", "xlmr"]:
        X = c.extraer_todas_las_estrategias(oraciones, modelo)["mean_pooling"]
        s_aum = silhouette_samples(X, y, metric="cosine")
        s_orig = silhouette_samples(X[es_real], y[es_real], metric="cosine")   # las 700 reales entre si (= corpus original)
        filas.append({"variante": variante, "modelo": modelo, "silueta_original_700": s_orig.mean(),
                      "silueta_aumentado_todas": s_aum.mean(), "silueta_aumentado_solo_reales": s_aum[es_real].mean(),
                      "silueta_aumentado_solo_sinteticas": s_aum[~es_real].mean()})
R = pd.DataFrame(filas).round(4)
R.to_csv(c.CARACTERIZACION_RESULTADOS / "silueta_real_vs_sintetico.csv", index=False)
print(R.to_string(index=False))
