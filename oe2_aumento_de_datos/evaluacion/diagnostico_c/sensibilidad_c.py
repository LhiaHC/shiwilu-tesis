"""
Sensibilidad del F1 al hiperparametro C de la Regresion Logistica.

`validacion_cruzada.py` elige C en un "dev interno" de ~90-104 oraciones por fold,
una eleccion ruidosa que puede mover el F1 de una tecnica unas milesimas hacia
arriba o abajo. Este script repite EXACTAMENTE la validacion cruzada vigente (mismos
folds, mismos sinteticos, misma eleccion de C) y ademas registra, para cada
(modelo, tecnica) y para cada C de una grilla amplia FIJA, el F1 macro agrupado de
las 700 predicciones, sin elegir C en ningun dev. Sirve para responder:

  - la grilla de C de la validacion cruzada (0.01-10) se queda corta?
  - la conclusion "sin aumento gana" depende de como se eligio C?

No modifica ningun resultado vigente: la corrida base escribe en una carpeta
temporal que se descarta.

Entrada: las mismas que validacion_cruzada.py.
Salida:  evaluacion/diagnostico_c/resultados/sensibilidad_c.csv   (modelo, tecnica, C, f1_macro_agrupado)

Uso (desde la raiz del repositorio; tarda unos 10-15 minutos en CPU):
    python oe2_aumento_de_datos/evaluacion/diagnostico_c/sensibilidad_c.py
"""

from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # evaluacion/: validacion_cruzada*, _ruta_raiz
import _ruta_raiz  # noqa: F401  (deja importable el paquete `shiwilu`)

import validacion_cruzada as vc  # noqa: E402
from shiwilu.clasificacion import MODELOS, N_FOLDS, SEMILLA, cargar_corpus, cargar_folds  # noqa: E402
from shiwilu.rutas import DIAGNOSTICO_C_RESULTADOS  # noqa: E402

GRILLA_C = [0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 1000.0]
SALIDA = DIAGNOSTICO_C_RESULTADOS / "sensibilidad_c.csv"


def main() -> int:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)   # esperable con C >= 100

    llamadas: list[dict[float, np.ndarray]] = []
    ajustar_original = vc.ajustar_y_predecir

    def ajustar_y_registrar(X_core, y_core, X_dev, y_dev, X_pool, y_pool, X_test):
        resultado = ajustar_original(X_core, y_core, X_dev, y_dev, X_pool, y_pool, X_test)
        llamadas.append({
            c: LogisticRegression(C=c, max_iter=2000, random_state=SEMILLA).fit(X_pool, y_pool).predict(X_test)
            for c in GRILLA_C
        })
        return resultado

    vc.ajustar_y_predecir = ajustar_y_registrar
    with tempfile.TemporaryDirectory() as tmp:
        vc.EVALUACION_RESULTADOS = Path(tmp)
        sys.argv = ["validacion_cruzada.py"]
        vc.main()

    df = cargar_corpus()
    y = df["intencion"].to_numpy()
    folds = cargar_folds(df)
    orden_modelos = ["labse"] + [m for m in MODELOS if m != "labse"]   # el de validacion_cruzada.main
    filas = []
    for i_modelo, modelo in enumerate(orden_modelos):
        for i_tecnica, tecnica in enumerate(vc.TECNICAS):
            for c in GRILLA_C:
                pred = np.empty(len(y), dtype=object)
                for f in range(N_FOLDS):
                    pred[np.where(folds == f)[0]] = llamadas[(i_modelo * N_FOLDS + f) * len(vc.TECNICAS) + i_tecnica][c]
                filas.append({"modelo": modelo, "tecnica": tecnica, "C": c,
                              "f1_macro_agrupado": float(f1_score(y, pred, average="macro", zero_division=0))})

    R = pd.DataFrame(filas)
    R.to_csv(SALIDA, index=False, encoding="utf-8")
    for modelo in MODELOS:
        print(f"\n{modelo}: F1 macro agrupado con C fijo")
        tabla = R[R["modelo"] == modelo].pivot(index="tecnica", columns="C", values="f1_macro_agrupado")
        print(tabla.reindex(vc.TECNICAS).round(4).to_string())
    print(f"\nGuardado en {SALIDA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
