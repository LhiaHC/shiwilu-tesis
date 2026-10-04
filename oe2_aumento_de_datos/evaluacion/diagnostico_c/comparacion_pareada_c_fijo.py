"""
Comparacion pareada de cada tecnica de aumento contra "sin aumento" con el MISMO C fijo.

`comparacion_pareada.py` compara las predicciones de la validacion cruzada, donde cada configuracion
elige su C en un dev interno de ~90 oraciones (una eleccion ruidosa). Aqui se fija C (el mismo para
las dos configuraciones comparadas, sin elegirlo en ningun dev) y se estima la diferencia de F1 macro
con el mismo bootstrap pareado (se remuestrean las 700 oraciones una vez por repeticion y se evalua
ambas configuraciones sobre esa muestra).

Configuraciones: sin aumento y Mixup (evaluacion vigente), Generate-then-Refine (20, 40, 80, 120 por
categoria y etapa) y Retrotraduccion (1 parafrasis), estas ultimas desde la cache de la evaluacion en
linea (para GtR, refiltrada con `tecnicas_aumento/refiltrar_marcadores.py`).

Entrada: --cache = carpeta con generate_then_refine/ y retrotraduccion/.
Salida:  evaluacion/diagnostico_c/resultados/comparacion_pareada_c_fijo.csv
         evaluacion/diagnostico_c/resultados/predicciones_c_fijo.csv   (configuracion, modelo, C, pos, real, prediccion)

Uso (desde la raiz del repositorio; unos 30-40 minutos en CPU):
    python oe2_aumento_de_datos/evaluacion/diagnostico_c/comparacion_pareada_c_fijo.py --cache <carpeta>
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # evaluacion/: validacion_cruzada*, _ruta_raiz
import _ruta_raiz  # noqa: F401  (deja importable el paquete `shiwilu`)

import validacion_cruzada as vc  # noqa: E402
from sensibilidad_c_en_linea import CONFIGURACIONES, predicciones_por_c  # noqa: E402
from shiwilu.clasificacion import MODELOS, N_FOLDS, SEMILLA, cargar_corpus, cargar_folds  # noqa: E402
from shiwilu.rutas import AUMENTO_SALIDA, DIAGNOSTICO_C_RESULTADOS  # noqa: E402

C_COMPARADOS = [0.1, 1.0, 10.0, 100.0]
REPETICIONES = 2000


def predicciones_vigentes(y, folds, grilla) -> dict[str, dict[str, dict[float, np.ndarray]]]:
    """{tecnica: {modelo: {C: predicciones}}} de la evaluacion vigente (aqui, sin_aumento y mixup)."""
    llamadas: list[dict[float, np.ndarray]] = []
    ajustar_original = vc.ajustar_y_predecir

    def ajustar_y_registrar(X_core, y_core, X_dev, y_dev, X_pool, y_pool, X_test):
        resultado = ajustar_original(X_core, y_core, X_dev, y_dev, X_pool, y_pool, X_test)
        llamadas.append({
            c: LogisticRegression(C=c, max_iter=2000, random_state=SEMILLA).fit(X_pool, y_pool).predict(X_test)
            for c in grilla
        })
        return resultado

    vc.ajustar_y_predecir = ajustar_y_registrar
    try:
        with tempfile.TemporaryDirectory() as tmp:
            vc.EVALUACION_RESULTADOS = Path(tmp)
            sys.argv = ["validacion_cruzada.py"]
            vc.main()
    finally:
        vc.ajustar_y_predecir = ajustar_original

    orden_modelos = ["labse"] + [m for m in MODELOS if m != "labse"]   # el de validacion_cruzada.main
    salida: dict = {}
    for tecnica in ("sin_aumento", "mixup"):
        i_tecnica = vc.TECNICAS.index(tecnica)
        salida[tecnica] = {}
        for i_modelo, modelo in enumerate(orden_modelos):
            salida[tecnica][modelo] = {}
            for c in grilla:
                pred = np.empty(len(y), dtype=object)
                for f in range(N_FOLDS):
                    pred[np.where(folds == f)[0]] = llamadas[(i_modelo * N_FOLDS + f) * len(vc.TECNICAS) + i_tecnica][c]
                salida[tecnica][modelo][c] = pred
    return salida


def f1_macro_rapido(y_int: np.ndarray, p_int: np.ndarray, k: int) -> float:
    """F1 macro (misma definicion que sklearn con zero_division=0) sobre etiquetas enteras 0..k-1."""
    conf = np.bincount(y_int * k + p_int, minlength=k * k).reshape(k, k)
    tp = np.diag(conf).astype(float)
    pred, real = conf.sum(axis=0), conf.sum(axis=1)
    denom = pred + real
    return float(np.mean(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)))


def diferencia_pareada(y_int, pa, pb, k, rng):
    n = len(y_int)
    indices = rng.integers(0, n, (REPETICIONES, n))
    difs = np.array([f1_macro_rapido(y_int[i], pa[i], k) - f1_macro_rapido(y_int[i], pb[i], k) for i in indices])
    return f1_macro_rapido(y_int, pa, k) - f1_macro_rapido(y_int, pb, k), \
        float(np.percentile(difs, 2.5)), float(np.percentile(difs, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=AUMENTO_SALIDA / "en_linea_marcadores_fuentes")
    ap.add_argument("--c", nargs="+", type=float, default=C_COMPARADOS, help="Valores de C fijos a comparar.")
    ap.add_argument("--etiqueta", default="", help="Se agrega al nombre de los CSV de salida (p. ej. _fina).")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)   # esperable con C >= 100

    df = cargar_corpus()
    y = df["intencion"].to_numpy()
    folds = cargar_folds(df)
    orden_modelos = list(MODELOS)
    grilla = sorted(set(args.c))
    sufijo = args.etiqueta

    print("######## evaluacion vigente (sin aumento y Mixup) ########")
    pred: dict[str, dict[str, dict[float, np.ndarray]]] = predicciones_vigentes(y, folds, grilla)
    for nombre, argumentos in CONFIGURACIONES:
        print(f"\n######## {nombre} ########")
        pred[nombre] = predicciones_por_c(argumentos, args.cache, y, folds, orden_modelos, grilla)

    filas_pred = [{"configuracion": cfg, "modelo": m, "C": c, "pos": i, "real": y[i], "prediccion": p[i]}
                  for cfg, por_modelo in pred.items() for m, por_c in por_modelo.items()
                  for c, p in por_c.items() for i in range(len(y))]
    pd.DataFrame(filas_pred).to_csv(DIAGNOSTICO_C_RESULTADOS / f"predicciones_c_fijo{sufijo}.csv", index=False, encoding="utf-8")

    clases = sorted(set(y))
    codigo = {c: i for i, c in enumerate(clases)}
    y_int = np.array([codigo[v] for v in y])
    rng = np.random.default_rng(SEMILLA)
    filas = []
    for modelo in orden_modelos:
        for c in grilla:
            base = np.array([codigo[v] for v in pred["sin_aumento"][modelo][c]])
            for cfg in [n for n in pred if n != "sin_aumento"]:
                alt = np.array([codigo[v] for v in pred[cfg][modelo][c]])
                d, lo, hi = diferencia_pareada(y_int, alt, base, len(clases), rng)
                filas.append({"modelo": modelo, "C": c, "configuracion": cfg,
                              "f1_sin_aumento": f1_macro_rapido(y_int, base, len(clases)),
                              "f1_configuracion": f1_macro_rapido(y_int, alt, len(clases)),
                              "diferencia_f1": d, "ic95_bajo": lo, "ic95_alto": hi,
                              "distinguible_de_cero": not (lo <= 0 <= hi)})
    R = pd.DataFrame(filas)
    R.to_csv(DIAGNOSTICO_C_RESULTADOS / f"comparacion_pareada_c_fijo{sufijo}.csv", index=False, encoding="utf-8")
    for modelo in orden_modelos:
        print(f"\n{modelo}: diferencia de F1 frente a sin aumento con el MISMO C  [* = IC95% no incluye 0]")
        g = R[R["modelo"] == modelo].copy()
        g["txt"] = [f"{d:+.3f}{'*' if s else ' '}" for d, s in zip(g["diferencia_f1"], g["distinguible_de_cero"])]
        print(g.pivot(index="configuracion", columns="C", values="txt").reindex([n for n in pred if n != "sin_aumento"]).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
