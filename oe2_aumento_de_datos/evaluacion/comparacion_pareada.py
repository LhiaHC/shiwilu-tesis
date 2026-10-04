"""
Comparaciones pareadas sobre las predicciones de la validacion cruzada.

Dos configuraciones se evaluaron sobre LAS MISMAS 700 oraciones, asi que su
diferencia de F1 se puede estimar mucho mas finamente que comparando dos
intervalos por separado: se remuestrean las 700 oraciones (con reemplazo) UNA
vez por repeticion y se calcula el F1 de ambas configuraciones sobre esa misma
muestra. Si el intervalo de la diferencia (A - B) no incluye 0, la diferencia
es estadisticamente distinguible del azar de muestreo.

Compara (a) cada tecnica de aumento contra "sin aumento" dentro de cada modelo,
y (b) los modelos entre si en "sin aumento".

Entrada: evaluacion/resultados/validacion_cruzada_predicciones_sin_puntuacion.csv
Salida:  evaluacion/resultados/comparacion_pareada_sin_puntuacion.csv

Con --condicion original usa las predicciones del texto crudo (historico:
historico/atajos_texto_crudo/), que ya no forman parte de los resultados vigentes.
Con --entrada [--etiqueta X] compara las predicciones de otro CSV (p. ej. el de una corrida
en linea de los cuadernos de Colab) y guarda `comparacion_pareada_sin_puntuacion_X.csv`.

Uso (desde la raiz del repositorio):
    python oe2_aumento_de_datos/evaluacion/comparacion_pareada.py
    python oe2_aumento_de_datos/evaluacion/comparacion_pareada.py --condicion original
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import _ruta_raiz  # noqa: F401  (deja importable el paquete `shiwilu`)

from shiwilu.rutas import ATAJOS_TEXTO_CRUDO, EVALUACION_RESULTADOS  # noqa: E402

REPETICIONES = 2000
SEMILLA = 42


def f1m(y, p) -> float:
    return float(f1_score(y, p, average="macro", zero_division=0))


def diferencia_pareada(P, a, b, rng):
    """a, b = (modelo, tecnica). Devuelve (dif, ic_bajo, ic_alto) de F1(a) - F1(b)."""
    A = P[(P["modelo"] == a[0]) & (P["tecnica"] == a[1])].sort_values("pos")
    B = P[(P["modelo"] == b[0]) & (P["tecnica"] == b[1])].sort_values("pos")
    assert (A["pos"].to_numpy() == B["pos"].to_numpy()).all()
    y, pa, pb = A["real"].to_numpy(), A["prediccion"].to_numpy(), B["prediccion"].to_numpy()
    n = len(y)
    difs = []
    for _ in range(REPETICIONES):
        i = rng.integers(0, n, n)
        difs.append(f1m(y[i], pa[i]) - f1m(y[i], pb[i]))
    return f1m(y, pa) - f1m(y, pb), float(np.percentile(difs, 2.5)), float(np.percentile(difs, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condicion", choices=["sin_puntuacion", "original"], default="sin_puntuacion",
                     help="Condicion de texto de las predicciones a comparar (por defecto, la vigente).")
    ap.add_argument("--entrada", type=Path, default=None,
                     help="CSV de predicciones a comparar (por defecto, el vigente de la condicion).")
    ap.add_argument("--etiqueta", default="", help="Se agrega al nombre del CSV de salida.")
    ap.add_argument("--carpeta", type=Path, default=None,
                     help="Carpeta de salida (por defecto, evaluacion/resultados/; para la CV interna, "
                          "evaluacion/resultados/cv_interna/).")
    args = ap.parse_args()
    sufijo = "" if args.condicion == "original" else f"_{args.condicion}"
    carpeta = args.carpeta or (EVALUACION_RESULTADOS if args.condicion == "sin_puntuacion" else ATAJOS_TEXTO_CRUDO)
    P = pd.read_csv(args.entrada or carpeta / f"validacion_cruzada_predicciones{sufijo}.csv")
    sufijo += f"_{args.etiqueta}" if args.etiqueta else ""
    rng = np.random.default_rng(SEMILLA)
    filas = []

    for modelo in ["labse", "mbert", "xlmr"]:
        for tecnica in ["mixup", "retrotraduccion", "generate_then_refine"]:
            if not ((P["modelo"] == modelo) & (P["tecnica"] == tecnica)).any():
                continue   # corrida parcial (p. ej. la CV interna sin Generate-then-Refine)
            d, lo, hi = diferencia_pareada(P, (modelo, tecnica), (modelo, "sin_aumento"), rng)
            filas.append({"comparacion": f"{modelo}: {tecnica} - sin_aumento", "diferencia_f1": d,
                          "ic95_bajo": lo, "ic95_alto": hi, "distinguible_de_cero": not (lo <= 0 <= hi)})
    for a, b in [("xlmr", "mbert"), ("xlmr", "labse"), ("mbert", "labse")]:
        d, lo, hi = diferencia_pareada(P, (a, "sin_aumento"), (b, "sin_aumento"), rng)
        filas.append({"comparacion": f"sin_aumento: {a} - {b}", "diferencia_f1": d,
                      "ic95_bajo": lo, "ic95_alto": hi, "distinguible_de_cero": not (lo <= 0 <= hi)})

    R = pd.DataFrame(filas)
    R.to_csv(carpeta / f"comparacion_pareada{sufijo}.csv", index=False, encoding="utf-8")
    print(R.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
