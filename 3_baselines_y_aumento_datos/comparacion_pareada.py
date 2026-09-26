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

Entrada: 3_baselines_y_aumento_datos/validacion_cruzada_predicciones.csv
Salida:  3_baselines_y_aumento_datos/comparacion_pareada.csv

Con --sin-puntuacion usa las predicciones de la ablacion sin signos de puntuacion.

Uso:
    python 3_baselines_y_aumento_datos/comparacion_pareada.py
    python 3_baselines_y_aumento_datos/comparacion_pareada.py --sin-puntuacion
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

DIR = Path(__file__).resolve().parent
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
    sufijo = "_sin_puntuacion" if "--sin-puntuacion" in sys.argv else ""
    P = pd.read_csv(DIR / f"validacion_cruzada_predicciones{sufijo}.csv")
    rng = np.random.default_rng(SEMILLA)
    filas = []

    for modelo in ["labse", "mbert", "xlmr"]:
        for tecnica in ["mixup", "retrotraduccion", "generate_then_refine"]:
            d, lo, hi = diferencia_pareada(P, (modelo, tecnica), (modelo, "sin_aumento"), rng)
            filas.append({"comparacion": f"{modelo}: {tecnica} - sin_aumento", "diferencia_f1": d,
                          "ic95_bajo": lo, "ic95_alto": hi, "distinguible_de_cero": not (lo <= 0 <= hi)})
    for a, b in [("xlmr", "mbert"), ("xlmr", "labse"), ("mbert", "labse")]:
        d, lo, hi = diferencia_pareada(P, (a, "sin_aumento"), (b, "sin_aumento"), rng)
        filas.append({"comparacion": f"sin_aumento: {a} - {b}", "diferencia_f1": d,
                      "ic95_bajo": lo, "ic95_alto": hi, "distinguible_de_cero": not (lo <= 0 <= hi)})
    d, lo, hi = diferencia_pareada(P, ("xlmr", "generate_then_refine"), ("xlmr", "sin_aumento"), rng)

    R = pd.DataFrame(filas)
    R.to_csv(DIR / f"comparacion_pareada{sufijo}.csv", index=False, encoding="utf-8")
    print(R.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
