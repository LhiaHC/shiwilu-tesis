"""
Junta las predicciones de varias corridas (una tecnica por archivo) en un solo CSV con el formato de la
evaluacion vigente, para pasarlo a `comparacion_pareada.py` y `curvas_roc.py` (--entrada / --etiqueta).

Pensado para `validacion_cruzada_cv_interna.py`, que corre una tecnica por vez. Calcula tambien el resumen
(F1 macro agrupado, IC95%).

Uso (desde la raiz del repositorio):
    python oe2_aumento_de_datos/evaluacion/unir_predicciones.py --etiqueta cv_interna \
        --entradas validacion_cruzada_predicciones_sin_puntuacion_cv_interna_{sin_aumento,mixup,retrotraduccion,generate_then_refine}.csv
Salida: evaluacion/resultados/cv_interna/validacion_cruzada_{predicciones,resumen}_sin_puntuacion_<etiqueta>.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import _ruta_raiz  # noqa: F401  (deja importable el paquete `shiwilu`)

import validacion_cruzada as vc  # noqa: E402
from shiwilu.rutas import CV_INTERNA_RESULTADOS  # noqa: E402



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entradas", nargs="+", required=True, help="CSV de predicciones (nombres de archivo en evaluacion/resultados/cv_interna/).")
    ap.add_argument("--etiqueta", required=True, help="Se usa en el nombre de los CSV de salida.")
    ap.add_argument("--carpeta", type=Path, default=CV_INTERNA_RESULTADOS,
                    help="Carpeta de los CSV de entrada y de salida (por defecto, evaluacion/resultados/cv_interna/; p. ej. cv_interna_sin_DES/).")
    args = ap.parse_args()

    P = pd.concat([pd.read_csv(args.carpeta / e) for e in args.entradas], ignore_index=True)
    clave = ["modelo", "tecnica", "fold", "pos"]
    assert not P.duplicated(clave).any(), "hay predicciones repetidas para el mismo (modelo, tecnica, fold, pos)"
    P.to_csv(args.carpeta / f"validacion_cruzada_predicciones_sin_puntuacion_{args.etiqueta}.csv", index=False, encoding="utf-8")
    R = vc.resumir_predicciones(P)
    R.to_csv(args.carpeta / f"validacion_cruzada_resumen_sin_puntuacion_{args.etiqueta}.csv", index=False, encoding="utf-8")
    print(R.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
