"""
Corre los 3 baselines de clasificacion (LaBSE, mBERT, XLM-R) en una sola
ejecucion y muestra una tabla comparativa al final.

Cada uno guarda sus resultados por separado en
2_baselines/resultados/<modelo>/ (ver baseline.py).

Uso:
    python 2_baselines/correr_todos.py
"""

from __future__ import annotations

import _ruta_raiz  # noqa: F401  (deja importable el paquete `shiwilu`)
from baseline import correr_baseline
from comun import MODELOS
from shiwilu.rutas import preparar_directorios


def main() -> int:
    preparar_directorios()

    resumen = {}
    for nombre_modelo in MODELOS:
        resumen[nombre_modelo] = correr_baseline(nombre_modelo)

    print("\n=== Comparacion de los 3 baselines (sin aumento de datos) ===")
    print(f"{'Modelo':<8}{'F1 macro':>12}{'F1 ponderado':>16}{'Exactitud':>12}")
    for nombre_modelo, m in resumen.items():
        print(f"{nombre_modelo:<8}{m['f1_macro']:>12.4f}"
              f"{m['f1_ponderado']:>16.4f}{m['exactitud']:>12.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
