"""
Consolida los resultados de todos los experimentos (baselines + tecnicas de
aumento) en una sola tabla comparativa. Recorre los `metricas.json` que cada
script ya escribio y no vuelve a entrenar nada.

Uso:
    python 3_baselines_y_aumento_datos/resumen_experimentos.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
MODELOS = ["labse", "mbert", "xlmr"]


def recolectar() -> pd.DataFrame:
    patrones = [
        RAIZ / "2_baselines" / "resultados" / "*" / "metricas.json",
        RAIZ / "3_baselines_y_aumento_datos" / "tecnicas_aumento" / "resultados" / "*" / "*" / "metricas.json",
    ]
    filas = []
    for patron in patrones:
        for ruta in Path(patron.anchor).glob(str(patron.relative_to(patron.anchor))):
            with ruta.open(encoding="utf-8") as f:
                m = json.load(f)
            modelo = next((mo for mo in MODELOS if mo in ruta.parts), "?")
            tecnica = m.get("tecnica_aumento") or "sin_aumento"
            filas.append({
                "modelo": modelo,
                "tecnica": tecnica,
                "f1_macro": m["f1_macro"],
                "f1_ponderado": m["f1_ponderado"],
                "exactitud": m["exactitud"],
            })
    return pd.DataFrame(filas)


def main() -> int:
    df = recolectar()
    if df.empty:
        raise SystemExit("No se encontraron metricas.json. Corre primero los baselines y las tecnicas.")

    tabla = df.pivot(index="modelo", columns="tecnica", values="f1_macro").reindex(MODELOS)
    if "sin_aumento" in tabla.columns:
        orden = ["sin_aumento"] + [c for c in tabla.columns if c != "sin_aumento"]
        tabla = tabla[orden]

    print("=== F1 macro por modelo x tecnica ===")
    print(tabla.round(4).to_string())

    salida = Path(__file__).resolve().parent / "resumen_experimentos.csv"
    df.sort_values(["modelo", "tecnica"]).to_csv(salida, index=False, encoding="utf-8")
    print(f"\nDetalle completo (f1_macro, f1_ponderado, exactitud) -> {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
