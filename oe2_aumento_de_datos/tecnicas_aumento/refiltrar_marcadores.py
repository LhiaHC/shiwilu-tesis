"""
Reaplica el filtro de marcador de Generate-then-Refine a candidatas YA generadas, sin llamar a la API.

Los archivos de `generate_then_refine.py` guardan TODAS las candidatas de Claude (aprobadas o no) con su
`estado_filtro`. Si cambia la lista de marcadores (`analisis_intrinseco/marcadores_fuentes.csv`) no hace
falta volver a generar: este script recalcula, para cada candidata, si cumple el filtro de marcador con la
lista vigente, conserva el resultado de los otros dos filtros (idioma y semantico, tal como se registro en
`detalle_filtro`) y escribe copias con el `estado_filtro` actualizado. Esas copias se pueden evaluar con
`evaluacion/validacion_cruzada_en_linea.py --tecnica generate_then_refine --cache <salida>`, que reutiliza
los archivos en vez de generar.

Entrada: carpeta con fold<N>_{core,pool}.csv (por defecto, salidas/en_linea/generate_then_refine)
Salida:  carpeta con los mismos archivos refiltrados (por defecto, salidas/en_linea_marcadores_fuentes/generate_then_refine)

Uso (desde la raiz del repositorio):
    python oe2_aumento_de_datos/tecnicas_aumento/refiltrar_marcadores.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import _ruta_raiz  # noqa: F401  (deja importable el paquete `shiwilu`)

import generate_then_refine as gtr  # noqa: E402
from shiwilu.rutas import AUMENTO_SALIDA  # noqa: E402
from shiwilu.taxonomia import INTENCIONES  # noqa: E402


def refiltrar(df: pd.DataFrame, marcadores: dict[str, list[dict]]) -> pd.DataFrame:
    df = df.copy()
    pasa_marcador = [gtr.filtro_marcador(t, marcadores.get(c, [])) for t, c in zip(df["shiwilu"], df["intencion"])]
    otros = df["detalle_filtro"].fillna("").map(
        lambda s: [x for x in str(s).split(";") if x and x != "marcador"]
    )
    detalle = [(["marcador"] if not ok else []) + o for ok, o in zip(pasa_marcador, otros)]
    df["detalle_filtro"] = [";".join(d) for d in detalle]
    df["estado_filtro"] = ["aprobado" if not d else "revisar_hablante_nativo" for d in detalle]
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", type=Path, default=AUMENTO_SALIDA / "en_linea" / "generate_then_refine")
    ap.add_argument("--salida", type=Path, default=AUMENTO_SALIDA / "en_linea_marcadores_fuentes" / "generate_then_refine")
    args = ap.parse_args()

    marcadores = gtr.cargar_marcadores_validados()
    print("Marcadores vigentes:", {c: [m["forma"] for m in v] for c, v in marcadores.items()})
    args.salida.mkdir(parents=True, exist_ok=True)
    filas = []
    for ruta in sorted(args.entrada.glob("fold*_*.csv")):
        antes = pd.read_csv(ruta)
        despues = refiltrar(antes, marcadores)
        despues.to_csv(args.salida / ruta.name, index=False, encoding="utf-8")
        for cat in INTENCIONES:
            filas.append({"archivo": ruta.name, "categoria": cat,
                          "candidatas": int((antes["intencion"] == cat).sum()),
                          "aprobadas_antes": int(((antes["intencion"] == cat) & (antes["estado_filtro"] == "aprobado")).sum()),
                          "aprobadas_despues": int(((despues["intencion"] == cat) & (despues["estado_filtro"] == "aprobado")).sum())})
    R = pd.DataFrame(filas)
    resumen = R.groupby("categoria")[["candidatas", "aprobadas_antes", "aprobadas_despues"]].sum().reindex(INTENCIONES)
    resumen["tasa_antes"] = (resumen["aprobadas_antes"] / resumen["candidatas"]).round(2)
    resumen["tasa_despues"] = (resumen["aprobadas_despues"] / resumen["candidatas"]).round(2)
    print("\nCandidatas aprobadas por categoria (suma de todos los archivos):")
    print(resumen.to_string())
    print(f"\nTotal aprobadas: {int(resumen['aprobadas_antes'].sum())} -> {int(resumen['aprobadas_despues'].sum())}")
    print(f"Archivos refiltrados en {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
