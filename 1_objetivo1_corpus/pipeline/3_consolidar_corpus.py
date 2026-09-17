"""
Etapa 3 — Consolidacion del corpus final (R3 del OE1).

Une las dos fuentes del corpus en un unico archivo CSV:
  1. Pares bilingue extraidos de las flashcards (Etapa 1)
  2. Oraciones generadas con la API y traducidas por el hablante nativo (Etapa 2)

Entradas: 1_objetivo1_corpus/intermedios/1_corpus_pares_anotados.xlsx
          1_objetivo1_corpus/intermedios/2_oraciones_traducidas.xlsx
Salida:   corpus/corpus_shiwilu_final.csv
Columnas: id, espanol, shiwilu, intencion, fuente

Esta es la unica etapa que escribe en corpus/, la frontera entre la Fase 1 y la
Fase 2. No requiere llamadas a la API.
"""

import pandas as pd

import _ruta_raiz  # noqa: F401  (deja importable el paquete `shiwilu`)
from shiwilu.rutas import (
    CORPUS_CSV as SALIDA_CSV,
    ORACIONES_TRAD,
    PARES_ANOTADOS,
    preparar_directorios,
)
from shiwilu.taxonomia import INTENCIONES

preparar_directorios()


def leer_pares_anotados() -> pd.DataFrame:
    """Lee los pares extraidos de las flashcards (ya bilingue)."""
    df = pd.read_excel(PARES_ANOTADOS, sheet_name="Pares anotados")
    df = df.rename(columns={
        "ID": "id",
        "Español": "espanol",
        "Shiwilu": "shiwilu",
        "Intención": "intencion",
    })
    df = df[["id", "espanol", "shiwilu", "intencion"]].copy()
    df["fuente"] = "flashcards"
    return df


def leer_oraciones_traducidas() -> pd.DataFrame:
    """Lee las oraciones generadas con API y traducidas por el hablante nativo."""
    df = pd.read_excel(ORACIONES_TRAD, sheet_name="Para traducir")
    df = df.rename(columns={
        "ID": "id",
        "Espanol": "espanol",
        "Intencion": "intencion",
        "Shiwilu (completar)": "shiwilu",
    })
    df = df[["id", "espanol", "shiwilu", "intencion"]].copy()
    df["fuente"] = "api_generada"
    return df


def validar(df: pd.DataFrame) -> list[str]:
    """Devuelve la lista de problemas encontrados en el corpus consolidado."""
    problemas = []

    vacias = df[df["shiwilu"].isna() | (df["shiwilu"].astype(str).str.strip() == "")]
    if len(vacias) > 0:
        problemas.append(f"{len(vacias)} oraciones sin traduccion al shiwilu")

    dup_id = df[df["id"].duplicated()]
    if len(dup_id) > 0:
        problemas.append(f"{len(dup_id)} IDs duplicados")

    dup_esp = df[df["espanol"].str.strip().str.lower().duplicated()]
    if len(dup_esp) > 0:
        problemas.append(f"{len(dup_esp)} oraciones en espanol duplicadas")

    fuera = set(df["intencion"].unique()) - set(INTENCIONES)
    if fuera:
        problemas.append(f"Intenciones no reconocidas: {sorted(fuera)}")

    return problemas


def main():
    if not PARES_ANOTADOS.exists():
        raise FileNotFoundError(f"[ERROR] No se encontro {PARES_ANOTADOS}")
    if not ORACIONES_TRAD.exists():
        raise FileNotFoundError(f"[ERROR] No se encontro {ORACIONES_TRAD}")

    print("[1/4] Leyendo pares de flashcards...")
    df_flash = leer_pares_anotados()
    print(f"      {len(df_flash)} pares bilingue")

    print("[2/4] Leyendo oraciones traducidas...")
    df_api = leer_oraciones_traducidas()
    print(f"      {len(df_api)} oraciones")

    print("[3/4] Consolidando...")
    df = pd.concat([df_flash, df_api], ignore_index=True)
    df["espanol"] = df["espanol"].astype(str).str.strip()
    df["shiwilu"] = df["shiwilu"].astype(str).str.strip()
    df = df.sort_values(["intencion", "id"]).reset_index(drop=True)

    print("[4/4] Validando...")
    problemas = validar(df)
    if problemas:
        print("      ADVERTENCIAS:")
        for pr in problemas:
            print(f"        - {pr}")
    else:
        print("      Sin problemas detectados")

    df.to_csv(SALIDA_CSV, index=False, encoding="utf-8")

    print()
    print(f"Corpus consolidado: {SALIDA_CSV}")
    print(f"Total de pares: {len(df)}")
    print()
    print("Distribucion por intencion:")
    dist = df.groupby(["intencion", "fuente"]).size().unstack(fill_value=0)
    for intencion in INTENCIONES:
        if intencion in dist.index:
            fila = dist.loc[intencion]
            flash = fila.get("flashcards", 0)
            api = fila.get("api_generada", 0)
            print(f"  {intencion:<8} flashcards={flash:>4}  api={api:>4}  total={flash + api:>4}")


if __name__ == "__main__":
    main()
