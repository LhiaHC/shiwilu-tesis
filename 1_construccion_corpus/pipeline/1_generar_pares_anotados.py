"""
Etapa 1 — Anotacion por reglas de los pares de flashcards (OE1).

Anota cada par bilingue de las flashcards con su intencion comunicativa y su
dominio semantico, y selecciona hasta 100 pares por intencion.

Entrada: 1_construccion_corpus/datos/flashcards2.csv
Salida:  1_construccion_corpus/intermedios/1_corpus_pares_anotados.xlsx

No requiere llamadas a la API.
"""

import pandas as pd
from openpyxl import Workbook

import _ruta_raiz  # noqa: F401  (deja importable el paquete `shiwilu`)
from shiwilu.anotacion import anotar_dominio, anotar_intencion
from shiwilu.excel import ancho, aplicar_fila, aplicar_header
from shiwilu.rutas import FLASHCARDS, PARES_ANOTADOS, preparar_directorios
from shiwilu.taxonomia import (
    COLOR_INT, DESC_INT, INTENCIONES, NORMALIZAR_INT, TIPO_SEARLE,
)

preparar_directorios()


VOCABULARIO_EXCLUIDO = ["mago", "cerveza"] #identificado tras revisión manual

def leer_flashcards() -> pd.DataFrame:
    df = pd.read_csv(FLASHCARDS, encoding="utf-8")
    patron_excluir = "|".join(VOCABULARIO_EXCLUIDO)
    validos = df[
        (df["Defectuosos?"] == False) &
        df["SHIWILU"].notna() &
        df["ESP"].notna() &
        ~df["ESP"].str.contains(patron_excluir, case=False, na=False)
    ][["CODIGO", "ESP", "SHIWILU"]].copy()
    validos.columns = ["ID", "Español", "Shiwilu"]
    print(f"  Flashcards válidos: {len(validos)}")
    return validos.reset_index(drop=True)


def anotar(df: pd.DataFrame) -> pd.DataFrame:
    resultados      = df["Español"].apply(anotar_intencion)
    df["Intención"] = resultados.apply(lambda x: NORMALIZAR_INT.get(x[0], x[0]))

    #dsp de revisión manual
    FORZAR_REQUEST = [
        "FL2_0028","FL2_0027","FL2_0607","FL2_0606","FL2_0605",
        "FL2_0604","FL2_0603","FL2_0601","FL2_0600","FL2_0599",
        "FL2_0598","FL2_0597","FL2_0596","FL2_0595","FL2_0626",
        "FL2_0624","FL2_0623","FL2_0619","FL2_0618","FL2_0617",
        "FL2_0613","FL2_0612","FL2_0611","FL2_0610","FL2_0609",
        "FL2_0608","FL2_0634","FL2_0633","FL2_0631","FL2_0628",
        "FL2_0627",
    ]
    FORZAR_REQUEST_PRG = [
        "FL2_1935",
        "FL2_1936",
    ]
    ELIMINAR_IDS = [
        "FL2_0555",
        "FL2_0554",
        "FL2_0064",
    ]

    df["Confianza"] = resultados.apply(lambda x: round(x[1], 2))

    df.loc[df["ID"].isin(FORZAR_REQUEST), "Intención"] = "REQUEST"
    df.loc[df["ID"].isin(FORZAR_REQUEST_PRG), "Intención"] = "REQUEST"
    df = df[~df["ID"].isin(ELIMINAR_IDS)].reset_index(drop=True)
    df["Dominio"]   = df["Español"].apply(anotar_dominio)
    return df


def seleccionar_100(df: pd.DataFrame) -> pd.DataFrame:
    return (df
            .sort_values("Confianza", ascending=False)
            .groupby("Intención", group_keys=False)
            .head(100)
            .reset_index(drop=True))


def crear_excel(df: pd.DataFrame): # por revisión manual
    FORZAR_REQUEST = [
        "FL2_0028","FL2_0027","FL2_0607","FL2_0606","FL2_0605",
        "FL2_0604","FL2_0603","FL2_0601","FL2_0600","FL2_0599",
        "FL2_0598","FL2_0597","FL2_0596","FL2_0595","FL2_0626",
        "FL2_0624","FL2_0623","FL2_0619","FL2_0618","FL2_0617",
        "FL2_0613","FL2_0612","FL2_0611","FL2_0610","FL2_0609",
        "FL2_0608","FL2_0634","FL2_0633","FL2_0631","FL2_0628",
        "FL2_0627",
    ]
    ELIMINAR_IDS = ["FL2_0555", "FL2_0554"]

    df.loc[df["ID"].isin(FORZAR_REQUEST), "Intención"] = "REQUEST"
    df = df[~df["ID"].isin(ELIMINAR_IDS)].reset_index(drop=True)

    wb = Workbook()

    ws = wb.active
    ws.title = "Pares anotados"
    ws.append(["#", "ID", "Español", "Shiwilu",
                "Intención", "Tipo Searle", "Confianza"])
    aplicar_header(ws, "1565C0")

    for i, (_, row) in enumerate(df.iterrows(), 1):
        intent = str(row["Intención"]).strip()
        ws.append([i, row["ID"], row["Español"], row["Shiwilu"],
                   intent, TIPO_SEARLE.get(intent, ""), row["Confianza"]])
        aplicar_fila(ws, COLOR_INT.get(intent, "FFFFFF"))

    for col, w in [("A",5),("B",14),("C",48),("D",48),
                   ("E",12),("F",13),("G",11)]:
        ancho(ws, col, w)
    ws.freeze_panes = "A2"

    ws2 = wb.create_sheet("Distribución por intención")
    ws2.append(["Intención", "Descripción", "Tipo Searle",
                "Pares en corpus", "Déficit para 100"])
    aplicar_header(ws2, "1565C0")

    dist = df["Intención"].value_counts().to_dict()
    for intent in INTENCIONES:
        sel     = dist.get(intent, 0)
        deficit = max(0, 100 - sel)
        ws2.append([intent, DESC_INT[intent], TIPO_SEARLE[intent],
                    sel, deficit if deficit > 0 else "—"])
        aplicar_fila(ws2, COLOR_INT.get(intent, "FFFFFF"))

    for col, w in [("A",12),("B",35),("C",13),("D",16),("E",18)]:
        ancho(ws2, col, w)

    wb.save(PARES_ANOTADOS)
    print(f"  [OK] {PARES_ANOTADOS.name}  ({len(df)} pares)")


if __name__ == "__main__":
    print("Generando 1_corpus_pares_anotados.xlsx ...")
    df = leer_flashcards()
    df = anotar(df)
    df = seleccionar_100(df)
    crear_excel(df)
