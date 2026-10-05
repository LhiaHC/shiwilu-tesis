"""
Auditoria de las etiquetas de DES ("descripciones de estados o eventos") y su solape con REQUEST.

Motivo: en los experimentos DES es la categoria con menor F1 en los tres modelos (0.38-0.55) y 24 de sus 100 oraciones las
clasifican mal los tres modelos a la vez. La metodologia del corpus (docs/metodologia_construccion_corpus.md) indica que las
flashcards se anotaron con reglas de expresiones regulares en cascada y que DES era la clase RESIDUAL ("si ningun patron
coincide"); este script revisa que tan homogenea es esa clase y donde choca con REQUEST.

Usa las predicciones del protocolo B de "sin aumento" en las DOS condiciones de texto (sin puntuacion y con `¿?`):
  evaluacion/resultados/cv_interna/...sin_aumento.csv  y  evaluacion/resultados/cv_interna_con_interrogacion/...sin_aumento.csv
No modifica el corpus ni ningun resultado: solo lista casos para revisar con la autora / un hablante.

Salida (analisis_intrinseco/resultados/auditoria_des/):
  des_oraciones_con_errores.csv      las 100 oraciones DES con el numero de errores (de 6 corridas) y lo que predijo cada una
  des_vs_request_indicios.csv        cuantas oraciones de cada categoria contienen cada indicio lexical (modales, volitivos, ...)
  des_vs_request_pares_similares.csv pares (DES, REQUEST) mas parecidos en español (LaBSE)
  request_predichas_como_des.csv     oraciones REQUEST que los modelos mandan a DES

Uso (desde la raiz del repositorio):
    python oe2_aumento_de_datos/analisis_intrinseco/auditoria_etiquetas_des.py
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # raiz del repo: deja importable `shiwilu`

from shiwilu.clasificacion import cargar_corpus, extraer_embeddings
from shiwilu.rutas import ANALISIS_INTRINSECO, EVALUACION_RESULTADOS

SALIDA = ANALISIS_INTRINSECO / "resultados" / "auditoria_des"
MODELOS = ["labse", "mbert", "xlmr"]
CONDICIONES = {
    "sin": EVALUACION_RESULTADOS / "cv_interna" / "validacion_cruzada_predicciones_sin_puntuacion_cv_interna_sin_aumento.csv",
    "con": EVALUACION_RESULTADOS / "cv_interna_con_interrogacion" / "validacion_cruzada_predicciones_sin_puntuacion_cv_interna_sin_aumento.csv",
}

# Indicios lexicales (en español) de otras intenciones; sirven para ver si las mismas formas aparecen etiquetadas como DES y como REQUEST
INDICIOS = {
    "modal_permiso_capacidad": r"\b(puedo|puedes|puede|podemos|pueden|podr[eé]|podr[aá]s|podr[aá]|podr[ií]a\w*)\b",
    "deber_obligacion": r"\b(debo|debes|debe|debemos|deber[ií]as?|tengo que|tienes que|tiene que|hay que)\b",
    "volitivo_necesidad": r"\b(quiero|quieres|quiere|queremos|necesito|necesitas|necesita|necesitamos|deseo|ojal[aá]|espero)\b",
    "emocion_valoracion": r"\b(feliz|triste|enojad\w*|contento|contenta|alegr\w*|odia\w*|odio|ama\w*|gust\w*|miedo|asust\w*|preocup\w*|cansad\w*|orgullos\w*|celos\w*|desesper\w*)\b",
    "imperativo_con_clitico": r"\b\w+(ame|anos|ale|alo|alos|elo|elos|ale|enos|adme|edme)\b",
    "por_favor_cortesia": r"\b(por favor|favor|gracias)\b",
    "primera_persona_accion_inmediata": r"\b(me voy|nos vamos|voy a|vamos a|salgo|salí)\b",
}


def main() -> int:
    SALIDA.mkdir(parents=True, exist_ok=True)
    corpus = cargar_corpus()
    corpus["pos"] = np.arange(len(corpus))

    # --- predicciones de las 6 corridas de "sin aumento" (3 modelos x 2 condiciones)
    cols = {}
    for cond, ruta in CONDICIONES.items():
        P = pd.read_csv(ruta)
        P = P[P["tecnica"] == "sin_aumento"]
        for m in MODELOS:
            g = P[P["modelo"] == m].set_index("pos")["prediccion"]
            cols[f"{m}_{cond}"] = g
    pred = pd.DataFrame(cols).sort_index()
    pred.index.name = "pos"
    base = corpus.set_index("pos").join(pred)
    corridas = list(cols)

    # --- A. oraciones DES con errores
    des = base[base["intencion"] == "DES"].copy()
    des["errores_de_6"] = sum((des[c] != "DES").astype(int) for c in corridas)
    des["prediccion_mas_frecuente"] = des[corridas].apply(
        lambda f: pd.Series([x for x in f if x != "DES"]).mode().iloc[0] if (f != "DES").any() else "", axis=1)
    des["a_request_en"] = sum((des[c] == "REQUEST").astype(int) for c in corridas)
    for nombre, rx in INDICIOS.items():
        des[nombre] = des["espanol"].str.lower().str.contains(rx, regex=True)
    des["indicios"] = des[list(INDICIOS)].apply(lambda f: ",".join(k for k, v in f.items() if v), axis=1)
    cols_out = ["id", "espanol", "shiwilu", "fuente", "errores_de_6", "prediccion_mas_frecuente", "a_request_en", "indicios"] + corridas
    des = des.sort_values(["errores_de_6", "a_request_en"], ascending=False)
    des[cols_out].to_csv(SALIDA / "des_oraciones_con_errores.csv", index=False, encoding="utf-8")

    # --- B1. indicios lexicales por categoria
    filas = []
    texto = corpus["espanol"].str.lower()
    for nombre, rx in INDICIOS.items():
        tiene = texto.str.contains(rx, regex=True)
        fila = {"indicio": nombre}
        fila.update({cat: int((tiene & (corpus["intencion"] == cat)).sum()) for cat in sorted(corpus["intencion"].unique())})
        filas.append(fila)
    pd.DataFrame(filas).to_csv(SALIDA / "des_vs_request_indicios.csv", index=False, encoding="utf-8")

    # --- B2. pares DES-REQUEST mas parecidos (LaBSE sobre el texto en español)
    E = extraer_embeddings(corpus["espanol"].astype(str).tolist(), "labse")
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    S = E @ E.T
    i_des = np.where(corpus["intencion"] == "DES")[0]
    i_req = np.where(corpus["intencion"] == "REQUEST")[0]
    pares = []
    for i in i_des:
        j = i_req[np.argmax(S[i, i_req])]
        pares.append({"id_des": corpus.loc[i, "id"], "espanol_des": corpus.loc[i, "espanol"], "shiwilu_des": corpus.loc[i, "shiwilu"],
                      "id_request": corpus.loc[j, "id"], "espanol_request": corpus.loc[j, "espanol"], "shiwilu_request": corpus.loc[j, "shiwilu"],
                      "similitud": float(S[i, j])})
    pares = pd.DataFrame(pares).sort_values("similitud", ascending=False)
    pares.to_csv(SALIDA / "des_vs_request_pares_similares.csv", index=False, encoding="utf-8")

    # --- B3. REQUEST que los modelos mandan a DES
    req = base[base["intencion"] == "REQUEST"].copy()
    req["a_des_en"] = sum((req[c] == "DES").astype(int) for c in corridas)
    req = req[req["a_des_en"] >= 3].sort_values("a_des_en", ascending=False)
    req[["id", "espanol", "shiwilu", "fuente", "a_des_en"] + corridas].to_csv(SALIDA / "request_predichas_como_des.csv", index=False, encoding="utf-8")

    # --- resumen en pantalla
    print("DES con errores en cuantas de las 6 corridas (3 modelos x sin/con ¿?):")
    print(des["errores_de_6"].value_counts().sort_index().to_string())
    print("\nPrediccion mas frecuente de las DES con >=4 errores:")
    print(des[des["errores_de_6"] >= 4]["prediccion_mas_frecuente"].value_counts().to_string())
    print("\nIndicios lexicales por categoria (n de oraciones):")
    print(pd.DataFrame(filas).to_string(index=False))
    print(f"\npares DES-REQUEST con similitud >= 0.80: {(pares['similitud'] >= 0.80).sum()}; >= 0.70: {(pares['similitud'] >= 0.70).sum()}")
    print(f"REQUEST mandadas a DES por >=3 de 6 corridas: {len(req)}")
    print("\nGuardado en", SALIDA)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
