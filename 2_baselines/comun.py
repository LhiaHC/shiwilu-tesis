"""
Funciones compartidas por los baselines de clasificacion (congelado + Regresion
Logistica) y, mas adelante, por los experimentos con tecnicas de aumento.

Cada "baseline" es una combinacion (modelo de embeddings, sin ajuste fino) +
(Regresion Logistica). Todos comparten el mismo corpus, la misma division
train/dev/test y el mismo procedimiento de ajuste/evaluacion — lo unico que
cambia es como se extraen los embeddings.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from shiwilu.rutas import CORPUS_CSV
from shiwilu.taxonomia import INTENCIONES

SEMILLA = 42
PROP_DEV_TEST = 0.30   # 70% train, 15% dev, 15% test
VALORES_C = [0.01, 0.1, 1.0, 3.0, 10.0]   # grilla de ajuste sobre el dev

# Modelos de embeddings soportados: nombre corto -> (id de HuggingFace, tipo)
#   "sentence_transformers": el modelo ya trae su propio pooling de oracion.
#   "mean_pooling":          encoder generico (mBERT, XLM-R); se aplica mean
#                            pooling sobre la ultima capa oculta, ponderado
#                            por la mascara de atencion.
MODELOS = {
    "labse": ("sentence-transformers/LaBSE", "sentence_transformers"),
    "mbert": ("bert-base-multilingual-cased", "mean_pooling"),
    "xlmr":  ("xlm-roberta-base", "mean_pooling"),
}


def cargar_corpus(corpus_csv=CORPUS_CSV) -> pd.DataFrame:
    df = pd.read_csv(corpus_csv)
    faltantes = {"shiwilu", "intencion"} - set(df.columns)
    if faltantes:
        raise SystemExit(f"Al corpus le faltan columnas: {faltantes}")
    return df


def dividir_train_dev_test(df: pd.DataFrame):
    """Divide train/dev/test agrupando por texto en shiwilu.

    El corpus tiene oraciones tan cortas que 28 textos en shiwilu se repiten
    con distinta glosa en espanol (ej. "MUPALLI"). Dividir por fila suelta
    podia mandar el mismo texto shiwilu a train Y a test a la vez, y el
    clasificador "adivinaba" esas filas de memoria en vez de generalizar.
    Agrupar por texto antes de dividir asegura que cada oracion en shiwilu
    caiga entera en un solo split.
    """
    grupos = df.groupby("shiwilu", as_index=False)["intencion"].first()
    train_g, resto_g = train_test_split(
        grupos, test_size=PROP_DEV_TEST, stratify=grupos["intencion"], random_state=SEMILLA,
    )
    dev_g, test_g = train_test_split(
        resto_g, test_size=0.5, stratify=resto_g["intencion"], random_state=SEMILLA,
    )
    train = df[df["shiwilu"].isin(train_g["shiwilu"])]
    dev = df[df["shiwilu"].isin(dev_g["shiwilu"])]
    test = df[df["shiwilu"].isin(test_g["shiwilu"])]
    return train, dev, test


def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


def extraer_embeddings(oraciones: list[str], nombre_modelo: str) -> np.ndarray:
    """Extrae embeddings de oracion, congelados (sin ajuste fino)."""
    if nombre_modelo not in MODELOS:
        raise ValueError(f"Modelo desconocido {nombre_modelo!r}. Opciones: {list(MODELOS)}")
    hf_id, tipo = MODELOS[nombre_modelo]

    if tipo == "sentence_transformers":
        from sentence_transformers import SentenceTransformer

        modelo = SentenceTransformer(hf_id)
        return modelo.encode(oraciones, convert_to_numpy=True, show_progress_bar=True)

    # tipo == "mean_pooling": mBERT / XLM-R, sin cabeza de embedding de oracion
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    modelo = AutoModel.from_pretrained(hf_id)
    modelo.eval()

    vectores = []
    batch_size = 16
    with torch.no_grad():
        for inicio in range(0, len(oraciones), batch_size):
            lote = oraciones[inicio : inicio + batch_size]
            enc = tokenizer(lote, padding=True, truncation=True, max_length=64, return_tensors="pt")
            salida = modelo(**enc)
            pooled = _mean_pooling(salida, enc["attention_mask"])
            vectores.append(pooled.numpy())
    return np.concatenate(vectores, axis=0)


def ajustar_clasificador(X_train, y_train, X_dev, y_dev) -> LogisticRegression:
    """Ajusta el hiperparametro C sobre el conjunto de desarrollo (F1 macro)."""
    mejor_c, mejor_f1, mejor_clf = None, -1.0, None
    for c in VALORES_C:
        clf = LogisticRegression(C=c, max_iter=2000, random_state=SEMILLA)
        clf.fit(X_train, y_train)
        f1_dev = f1_score(y_dev, clf.predict(X_dev), average="macro", zero_division=0)
        print(f"  C={c:<6} F1-macro (dev) = {f1_dev:.4f}")
        if f1_dev > mejor_f1:
            mejor_c, mejor_f1, mejor_clf = c, f1_dev, clf
    print(f"Mejor C sobre dev: {mejor_c} (F1-macro={mejor_f1:.4f})")
    return mejor_clf


def evaluar(clf: LogisticRegression, X_test, y_test) -> dict:
    y_pred = clf.predict(X_test)
    metricas = {
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_ponderado": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "exactitud": accuracy_score(y_test, y_pred),
    }
    reporte = classification_report(
        y_test, y_pred, labels=INTENCIONES, output_dict=True, zero_division=0,
    )
    matriz = confusion_matrix(y_test, y_pred, labels=INTENCIONES)
    return {
        "metricas": metricas, "reporte": reporte, "matriz": matriz,
        "y_test": np.asarray(y_test), "y_pred": y_pred,
    }


def guardar_resultados(carpeta, titulo: str, info_extra: dict, resultado: dict) -> None:
    carpeta.mkdir(parents=True, exist_ok=True)
    m = resultado["metricas"]

    with (carpeta / "metricas.json").open("w", encoding="utf-8") as f:
        json.dump({**info_extra, **m}, f, ensure_ascii=False, indent=2)

    pd.DataFrame(resultado["reporte"]).T.to_csv(
        carpeta / "reporte_clasificacion.csv", encoding="utf-8"
    )

    # una fila por oracion de test: insumo para el bootstrap de intervalos
    # de confianza (ver bootstrap_ic.py), sin tener que reentrenar nada.
    pd.DataFrame({"real": resultado["y_test"], "prediccion": resultado["y_pred"]}).to_csv(
        carpeta / "predicciones.csv", index=False, encoding="utf-8"
    )

    disp = ConfusionMatrixDisplay(resultado["matriz"], display_labels=INTENCIONES)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title(titulo)
    fig.tight_layout()
    fig.savefig(carpeta / "matriz_confusion.png", dpi=150)
    plt.close(fig)

    print(f"F1 macro={m['f1_macro']:.4f}  F1 ponderado={m['f1_ponderado']:.4f}  "
          f"Exactitud={m['exactitud']:.4f}")
    print(f"Resultados guardados en {carpeta}")
