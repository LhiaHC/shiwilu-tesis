"""
Baseline de clasificacion: LaBSE congelado + Regresion Logistica, sin aumento.

Extrae embeddings de las oraciones en SHIWILU del corpus con LaBSE (modelo
congelado, sin ajuste fino) y entrena un clasificador de regresion logistica
sobre esas representaciones para predecir la categoria de intencion. No se
aplica ninguna tecnica de aumento de datos: se usa unicamente
corpus/corpus_shiwilu_final.csv tal como esta.

Sirve como linea base minima para comparar, mas adelante, el efecto de las
tecnicas de aumento (OE2, R5/R6) y de los demas metodos de caracterizacion de
embeddings (OE3, R7/R8).

Division: 70/15/15 (train/dev/test), estratificada por categoria de
intencion, semilla fija para reproducibilidad.

Entrada: corpus/corpus_shiwilu_final.csv
Salida:  3_baseline_clasificacion/resultados/
    metricas.json              F1 macro, F1 ponderado, exactitud
    reporte_clasificacion.csv  metricas por categoria de intencion
    matriz_confusion.png       matriz de confusion (conjunto de prueba)

Uso:
    python 3_baseline_clasificacion/baseline.py
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

import _ruta_raiz  # noqa: F401  (deja importable el paquete `shiwilu`)
from shiwilu.rutas import BASELINE_RESULTADOS, CORPUS_CSV, preparar_directorios
from shiwilu.taxonomia import INTENCIONES

MODELO_LABSE = "sentence-transformers/LaBSE"
SEMILLA = 42
PROP_DEV_TEST = 0.30   # 70% train, 15% dev, 15% test
VALORES_C = [0.01, 0.1, 1.0, 3.0, 10.0]   # grilla de ajuste sobre el dev


def cargar_corpus() -> pd.DataFrame:
    df = pd.read_csv(CORPUS_CSV)
    faltantes = {"shiwilu", "intencion"} - set(df.columns)
    if faltantes:
        raise SystemExit(f"Al corpus le faltan columnas: {faltantes}")
    return df


def dividir_train_dev_test(df: pd.DataFrame):
    train, resto = train_test_split(
        df, test_size=PROP_DEV_TEST, stratify=df["intencion"], random_state=SEMILLA,
    )
    dev, test = train_test_split(
        resto, test_size=0.5, stratify=resto["intencion"], random_state=SEMILLA,
    )
    return train, dev, test


def extraer_embeddings(oraciones: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    modelo = SentenceTransformer(MODELO_LABSE)
    return modelo.encode(oraciones, convert_to_numpy=True, show_progress_bar=True)


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
    return {"metricas": metricas, "reporte": reporte, "matriz": matriz, "y_pred": y_pred}


def main() -> int:
    preparar_directorios()

    print("Cargando corpus...")
    df = cargar_corpus()
    train, dev, test = dividir_train_dev_test(df)
    print(f"train={len(train)}  dev={len(dev)}  test={len(test)}")

    print("Extrayendo embeddings LaBSE (congelado, sin ajuste fino)...")
    X_train = extraer_embeddings(train["shiwilu"].astype(str).tolist())
    X_dev = extraer_embeddings(dev["shiwilu"].astype(str).tolist())
    X_test = extraer_embeddings(test["shiwilu"].astype(str).tolist())

    print("Ajustando Regresion Logistica sobre el conjunto de desarrollo...")
    clf = ajustar_clasificador(X_train, train["intencion"], X_dev, dev["intencion"])

    print("Evaluando sobre el conjunto de prueba...")
    resultado = evaluar(clf, X_test, test["intencion"])
    m = resultado["metricas"]
    print(f"F1 macro={m['f1_macro']:.4f}  F1 ponderado={m['f1_ponderado']:.4f}  "
          f"Exactitud={m['exactitud']:.4f}")

    BASELINE_RESULTADOS.mkdir(parents=True, exist_ok=True)

    with (BASELINE_RESULTADOS / "metricas.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "modelo_embeddings": MODELO_LABSE,
                "clasificador": "LogisticRegression",
                "aumento_de_datos": False,
                "n_train": len(train), "n_dev": len(dev), "n_test": len(test),
                **m,
            },
            f, ensure_ascii=False, indent=2,
        )

    pd.DataFrame(resultado["reporte"]).T.to_csv(
        BASELINE_RESULTADOS / "reporte_clasificacion.csv", encoding="utf-8"
    )

    disp = ConfusionMatrixDisplay(resultado["matriz"], display_labels=INTENCIONES)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title("Baseline: LaBSE congelado + Regresion Logistica (sin aumento)")
    fig.tight_layout()
    fig.savefig(BASELINE_RESULTADOS / "matriz_confusion.png", dpi=150)

    print(f"Resultados guardados en {BASELINE_RESULTADOS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
