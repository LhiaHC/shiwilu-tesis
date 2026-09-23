"""
Baselines triviales: el piso contra el que hay que leer los F1 de R4-R6.

Ninguno de los dos usa un modelo de embeddings ni aprendizaje real — son la
referencia minima para responder "¿que tanto de la F1 de LaBSE/mBERT/XLM-R
viene de que el corpus es facil (mucha repeticion lexica por categoria), y
no de que el modelo entienda shiwilu?":

  mayoria     Predice siempre la categoria mas frecuente del train. El piso
              absoluto — si un modelo real no le gana con margen, no esta
              aprendiendo nada util.
  nn_palabras 1-vecino-mas-cercano por solapamiento de palabras EXACTAS
              (interseccion de tokens, sin ningun embedding ni nocion de
              significado). Si esto ya saca una F1 alta, es señal de que el
              corpus tiene mucha reutilizacion lexica por categoria (el
              mismo lexema con variantes de puntuacion/genero), y que los
              modelos de embeddings podrian estar aprovechando esa
              repeticion superficial, no necesariamente semantica real.

Entrada: corpus/corpus_shiwilu_final.csv (mismo split que los baselines reales)
Salida:  2_baselines/resultados_triviales.csv

Uso:
    python 2_baselines/baseline_trivial.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

import _ruta_raiz  # noqa: F401  (deja importable el paquete `shiwilu`)
from comun import cargar_corpus, dividir_train_dev_test

SALIDA = Path(__file__).resolve().parent / "resultados_triviales.csv"


def predecir_mayoria(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    mas_frecuente = train["intencion"].value_counts().idxmax()
    return [mas_frecuente] * len(test)


def _tokeniza(oracion: str) -> set[str]:
    return set(str(oracion).lower().replace("'", "").split())


def predecir_nn_palabras(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    mas_frecuente = train["intencion"].value_counts().idxmax()
    train_tokens = [_tokeniza(s) for s in train["shiwilu"]]
    train_labels = train["intencion"].tolist()

    predicciones = []
    for oracion in test["shiwilu"]:
        tokens = _tokeniza(oracion)
        mejor_solape, mejor_label = -1, mas_frecuente
        for tok, label in zip(train_tokens, train_labels):
            solape = len(tokens & tok)
            if solape > mejor_solape:
                mejor_solape, mejor_label = solape, label
        predicciones.append(mejor_label)
    return predicciones


def main() -> int:
    df = cargar_corpus()
    train, _dev, test = dividir_train_dev_test(df)
    y_test = test["intencion"].to_numpy()

    filas = []
    for nombre, predecir in [
        ("mayoria", predecir_mayoria),
        ("nn_palabras", predecir_nn_palabras),
    ]:
        y_pred = predecir(train, test)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        print(f"{nombre:<12} F1 macro = {f1:.4f}")
        filas.append({"baseline": nombre, "f1_macro": f1, "n_test": len(test)})

    pd.DataFrame(filas).to_csv(SALIDA, index=False, encoding="utf-8")
    print(f"\nGuardado en {SALIDA}")
    print("\nComparar contra 2_baselines/resultados/<modelo>/metricas.json "
          "(f1_macro con LaBSE/mBERT/XLM-R + Regresion Logistica).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
