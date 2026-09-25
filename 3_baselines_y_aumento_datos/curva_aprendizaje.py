"""
Curva de aprendizaje: cuanto mejora el F1 macro al darle mas oraciones de
train al clasificador (embeddings congelados + Regresion Logistica).

Sirve para decidir si repartir mas datos a train (o pasar a validacion
cruzada) tiene sentido: si la curva ya esta plana con todo el train actual,
darle mas train casi no ayuda, y conviene usar los datos extra para un test
mas grande y mas confiable.

Para cada modelo y para "sin aumento" y "Mixup" (las dos configuraciones que
no dependen de generar texto fuera de este pipeline), se entrena con el 25%,
50%, 75% y 100% del train ACTUAL (split congelado, estratificado por
categoria), y se mide siempre sobre el mismo test. Con fracciones menores a
100% se repite con varios submuestreos aleatorios y se reporta media +- sd.

No toca dev ni test: `C` se elige sobre dev y el F1 se mide sobre test, igual
que en los experimentos principales. No genera texto sintetico nuevo
(retrotraduccion y Generate-then-Refine quedan fuera).

Salida:
    3_baselines_y_aumento_datos/curva_aprendizaje.csv
    3_baselines_y_aumento_datos/curva_aprendizaje.png

Uso:
    python 3_baselines_y_aumento_datos/curva_aprendizaje.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

RAIZ = Path(__file__).resolve().parents[1]
for ruta in (RAIZ, RAIZ / "2_baselines", RAIZ / "3_baselines_y_aumento_datos" / "tecnicas_aumento"):
    sys.path.insert(0, str(ruta))

from comun import (  # noqa: E402
    MODELOS,
    SEMILLA,
    VALORES_C,
    cargar_corpus,
    dividir_train_dev_test,
    extraer_embeddings,
)
from mixup import generar_sinteticos_mixup  # noqa: E402

FRACCIONES = [0.25, 0.5, 0.75, 1.0]
REPETICIONES = 5          # submuestreos por fraccion < 100%
ALPHA_MIXUP, MULT_MIXUP = 0.4, 1.0
SALIDA_CSV = Path(__file__).resolve().parent / "curva_aprendizaje.csv"
SALIDA_PNG = Path(__file__).resolve().parent / "curva_aprendizaje.png"


def f1_test(X_train, y_train, X_dev, y_dev, X_test, y_test) -> float:
    """Elige C sobre dev (F1 macro) y devuelve el F1 macro sobre test."""
    mejor_f1_dev, mejor_clf = -1.0, None
    for c in VALORES_C:
        clf = LogisticRegression(C=c, max_iter=2000, random_state=SEMILLA).fit(X_train, y_train)
        f1_dev = f1_score(y_dev, clf.predict(X_dev), average="macro", zero_division=0)
        if f1_dev > mejor_f1_dev:
            mejor_f1_dev, mejor_clf = f1_dev, clf
    return float(f1_score(y_test, mejor_clf.predict(X_test), average="macro", zero_division=0))


def main() -> int:
    df = cargar_corpus()
    train, dev, test = dividir_train_dev_test(df)
    y_train = train["intencion"].to_numpy()
    y_dev = dev["intencion"].to_numpy()
    y_test = test["intencion"].to_numpy()
    print(f"train={len(train)}  dev={len(dev)}  test={len(test)}")

    filas = []
    for modelo in MODELOS:
        print(f"\n=== {modelo} ===")
        X_train = extraer_embeddings(train["shiwilu"].astype(str).tolist(), modelo)
        X_dev = extraer_embeddings(dev["shiwilu"].astype(str).tolist(), modelo)
        X_test = extraer_embeddings(test["shiwilu"].astype(str).tolist(), modelo)

        for fraccion in FRACCIONES:
            n_rep = 1 if fraccion == 1.0 else REPETICIONES
            resultados = {"sin_aumento": [], "mixup": []}
            n_usado = int(round(len(train) * fraccion))
            for rep in range(n_rep):
                if fraccion == 1.0:
                    idx = np.arange(len(train))
                else:
                    idx, _ = train_test_split(
                        np.arange(len(train)), train_size=fraccion,
                        stratify=y_train, random_state=rep,
                    )
                Xs, ys = X_train[idx], y_train[idx]
                resultados["sin_aumento"].append(f1_test(Xs, ys, X_dev, y_dev, X_test, y_test))

                rng = np.random.default_rng(SEMILLA + rep)
                X_sint, y_sint = generar_sinteticos_mixup(Xs, ys, ALPHA_MIXUP, MULT_MIXUP, rng)
                resultados["mixup"].append(f1_test(
                    np.concatenate([Xs, X_sint]), np.concatenate([ys, y_sint]),
                    X_dev, y_dev, X_test, y_test,
                ))

            for tecnica, valores in resultados.items():
                media, sd = float(np.mean(valores)), float(np.std(valores))
                filas.append({"modelo": modelo, "tecnica": tecnica, "fraccion_train": fraccion,
                              "n_train": n_usado, "f1_macro_media": media, "f1_macro_sd": sd,
                              "repeticiones": n_rep})
                print(f"  {fraccion:>4.0%} (n={n_usado:>3})  {tecnica:<12} F1={media:.4f} +- {sd:.4f}")

    tabla = pd.DataFrame(filas)
    tabla.to_csv(SALIDA_CSV, index=False, encoding="utf-8")

    fig, ejes = plt.subplots(1, len(MODELOS), figsize=(5 * len(MODELOS), 4), sharey=True)
    for ax, modelo in zip(np.atleast_1d(ejes), MODELOS):
        for tecnica, estilo in (("sin_aumento", "o-"), ("mixup", "s--")):
            sub = tabla[(tabla["modelo"] == modelo) & (tabla["tecnica"] == tecnica)]
            ax.errorbar(sub["n_train"], sub["f1_macro_media"], yerr=sub["f1_macro_sd"],
                        fmt=estilo, capsize=3, label=tecnica)
        ax.set_title(modelo)
        ax.set_xlabel("oraciones de train usadas")
        ax.grid(alpha=0.3)
    np.atleast_1d(ejes)[0].set_ylabel("F1 macro (test)")
    np.atleast_1d(ejes)[0].legend()
    fig.suptitle("Curva de aprendizaje (test fijo, media +- sd de submuestreos)")
    fig.tight_layout()
    fig.savefig(SALIDA_PNG, dpi=150)
    plt.close(fig)

    print(f"\nGuardado en {SALIDA_CSV} y {SALIDA_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
