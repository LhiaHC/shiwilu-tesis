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

Usa el corpus con `shiwilu` normalizado (`clasificacion.cargar_corpus_normalizado`,
minusculas/sin puntuacion/sin tildes-ñ) - la condicion `sin_puntuacion`,
unica vigente desde 2026-09-27 - y el split unico congelado
(`split_fijo.csv`, en esta carpeta), no los folds de la validacion cruzada.

Salida:
    historico/split_unico/curva_aprendizaje.csv
    historico/split_unico/curva_aprendizaje.png

Uso (desde la raiz del repositorio):
    python oe2_aumento_de_datos/historico/split_unico/curva_aprendizaje.py

    # un modelo a la vez (evita el segmentation fault de Windows al cargar
    # sentence-transformers y transformers en el mismo proceso, viendo
    # transicion labse -> mbert): cada corrida se fusiona con el CSV existente
    python oe2_aumento_de_datos/historico/split_unico/curva_aprendizaje.py --modelo labse
    python oe2_aumento_de_datos/historico/split_unico/curva_aprendizaje.py --modelo mbert
    python oe2_aumento_de_datos/historico/split_unico/curva_aprendizaje.py --modelo xlmr
"""

from __future__ import annotations

import argparse
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

import _ruta_raiz  # noqa: F401  (deja importable el paquete `shiwilu`)
from shiwilu.rutas import TECNICAS_AUMENTO  # noqa: E402

sys.path.insert(0, str(TECNICAS_AUMENTO))

from shiwilu.clasificacion import (  # noqa: E402
    MODELOS,
    SEMILLA,
    VALORES_C,
    cargar_corpus,
    cargar_corpus_normalizado,
    dividir_train_dev_test,
    extraer_embeddings,
    liberar_modelo,
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


def graficar(tabla: pd.DataFrame) -> None:
    modelos = [m for m in MODELOS if m in set(tabla["modelo"])]
    fig, ejes = plt.subplots(1, len(modelos), figsize=(5 * len(modelos), 4), sharey=True)
    for ax, modelo in zip(np.atleast_1d(ejes), modelos):
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", choices=list(MODELOS), default=None,
                     help="Corre solo este modelo y fusiona el resultado con el CSV existente. "
                          "Por defecto corre los 3 modelos en un mismo proceso; usar esta opcion "
                          "si eso falla con 'Segmentation fault' (conflicto de librerias al cargar "
                          "sentence-transformers y transformers en el mismo proceso, visto en Windows).")
    ap.add_argument("--solo-graficar", action="store_true",
                     help="No calcula nada nuevo: regenera curva_aprendizaje.png a partir del CSV ya guardado.")
    args = ap.parse_args()

    if args.solo_graficar:
        if not SALIDA_CSV.exists():
            raise SystemExit(f"No existe {SALIDA_CSV} todavia.")
        graficar(pd.read_csv(SALIDA_CSV))
        print(f"Grafico regenerado en {SALIDA_PNG}")
        return 0

    df = cargar_corpus()
    # El split unico (split_fijo.csv) esta congelado sobre claves calculadas
    # con el texto ORIGINAL (acentos/ñ incluidos, ver clasificacion._normalizar_shiwilu);
    # hay que dividir primero y normalizar despues, o esas claves dejan de
    # coincidir (a diferencia de folds_fijos.csv, que agrupa con la
    # normalizacion estricta y por eso es idempotente con el texto ya limpio).
    train, dev, test = dividir_train_dev_test(df)
    normalizado = cargar_corpus_normalizado().set_index("id")["shiwilu"]
    train = train.copy()
    train["shiwilu"] = train["id"].map(normalizado)
    dev = dev.copy()
    dev["shiwilu"] = dev["id"].map(normalizado)
    test = test.copy()
    test["shiwilu"] = test["id"].map(normalizado)
    y_train = train["intencion"].to_numpy()
    y_dev = dev["intencion"].to_numpy()
    y_test = test["intencion"].to_numpy()
    print(f"train={len(train)}  dev={len(dev)}  test={len(test)}")

    modelos_a_correr = [args.modelo] if args.modelo else list(MODELOS)
    filas = []
    for modelo in modelos_a_correr:
        print(f"\n=== {modelo} ===")
        X_train = extraer_embeddings(train["shiwilu"].astype(str).tolist(), modelo)
        X_dev = extraer_embeddings(dev["shiwilu"].astype(str).tolist(), modelo)
        X_test = extraer_embeddings(test["shiwilu"].astype(str).tolist(), modelo)
        liberar_modelo(modelo)  # ya se extrajeron los embeddings; no hace falta mantener los pesos en RAM

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

    tabla_nueva = pd.DataFrame(filas)
    if SALIDA_CSV.exists():
        previa = pd.read_csv(SALIDA_CSV)
        previa = previa[~previa["modelo"].isin(modelos_a_correr)]
        tabla = pd.concat([previa, tabla_nueva], ignore_index=True)
    else:
        tabla = tabla_nueva
    tabla.to_csv(SALIDA_CSV, index=False, encoding="utf-8")

    graficar(tabla)

    faltantes = [m for m in MODELOS if m not in set(tabla["modelo"])]
    if faltantes:
        print(f"\nFalta correr: {faltantes} (el grafico solo tiene los modelos ya calculados)")
    print(f"\nGuardado en {SALIDA_CSV} y {SALIDA_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
