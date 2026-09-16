"""
Tecnica de aumento de datos: Mixup sobre embeddings (Guo et al., 2019).

No genera texto nuevo en shiwilu. En su lugar, interpola los VECTORES de
representacion de dos enunciados shiwilu existentes de la MISMA categoria de
intencion, generando un vector sintetico que hereda la etiqueta compartida:

    lambda ~ Beta(alpha, alpha)
    x_sintetico = lambda * x_i + (1 - lambda) * x_j      (i, j de la misma categoria)
    y_sintetico = categoria de i (== categoria de j)

Como ambos vectores de origen son de la misma categoria, no hace falta
mezclar etiquetas (a diferencia del Mixup original entre clases distintas):
el vector sintetico conserva la etiqueta sin ambiguedad.

Solo se aumenta el conjunto de ENTRENAMIENTO; dev y test se mantienen sin
modificar (mismo criterio que el resto de la Fase 3/4), para que la
comparacion contra el baseline sin aumento sea directa.

Entrada: corpus/corpus_shiwilu_final.csv
Salida:  4_aumento_datos/resultados/mixup/<modelo>/
    metricas.json, reporte_clasificacion.csv, matriz_confusion.png

Uso:
    python 4_aumento_datos/mixup.py --modelo labse
    python 4_aumento_datos/mixup.py --modelo labse --alpha 0.4 --multiplicador 1.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import _ruta_raiz  # noqa: F401  (deja importable el paquete `shiwilu`)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3_baseline_clasificacion"))
from comun import (  # noqa: E402
    MODELOS,
    SEMILLA,
    ajustar_clasificador,
    cargar_corpus,
    dividir_train_dev_test,
    evaluar,
    extraer_embeddings,
    guardar_resultados,
)

from shiwilu.rutas import FASE4, preparar_directorios  # noqa: E402
from shiwilu.taxonomia import INTENCIONES  # noqa: E402

RESULTADOS_MIXUP = FASE4 / "resultados" / "mixup"


def generar_sinteticos_mixup(
    X_train: np.ndarray,
    y_train,
    alpha: float,
    multiplicador: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Genera vectores sinteticos por interpolacion intra-categoria (Mixup)."""
    y_train = np.asarray(y_train)
    sinteticos, etiquetas = [], []

    for categoria in INTENCIONES:
        idx = np.where(y_train == categoria)[0]
        if len(idx) < 2:
            continue   # no hay pares posibles en esta categoria
        n_sinteticos = int(round(len(idx) * multiplicador))
        for _ in range(n_sinteticos):
            i, j = rng.choice(idx, size=2, replace=True)
            lam = rng.beta(alpha, alpha)
            sinteticos.append(lam * X_train[i] + (1 - lam) * X_train[j])
            etiquetas.append(categoria)

    return np.array(sinteticos), np.array(etiquetas)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modelo", choices=list(MODELOS), default="labse")
    ap.add_argument("--alpha", type=float, default=0.4,
                     help="Parametro de la Beta(alpha, alpha) para el coeficiente de mezcla.")
    ap.add_argument("--multiplicador", type=float, default=1.0,
                     help="Sinteticos por categoria = round(n_categoria * multiplicador). "
                          "1.0 = incremento del 100%% del train, por categoria.")
    args = ap.parse_args()
    hf_id, _ = MODELOS[args.modelo]
    rng = np.random.default_rng(SEMILLA)

    preparar_directorios()

    print(f"=== Mixup ({args.modelo}: {hf_id}) alpha={args.alpha} multiplicador={args.multiplicador} ===")
    df = cargar_corpus()
    train, dev, test = dividir_train_dev_test(df)
    print(f"train={len(train)}  dev={len(dev)}  test={len(test)}")

    print("Extrayendo embeddings del conjunto de entrenamiento...")
    X_train = extraer_embeddings(train["shiwilu"].astype(str).tolist(), args.modelo)
    y_train = train["intencion"].to_numpy()

    print("Generando vectores sinteticos (Mixup intra-categoria)...")
    X_sint, y_sint = generar_sinteticos_mixup(X_train, y_train, args.alpha, args.multiplicador, rng)
    print(f"  {len(X_sint)} vectores sinteticos generados "
          f"(train original {len(X_train)} -> train aumentado {len(X_train) + len(X_sint)})")

    X_train_aum = np.concatenate([X_train, X_sint], axis=0)
    y_train_aum = np.concatenate([y_train, y_sint], axis=0)

    print("Extrayendo embeddings de dev y test (sin modificar)...")
    X_dev = extraer_embeddings(dev["shiwilu"].astype(str).tolist(), args.modelo)
    X_test = extraer_embeddings(test["shiwilu"].astype(str).tolist(), args.modelo)

    print("Ajustando Regresion Logistica sobre el conjunto de desarrollo...")
    clf = ajustar_clasificador(X_train_aum, y_train_aum, X_dev, dev["intencion"])

    print("Evaluando sobre el conjunto de prueba...")
    resultado = evaluar(clf, X_test, test["intencion"])

    guardar_resultados(
        carpeta=RESULTADOS_MIXUP / args.modelo,
        titulo=f"Mixup + {args.modelo.upper()} congelado + Regresion Logistica",
        info_extra={
            "modelo_embeddings": hf_id,
            "clasificador": "LogisticRegression",
            "tecnica_aumento": "mixup",
            "alpha": args.alpha,
            "multiplicador": args.multiplicador,
            "n_train_original": len(train),
            "n_train_sinteticos": len(X_sint),
            "n_train_aumentado": len(X_train_aum),
            "n_dev": len(dev), "n_test": len(test),
        },
        resultado=resultado,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
