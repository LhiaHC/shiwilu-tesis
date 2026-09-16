"""
Baseline de clasificacion: embedding multilingue congelado + Regresion
Logistica, sin ningun aumento de datos.

Extrae embeddings de las oraciones en SHIWILU del corpus con el modelo
elegido (congelado, sin ajuste fino) y entrena un clasificador de regresion
logistica sobre esas representaciones para predecir la categoria de
intencion. No se aplica ninguna tecnica de aumento de datos: se usa
unicamente corpus/corpus_shiwilu_final.csv tal como esta.

Sirve como linea base minima para comparar, mas adelante, el efecto de las
tecnicas de aumento (OE2, R5/R6) y de los demas metodos de caracterizacion de
embeddings (OE3, R7/R8).

Division: 70/15/15 (train/dev/test), estratificada por categoria de
intencion, semilla fija para reproducibilidad. Misma division para los tres
modelos, para que los resultados sean comparables entre si.

Entrada: corpus/corpus_shiwilu_final.csv
Salida:  3_baseline_clasificacion/resultados/<modelo>/
    metricas.json              F1 macro, F1 ponderado, exactitud
    reporte_clasificacion.csv  metricas por categoria de intencion
    matriz_confusion.png       matriz de confusion (conjunto de prueba)

Uso:
    python 3_baseline_clasificacion/baseline.py --modelo labse
    python 3_baseline_clasificacion/baseline.py --modelo mbert
    python 3_baseline_clasificacion/baseline.py --modelo xlmr
"""

from __future__ import annotations

import argparse

import _ruta_raiz  # noqa: F401  (deja importable el paquete `shiwilu`)
from comun import (
    MODELOS,
    ajustar_clasificador,
    cargar_corpus,
    dividir_train_dev_test,
    evaluar,
    extraer_embeddings,
    guardar_resultados,
)
from shiwilu.rutas import BASELINE_RESULTADOS, preparar_directorios


def correr_baseline(nombre_modelo: str) -> dict:
    """Corre el pipeline completo para un modelo y devuelve sus metricas."""
    hf_id, _ = MODELOS[nombre_modelo]

    print(f"\n=== Baseline: {nombre_modelo} ({hf_id}), congelado, sin ajuste fino ===")
    print("Cargando corpus...")
    df = cargar_corpus()
    train, dev, test = dividir_train_dev_test(df)
    print(f"train={len(train)}  dev={len(dev)}  test={len(test)}")

    print("Extrayendo embeddings...")
    X_train = extraer_embeddings(train["shiwilu"].astype(str).tolist(), nombre_modelo)
    X_dev = extraer_embeddings(dev["shiwilu"].astype(str).tolist(), nombre_modelo)
    X_test = extraer_embeddings(test["shiwilu"].astype(str).tolist(), nombre_modelo)

    print("Ajustando Regresion Logistica sobre el conjunto de desarrollo...")
    clf = ajustar_clasificador(X_train, train["intencion"], X_dev, dev["intencion"])

    print("Evaluando sobre el conjunto de prueba...")
    resultado = evaluar(clf, X_test, test["intencion"])

    guardar_resultados(
        carpeta=BASELINE_RESULTADOS / nombre_modelo,
        titulo=f"Baseline: {nombre_modelo.upper()} congelado + Regresion Logistica (sin aumento)",
        info_extra={
            "modelo_embeddings": hf_id,
            "clasificador": "LogisticRegression",
            "aumento_de_datos": False,
            "n_train": len(train), "n_dev": len(dev), "n_test": len(test),
        },
        resultado=resultado,
    )
    return resultado["metricas"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modelo", choices=list(MODELOS), default="labse")
    args = ap.parse_args()

    preparar_directorios()
    correr_baseline(args.modelo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
