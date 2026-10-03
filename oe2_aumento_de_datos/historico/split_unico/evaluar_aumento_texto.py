"""
Evalua, contra el pipeline de clasificacion, un corpus aumentado por una
tecnica que genera TEXTO nuevo (retrotraduccion o Generate-then-Refine) —
a diferencia de Mixup, que interpola embeddings directamente (ver mixup.py).

Toma el CSV de salida de la tecnica (columnas id, espanol, shiwilu,
intencion, fuente, estado_filtro, ...), se queda solo con las filas
`estado_filtro == "aprobado"`, y las agrega al conjunto de ENTRENAMIENTO del
baseline (dev y test se mantienen sin modificar, igual que en mixup.py).

Entrada: corpus/corpus_shiwilu_final.csv +
         oe2_aumento_de_datos/tecnicas_aumento/salidas/<tecnica>.csv
Salida:  historico/split_unico/resultados_tecnicas/<tecnica>/<modelo>/
    metricas.json, reporte_clasificacion.csv, matriz_confusion.png

Uso:
    python oe2_aumento_de_datos/historico/split_unico/evaluar_aumento_texto.py --tecnica retrotraduccion --modelo labse
    python oe2_aumento_de_datos/historico/split_unico/evaluar_aumento_texto.py --tecnica generate_then_refine --modelo mbert
"""

from __future__ import annotations

import argparse
import pandas as pd

import _ruta_raiz  # noqa: F401  (deja importable el paquete `shiwilu`)

from shiwilu.clasificacion import (  # noqa: E402
    MODELOS,
    ajustar_clasificador,
    cargar_corpus,
    dividir_train_dev_test,
    evaluar,
    extraer_embeddings,
    guardar_resultados,
)

from shiwilu.rutas import AUMENTO_SALIDA, TECNICAS_RESULTADOS, preparar_directorios  # noqa: E402

TECNICAS = {
    "retrotraduccion": AUMENTO_SALIDA / "retrotraduccion.csv",
    "generate_then_refine": AUMENTO_SALIDA / "generate_then_refine.csv",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tecnica", choices=list(TECNICAS), required=True)
    ap.add_argument("--modelo", choices=list(MODELOS), default="labse")
    args = ap.parse_args()
    hf_id, _ = MODELOS[args.modelo]

    ruta_csv = TECNICAS[args.tecnica]
    if not ruta_csv.exists():
        raise SystemExit(f"No se encontro {ruta_csv}. Corre primero la tecnica correspondiente.")

    preparar_directorios()

    print(f"=== {args.tecnica} + {args.modelo} ({hf_id}) ===")
    df = cargar_corpus()
    train, dev, test = dividir_train_dev_test(df)
    print(f"train original={len(train)}  dev={len(dev)}  test={len(test)}")

    aumento = pd.read_csv(ruta_csv)
    aumento_aprobado = aumento[aumento["estado_filtro"] == "aprobado"]
    print(f"{args.tecnica}: {len(aumento_aprobado)}/{len(aumento)} filas aprobadas se usan para aumentar train")

    train_aum = pd.concat(
        [train[["shiwilu", "intencion"]], aumento_aprobado[["shiwilu", "intencion"]]],
        ignore_index=True,
    )

    print("Extrayendo embeddings del conjunto de entrenamiento aumentado...")
    X_train_aum = extraer_embeddings(train_aum["shiwilu"].astype(str).tolist(), args.modelo)
    y_train_aum = train_aum["intencion"].to_numpy()

    print("Extrayendo embeddings de dev y test (sin modificar)...")
    X_dev = extraer_embeddings(dev["shiwilu"].astype(str).tolist(), args.modelo)
    X_test = extraer_embeddings(test["shiwilu"].astype(str).tolist(), args.modelo)

    print("Ajustando Regresion Logistica sobre el conjunto de desarrollo...")
    clf = ajustar_clasificador(X_train_aum, y_train_aum, X_dev, dev["intencion"])

    print("Evaluando sobre el conjunto de prueba...")
    resultado = evaluar(clf, X_test, test["intencion"])

    guardar_resultados(
        carpeta=TECNICAS_RESULTADOS / args.tecnica / args.modelo,
        titulo=f"{args.tecnica} + {args.modelo.upper()} congelado + Regresion Logistica",
        info_extra={
            "modelo_embeddings": hf_id,
            "clasificador": "LogisticRegression",
            "tecnica_aumento": args.tecnica,
            "n_train_original": len(train),
            "n_train_aumento_aprobado": len(aumento_aprobado),
            "n_train_aumentado": len(train_aum),
            "n_dev": len(dev), "n_test": len(test),
        },
        resultado=resultado,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
