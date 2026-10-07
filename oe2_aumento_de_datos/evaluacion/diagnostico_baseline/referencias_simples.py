"""
Referencias simples (pisos) contra las que se comparan los baselines con embeddings (OE2, R4).

No usan ningun modelo de lenguaje; sirven para saber cuanto aportan realmente mBERT, LaBSE y XLM-R:

  mayoria           predice siempre la clase mas frecuente del entrenamiento de cada fold (el corpus esta balanceado,
                    asi que equivale a un azar casi sin informacion).
  vecino_lexico     a cada oracion de prueba le asigna la clase de la oracion de entrenamiento con la que comparte mas
                    palabras (conteo de palabras distintas compartidas; ante un empate gana la primera del entrenamiento).
  vecino_lexico_jaccard   igual, pero con la similitud de Jaccard (palabras compartidas / palabras totales).

La referencia de n-gramas de caracteres esta en `ngramas_caracteres.py`.

Mismos folds congelados y mismo texto normalizado que los baselines; el F1 macro es el agrupado sobre las 700 predicciones.
No usa la API, ni GPU, ni modelos: corre en segundos.

Uso (desde la raiz del repositorio):  python oe2_aumento_de_datos/evaluacion/diagnostico_baseline/referencias_simples.py
Salida: oe2_aumento_de_datos/evaluacion/diagnostico_baseline/resultados/referencias_simples.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ))

from shiwilu.clasificacion import cargar_corpus, cargar_corpus_normalizado, cargar_folds  # noqa: E402

SAL = Path(__file__).resolve().parent / "resultados"


def f1m(a, b) -> float:
    return float(f1_score(a, b, average="macro", zero_division=0))


def main() -> int:
    corpus = cargar_corpus()
    y = corpus["intencion"].to_numpy()
    folds = cargar_folds(corpus)
    texto = cargar_corpus_normalizado()["shiwilu"].astype(str).tolist()
    palabras = [set(t.split()) for t in texto]

    def vecino(similitud) -> np.ndarray:
        pred = np.empty(len(y), dtype=object)
        for f in sorted(set(folds)):
            tr = np.where(folds != f)[0]
            for i in np.where(folds == f)[0]:
                s = [similitud(palabras[i], palabras[j]) for j in tr]
                pred[i] = y[tr[int(np.argmax(s))]]   # argmax devuelve el primer maximo
        return pred

    mayoria = np.empty(len(y), dtype=object)
    for f in sorted(set(folds)):
        clases, cuentas = np.unique(y[folds != f], return_counts=True)
        mayoria[folds == f] = clases[int(cuentas.argmax())]

    filas = [
        {"referencia": "mayoria", "f1_macro_agrupado": f1m(y, mayoria)},
        {"referencia": "vecino_lexico", "f1_macro_agrupado": f1m(y, vecino(lambda a, b: len(a & b)))},
        {"referencia": "vecino_lexico_jaccard", "f1_macro_agrupado": f1m(y, vecino(lambda a, b: len(a & b) / max(1, len(a | b))))},
    ]
    SAL.mkdir(parents=True, exist_ok=True)
    R = pd.DataFrame(filas)
    R.to_csv(SAL / "referencias_simples.csv", index=False)
    print(R.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
