"""
Intervalo de confianza del F1 macro, por remuestreo bootstrap.

Con un conjunto de prueba de ~105 oraciones, un solo F1 macro es un punto
sin margen de error: no se sabe si una diferencia de 2-3 puntos entre dos
modelos es real o solo ruido de muestreo. En vez de reentrenar cada
experimento con varias semillas (caro para retrotraduccion/Generate-then-
Refine, que dependen de GPU/API), este script reutiliza las predicciones
que cada experimento ya guardo (`predicciones.csv`, columnas `real` y
`prediccion`) y remuestrea esas mismas ~105 filas con reemplazo, muchas
veces, para estimar cuanto podria variar el F1 macro por puro azar de
muestreo.

No reentrena ningun clasificador ni cambia ningun resultado ya reportado:
solo le agrega un margen de incertidumbre a los mismos F1 macro de
`resumen_experimentos.csv`.

Entrada: **/predicciones.csv bajo 2_baselines/resultados/ y
          3_baselines_y_aumento_datos/tecnicas_aumento/resultados/
Salida:  3_baselines_y_aumento_datos/intervalos_confianza.csv

Uso:
    python 3_baselines_y_aumento_datos/bootstrap_ic.py
    python 3_baselines_y_aumento_datos/bootstrap_ic.py --repeticiones 5000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

RAIZ = Path(__file__).resolve().parents[1]
MODELOS = ["labse", "mbert", "xlmr"]
SEMILLA = 42


def recolectar_predicciones() -> list[Path]:
    patrones = [
        RAIZ / "2_baselines" / "resultados" / "*" / "predicciones.csv",
        RAIZ / "3_baselines_y_aumento_datos" / "tecnicas_aumento" / "resultados" / "*" / "*" / "predicciones.csv",
    ]
    rutas = []
    for patron in patrones:
        rutas.extend(Path(patron.anchor).glob(str(patron.relative_to(patron.anchor))))
    return rutas


def bootstrap_f1(y_real: np.ndarray, y_pred: np.ndarray, repeticiones: int, rng: np.random.Generator) -> dict:
    n = len(y_real)
    muestras = np.empty(repeticiones)
    for i in range(repeticiones):
        idx = rng.integers(0, n, size=n)
        muestras[i] = f1_score(y_real[idx], y_pred[idx], average="macro", zero_division=0)
    return {
        "f1_macro": float(f1_score(y_real, y_pred, average="macro", zero_division=0)),
        "ic95_bajo": float(np.percentile(muestras, 2.5)),
        "ic95_alto": float(np.percentile(muestras, 97.5)),
        "desviacion_estandar": float(muestras.std()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeticiones", type=int, default=2000,
                     help="Numero de remuestreos bootstrap por experimento.")
    args = ap.parse_args()
    rng = np.random.default_rng(SEMILLA)

    rutas = recolectar_predicciones()
    if not rutas:
        raise SystemExit(
            "No se encontraron predicciones.csv. Corre primero los experimentos "
            "(re-generan predicciones.csv automaticamente desde comun.guardar_resultados)."
        )

    filas = []
    for ruta in sorted(rutas):
        carpeta = ruta.parent
        metricas_json = carpeta / "metricas.json"
        with metricas_json.open(encoding="utf-8") as f:
            info = json.load(f)
        modelo = next((mo for mo in MODELOS if mo in ruta.parts), "?")
        tecnica = info.get("tecnica_aumento") or "sin_aumento"

        pred = pd.read_csv(ruta)
        resultado = bootstrap_f1(
            pred["real"].to_numpy(), pred["prediccion"].to_numpy(), args.repeticiones, rng,
        )
        print(f"{modelo:<6} {tecnica:<22} F1={resultado['f1_macro']:.4f}  "
              f"IC95%=[{resultado['ic95_bajo']:.4f}, {resultado['ic95_alto']:.4f}]  "
              f"sd={resultado['desviacion_estandar']:.4f}")
        filas.append({"modelo": modelo, "tecnica": tecnica, **resultado})

    tabla = pd.DataFrame(filas).sort_values(["modelo", "tecnica"])
    salida = Path(__file__).resolve().parent / "intervalos_confianza.csv"
    tabla.to_csv(salida, index=False, encoding="utf-8")
    print(f"\nGuardado en {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
