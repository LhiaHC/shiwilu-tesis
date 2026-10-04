"""
Sensibilidad del F1 al hiperparametro C para los niveles de volumen de la evaluacion en linea.

Igual que `sensibilidad_c.py`, pero sobre `validacion_cruzada_en_linea.py`: repite la validacion
cruzada de cada configuracion (misma cache de lo generado, mismos folds, misma eleccion de C) y
ademas registra, para cada (modelo, configuracion) y cada C de una grilla FIJA, el F1 macro agrupado
de las 700 predicciones, sin elegir C en ningun dev. Asi se separa el efecto del aumento del efecto
(ruidoso) de elegir C en ~90 oraciones. No llama a la API ni a la GPU: todo sale de la cache.

Configuraciones: Generate-then-Refine (20, 40, 80, 120 por categoria y etapa) y Retrotraduccion
(1 parafrasis por oracion). Mixup y "sin aumento" ya estan en `sensibilidad_c.csv`.

Entrada: --cache = carpeta con generate_then_refine/ y retrotraduccion/ (fold<N>_{core,pool}[_l<j>].csv).
         Para GtR debe haber sido refiltrada con `tecnicas_aumento/refiltrar_marcadores.py`.
Salida:  evaluacion/diagnostico_c/resultados/sensibilidad_c_en_linea.csv   (modelo, configuracion, C, f1_macro_agrupado)

Uso (desde la raiz del repositorio):
    python oe2_aumento_de_datos/evaluacion/diagnostico_c/sensibilidad_c_en_linea.py --cache <carpeta>
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # evaluacion/: validacion_cruzada*, _ruta_raiz
import _ruta_raiz  # noqa: F401  (deja importable el paquete `shiwilu`)

import validacion_cruzada as vc  # noqa: E402
import validacion_cruzada_en_linea as ven  # noqa: E402
from sensibilidad_c import GRILLA_C  # noqa: E402
from shiwilu.clasificacion import MODELOS, N_FOLDS, SEMILLA, cargar_corpus, cargar_folds  # noqa: E402
from shiwilu.rutas import AUMENTO_SALIDA, DIAGNOSTICO_C_RESULTADOS  # noqa: E402

CONFIGURACIONES = [   # (nombre, argumentos de validacion_cruzada_en_linea.py)
    ("gtr_20", ["--tecnica", "generate_then_refine", "--cantidad", "20"]),
    ("gtr_40", ["--tecnica", "generate_then_refine", "--cantidad", "40"]),
    ("gtr_80", ["--tecnica", "generate_then_refine", "--cantidad", "80"]),
    ("gtr_120", ["--tecnica", "generate_then_refine", "--cantidad", "120"]),
    ("retro_x1", ["--tecnica", "retrotraduccion", "--multiplicador", "1", "--checkpoint", "no_se_usa_con_cache"]),
]
SALIDA = DIAGNOSTICO_C_RESULTADOS / "sensibilidad_c_en_linea.csv"


def predicciones_por_c(argumentos: list[str], cache: Path, y: np.ndarray, folds: np.ndarray,
                       orden_modelos: list[str], grilla: list[float] = GRILLA_C) -> dict[str, dict[float, np.ndarray]]:
    """Corre la validacion cruzada en linea de una configuracion y devuelve, por modelo y C fija,
    las 700 predicciones del test (en el orden de las oraciones del corpus)."""
    llamadas: list[dict[float, np.ndarray]] = []
    ajustar_original = vc.ajustar_y_predecir

    def ajustar_y_registrar(X_core, y_core, X_dev, y_dev, X_pool, y_pool, X_test):
        resultado = ajustar_original(X_core, y_core, X_dev, y_dev, X_pool, y_pool, X_test)
        llamadas.append({
            c: LogisticRegression(C=c, max_iter=2000, random_state=SEMILLA).fit(X_pool, y_pool).predict(X_test)
            for c in grilla
        })
        return resultado

    vc.ajustar_y_predecir = ajustar_y_registrar
    ven.combinar_con_vigente = lambda P, tecnica: P   # en la carpeta temporal no esta la evaluacion vigente
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ven.EVALUACION_RESULTADOS = Path(tmp)
            sys.argv = ["validacion_cruzada_en_linea.py", "--cache", str(cache)] + argumentos
            ven.main()
    finally:
        vc.ajustar_y_predecir = ajustar_original

    salida: dict[str, dict[float, np.ndarray]] = {}
    for i_modelo, modelo in enumerate(orden_modelos):
        salida[modelo] = {}
        for c in grilla:
            pred = np.empty(len(y), dtype=object)
            for f in range(N_FOLDS):
                pred[np.where(folds == f)[0]] = llamadas[f * len(orden_modelos) + i_modelo][c]
            salida[modelo][c] = pred
    return salida


def f1_por_c(argumentos, cache, y, folds, orden_modelos) -> list[dict]:
    pred = predicciones_por_c(argumentos, cache, y, folds, orden_modelos)
    return [{"modelo": m, "C": c, "f1_macro_agrupado": float(f1_score(y, p, average="macro", zero_division=0))}
            for m, por_c in pred.items() for c, p in por_c.items()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=AUMENTO_SALIDA / "en_linea_marcadores_fuentes")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)   # esperable con C >= 100

    df = cargar_corpus()
    y = df["intencion"].to_numpy()
    folds = cargar_folds(df)
    orden_modelos = list(MODELOS)   # mismo orden que el bucle de validacion_cruzada_en_linea

    filas = []
    for nombre, argumentos in CONFIGURACIONES:
        print(f"\n######## {nombre} ########")
        for fila in f1_por_c(argumentos, args.cache, y, folds, orden_modelos):
            filas.append({"configuracion": nombre, **fila})

    R = pd.DataFrame(filas)
    R.to_csv(SALIDA, index=False, encoding="utf-8")

    base = pd.read_csv(DIAGNOSTICO_C_RESULTADOS / "sensibilidad_c.csv")
    base = base[base["tecnica"] == "sin_aumento"].set_index(["modelo", "C"])["f1_macro_agrupado"]
    R["f1_sin_aumento"] = [base[(m, c)] for m, c in zip(R["modelo"], R["C"])]
    R["diferencia"] = R["f1_macro_agrupado"] - R["f1_sin_aumento"]
    for modelo in MODELOS:
        print(f"\n{modelo}: F1 macro agrupado con C fijo (sin aumento: {', '.join(f'C={c}:{base[(modelo, c)]:.3f}' for c in GRILLA_C)})")
        g = R[R["modelo"] == modelo].pivot(index="configuracion", columns="C", values="diferencia")
        print("Diferencia frente a sin aumento con el MISMO C:")
        print(g.reindex([n for n, _ in CONFIGURACIONES]).round(3).to_string())
    print(f"\nGuardado en {SALIDA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
