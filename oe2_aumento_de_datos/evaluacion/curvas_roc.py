"""
Curvas ROC (One-vs-Rest, una por categoria, mas el promedio macro) para las
12 combinaciones de la validacion cruzada (3 modelos x 4 tecnicas de aumento),
sobre la condicion vigente (`sin_puntuacion`).

Con 7 categorias no existe UNA sola curva ROC (eso es para 2 clases): se traza
una curva por categoria ("esta categoria vs. todas las demas") mas el promedio
macro de las 7, para cada una de las 12 combinaciones.

No reentrena nada: toma las probabilidades por categoria que ya guarda
`validacion_cruzada.py` en `validacion_cruzada_predicciones_sin_puntuacion.csv`
(columnas `prob_<categoria>`, una fila por oracion de cada fold de test, con
la probabilidad que le asigno el clasificador de ESE fold).

Entrada: evaluacion/resultados/validacion_cruzada_predicciones_sin_puntuacion.csv
         (con columnas prob_<categoria> - corre primero `validacion_cruzada.py`
         si no existen)
Salida:  evaluacion/resultados/curvas_roc.csv  (fpr, tpr, auc por categoria y combinacion)
         evaluacion/resultados/curvas_roc.png   (grilla 3 modelos x 4 tecnicas)
         evaluacion/resultados/curvas_roc/      (12 imagenes, una por combinacion)

Con --entrada X.csv --etiqueta T grafica las predicciones de otro CSV (p. ej. el de una corrida en
linea de los cuadernos de Colab, que trae los 12 experimentos) y guarda `curvas_roc_T.csv`,
`curvas_roc_T.png` y la carpeta `curvas_roc_T/`.

Uso (desde la raiz del repositorio):
    python oe2_aumento_de_datos/evaluacion/curvas_roc.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

import _ruta_raiz  # noqa: F401  (deja importable el paquete `shiwilu`)

from shiwilu.clasificacion import MODELOS  # noqa: E402
from shiwilu.rutas import EVALUACION_RESULTADOS  # noqa: E402

ENTRADA = EVALUACION_RESULTADOS / "validacion_cruzada_predicciones_sin_puntuacion.csv"
SALIDA_CSV = EVALUACION_RESULTADOS / "curvas_roc.csv"
SALIDA_PNG = EVALUACION_RESULTADOS / "curvas_roc.png"
SALIDA_DIR = EVALUACION_RESULTADOS / "curvas_roc"

TECNICAS = ["sin_aumento", "mixup", "retrotraduccion", "generate_then_refine"]
NOMBRES_TECNICA = {
    "sin_aumento": "Sin aumento",
    "mixup": "Mixup",
    "retrotraduccion": "Retrotraduccion",
    "generate_then_refine": "Generate-then-Refine",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", type=Path, default=ENTRADA,
                     help="CSV de predicciones con columnas prob_<categoria> (por defecto, el vigente).")
    ap.add_argument("--etiqueta", default="", help="Se agrega al nombre de las salidas (CSV, PNG y carpeta).")
    args = ap.parse_args()
    sufijo = f"_{args.etiqueta}" if args.etiqueta else ""
    entrada = args.entrada
    salida_csv = EVALUACION_RESULTADOS / f"curvas_roc{sufijo}.csv"
    salida_png = EVALUACION_RESULTADOS / f"curvas_roc{sufijo}.png"
    salida_dir = EVALUACION_RESULTADOS / f"curvas_roc{sufijo}"

    if not entrada.exists():
        raise SystemExit(
            f"No se encontro {entrada}. Corre primero:\n"
            "  python oe2_aumento_de_datos/evaluacion/validacion_cruzada.py"
        )
    P = pd.read_csv(entrada)
    columnas_prob = [c for c in P.columns if c.startswith("prob_")]
    if not columnas_prob:
        raise SystemExit(
            f"{entrada} no tiene columnas prob_<categoria>. Corre de nuevo:\n"
            "  python oe2_aumento_de_datos/evaluacion/validacion_cruzada.py\n"
            "(con la version actualizada de validacion_cruzada.py que ya guarda las probabilidades)."
        )
    categorias = sorted(c.removeprefix("prob_") for c in columnas_prob)
    print(f"{len(P)} predicciones, {len(categorias)} categorias: {categorias}")

    def dibujar(ax, g, modelo, tecnica, filas, fuente_leyenda):
        y_real = label_binarize(g["real"], classes=categorias)
        fpr_grid = np.linspace(0.0, 1.0, 200)
        tpr_acum = np.zeros_like(fpr_grid)
        for k, cat in enumerate(categorias):
            fpr, tpr, _ = roc_curve(y_real[:, k], g[f"prob_{cat}"].to_numpy())
            area = auc(fpr, tpr)
            tpr_acum += np.interp(fpr_grid, fpr, tpr)
            ax.plot(fpr, tpr, lw=1, alpha=0.6, label=f"{cat} (AUC={area:.2f})")
            for a, b in zip(fpr, tpr):
                filas.append({"modelo": modelo, "tecnica": tecnica, "categoria": cat,
                              "fpr": float(a), "tpr": float(b), "auc": float(area)})
        tpr_macro = tpr_acum / len(categorias)
        auc_macro = auc(fpr_grid, tpr_macro)
        ax.plot(fpr_grid, tpr_macro, "k--", lw=2, label=f"Macro (AUC={auc_macro:.2f})")
        filas.append({"modelo": modelo, "tecnica": tecnica, "categoria": "macro",
                      "fpr": np.nan, "tpr": np.nan, "auc": float(auc_macro)})
        ax.plot([0, 1], [0, 1], color="grey", lw=0.8, linestyle=":")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=fuente_leyenda, loc="lower right")
        ax.grid(alpha=0.3)
        return auc_macro

    filas = []
    salida_dir.mkdir(exist_ok=True)
    fig, ejes = plt.subplots(
        len(MODELOS), len(TECNICAS), figsize=(4.2 * len(TECNICAS), 4 * len(MODELOS)),
        sharex=True, sharey=True,
    )
    for i, modelo in enumerate(MODELOS):
        for j, tecnica in enumerate(TECNICAS):
            ax = ejes[i, j]
            g = P[(P["modelo"] == modelo) & (P["tecnica"] == tecnica)]
            auc_macro = dibujar(ax, g, modelo, tecnica, filas, 6)
            if i == 0:
                ax.set_title(NOMBRES_TECNICA[tecnica])
            if j == 0:
                ax.set_ylabel(f"{modelo.upper()}\nTPR (sensibilidad)")
            if i == len(MODELOS) - 1:
                ax.set_xlabel("FPR (1 - especificidad)")
            print(f"  {modelo:<6} {tecnica:<22} AUC macro = {auc_macro:.4f}")

            fig1, ax1 = plt.subplots(figsize=(6, 5.2))
            dibujar(ax1, g, modelo, tecnica, [], 8)
            ax1.set_title(f"{modelo.upper()} - {NOMBRES_TECNICA[tecnica]}\nCurvas ROC One-vs-Rest, texto normalizado, CV 5 folds")
            ax1.set_xlabel("FPR (1 - especificidad)")
            ax1.set_ylabel("TPR (sensibilidad)")
            fig1.tight_layout()
            fig1.savefig(salida_dir / f"roc_{modelo}_{tecnica}.png", dpi=150)
            plt.close(fig1)

    fig.suptitle("Curvas ROC One-vs-Rest (7 categorias) - texto normalizado, validacion cruzada de 5 folds")
    fig.tight_layout()
    fig.savefig(salida_png, dpi=150)
    plt.close(fig)

    pd.DataFrame(filas).to_csv(salida_csv, index=False, encoding="utf-8")
    print(f"\nGuardado en {salida_csv} y {salida_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
