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

Entrada: 3_baselines_y_aumento_datos/validacion_cruzada_predicciones_sin_puntuacion.csv
         (con columnas prob_<categoria> - corre primero
         `validacion_cruzada.py --sin-puntuacion` si no existen)
Salida:  3_baselines_y_aumento_datos/curvas_roc.csv  (fpr, tpr, auc por categoria y combinacion)
         3_baselines_y_aumento_datos/curvas_roc.png   (grilla 3 modelos x 4 tecnicas)

Uso:
    python 3_baselines_y_aumento_datos/curvas_roc.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "2_baselines"))
from comun import MODELOS  # noqa: E402

DIR = Path(__file__).resolve().parent
ENTRADA = DIR / "validacion_cruzada_predicciones_sin_puntuacion.csv"
SALIDA_CSV = DIR / "curvas_roc.csv"
SALIDA_PNG = DIR / "curvas_roc.png"

TECNICAS = ["sin_aumento", "mixup", "retrotraduccion", "generate_then_refine"]
NOMBRES_TECNICA = {
    "sin_aumento": "Sin aumento",
    "mixup": "Mixup",
    "retrotraduccion": "Retrotraduccion",
    "generate_then_refine": "Generate-then-Refine",
}


def main() -> int:
    if not ENTRADA.exists():
        raise SystemExit(
            f"No se encontro {ENTRADA}. Corre primero:\n"
            "  python 3_baselines_y_aumento_datos/validacion_cruzada.py --sin-puntuacion"
        )
    P = pd.read_csv(ENTRADA)
    columnas_prob = [c for c in P.columns if c.startswith("prob_")]
    if not columnas_prob:
        raise SystemExit(
            f"{ENTRADA} no tiene columnas prob_<categoria>. Corre de nuevo:\n"
            "  python 3_baselines_y_aumento_datos/validacion_cruzada.py --sin-puntuacion\n"
            "(con la version actualizada de validacion_cruzada.py que ya guarda las probabilidades)."
        )
    categorias = sorted(c.removeprefix("prob_") for c in columnas_prob)
    print(f"{len(P)} predicciones, {len(categorias)} categorias: {categorias}")

    filas = []
    fig, ejes = plt.subplots(
        len(MODELOS), len(TECNICAS), figsize=(4.2 * len(TECNICAS), 4 * len(MODELOS)),
        sharex=True, sharey=True,
    )
    for i, modelo in enumerate(MODELOS):
        for j, tecnica in enumerate(TECNICAS):
            ax = ejes[i, j]
            g = P[(P["modelo"] == modelo) & (P["tecnica"] == tecnica)]
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
            if i == 0:
                ax.set_title(NOMBRES_TECNICA[tecnica])
            if j == 0:
                ax.set_ylabel(f"{modelo.upper()}\nTPR (sensibilidad)")
            if i == len(MODELOS) - 1:
                ax.set_xlabel("FPR (1 - especificidad)")
            ax.legend(fontsize=6, loc="lower right")
            ax.grid(alpha=0.3)
            print(f"  {modelo:<6} {tecnica:<22} AUC macro = {auc_macro:.4f}")

    fig.suptitle("Curvas ROC One-vs-Rest (7 categorias) - texto normalizado, validacion cruzada de 5 folds")
    fig.tight_layout()
    fig.savefig(SALIDA_PNG, dpi=150)
    plt.close(fig)

    pd.DataFrame(filas).to_csv(SALIDA_CSV, index=False, encoding="utf-8")
    print(f"\nGuardado en {SALIDA_CSV} y {SALIDA_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
