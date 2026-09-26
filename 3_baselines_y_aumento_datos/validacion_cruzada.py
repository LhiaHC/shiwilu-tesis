"""
Validacion cruzada de 5 folds de los 12 experimentos (3 modelos x 4 configuraciones).

Con un test de ~99 oraciones, un solo F1 tiene un margen de error de ~+-0.045.
Aqui cada oracion del corpus se usa como test exactamente una vez (5 folds
agrupados por texto shiwilu normalizado y estratificados por categoria, ver
`comun.cargar_folds`), y se reporta el F1 promedio de los folds y el F1
"agrupado" sobre las 700 predicciones, con su intervalo bootstrap.

Protocolo por fold (el test nunca se toca hasta el final):
  1. pool = oraciones fuera del fold de test (~560).
  2. Un ~1/6 del pool se aparta como dev interno (agrupado) SOLO para elegir C.
  3. Se elige C entrenando con el resto del pool (+ su aumento) y midiendo el
     F1 macro en el dev interno.
  4. Con ese C se reentrena sobre TODO el pool (+ su aumento) y se predice el fold.

Aumento, siempre derivado solo del pool del fold:
  sin_aumento           nada
  mixup                 vectores interpolados a partir de los embeddings del pool
  retrotraduccion       filas del catalogo (`retrotraduccion_pool.csv`) cuya
                        oracion de origen esta en el pool, filtradas con el
                        centroide LaBSE del pool
  generate_then_refine  `generate_then_refine_fold<N>.csv` (generado solo con
                        ejemplos del pool de ese fold), filas aprobadas

Deduplicacion contra el test: en cada fold se descartan las filas sinteticas
(retrotraduccion, Generate-then-Refine) cuyo texto normalizado coincide con el de
una oracion del fold de test. Motivo: el NMT de F. Prado se entreno con datos que
incluyen ~46% de las oraciones de este corpus (ambos parten de flashcards2) y las
memorizo, asi que ~15% de sus traducciones reproducen una oracion real del corpus
y algunas caerian justo en el test de otro fold.

Entrada: corpus, 2_baselines/folds_fijos.csv, salidas/retrotraduccion_pool.csv,
         salidas/generate_then_refine_fold<0..4>.csv
Salida:  3_baselines_y_aumento_datos/validacion_cruzada_resumen.csv
         3_baselines_y_aumento_datos/validacion_cruzada_predicciones.csv

Opcion --condicion (o --sin-puntuacion): transforma el texto ANTES de extraer los
embeddings, como ablacion de dos atajos del corpus (ver CONDICIONES): los signos de
puntuacion y las mayusculas. `sin_puntuacion` quita ambos (minusculas + sin signos);
`minusculas` y `sin_puntuacion_mayusculas` separan el efecto de cada uno. Guarda los
resultados con el sufijo `_<condicion>`. `--tecnicas` limita las configuraciones.

Uso:
    python 3_baselines_y_aumento_datos/validacion_cruzada.py
    python 3_baselines_y_aumento_datos/validacion_cruzada.py --sin-puntuacion
    python 3_baselines_y_aumento_datos/validacion_cruzada.py --condicion minusculas --tecnicas sin_aumento
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

RAIZ = Path(__file__).resolve().parents[1]
for ruta in (RAIZ, RAIZ / "2_baselines", RAIZ / "3_baselines_y_aumento_datos" / "tecnicas_aumento"):
    sys.path.insert(0, str(ruta))

from baseline_trivial import predecir_mayoria, predecir_nn_palabras  # noqa: E402
from comun import (  # noqa: E402
    MODELOS,
    N_FOLDS,
    SEMILLA,
    VALORES_C,
    _normalizar_estricto,
    quitar_puntuacion,
    cargar_corpus,
    cargar_folds,
    extraer_embeddings,
)
from mixup import generar_sinteticos_mixup  # noqa: E402
from retrotraduccion import UMBRAL_SIMILITUD_MAX, UMBRAL_SIMILITUD_MIN  # noqa: E402

from shiwilu.rutas import AUMENTO_SALIDA  # noqa: E402
from shiwilu.taxonomia import INTENCIONES  # noqa: E402

TECNICAS = ["sin_aumento", "mixup", "retrotraduccion", "generate_then_refine"]
# Transformaciones del texto. Dos atajos del corpus: los signos de puntuacion (los
# `¿?` delatan PRG) y las mayusculas (DES, PRG y REQUEST estan 100% en MAYUSCULAS;
# las demas categorias, 10-21%). `sin_puntuacion` (la ablacion principal) quita
# AMBOS; las otras dos condiciones sirven para atribuir el efecto a cada uno.
CONDICIONES = {
    "original": lambda t: str(t),
    "minusculas": lambda t: str(t).lower(),
    "sin_puntuacion_mayusculas": lambda t: quitar_puntuacion(t, minusculas=False),
    "sin_puntuacion": lambda t: quitar_puntuacion(t, minusculas=True),
}
ALPHA_MIXUP, MULT_MIXUP = 0.4, 1.0
REPETICIONES_BOOTSTRAP = 2000
DIR = Path(__file__).resolve().parent


def f1m(y, p) -> float:
    return float(f1_score(y, p, average="macro", zero_division=0))


def ajustar_y_predecir(X_core, y_core, X_dev, y_dev, X_pool, y_pool, X_test):
    """Elige C sobre el dev interno y predice el test con el modelo reentrenado sobre el pool."""
    mejor_c, mejor_f1 = VALORES_C[0], -1.0
    for c in VALORES_C:
        clf = LogisticRegression(C=c, max_iter=2000, random_state=SEMILLA).fit(X_core, y_core)
        f1 = f1m(y_dev, clf.predict(X_dev))
        if f1 > mejor_f1:
            mejor_c, mejor_f1 = c, f1
    clf = LogisticRegression(C=mejor_c, max_iter=2000, random_state=SEMILLA).fit(X_pool, y_pool)
    return clf.predict(X_test), mejor_c


def filtrar_catalogo_retro(pool_retro, L_orig, L_retro, y, idx_pool) -> np.ndarray:
    """Mascara de filas del catalogo que pasan los filtros, con el centroide LaBSE del pool."""
    en_pool = pool_retro["id_origen"].isin(set(idx_pool)).to_numpy()
    sim = np.zeros(len(pool_retro))
    for cat in INTENCIONES:
        sel_pool = idx_pool[y[idx_pool] == cat]
        centroide = L_orig[sel_pool].mean(axis=0)
        filas = (pool_retro["intencion"] == cat).to_numpy()
        v = L_retro[filas]
        sim[filas] = (v @ centroide) / (np.linalg.norm(v, axis=1) * np.linalg.norm(centroide) + 1e-9)
    idioma_ok = (pool_retro["shiwilu"].str.strip().str.lower()
                 != pool_retro["espanol"].str.strip().str.lower()).to_numpy()
    return en_pool & idioma_ok & (sim >= UMBRAL_SIMILITUD_MIN) & (sim <= UMBRAL_SIMILITUD_MAX)


def bootstrap_ic(y, p, rng) -> tuple[float, float]:
    n = len(y)
    muestras = [f1m(y[i], p[i]) for i in (rng.integers(0, n, n) for _ in range(REPETICIONES_BOOTSTRAP))]
    return float(np.percentile(muestras, 2.5)), float(np.percentile(muestras, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condicion", choices=list(CONDICIONES), default="original",
                     help="Transformacion del texto ANTES de extraer embeddings (ver CONDICIONES).")
    ap.add_argument("--sin-puntuacion", action="store_true",
                     help="Atajo de --condicion sin_puntuacion.")
    ap.add_argument("--tecnicas", nargs="+", choices=TECNICAS, default=TECNICAS,
                     help="Subconjunto de configuraciones a correr (por defecto, las 4).")
    args = ap.parse_args()
    condicion = "sin_puntuacion" if args.sin_puntuacion else args.condicion
    sufijo = "" if condicion == "original" else f"_{condicion}"
    tx = CONDICIONES[condicion]

    df = cargar_corpus()
    n = len(df)
    y = df["intencion"].to_numpy()
    claves = df["shiwilu"].map(_normalizar_estricto).to_numpy()
    folds = cargar_folds(df)
    print(f"{n} oraciones, folds: {np.bincount(folds).tolist()}")

    pool_retro = pd.read_csv(AUMENTO_SALIDA / "retrotraduccion_pool.csv")
    gtr = {}
    for f in range(N_FOLDS):
        g = pd.read_csv(AUMENTO_SALIDA / f"generate_then_refine_fold{f}.csv")
        gtr[f] = g[g["estado_filtro"] == "aprobado"].reset_index(drop=True)
        print(f"  Generate-then-Refine fold {f}: {len(gtr[f])} filas aprobadas de {len(g)}")

    textos = [tx(t) for t in df["shiwilu"]] + [tx(t) for t in pool_retro["shiwilu"]]
    corte_gtr, pos = {}, len(textos)
    for f in range(N_FOLDS):
        corte_gtr[f] = (pos, pos + len(gtr[f]))
        textos += [tx(t) for t in gtr[f]["shiwilu"]]
        pos += len(gtr[f])
    m = len(pool_retro)
    claves_pool_retro = pool_retro["shiwilu"].map(_normalizar_estricto).to_numpy()
    claves_gtr = {f: gtr[f]["shiwilu"].map(_normalizar_estricto).to_numpy() for f in range(N_FOLDS)}

    predicciones, L_orig, L_retro = [], None, None
    for modelo in ["labse"] + [k for k in MODELOS if k != "labse"]:
        print(f"\n=== {modelo}: extrayendo embeddings de {len(textos)} textos ===")
        E = extraer_embeddings(textos, modelo)
        E_orig, E_retro = E[:n], E[n:n + m]
        if modelo == "labse":
            L_orig, L_retro = E_orig, E_retro

        for f in range(N_FOLDS):
            idx_test = np.where(folds == f)[0]
            idx_pool = np.where(folds != f)[0]
            cv_interno = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=SEMILLA + f)
            _, dev_rel = next(cv_interno.split(idx_pool, y[idx_pool], groups=claves[idx_pool]))
            idx_dev = idx_pool[dev_rel]
            idx_core = np.setdiff1d(idx_pool, idx_dev)

            ok_retro = filtrar_catalogo_retro(pool_retro, L_orig, L_retro, y, idx_pool)
            claves_test = np.unique(claves[idx_test])
            copia_retro = ok_retro & np.isin(claves_pool_retro, claves_test)
            ok_retro &= ~copia_retro
            origen = pool_retro["id_origen"].to_numpy()
            y_retro = pool_retro["intencion"].to_numpy()
            a, b = corte_gtr[f]
            copia_gtr = np.isin(claves_gtr[f], claves_test)
            X_gtr, y_gtr = E[a:b][~copia_gtr], gtr[f]["intencion"].to_numpy()[~copia_gtr]
            if modelo == "labse":
                print(f"  fold {f}: se descartan {int(copia_retro.sum())} filas de retrotraduccion y "
                      f"{int(copia_gtr.sum())} de Generate-then-Refine que copian una oracion del test")

            for tecnica in args.tecnicas:
                rng = np.random.default_rng(SEMILLA + f)
                if tecnica == "sin_aumento":
                    aug_core = (E_orig[:0], y[:0]); aug_pool = (E_orig[:0], y[:0])
                elif tecnica == "mixup":
                    aug_core = generar_sinteticos_mixup(E_orig[idx_core], y[idx_core], ALPHA_MIXUP, MULT_MIXUP, rng)
                    aug_pool = generar_sinteticos_mixup(E_orig[idx_pool], y[idx_pool], ALPHA_MIXUP, MULT_MIXUP, rng)
                elif tecnica == "retrotraduccion":
                    en_core = ok_retro & np.isin(origen, idx_core)
                    aug_core = (E_retro[en_core], y_retro[en_core])
                    aug_pool = (E_retro[ok_retro], y_retro[ok_retro])
                else:
                    aug_core = (X_gtr, y_gtr); aug_pool = (X_gtr, y_gtr)

                X_core = np.concatenate([E_orig[idx_core], aug_core[0]]); y_core = np.concatenate([y[idx_core], aug_core[1]])
                X_pool = np.concatenate([E_orig[idx_pool], aug_pool[0]]); y_pool = np.concatenate([y[idx_pool], aug_pool[1]])
                pred, c = ajustar_y_predecir(X_core, y_core, E_orig[idx_dev], y[idx_dev], X_pool, y_pool, E_orig[idx_test])
                for pos_i, real_i, pred_i in zip(idx_test, y[idx_test], pred):
                    predicciones.append({"modelo": modelo, "tecnica": tecnica, "fold": f, "pos": int(pos_i),
                                         "real": real_i, "prediccion": pred_i, "C": c})
            print(f"  fold {f} listo (test={len(idx_test)}, pool={len(idx_pool)}, dev interno={len(idx_dev)})")

    # baselines triviales por fold
    for f in range(N_FOLDS):
        idx_test = np.where(folds == f)[0]
        df_txt = df.assign(shiwilu=df["shiwilu"].map(tx))
        train_f, test_f = df_txt.iloc[np.where(folds != f)[0]], df_txt.iloc[idx_test]
        for nombre, fn in (("trivial_mayoria", predecir_mayoria), ("trivial_nn_palabras", predecir_nn_palabras)):
            if args.tecnicas != TECNICAS:
                continue
            for pos_i, real_i, pred_i in zip(idx_test, y[idx_test], fn(train_f, test_f)):
                predicciones.append({"modelo": "-", "tecnica": nombre, "fold": f, "pos": int(pos_i),
                                     "real": real_i, "prediccion": pred_i, "C": np.nan})

    P = pd.DataFrame(predicciones)
    P.to_csv(DIR / f"validacion_cruzada_predicciones{sufijo}.csv", index=False, encoding="utf-8")

    rng = np.random.default_rng(SEMILLA)
    filas = []
    for (modelo, tecnica), g in P.groupby(["modelo", "tecnica"], sort=False):
        por_fold = [f1m(gf["real"], gf["prediccion"]) for _, gf in g.groupby("fold")]
        real, pred = g["real"].to_numpy(), g["prediccion"].to_numpy()
        lo, hi = bootstrap_ic(real, pred, rng)
        filas.append({"modelo": modelo, "tecnica": tecnica,
                      "f1_macro_agrupado": f1m(real, pred),
                      "ic95_bajo": lo, "ic95_alto": hi, "ancho_ic": hi - lo,
                      "f1_media_folds": float(np.mean(por_fold)), "f1_sd_folds": float(np.std(por_fold, ddof=1)),
                      "n_predicciones": len(g)})
    R = pd.DataFrame(filas)
    R.to_csv(DIR / f"validacion_cruzada_resumen{sufijo}.csv", index=False, encoding="utf-8")
    print("\n" + R.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
