"""
Sintesis del OE4 (R9): reune en tablas (CSV) y figuras (PNG) todo lo evaluado en OE2, OE3 y el fine-tuning,
con el protocolo B (C elegido por validacion cruzada interna, K=5), texto normalizado y las 5 particiones congeladas.
No entrena modelos ni usa la API: lee predicciones ya generadas. Lo unico que calcula si no esta en cache es el
clasificador de n-gramas de letras (referencia), que guarda en `resultados/ngramas_predicciones.csv`.

ENTRADAS
  OE2 (extrinseco, extraccion de fabrica)  oe2_aumento_de_datos/evaluacion/resultados/cv_interna/
        validacion_cruzada_predicciones_sin_puntuacion_cv_interna_c120.csv   (3 modelos x sin aumento, Mixup, Retrotraduccion, GtR)
  OE3 (4 estrategias de pooling)           oe2_aumento_de_datos/evaluacion/resultados/cv_interna_pooling/
        validacion_cruzada_predicciones_sin_puntuacion_cv_interna_<tecnica>_<pooling>.csv   (3 modelos x 3 corpus x 4 poolings = 36)
  OE3 (intrinseco)                         oe3_caracterizacion_embeddings/resultados/<corpus>/metricas_intrinsecas.csv
  Fine-tuning                              oe4_sintesis/finetuning/resultados/predicciones_<modelo>_<variante>.csv

SALIDAS (oe4_sintesis/resultados/)
  tabla_36_combinaciones.csv              F1 macro (IC 95%), F1 ponderado, exactitud, F1 por intencion, metricas intrinsecas y
                                          diferencias pareadas (frente al baseline de OE2 y frente al mismo pooling sin aumento)
  mejores_por_modelo.csv                  mejor pooling por F1 y por silueta, para cada modelo (sin aumento)
  top_configuraciones_vs_mejor.csv        las mejores configuraciones congeladas frente a la mejor (diferencia pareada)
  correlaciones_intrinseco_extrinseco.csv Spearman entre cada metrica intrinseca y el F1
  enfoques_f1.csv                         F1 (IC 95%) de todos los enfoques: congelado, mejor pooling, +GtR, fine-tuning, n-gramas
  finetuning_vs_congelado.csv             fine-tuning de cada modelo y variante frente a su baseline congelado y frente a los n-gramas
  f1_por_intencion_enfoques.csv           F1 por intencion de cada enfoque
  matriz_confusion_<enfoque>.csv          matrices de confusion (conteos) de la mejor configuracion congelada y del fine-tuning de mBERT
  des_oraciones_dificiles.csv             oraciones de DES que ningun metodo clasifica bien
  figuras/fig_oe4_*.png

Uso (desde la raiz del repositorio):
    python oe4_sintesis/sintesis.py
    python oe4_sintesis/sintesis.py --recalcular-ngramas     # vuelve a calcular los n-gramas en vez de leer la cache
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import _ruta_raiz  # noqa: F401  (deja importable el paquete `shiwilu`)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from shiwilu.rutas import CARACTERIZACION_RESULTADOS, CV_INTERNA_RESULTADOS, EVALUACION_RESULTADOS, SINTESIS_RESULTADOS  # noqa: E402
from shiwilu.taxonomia import INTENCIONES  # noqa: E402

warnings.filterwarnings("ignore")

CLASES = sorted(INTENCIONES)                     # AFI, DES, EMO, NEG, PRG, REQUEST, SAL (el orden de las columnas prob_<clase>)
K = len(CLASES)
MODELOS = ["mbert", "labse", "xlmr"]
NOM = {"mbert": "mBERT", "labse": "LaBSE", "xlmr": "XLM-R"}
COLOR = {"mbert": "#1976d2", "labse": "#f57c00", "xlmr": "#2e7d32"}
POOLINGS = ["cls", "mean_pooling", "max_pooling", "combinacion_capas"]
NOM_POOL = {"cls": "CLS", "mean_pooling": "Mean", "max_pooling": "Max", "combinacion_capas": "Comb. capas"}
CORPUS = {"sin_aumento": "sin aumento", "retrotraduccion": "Retrotraduccion", "generate_then_refine": "GtR"}
CARPETA_INTRINSECA = {"sin aumento": "original", "Retrotraduccion": "retrotraduccion", "GtR": "generate_then_refine"}
EXTRACCION_OE2 = "extraccion OE2"                # Mixup solo existe con la extraccion de fabrica de OE2
POOL_RESULTADOS = EVALUACION_RESULTADOS / "cv_interna_pooling"
FT_RESULTADOS = SINTESIS_RESULTADOS.parent / "finetuning" / "resultados"
FIGURAS = SINTESIS_RESULTADOS / "figuras"
SEMILLA_BOOTSTRAP = 42

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})


# --------------------------------------------------------------------------------------------------- medidas
def matriz_confusion(y, p):
    return np.bincount(y * K + p, minlength=K * K).reshape(K, K)


def f1_por_clase(y, p):
    cm = matriz_confusion(y, p)
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)


def f1_macro(y, p):
    return float(f1_por_clase(y, p).mean())


def metricas(y, p):
    cm = matriz_confusion(y, p)
    f1c = f1_por_clase(y, p)
    soporte = cm.sum(1)
    return {"f1_macro": float(f1c.mean()), "f1_ponderado": float((f1c * soporte).sum() / soporte.sum()),
            "exactitud": float(np.diag(cm).sum() / cm.sum()), **{f"f1_{c}": float(v) for c, v in zip(CLASES, f1c)}}


def a_indices(serie) -> np.ndarray:
    return np.array([CLASES.index(c) for c in serie])


def ic_f1(y, preds, rng, repeticiones):
    """IC 95% bootstrap del F1 macro (media de las predicciones de varias semillas si hay mas de una)."""
    P = np.atleast_2d(preds)
    n = len(y); v = []
    for _ in range(repeticiones):
        i = rng.integers(0, n, n)
        v.append(np.mean([f1_macro(y[i], p[i]) for p in P]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def dif_pareada(y, A, B, rng, repeticiones):
    """F1(A) - F1(B) sobre las mismas oraciones; A y B pueden traer varias semillas (se promedia). Devuelve (dif, bajo, alto, distinguible)."""
    A, B = np.atleast_2d(A), np.atleast_2d(B)
    n = len(y); v = []
    for _ in range(repeticiones):
        i = rng.integers(0, n, n)
        v.append(np.mean([f1_macro(y[i], p[i]) for p in A]) - np.mean([f1_macro(y[i], p[i]) for p in B]))
    dif = float(np.mean([f1_macro(y, p) for p in A]) - np.mean([f1_macro(y, p) for p in B]))
    lo, hi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
    return dif, lo, hi, bool(lo > 0 or hi < 0)


# --------------------------------------------------------------------------------------------------- carga
def cargar_predicciones(ruta: Path, modelo=None, tecnica=None) -> pd.DataFrame:
    d = pd.read_csv(ruta)
    if modelo is not None:
        d = d[d["modelo"] == modelo]
    if tecnica is not None:
        d = d[d["tecnica"] == tecnica]
    return d.sort_values("pos")


def cargar_oe2() -> dict:
    d = pd.read_csv(CV_INTERNA_RESULTADOS / "validacion_cruzada_predicciones_sin_puntuacion_cv_interna_c120.csv")
    return {(m, t): g.sort_values("pos") for (m, t), g in d.groupby(["modelo", "tecnica"])}


def cargar_pooling() -> dict:
    salida = {}
    for t, nombre in CORPUS.items():
        for p in POOLINGS:
            d = pd.read_csv(POOL_RESULTADOS / f"validacion_cruzada_predicciones_sin_puntuacion_cv_interna_{t}_{p}.csv")
            for m in MODELOS:
                salida[(m, p, nombre)] = d[(d["modelo"] == m) & (d["tecnica"] == t)].sort_values("pos")
    return salida


def cargar_finetuning() -> dict:
    """{(modelo, variante): (semillas x 700 predicciones en indices)} a partir de predicciones_<modelo>_<variante>.csv."""
    salida = {}
    for ruta in sorted(FT_RESULTADOS.glob("predicciones_*.csv")):
        d = pd.read_csv(ruta).sort_values(["semilla", "pos"])
        modelo, variante = d["modelo"].iloc[0], d["variante"].iloc[0]
        salida[(modelo, variante)] = np.stack([a_indices(g["prediccion"]) for _, g in d.groupby("semilla")])
    return salida


def ngramas_de_letras(y_txt, recalcular: bool) -> np.ndarray:
    """Referencia: TF-IDF de n-gramas de caracteres (2-5) + Regresion Logistica, con las mismas particiones y el mismo protocolo de C
    (CV interna K=5, grilla 0.1-100). Es el mismo clasificador de oe2.../diagnostico_baseline/ngramas_caracteres.py; aqui se guardan sus predicciones."""
    ruta = SINTESIS_RESULTADOS / "ngramas_predicciones.csv"
    if ruta.exists() and not recalcular:
        d = pd.read_csv(ruta).sort_values("pos")
        return a_indices(d["prediccion"])
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from shiwilu.clasificacion import SEMILLA, _normalizar_estricto, cargar_corpus, cargar_corpus_normalizado, cargar_folds

    corpus = cargar_corpus(); y = corpus["intencion"].to_numpy(); folds = cargar_folds(corpus)
    txt = cargar_corpus_normalizado()["shiwilu"].astype(str).tolist()
    claves = np.array([_normalizar_estricto(t) for t in txt])
    grilla = [0.1, 0.3, 1, 3, 10, 30, 100]

    def ajustar(tr, te, C):
        v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
        Xtr = v.fit_transform([txt[i] for i in tr]); Xte = v.transform([txt[i] for i in te])
        return LogisticRegression(C=C, max_iter=3000, random_state=SEMILLA).fit(Xtr, y[tr]).predict(Xte)

    pred = np.empty(len(y), dtype=object)
    for f in sorted(set(folds)):
        test = np.where(folds == f)[0]; pool = np.where(folds != f)[0]
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEMILLA + int(f) + 500)   # mismas particiones internas que el protocolo B
        partes = [(pool[a], pool[b]) for a, b in cv.split(pool, y[pool], groups=claves[pool])]
        posicion = {int(i): n for n, i in enumerate(pool)}
        puntaje = {}
        for C in grilla:
            p = np.empty(len(pool), dtype=object)
            for tr, va in partes:
                p[[posicion[int(i)] for i in va]] = ajustar(tr, va, C)
            puntaje[C] = f1_macro(a_indices(y[pool]), a_indices(p))
        mejor = max(grilla, key=lambda c: (puntaje[c], -c))
        pred[test] = ajustar(pool, test, mejor)
    pd.DataFrame({"pos": np.arange(len(y)), "real": y, "prediccion": pred}).to_csv(ruta, index=False)
    return a_indices(pred)


# --------------------------------------------------------------------------------------------------- figuras
def guardar(fig, nombre, ajustar=True):
    FIGURAS.mkdir(parents=True, exist_ok=True)
    if ajustar:
        fig.tight_layout()
    fig.savefig(FIGURAS / nombre, bbox_inches="tight")
    plt.close(fig)
    print("figura:", nombre)


def fig_f1_36(T):
    fig, ejes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
    for ax, corpus in zip(ejes, CORPUS.values()):
        sub = T[(T.corpus == corpus) & (T.pooling != EXTRACCION_OE2)]
        M = np.array([[sub[(sub.modelo == m) & (sub.pooling == p)].f1_macro.iloc[0] for p in POOLINGS] for m in MODELOS])
        S = np.array([[bool(sub[(sub.modelo == m) & (sub.pooling == p)].distinguible_vs_baseline.iloc[0]) for p in POOLINGS] for m in MODELOS])
        im = ax.imshow(M, cmap="YlGnBu", vmin=0.52, vmax=0.68, aspect="auto")
        for i in range(len(MODELOS)):
            for j in range(len(POOLINGS)):
                ax.text(j, i, f"{M[i, j]:.3f}" + ("*" if S[i, j] else ""), ha="center", va="center", color="white" if M[i, j] > 0.62 else "black", fontsize=10)
        ax.set_xticks(range(4)); ax.set_xticklabels([NOM_POOL[p] for p in POOLINGS], fontsize=9)
        ax.set_yticks(range(3)); ax.set_yticklabels([NOM[m] for m in MODELOS])
        ax.set_title(corpus); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.subplots_adjust(left=0.06, right=0.9, top=0.82, bottom=0.12, wspace=0.08)
    cb = fig.colorbar(im, cax=fig.add_axes([0.92, 0.15, 0.012, 0.6])); cb.set_label("F1 macro")
    fig.suptitle("F1 macro de las 36 combinaciones (* = diferencia distinguible frente al baseline de OE2 del mismo modelo)", fontsize=10, y=0.98)
    guardar(fig, "fig_oe4_1_f1_36_combinaciones.png", ajustar=False)


def fig_diferencias(T):
    sub = T[T.pooling != EXTRACCION_OE2].copy()
    sub["etiqueta"] = sub.modelo.map(NOM) + " | " + sub.pooling.map(NOM_POOL) + " | " + sub.corpus
    sub["orden_modelo"] = sub.modelo.map({m: i for i, m in enumerate(MODELOS)})
    sub = sub.sort_values(["orden_modelo", "dif_vs_baseline_oe2"])
    fig, ax = plt.subplots(figsize=(7.5, 9.5))
    y = np.arange(len(sub))[::-1]
    for yi, (_, r) in zip(y, sub.iterrows()):
        c = COLOR[r.modelo]
        ax.errorbar(r.dif_vs_baseline_oe2, yi, xerr=[[r.dif_vs_baseline_oe2 - r.dif_baseline_ic95_bajo], [r.dif_baseline_ic95_alto - r.dif_vs_baseline_oe2]],
                    fmt="o" if r.distinguible_vs_baseline else "s", color=c, mfc=c if r.distinguible_vs_baseline else "white", capsize=2, markersize=5)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(sub.etiqueta, fontsize=7)
    ax.set_xlabel("Diferencia de F1 frente al baseline de OE2 (IC 95% pareado)")
    ax.set_title("Punto relleno = distinguible de cero", fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    guardar(fig, "fig_oe4_2_diferencias_vs_baseline.png")


def fig_intrinseco(T):
    sub = T[T.silueta_coseno.notna()]
    marcas = {"sin aumento": "o", "Retrotraduccion": "^", "GtR": "s"}
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for m in MODELOS:
        for c, mk in marcas.items():
            x = sub[(sub.modelo == m) & (sub.corpus == c)]
            ax.scatter(x.silueta_coseno, x.f1_macro, color=COLOR[m], marker=mk, s=45, alpha=0.85, label=f"{NOM[m]} - {c}")
    rho = spearmanr(sub.silueta_coseno, sub.f1_macro)[0]
    ax.set_xlabel("Silueta (distancia coseno, evaluacion intrinseca)"); ax.set_ylabel("F1 macro (evaluacion extrinseca)")
    ax.set_title(f"Intrinseco vs. extrinseco, 36 combinaciones (Spearman = {rho:.2f})", fontsize=10)
    ax.grid(alpha=0.25)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, fontsize=6.5, ncol=3, loc="lower right", frameon=False)
    guardar(fig, "fig_oe4_3_intrinseco_vs_extrinseco.png")


def fig_enfoques(E):
    E = E.copy()
    fig, ax = plt.subplots(figsize=(7.6, 0.36 * len(E) + 1.2))
    y = np.arange(len(E))[::-1]
    for yi, (_, r) in zip(y, E.iterrows()):
        ax.barh(yi, r.f1_macro, color=r.color, height=0.65)
        ax.errorbar(r.f1_macro, yi, xerr=[[r.f1_macro - r.ic95_bajo], [r.ic95_alto - r.f1_macro]], color="k", capsize=2, lw=0.9)
        ax.text(r.ic95_alto + 0.006, yi, f"{r.f1_macro:.3f}", va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(E.enfoque, fontsize=8)
    ax.set_xlim(0.45, 0.82); ax.set_xlabel("F1 macro (IC 95% bootstrap)"); ax.grid(axis="x", alpha=0.25)
    guardar(fig, "fig_oe4_4_enfoques.png")


def fig_por_intencion(P):
    fig, ax = plt.subplots(figsize=(0.95 * len(P.columns) + 1.5, 3.8))
    im = ax.imshow(P.values, cmap="viridis", vmin=0.3, vmax=0.9, aspect="auto")
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            ax.text(j, i, f"{P.values[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    ax.set_xticks(range(P.shape[1])); ax.set_xticklabels(P.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(P.shape[0])); ax.set_yticklabels(P.index); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.85).set_label("F1 por intencion")
    guardar(fig, "fig_oe4_5_f1_por_intencion.png")


def fig_confusion(cm, titulo, nombre):
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.8, 5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(K):
        for j in range(K):
            v = cm[i, j]
            cuenta = f"{v:.0f}" if abs(v - round(v)) < 1e-9 else f"{v:.1f}"
            ax.text(j, i, f"{cmn[i, j]:.2f}\n({cuenta})", ha="center", va="center", fontsize=7.5, color="white" if cmn[i, j] > 0.45 else "black")
    ax.set_xticks(range(K)); ax.set_xticklabels(CLASES, rotation=30, ha="right"); ax.set_yticks(range(K)); ax.set_yticklabels(CLASES)
    ax.set_xlabel("Prediccion"); ax.set_ylabel("Categoria real"); ax.set_title(titulo, fontsize=10); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.8).set_label("Proporcion por categoria real")
    guardar(fig, nombre)


# --------------------------------------------------------------------------------------------------- principal
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recalcular-ngramas", action="store_true", help="vuelve a calcular el clasificador de n-gramas en vez de leer la cache")
    ap.add_argument("--repeticiones", type=int, default=2000, help="remuestreos bootstrap (por defecto 2000)")
    args = ap.parse_args()
    rng = np.random.default_rng(SEMILLA_BOOTSTRAP)
    R = args.repeticiones
    SINTESIS_RESULTADOS.mkdir(parents=True, exist_ok=True)

    oe2, pool, ft = cargar_oe2(), cargar_pooling(), cargar_finetuning()
    base_df = oe2[("mbert", "sin_aumento")]
    y_txt = base_df["real"].to_numpy()
    y = a_indices(y_txt)
    assert len(y) == 700
    for g in list(oe2.values()) + list(pool.values()):
        assert (g["real"].to_numpy() == y_txt).all(), "las predicciones no estan alineadas con el corpus"
    P_NG = ngramas_de_letras(y_txt, args.recalcular_ngramas)

    # ---------------------------------------------------------------- baselines de OE2 (extraccion de fabrica, sin aumento)
    P_BASE = {m: a_indices(oe2[(m, "sin_aumento")]["prediccion"]) for m in MODELOS}

    # ---------------------------------------------------------------- tabla de las 36 combinaciones (+ Mixup de OE2)
    intr = []
    for nombre, carpeta in CARPETA_INTRINSECA.items():
        d = pd.read_csv(CARACTERIZACION_RESULTADOS / carpeta / "metricas_intrinsecas.csv").rename(columns={"estrategia": "pooling"})
        d["corpus"] = nombre
        intr.append(d)
    INTR = pd.concat(intr)
    PRED = {}
    filas = []
    for (m, p, c), g in pool.items():
        pr = a_indices(g["prediccion"]); PRED[(m, p, c)] = pr
        lo, hi = ic_f1(y, pr, rng, R)
        dif_b = dif_pareada(y, pr, P_BASE[m], rng, R)
        if c == "sin aumento":
            dif_s = (np.nan, np.nan, np.nan, False)
        else:
            dif_s = dif_pareada(y, pr, a_indices(pool[(m, p, "sin aumento")]["prediccion"]), rng, R)
        filas.append({"modelo": m, "pooling": p, "corpus": c, **metricas(y, pr), "f1_ic95_bajo": lo, "f1_ic95_alto": hi,
                      "dif_vs_baseline_oe2": dif_b[0], "dif_baseline_ic95_bajo": dif_b[1], "dif_baseline_ic95_alto": dif_b[2], "distinguible_vs_baseline": dif_b[3],
                      "dif_vs_mismo_pooling_sin_aumento": dif_s[0], "dif_mismo_pooling_ic95_bajo": dif_s[1], "dif_mismo_pooling_ic95_alto": dif_s[2], "distinguible_vs_mismo_pooling": dif_s[3]})
    for m in MODELOS:
        pr = a_indices(oe2[(m, "mixup")]["prediccion"]); lo, hi = ic_f1(y, pr, rng, R)
        dif_b = dif_pareada(y, pr, P_BASE[m], rng, R)
        filas.append({"modelo": m, "pooling": EXTRACCION_OE2, "corpus": "Mixup", **metricas(y, pr), "f1_ic95_bajo": lo, "f1_ic95_alto": hi,
                      "dif_vs_baseline_oe2": dif_b[0], "dif_baseline_ic95_bajo": dif_b[1], "dif_baseline_ic95_alto": dif_b[2], "distinguible_vs_baseline": dif_b[3]})
    T = pd.DataFrame(filas).merge(INTR[["modelo", "pooling", "corpus", "silueta_coseno", "davies_bouldin", "calinski_harabasz"]], on=["modelo", "pooling", "corpus"], how="left")
    T.to_csv(SINTESIS_RESULTADOS / "tabla_36_combinaciones.csv", index=False, encoding="utf-8")
    T36 = T[T.pooling != EXTRACCION_OE2]
    print(f"tabla_36_combinaciones.csv ({len(T36)} combinaciones + {len(T) - len(T36)} de Mixup)")

    # ---------------------------------------------------------------- mejores por modelo (sin aumento): F1 vs silueta
    mejores = []
    for m in MODELOS:
        x = T36[(T36.modelo == m) & (T36.corpus == "sin aumento")]
        mf, ms = x.loc[x.f1_macro.idxmax()], x.loc[x.silueta_coseno.idxmax()]
        mejores.append({"modelo": m, "mejor_pooling_por_f1": mf.pooling, "f1_mejor": mf.f1_macro, "mejor_pooling_por_silueta": ms.pooling, "silueta_mejor": ms.silueta_coseno,
                        "coinciden": bool(mf.pooling == ms.pooling), "f1_baseline_oe2": f1_macro(y, P_BASE[m])})
    pd.DataFrame(mejores).to_csv(SINTESIS_RESULTADOS / "mejores_por_modelo.csv", index=False, encoding="utf-8")

    # ---------------------------------------------------------------- las mejores configuraciones congeladas frente a la mejor
    top = T36.sort_values("f1_macro", ascending=False).head(8).reset_index(drop=True)
    pm = PRED[(top.modelo[0], top.pooling[0], top.corpus[0])]
    cmp = []
    for i in range(1, len(top)):
        r = top.loc[i]
        d = dif_pareada(y, pm, PRED[(r.modelo, r.pooling, r.corpus)], rng, R)
        cmp.append({"mejor": f"{top.modelo[0]} | {top.pooling[0]} | {top.corpus[0]}", "f1_mejor": top.f1_macro[0], "otra": f"{r.modelo} | {r.pooling} | {r.corpus}", "f1_otra": r.f1_macro,
                    "dif_mejor_menos_otra": d[0], "ic95_bajo": d[1], "ic95_alto": d[2], "distinguible_de_cero": d[3]})
    pd.DataFrame(cmp).to_csv(SINTESIS_RESULTADOS / "top_configuraciones_vs_mejor.csv", index=False, encoding="utf-8")
    mejor_cong = (top.modelo[0], top.pooling[0], top.corpus[0])

    # ---------------------------------------------------------------- correlaciones intrinseco / extrinseco
    corr = []
    for ambito, x in [("las 36", T36), ("solo sin aumento (12)", T36[T36.corpus == "sin aumento"])] + [(f"solo {NOM[m]} (12)", T36[T36.modelo == m]) for m in MODELOS]:
        for col in ("silueta_coseno", "davies_bouldin", "calinski_harabasz"):
            corr.append({"ambito": ambito, "metrica_intrinseca": col, "n": len(x), "spearman_con_f1": float(spearmanr(x[col], x.f1_macro)[0])})
    pd.DataFrame(corr).to_csv(SINTESIS_RESULTADOS / "correlaciones_intrinseco_extrinseco.csv", index=False, encoding="utf-8")

    # ---------------------------------------------------------------- todos los enfoques en una tabla
    enfoques = []   # (nombre, color, predicciones (semillas x 700), grupo)

    def agregar(nombre, color, preds, grupo):
        P = np.atleast_2d(preds)
        lo, hi = ic_f1(y, P, rng, R)
        enfoques.append({"enfoque": nombre, "grupo": grupo, "color": color, "f1_macro": float(np.mean([f1_macro(y, p) for p in P])), "ic95_bajo": lo, "ic95_alto": hi,
                         "semillas": len(P), "_P": P})

    agregar("N-gramas de letras (sin modelo de lenguaje)", "#6a1b9a", P_NG, "referencia")
    for m in MODELOS:
        agregar(f"{NOM[m]} congelado (baseline OE2)", COLOR[m], P_BASE[m], "congelado")
    for m in MODELOS:
        x = T36[(T36.modelo == m) & (T36.corpus == "sin aumento")]; r = x.loc[x.f1_macro.idxmax()]
        agregar(f"{NOM[m]} congelado, mejor pooling ({NOM_POOL[r.pooling]})", COLOR[m], PRED[(m, r.pooling, "sin aumento")], "congelado_mejor_pooling")
    for m in MODELOS:
        x = T36[(T36.modelo == m) & (T36.corpus == "GtR")]; r = x.loc[x.f1_macro.idxmax()]
        agregar(f"{NOM[m]} congelado + GtR ({NOM_POOL[r.pooling]})", COLOR[m], PRED[(m, r.pooling, "GtR")], "congelado_gtr")
    etiqueta_ft = {("mbert", "completo"): "mBERT fine-tuning completo", ("labse", "completo"): "LaBSE fine-tuning completo", ("xlmr", "completo"): "XLM-R fine-tuning (rejilla comun, no convergio)",
                   ("xlmr_rejilla2", "completo"): "XLM-R fine-tuning (rejilla corregida)", ("mbert", "lora"): "mBERT LoRA (no concluyente)"}
    for clave, P in ft.items():
        agregar(etiqueta_ft.get(clave, f"{clave[0]} {clave[1]}"), COLOR[clave[0].split("_")[0]], P, "fine_tuning")
    E = pd.DataFrame(enfoques)
    E.drop(columns=["_P", "color"]).to_csv(SINTESIS_RESULTADOS / "enfoques_f1.csv", index=False, encoding="utf-8")

    # ---------------------------------------------------------------- fine-tuning frente al baseline congelado y frente a los n-gramas
    filas_ft = []
    for (modelo, variante), P in ft.items():
        m = modelo.split("_")[0]
        d_b = dif_pareada(y, P, P_BASE[m], rng, R)
        d_n = dif_pareada(y, P, P_NG, rng, R)
        f1s = [f1_macro(y, p) for p in P]
        filas_ft.append({"modelo": modelo, "variante": variante, "semillas": len(P), "f1_baseline_congelado": f1_macro(y, P_BASE[m]), "f1_finetuning_media": float(np.mean(f1s)),
                         "f1_sd_entre_semillas": float(np.std(f1s)), "f1_minimo": min(f1s), "f1_maximo": max(f1s),
                         "dif_vs_baseline": d_b[0], "dif_baseline_ic95_bajo": d_b[1], "dif_baseline_ic95_alto": d_b[2], "distinguible_vs_baseline": d_b[3],
                         "dif_vs_ngramas": d_n[0], "dif_ngramas_ic95_bajo": d_n[1], "dif_ngramas_ic95_alto": d_n[2], "distinguible_vs_ngramas": d_n[3]})
    pd.DataFrame(filas_ft).to_csv(SINTESIS_RESULTADOS / "finetuning_vs_congelado.csv", index=False, encoding="utf-8")

    # ---------------------------------------------------------------- F1 por intencion
    PI = {}
    for _, r in E.iterrows():
        if r.grupo in ("referencia", "congelado", "fine_tuning"):
            PI[r.enfoque] = np.mean([f1_por_clase(y, p) for p in r["_P"]], axis=0)
    P_INT = pd.DataFrame(PI, index=CLASES)
    P_INT.index.name = "intencion"
    P_INT.to_csv(SINTESIS_RESULTADOS / "f1_por_intencion_enfoques.csv", encoding="utf-8")

    # ---------------------------------------------------------------- matrices de confusion
    cm_cong = matriz_confusion(y, PRED[mejor_cong])
    nombre_cong = f"{NOM[mejor_cong[0]]} {NOM_POOL[mejor_cong[1]]} {mejor_cong[2]}"
    pd.DataFrame(cm_cong, index=CLASES, columns=CLASES).to_csv(SINTESIS_RESULTADOS / "matriz_confusion_mejor_congelada.csv", encoding="utf-8")
    cm_ft = None
    if ("mbert", "completo") in ft:
        cm_ft = sum(matriz_confusion(y, p) for p in ft[("mbert", "completo")]) / len(ft[("mbert", "completo")])
        pd.DataFrame(cm_ft, index=CLASES, columns=CLASES).to_csv(SINTESIS_RESULTADOS / "matriz_confusion_mbert_finetuning.csv", encoding="utf-8")

    # ---------------------------------------------------------------- DES: oraciones que ningun metodo clasifica bien
    from shiwilu.clasificacion import cargar_corpus
    corpus = cargar_corpus()
    metodos = {"baseline mBERT": P_BASE["mbert"][None, :], "n-gramas": P_NG[None, :]}
    for clave in (("mbert", "completo"), ("labse", "completo")):
        if clave in ft:
            metodos[f"{clave[0]} fine-tuning"] = ft[clave]
    des = np.where(y == CLASES.index("DES"))[0]
    filas_des = []
    for i in des:
        votos = [p[i] for P in metodos.values() for p in P]
        falla_todo = all(p[i] != y[i] for P in metodos.values() for p in P)
        filas_des.append({"pos": int(i), "espanol": corpus.loc[i, "espanol"], "shiwilu": corpus.loc[i, "shiwilu"], "fallan_todos": falla_todo,
                          "intencion_mas_votada": CLASES[int(np.bincount(votos, minlength=K).argmax())],
                          **{f"acierta_{n}": float(np.mean([p[i] == y[i] for p in P])) for n, P in metodos.items()}})
    pd.DataFrame(filas_des).sort_values("fallan_todos", ascending=False).to_csv(SINTESIS_RESULTADOS / "des_oraciones_dificiles.csv", index=False, encoding="utf-8")

    # ---------------------------------------------------------------- figuras
    fig_f1_36(T)
    fig_diferencias(T)
    fig_intrinseco(T)
    fig_enfoques(E[E.grupo.isin(["referencia", "congelado", "congelado_mejor_pooling", "fine_tuning"])].reset_index(drop=True))
    fig_por_intencion(P_INT)
    fig_confusion(cm_cong, f"Matriz de confusion: {nombre_cong} (mejor configuracion congelada)", "fig_oe4_6a_confusion_mejor_congelada.png")
    if cm_ft is not None:
        fig_confusion(cm_ft, "Matriz de confusion: mBERT con fine-tuning (promedio de 3 semillas)", "fig_oe4_6b_confusion_mbert_finetuning.png")

    # ---------------------------------------------------------------- resumen en pantalla
    pd.set_option("display.width", 220)
    print("\nEnfoques (F1 macro):")
    print(E[["enfoque", "f1_macro", "ic95_bajo", "ic95_alto"]].round(3).to_string(index=False))
    print("\nFine-tuning vs congelado:")
    print(pd.DataFrame(filas_ft)[["modelo", "variante", "f1_baseline_congelado", "f1_finetuning_media", "dif_vs_baseline", "dif_baseline_ic95_bajo", "dif_baseline_ic95_alto", "dif_vs_ngramas"]].round(3).to_string(index=False))
    print(f"\nSalidas en {SINTESIS_RESULTADOS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
