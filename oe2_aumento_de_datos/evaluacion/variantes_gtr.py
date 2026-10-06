"""
Variantes de seleccion del sintetico de Generate-then-Refine (GtR): balanceo, filtros de calidad y exclusion de DES.

Motivo (ver evaluacion/diagnostico_baseline/README.md): GtR 120 aporta un sintetico muy desbalanceado (mucho DES, SAL y NEG; poco EMO, AFI y REQUEST) y el de
DES, PRG, SAL y NEG perjudica a su propia categoria. Aqui se prueba si SELECCIONAR el sintetico (en vez de usar todo lo aprobado) mejora al baseline.

NO USA LA API: todo sale de las caches ya generadas (tecnicas_aumento/salidas/en_linea_marcadores_fuentes/generate_then_refine/: pool de cada fold y sus 5
particiones internas). Usa el protocolo B (C elegido por CV interna K=5 con los sinteticos de cada particion, 5 folds congelados, texto normalizado), asi que
las cifras son comparables con las de los 12 experimentos: la variante `ref_sin_aumento` debe dar 0.665 en mBERT y `ref_gtr_completo` 0.652.

Variantes (n = sinteticos por clase tras filtrar; si una clase tiene menos, se usan todos):
  ref_sin_aumento            solo reales (referencia)
  ref_gtr_completo           todo lo aprobado de GtR 120 (referencia)
  v1_c40_bal                 40 por clase, balanceado
  v2_c40_bal_sim060          40 por clase con similitud_labse >= 0.60
  v3_c40_bal_sim060_sinDES   idem, sin sintetico de DES
  v4_c20_bal_sim060          20 por clase con similitud_labse >= 0.60
  v5_c20_AFI_PRG_REQUEST_sim060   solo AFI, PRG y REQUEST (20 por clase, similitud >= 0.60)
  v6_c40_bal_margen          40 por clase, solo sinteticos mas cerca de su clase que de otra por un margen >= 0.05 (centroides LaBSE de las reales del entrenamiento)
  v7_c40_bal_margen_borde    40 por clase, solo los que estan del lado correcto pero cerca de la frontera (0 < margen < 0.05)
  v8_c40_bal_sinDES          40 por clase, sin DES (sin filtro de similitud)
Extras (--extras): v2_sim050, v2_sim055, v2_sim065 (umbrales de similitud) y v9_c20_bal, v10_c10_bal, v11_c5_bal (tamaño).

Nota de reproducibilidad: el C elegido por CV interna es sensible a diferencias numericas minimas (p. ej. Colab/Linux frente a Windows), asi que `ref_gtr_completo`
puede diferir ~0.003 del 0.652 oficial (en esta maquina 0.654: 4 de 5 folds eligen el mismo C); `ref_sin_aumento` reproduce 0.665 exacto. Las diferencias entre
variantes menores de ~0.005 no son interpretables.

Criterio de exito (en mBERT): F1 macro >= baseline, DES no cae mas de 0.01 y ninguna clase cae mas de 0.02 frente al baseline.

Salida: evaluacion/resultados/variantes_gtr/{resumen_variantes, f1_por_clase, conteos_por_clase}.csv y pred_<variante>.csv (se reanuda si se corta)
Uso (desde la raiz del repositorio):
    python oe2_aumento_de_datos/evaluacion/variantes_gtr.py                              # mBERT, 2 muestras aleatorias, 10 variantes
    python oe2_aumento_de_datos/evaluacion/variantes_gtr.py --modelos mbert labse xlmr --semillas 0 1 2
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import _ruta_raiz  # noqa: F401  (deja importable el paquete `shiwilu`)

import validacion_cruzada as vc  # noqa: E402
import validacion_cruzada_cv_interna as cv  # noqa: E402
import validacion_cruzada_en_linea as ven  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from shiwilu.clasificacion import (  # noqa: E402
    MODELOS, SEMILLA, _normalizar_estricto, cargar_corpus, cargar_corpus_normalizado, cargar_folds, extraer_embeddings,
)
from shiwilu.rutas import AUMENTO_SALIDA, EVALUACION_RESULTADOS  # noqa: E402
from shiwilu.taxonomia import INTENCIONES  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)
CACHE = AUMENTO_SALIDA / "en_linea_marcadores_fuentes" / "generate_then_refine"
SALIDA = EVALUACION_RESULTADOS / "variantes_gtr"
CLASES = list(INTENCIONES)

VARIANTES = {
    "ref_sin_aumento": {"nada": True},
    "ref_gtr_completo": {},
    "v1_c40_bal": {"n": 40},
    "v2_c40_bal_sim060": {"n": 40, "sim": 0.60},
    "v3_c40_bal_sim060_sinDES": {"n": 40, "sim": 0.60, "excluir": ["DES"]},
    "v4_c20_bal_sim060": {"n": 20, "sim": 0.60},
    "v5_c20_AFI_PRG_REQUEST_sim060": {"n": 20, "sim": 0.60, "solo": ["AFI", "PRG", "REQUEST"]},
    "v6_c40_bal_margen": {"n": 40, "margen": ("min", 0.05)},
    "v7_c40_bal_margen_borde": {"n": 40, "margen": ("borde", 0.05)},
    "v8_c40_bal_sinDES": {"n": 40, "excluir": ["DES"]},
}
EXTRAS = {
    "v2_sim050": {"n": 40, "sim": 0.50}, "v2_sim055": {"n": 40, "sim": 0.55}, "v2_sim065": {"n": 40, "sim": 0.65},
    "v9_c20_bal": {"n": 20}, "v10_c10_bal": {"n": 10}, "v11_c5_bal": {"n": 5},
}


# ----------------------------------------------------------------------------------------------------- datos
class Sint:
    """Sinteticos aprobados de una particion (DataFrame) y sus embeddings (por modelo y LaBSE)."""

    def __init__(self, df):
        self.df = df
        self.X, self.L = {}, None


def leer_cache(f, sufijo, claves_train, claves_excluir):
    nombres = [f"fold{f}_{sufijo}.csv"] + [f"fold{f}_{sufijo}_l{j}.csv" for j in range(1, 6)]
    faltan = [n for n in nombres if not (CACHE / n).exists()]
    if faltan:
        raise SystemExit(f"Falta la cache de GtR: {faltan[0]} (en {CACHE}). Este script no genera nada; trae esas carpetas del repositorio.")
    partes = [pd.read_csv(CACHE / n) for n in nombres]
    with contextlib.redirect_stdout(io.StringIO()):
        g = ven.combinar_sin_duplicados(partes, set(claves_train))
    ap = g[g["estado_filtro"] == "aprobado"].reset_index(drop=True)
    copia = ap["shiwilu"].map(_normalizar_estricto).isin(claves_excluir).to_numpy()
    return ap[~copia].reset_index(drop=True)


def preparar(corpus, y, folds, claves, tx, modelos, necesita_margen):
    """Lee las caches y calcula los embeddings de cada conjunto sintetico POR SEPARADO, igual que la evaluacion oficial (mismo agrupamiento
    en lotes), para que las referencias reproduzcan exactamente los resultados de OE2."""
    datos = {}
    for f in sorted(set(folds)):
        idx_test = np.where(folds == f)[0]; idx_pool = np.where(folds != f)[0]
        claves_test = set(claves[idx_test])
        partes = cv.particiones_internas(idx_pool, y, claves, f, cv.K_INTERNO)
        inner = [Sint(leer_cache(f, f"in{k}", claves[tr], claves_test | set(claves[va]))) for k, (tr, va) in enumerate(partes)]
        pool = Sint(leer_cache(f, "pool", claves[idx_pool], claves_test))
        datos[f] = {"test": idx_test, "pool": idx_pool, "partes": partes, "inner": inner, "sint_pool": pool}
    conjuntos = [(f, s) for f, d in datos.items() for s in d["inner"] + [d["sint_pool"]]]
    print(f"{len(conjuntos)} conjuntos sinteticos; {sum(len(s.df) for _, s in conjuntos)} oraciones por embeber por modelo", flush=True)
    for i, (f, s) in enumerate(conjuntos):
        t = [tx(x) for x in s.df["shiwilu"]]
        for m in modelos:
            s.X[m] = extraer_embeddings(t, m) if t else np.zeros((0, 768))
        if "labse" in modelos:
            s.L = s.X["labse"]
        elif necesita_margen:
            s.L = extraer_embeddings(t, "labse") if t else np.zeros((0, 768))
        if (i + 1) % 6 == 0:
            print(f"  embeddings: fold {f} listo", flush=True)
    return datos


# ----------------------------------------------------------------------------------------------- seleccion
def margen_a_la_frontera(L_real, y_real, L_sint, y_sint):
    """Coseno a su centroide de clase menos el maximo coseno a otro centroide (centroides LaBSE de las REALES del entrenamiento)."""
    n = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)
    cen = n(np.stack([n(L_real[y_real == c]).mean(axis=0) for c in CLASES]))
    cos = n(L_sint) @ cen.T
    propio = cos[np.arange(len(y_sint)), [CLASES.index(c) for c in y_sint]]
    cos[np.arange(len(y_sint)), [CLASES.index(c) for c in y_sint]] = -np.inf
    return propio - cos.max(axis=1)


def seleccionar(S, cfg, L_real, y_real, rng):
    """Posiciones (filas de S.df) que se conservan segun `cfg`."""
    if cfg.get("nada") or len(S.df) == 0:
        return np.array([], dtype=int)
    clase = S.df["intencion"].to_numpy()
    keep = np.ones(len(S.df), bool)
    if cfg.get("excluir"):
        keep &= ~np.isin(clase, cfg["excluir"])
    if cfg.get("solo"):
        keep &= np.isin(clase, cfg["solo"])
    if cfg.get("sim") is not None:
        keep &= S.df["similitud_labse"].to_numpy() >= cfg["sim"]
    if cfg.get("margen"):
        modo, tau = cfg["margen"]
        m = margen_a_la_frontera(L_real, y_real, S.L, clase)
        keep &= (m >= tau) if modo == "min" else ((m > 0) & (m < tau))
    idx = np.where(keep)[0]
    n = cfg.get("n")
    if n:
        elegidos = []
        for c in CLASES:
            ic = idx[clase[idx] == c]
            elegidos.append(ic if len(ic) <= n else rng.choice(ic, n, replace=False))
        idx = np.sort(np.concatenate(elegidos))
    return idx


# ------------------------------------------------------------------------------------------- evaluacion
def ajustar(X, y, C):
    return LogisticRegression(C=C, max_iter=2000, random_state=SEMILLA).fit(X, y)


def evaluar_fold(f, d, modelo, cfg, semilla, E, L_real, y):
    rng = np.random.default_rng(SEMILLA + 1000 * semilla + f)
    idx_pool, idx_test = d["pool"], d["test"]
    pos = {int(i): n for n, i in enumerate(idx_pool)}
    pred = {c: np.empty(len(idx_pool), dtype=object) for c in cv.VALORES_C_CV}
    for (tr, va), S in zip(d["partes"], d["inner"]):
        sel = seleccionar(S, cfg, L_real[tr], y[tr], rng)
        X = np.vstack([E[tr], S.X[modelo][sel]]) if len(sel) else E[tr]
        yy = np.concatenate([y[tr], S.df["intencion"].to_numpy()[sel]]) if len(sel) else y[tr]
        for c in cv.VALORES_C_CV:
            pred[c][[pos[int(i)] for i in va]] = ajustar(X, yy, c).predict(E[va])
    f1 = {c: vc.f1m(y[idx_pool], pred[c]) for c in cv.VALORES_C_CV}
    C = max(cv.VALORES_C_CV, key=lambda c: (f1[c], -c))
    S = d["sint_pool"]
    sel = seleccionar(S, cfg, L_real[idx_pool], y[idx_pool], rng)
    X = np.vstack([E[idx_pool], S.X[modelo][sel]]) if len(sel) else E[idx_pool]
    yy = np.concatenate([y[idx_pool], S.df["intencion"].to_numpy()[sel]]) if len(sel) else y[idx_pool]
    p = ajustar(X, yy, C).predict(E[idx_test])
    conteo = pd.Series(S.df["intencion"].to_numpy()[sel]).value_counts().reindex(CLASES, fill_value=0)
    filas = [{"modelo": modelo, "semilla": semilla, "fold": f, "pos": int(i), "real": y[i], "prediccion": q, "C": C} for i, q in zip(idx_test, p)]
    return filas, {"modelo": modelo, "semilla": semilla, "fold": f, **{c: int(conteo[c]) for c in CLASES}}


def f1_rapido(yi, pi, k):
    conf = np.bincount(yi * k + pi, minlength=k * k).reshape(k, k)
    tp = np.diag(conf).astype(float)
    den = conf.sum(axis=0) + conf.sum(axis=1)
    return float(np.mean(np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelos", nargs="+", choices=list(MODELOS), default=["mbert"])
    ap.add_argument("--semillas", nargs="+", type=int, default=[0, 1], help="Muestras aleatorias del balanceo (las variantes sin muestreo no cambian).")
    ap.add_argument("--variantes", nargs="+", default=None, help="Subconjunto (por defecto, las 10 principales).")
    ap.add_argument("--extras", action="store_true", help="Agrega los umbrales de similitud 0.50/0.55/0.65 y los tamaños 20/10/5.")
    ap.add_argument("--n-jobs", type=int, default=-1, help="Hilos para evaluar los folds en paralelo.")
    ap.add_argument("--rehacer", action="store_true", help="Recalcula aunque exista su archivo pred_<variante>.csv.")
    args = ap.parse_args()
    todas = {**VARIANTES, **EXTRAS} if args.extras else dict(VARIANTES)
    nombres = args.variantes or list(todas)
    desconocidas = [v for v in nombres if v not in {**VARIANTES, **EXTRAS}]
    if desconocidas:
        raise SystemExit(f"Variantes desconocidas: {desconocidas}")
    modelos = list(args.modelos)
    todas_cfg_ = {**VARIANTES, **EXTRAS}

    corpus = cargar_corpus()
    y = corpus["intencion"].to_numpy()
    folds = cargar_folds(corpus)
    texto = cargar_corpus_normalizado()["shiwilu"].astype(str).tolist()
    claves = np.array([_normalizar_estricto(t) for t in texto])
    tx = vc.CONDICIONES["sin_puntuacion"]
    SALIDA.mkdir(parents=True, exist_ok=True)
    pendientes = [n for n in nombres if args.rehacer or not (SALIDA / f"pred_{n}.csv").exists()]
    E = datos = None
    if pendientes:   # si todo esta calculado se salta el costo de los embeddings y se pasa directo al resumen
        E = {m: extraer_embeddings([tx(t) for t in texto], m) for m in dict.fromkeys(["labse"] + modelos)}
        necesita_margen = any(todas_cfg_[v].get("margen") for v in pendientes)
        datos = preparar(corpus, y, folds, claves, tx, modelos, necesita_margen)

    todas_cfg = {**VARIANTES, **EXTRAS}
    for nombre in nombres:
        archivo = SALIDA / f"pred_{nombre}.csv"
        if archivo.exists() and not args.rehacer:
            print(f"[reutiliza] {nombre}", flush=True)
            continue
        cfg = todas_cfg[nombre]
        trabajos = [(f, m, s) for s in (args.semillas if cfg.get("n") else [args.semillas[0]]) for m in modelos for f in sorted(datos)]
        res = Parallel(n_jobs=args.n_jobs, backend="threading")(
            delayed(evaluar_fold)(f, datos[f], m, cfg, s, E[m], E["labse"], y) for f, m, s in trabajos)
        pd.DataFrame([r for filas, _ in res for r in filas]).to_csv(archivo, index=False, encoding="utf-8")
        pd.DataFrame([c for _, c in res]).to_csv(SALIDA / f"conteo_{nombre}.csv", index=False, encoding="utf-8")
        p = pd.read_csv(archivo)
        print("OK", nombre, {m: round(float(np.mean([vc.f1m(g.real, g.prediccion) for _, g in x.groupby("semilla")])), 4) for m, x in p.groupby("modelo")}, flush=True)

    # ---------------------------------------------------------------- resumen
    P = {n: pd.read_csv(SALIDA / f"pred_{n}.csv") for n in todas_cfg if (SALIDA / f"pred_{n}.csv").exists()}
    if "ref_sin_aumento" not in P:
        print("Falta ref_sin_aumento: no se puede comparar."); return 0
    k = len(CLASES); cod = {c: i for i, c in enumerate(CLASES)}
    rng = np.random.default_rng(SEMILLA)
    filas, por_clase, conteos = [], [], []
    for nombre, p in P.items():
        for m, g in p.groupby("modelo"):
            semillas = sorted(g.semilla.unique())
            f1s = [vc.f1m(x.sort_values("pos").real, x.sort_values("pos").prediccion) for _, x in g.groupby("semilla")]
            cl = np.mean([f1_score(x.real, x.prediccion, average=None, labels=CLASES, zero_division=0) for _, x in g.groupby("semilla")], axis=0)
            por_clase.append({"variante": nombre, "modelo": m, **{c: float(v) for c, v in zip(CLASES, cl)}})
            fila = {"variante": nombre, "modelo": m, "f1_macro": float(np.mean(f1s)), "sd_entre_semillas": float(np.std(f1s)) if len(f1s) > 1 else np.nan}
            b = P["ref_sin_aumento"][P["ref_sin_aumento"].modelo == m].sort_values("pos")
            yi = np.array([cod[v] for v in b["real"]]); pb = np.array([cod[v] for v in b["prediccion"]]); n = len(yi)
            pas = [np.array([cod[v] for v in x.sort_values("pos")["prediccion"]]) for _, x in g.groupby("semilla")]
            if nombre != "ref_sin_aumento":
                d0 = np.mean([f1_rapido(yi, a, k) - f1_rapido(yi, pb, k) for a in pas])
                difs = []
                for _ in range(1000):
                    i = rng.integers(0, n, n)
                    difs.append(np.mean([f1_rapido(yi[i], a[i], k) - f1_rapido(yi[i], pb[i], k) for a in pas]))
                lo, hi = np.percentile(difs, [2.5, 97.5])
                fila.update({"dif_vs_sin_aumento": float(d0), "ic95_bajo": float(lo), "ic95_alto": float(hi), "distinguible_de_cero": not (lo <= 0 <= hi)})
            filas.append(fila)
            cnt = pd.read_csv(SALIDA / f"conteo_{nombre}.csv") if (SALIDA / f"conteo_{nombre}.csv").exists() else None
            if cnt is not None:
                conteos.append({"variante": nombre, "modelo": m, **cnt[cnt.modelo == m][CLASES].mean().round(1).to_dict()})
    R = pd.DataFrame(filas); PC = pd.DataFrame(por_clase)
    base = PC[PC.variante == "ref_sin_aumento"].set_index("modelo")[CLASES]
    for i, r in PC.iterrows():
        delta = r[CLASES].astype(float) - base.loc[r["modelo"]]
        j = R[(R.variante == r["variante"]) & (R.modelo == r["modelo"])].index[0]
        R.loc[j, "peor_clase"] = delta.idxmin(); R.loc[j, "peor_dif_clase"] = float(delta.min()); R.loc[j, "dif_DES"] = float(delta["DES"])
    R["cumple_criterio"] = (R["dif_vs_sin_aumento"] >= 0) & (R["dif_DES"] >= -0.01) & (R["peor_dif_clase"] >= -0.02)
    R.loc[R.variante == "ref_sin_aumento", "cumple_criterio"] = np.nan
    R.to_csv(SALIDA / "resumen_variantes.csv", index=False, encoding="utf-8")
    PC.to_csv(SALIDA / "f1_por_clase.csv", index=False, encoding="utf-8")
    pd.DataFrame(conteos).to_csv(SALIDA / "conteos_por_clase.csv", index=False, encoding="utf-8")
    pd.set_option("display.width", 220)
    print("\nF1 macro por variante (dif vs sin aumento; * = IC95% no incluye 0):")
    for m, g in R.groupby("modelo"):
        g = g.copy(); g["dif"] = [("" if pd.isna(d) else f"{d:+.3f}{'*' if s else ''}") for d, s in zip(g.get("dif_vs_sin_aumento"), g.get("distinguible_de_cero", False))]
        print(f"\n{m}"); print(g[["variante", "f1_macro", "dif", "dif_DES", "peor_clase", "peor_dif_clase", "cumple_criterio"]].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
