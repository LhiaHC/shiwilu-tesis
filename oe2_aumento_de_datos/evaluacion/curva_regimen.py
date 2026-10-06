"""
Curva de aprendizaje por REGIMEN DE DATOS: repite los experimentos con 10, 25, 50 y 80 (= todo el pool de entrenamiento) ejemplos REALES por clase.

Pregunta: ayuda el aumento cuando hay muy pocos datos reales y deja de ayudar cuando hay mas? Y si el F1 sin aumento sigue subiendo con mas datos reales
pero el sintetico no ayuda, el problema es la calidad del sintetico.

Diseño (mismos 5 folds congelados, texto normalizado, embeddings congelados, Regresion Logistica):
  - El TEST de cada fold no cambia. Del pool de entrenamiento (~80 por clase) se toman al azar N oraciones por clase (N = 10, 25, 50; en 80 se usa todo el pool).
    La misma muestra se usa para las cuatro tecnicas, asi que la comparacion es pareada.
  - El aumento se genera SOLO con esa muestra (Mixup con sus vectores; Retrotraduccion con las filas del catalogo cuyo origen esta en la muestra; GtR con Claude usando
    como ejemplos solo esas oraciones: por eso GtR hay que generarlo de nuevo en cada regimen).
  - C: se elige UNA vez por (modelo, regimen, fold) con CV interna K=5 sobre las oraciones REALES de la muestra (sin aumento) y se usa el mismo C en las cuatro
    tecnicas. (Elegir un C por tecnica requeriria generar GtR dentro de cada particion interna, 5 veces mas llamadas.)
  - Volumen de GtR por regimen: 20, 40, 80 y 120 por categoria (1, 2, 4 y 6 lotes), para que el sintetico quede en torno a 1x lo real.

Uso (desde la raiz del repositorio):
    python oe2_aumento_de_datos/evaluacion/curva_regimen.py                       # las cuatro tecnicas (GtR se omite si falta su cache; NUNCA llama a la API sin --permitir-api)
    python oe2_aumento_de_datos/evaluacion/curva_regimen.py --tecnicas generate_then_refine --anexar   # agrega GtR a una corrida previa
    python oe2_aumento_de_datos/evaluacion/curva_regimen.py --tecnicas sin_aumento mixup retrotraduccion generate_then_refine
    python oe2_aumento_de_datos/evaluacion/curva_regimen.py --solo-generar        # (Colab, requiere ANTHROPIC_API_KEY) solo genera la cache de GtR
Salida: evaluacion/resultados/curva_regimen/{predicciones, resumen_por_regimen, comparacion_pareada}.csv y curva_regimen.png
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import _ruta_raiz  # noqa: F401  (deja importable el paquete `shiwilu`)

import validacion_cruzada as vc  # noqa: E402
import validacion_cruzada_cv_interna as cv  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from shiwilu.clasificacion import (  # noqa: E402
    MODELOS, SEMILLA, _normalizar_estricto, cargar_corpus, cargar_corpus_normalizado, cargar_folds, extraer_embeddings,
)
from shiwilu.rutas import AUMENTO_SALIDA, EVALUACION_RESULTADOS  # noqa: E402
from shiwilu.taxonomia import INTENCIONES  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)
REGIMENES = [10, 25, 50, 80]
CANTIDAD_GTR = {10: 20, 25: 40, 50: 80, 80: 120}
TECNICAS = ["sin_aumento", "mixup", "retrotraduccion", "generate_then_refine"]
SALIDA = EVALUACION_RESULTADOS / "curva_regimen"
CACHE_REGIMENES = AUMENTO_SALIDA / "regimenes"                       # GtR de los regimenes 10, 25 y 50 (se genera en Colab)
CACHE_OE2 = AUMENTO_SALIDA / "en_linea_marcadores_fuentes"           # GtR de 120 por categoria del pool completo (regimen 80) y catalogo de retrotraduccion


def f1_macro_rapido(y_int, p_int, k):
    conf = np.bincount(y_int * k + p_int, minlength=k * k).reshape(k, k)
    tp = np.diag(conf).astype(float)
    den = conf.sum(axis=0) + conf.sum(axis=1)
    return float(np.mean(np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)))


def muestra(idx_pool, y, n, fold, semilla):
    """N oraciones por clase del pool (todo el pool si n >= lo disponible)."""
    rng = np.random.default_rng(SEMILLA + 1000 * semilla + fold)
    partes = []
    for c in INTENCIONES:
        idx_c = idx_pool[y[idx_pool] == c]
        partes.append(idx_c if n >= len(idx_c) else rng.choice(idx_c, n, replace=False))
    return np.sort(np.concatenate(partes))


def elegir_c(E, y, sub, claves, fold):
    """C con mejor F1 macro (CV interna K=5) sobre las oraciones reales de la muestra."""
    particiones = cv.particiones_internas(sub, y, claves, fold, 5)
    pos = {int(i): n for n, i in enumerate(sub)}
    pred = {c: np.empty(len(sub), dtype=object) for c in cv.VALORES_C_CV}
    for tr, va in particiones:
        for c in cv.VALORES_C_CV:
            pred[c][[pos[int(i)] for i in va]] = LogisticRegression(C=c, max_iter=2000, random_state=SEMILLA).fit(E[tr], y[tr]).predict(E[va])
    f1 = {c: vc.f1m(y[sub], pred[c]) for c in cv.VALORES_C_CV}
    return max(cv.VALORES_C_CV, key=lambda c: (f1[c], -c))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regimenes", nargs="+", type=int, default=REGIMENES)
    ap.add_argument("--tecnicas", nargs="+", choices=TECNICAS, default=["sin_aumento", "mixup", "retrotraduccion", "generate_then_refine"])
    ap.add_argument("--modelos", nargs="+", choices=list(MODELOS), default=list(MODELOS))
    ap.add_argument("--semillas", nargs="+", type=int, default=[0], help="Muestras aleatorias del pool (GtR solo se genera para la semilla 0).")
    ap.add_argument("--solo-generar", action="store_true", help="Solo genera con Claude la cache de GtR de los regimenes < 80 (Colab).")
    ap.add_argument("--cache-regimenes", type=Path, default=CACHE_REGIMENES)
    ap.add_argument("--anexar", action="store_true",
                    help="Une las filas nuevas a predicciones.csv ya existente (reemplaza solo las del mismo regimen y tecnica) y recalcula el resumen con todo.")
    ap.add_argument("--permitir-api", action="store_true",
                    help="Permite generar con Claude (GASTA CREDITOS) los lotes de GtR que falten. Sin esta opcion, un lote ausente se omite y NUNCA se llama a la API.")
    args = ap.parse_args()

    corpus = cargar_corpus()
    y = corpus["intencion"].to_numpy()
    folds = cargar_folds(corpus)
    texto = cargar_corpus_normalizado()["shiwilu"].astype(str).tolist()
    claves = np.array([_normalizar_estricto(t) for t in texto])
    tx = vc.CONDICIONES["sin_puntuacion"]
    modelos = list(args.modelos)
    modelos_emb = ["labse"] + [m for m in modelos if m != "labse"]
    E = {m: extraer_embeddings([tx(t) for t in texto], m) for m in modelos_emb}
    clases = list(INTENCIONES)
    cod = {c: i for i, c in enumerate(clases)}

    def sinteticos(tecnica, regimen, semilla, f, sub, claves_test):
        base = argparse.Namespace(cache=CACHE_OE2, cantidad=CANTIDAD_GTR[regimen], multiplicador=1, semilla_aumento=semilla, checkpoint=None,
                                  repo_nmt=None, excluir_categorias=[])
        if tecnica == "generate_then_refine" and regimen < 80:
            base.cache = args.cache_regimenes
        S = cv.Sinteticos(tecnica, base, corpus, y, claves, E, modelos_emb, tx, f, None)
        if tecnica == "retrotraduccion":
            S.cargar_retro(np.where(folds != f)[0])
        nombre = f"fold{f}_pool" if regimen == 80 else f"r{regimen}_s{semilla}_fold{f}"
        if tecnica == "generate_then_refine" and not args.permitir_api:
            lotes = -(-CANTIDAD_GTR[regimen] // 20)
            faltan = [n for n in [f"{nombre}.csv"] + [f"{nombre}_l{j}.csv" for j in range(1, lotes)] if not (base.cache / tecnica / n).exists()]
            if faltan:
                raise SystemExit(f"falta la cache de GtR ({faltan[0]}...): no se llama a la API sin --permitir-api")
        return S.para(sub, nombre, claves_test, semilla, modelos)

    filas = []
    for regimen in args.regimenes:
        for semilla in args.semillas:
            for f in sorted(set(folds)):
                idx_test = np.where(folds == f)[0]; idx_pool = np.where(folds != f)[0]
                sub = idx_pool if regimen >= 80 else muestra(idx_pool, y, regimen, f, semilla)
                claves_test = set(claves[idx_test])
                if args.solo_generar:
                    if regimen < 80 and semilla == 0:
                        print(f"[GtR] regimen {regimen}, fold {f}, {len(sub)} reales", flush=True)
                        sinteticos("generate_then_refine", regimen, semilla, f, sub, claves_test)
                    continue
                C = {m: elegir_c(E[m], y, sub, claves, f) for m in modelos}   # un C por (modelo, regimen, fold), con las reales de la muestra
                for tecnica in args.tecnicas:
                    if tecnica == "generate_then_refine" and semilla != 0:
                        continue
                    try:
                        sint = sinteticos(tecnica, regimen, semilla, f, sub, claves_test)
                    except SystemExit as e:
                        print(f"  [omitido] {tecnica} regimen {regimen} fold {f}: {str(e)[:80]}", flush=True)
                        continue
                    for m in modelos:
                        Xs, ys = sint[m]
                        X = np.vstack([E[m][sub], Xs]) if len(ys) else E[m][sub]
                        yy = np.concatenate([y[sub], ys]) if len(ys) else y[sub]
                        clf = LogisticRegression(C=C[m], max_iter=2000, random_state=SEMILLA).fit(X, yy)
                        p = clf.predict(E[m][idx_test])
                        for pos_i, real_i, pred_i in zip(idx_test, y[idx_test], p):
                            filas.append({"regimen": regimen, "semilla": semilla, "modelo": m, "tecnica": tecnica, "fold": f, "pos": int(pos_i),
                                          "real": real_i, "prediccion": pred_i, "C": C[m], "n_reales": len(sub), "n_sinteticos": len(ys)})
                print(f"regimen {regimen} semilla {semilla} fold {f}: {len(sub)} reales, C {C}", flush=True)

    if args.solo_generar:
        print("Generacion terminada: lleva la carpeta salidas/regimenes/ al repositorio y evalua sin --solo-generar.")
        return 0

    SALIDA.mkdir(parents=True, exist_ok=True)
    P = pd.DataFrame(filas)
    if args.anexar and (SALIDA / "predicciones.csv").exists():
        previo = pd.read_csv(SALIDA / "predicciones.csv")
        nuevas = set(zip(P["regimen"], P["tecnica"]))
        previo = previo[[(r, t) not in nuevas for r, t in zip(previo["regimen"], previo["tecnica"])]]
        P = pd.concat([previo, P], ignore_index=True)
    P.to_csv(SALIDA / "predicciones.csv", index=False, encoding="utf-8")
    # resumen: F1 macro agrupado y diferencias pareadas frente a sin aumento (bootstrap sobre las oraciones, promediando semillas)
    rng = np.random.default_rng(SEMILLA)
    resumen, pareadas = [], []
    for (regimen, modelo), g in P.groupby(["regimen", "modelo"]):
        base = {s: gg.sort_values("pos") for s, gg in g[g.tecnica == "sin_aumento"].groupby("semilla")}
        for tecnica, gt in g.groupby("tecnica"):
            f1s = []
            for s, gg in gt.groupby("semilla"):
                gg = gg.sort_values("pos")
                f1s.append(vc.f1m(gg["real"], gg["prediccion"]))
            fila = {"regimen": regimen, "modelo": modelo, "tecnica": tecnica, "f1_macro": float(np.mean(f1s)),
                    "sd_entre_semillas": float(np.std(f1s)) if len(f1s) > 1 else np.nan, "n_semillas": len(f1s),
                    "n_reales_medio": float(gt.groupby(["semilla", "fold"]).n_reales.first().mean()),
                    "n_sinteticos_medio": float(gt.groupby(["semilla", "fold"]).n_sinteticos.first().mean())}
            resumen.append(fila)
            if tecnica != "sin_aumento" and base:
                dif, bajo, alto = [], [], []
                semillas = [s for s in gt.semilla.unique() if s in base]
                pares = [(gt[gt.semilla == s].sort_values("pos"), base[s]) for s in semillas]
                yi = np.array([cod[v] for v in pares[0][0]["real"]]); n = len(yi)
                pa = [np.array([cod[v] for v in a["prediccion"]]) for a, _ in pares]; pb = [np.array([cod[v] for v in b["prediccion"]]) for _, b in pares]
                k = len(clases)
                difs = []
                for _ in range(1000):
                    i = rng.integers(0, n, n)
                    difs.append(np.mean([f1_macro_rapido(yi[i], a[i], k) - f1_macro_rapido(yi[i], b[i], k) for a, b in zip(pa, pb)]))
                d0 = np.mean([f1_macro_rapido(yi, a, k) - f1_macro_rapido(yi, b, k) for a, b in zip(pa, pb)])
                lo, hi = np.percentile(difs, [2.5, 97.5])
                pareadas.append({"regimen": regimen, "modelo": modelo, "tecnica": tecnica, "dif_vs_sin_aumento": float(d0), "ic95_bajo": float(lo),
                                 "ic95_alto": float(hi), "distinguible_de_cero": not (lo <= 0 <= hi)})
    R = pd.DataFrame(resumen); D = pd.DataFrame(pareadas)
    R.to_csv(SALIDA / "resumen_por_regimen.csv", index=False, encoding="utf-8"); D.to_csv(SALIDA / "comparacion_pareada.csv", index=False, encoding="utf-8")
    print("\nF1 macro por regimen (reales por clase):")
    print(R.pivot_table(index=["modelo", "tecnica"], columns="regimen", values="f1_macro").round(3).to_string())
    print("\nDiferencia vs sin aumento (* = IC95% no incluye 0):")
    D["t"] = [f"{d:+.3f}{'*' if s else ''}" for d, s in zip(D.dif_vs_sin_aumento, D.distinguible_de_cero)]
    print(D.pivot_table(index=["modelo", "tecnica"], columns="regimen", values="t", aggfunc="first").to_string())
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ejes = plt.subplots(1, len(modelos), figsize=(5 * len(modelos), 4), sharey=True)
        ejes = np.atleast_1d(ejes)
        for ax, m in zip(ejes, modelos):
            for t, g in R[R.modelo == m].groupby("tecnica"):
                g = g.sort_values("regimen"); ax.plot(g.regimen, g.f1_macro, marker="o", label=t)
            ax.set_title(m); ax.set_xlabel("ejemplos reales por clase"); ax.set_xticks(REGIMENES); ax.grid(alpha=0.3)
        ejes[0].set_ylabel("F1 macro"); ejes[0].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(SALIDA / "curva_regimen.png", dpi=130)
    except Exception as e:   # noqa: BLE001
        print("No se pudo dibujar la figura:", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
