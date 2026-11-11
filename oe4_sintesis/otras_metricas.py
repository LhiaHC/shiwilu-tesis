"""
Otras metricas de clasificacion ademas del F1 macro (complemento del OE4, observacion del asesor).

Pregunta: el F1 macro, ¿castiga demasiado?, ¿cambian las conclusiones si se usa otra metrica?
No entrena modelos ni usa la API: reutiliza las mismas predicciones que `sintesis.py` (protocolo B, 5 particiones congeladas,
700 oraciones). Todas las metricas se calculan sobre las 700 predicciones agrupadas, con IC 95% bootstrap (mismas reglas que el F1).

METRICAS
  Con las etiquetas predichas (todos los enfoques, incluidos fine-tuning y n-gramas)
    f1_macro              promedio del F1 de las 7 intenciones (la metrica de la tesis)
    exactitud             aciertos / 700 (= F1 micro)
    exactitud_balanceada  recall macro (promedio del recall de cada intencion); con 100 oraciones por intencion = exactitud
    precision_macro       promedio de la precision de cada intencion
    recall_macro          igual a exactitud_balanceada
    f1_ponderado          F1 por intencion ponderado por su soporte (100 c/u: = F1 macro)
    kappa                 Cohen: acuerdo corregido por azar (0 = azar, 1 = perfecto)
    mcc                   Matthews multiclase: correlacion entre real y predicho (0 = azar, 1 = perfecto)
  Con las probabilidades (solo modelos congelados + Regresion Logistica; el fine-tuning y los n-gramas no las guardaron)
    auc_macro             area bajo la curva ROC, uno-contra-resto, promedio de las 7 intenciones (0.5 = azar)
    top2, top3            la intencion correcta esta entre las 2 (3) mas probables
    log_loss              menor = mejor (azar entre 7 clases = 1.946)

SALIDAS (oe4_sintesis/resultados/otras_metricas/)
  metricas_por_enfoque.csv          cada enfoque de `enfoques_f1.csv` con todas las metricas (IC 95%) y su ranking por metrica
  metricas_36_combinaciones.csv     las 36 combinaciones (+ Mixup) con todas las metricas
  cambia_la_conclusion.csv          los 12 experimentos centrales: diferencia pareada frente al baseline en cada metrica, con IC 95%
  precision_recall_por_intencion.csv   precision y recall por intencion de los enfoques principales
  concordancia_entre_metricas.csv   Spearman entre rankings de metricas (sobre los enfoques y sobre las 36 combinaciones)

Uso (desde la raiz del repositorio):
    python oe4_sintesis/otras_metricas.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

import sintesis as S
from sintesis import CLASES, K, MODELOS, NOM, NOM_POOL, a_indices

warnings.filterwarnings("ignore")

SALIDA = S.SINTESIS_RESULTADOS / "otras_metricas"
REPS = 2000
SEMILLA = 42
COL_PROB = [f"prob_{c}" for c in CLASES]
METRICAS_ETQ = ["f1_macro", "exactitud", "precision_macro", "recall_macro", "f1_ponderado", "kappa", "mcc"]
METRICAS_PROB = ["auc_macro", "top2", "top3", "log_loss"]
TODAS = METRICAS_ETQ + METRICAS_PROB


# ------------------------------------------------------------------------------------------- metricas
def metricas_etiquetas(y, p):
    cm = S.matriz_confusion(y, p).astype(float)
    n = cm.sum(); tp = np.diag(cm); real = cm.sum(1); pred = cm.sum(0)
    prec = np.where(pred > 0, tp / np.maximum(pred, 1), 0.0)
    rec = np.where(real > 0, tp / np.maximum(real, 1), 0.0)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    acc = tp.sum() / n
    pe = (real * pred).sum() / n ** 2
    kappa = (acc - pe) / (1 - pe) if pe < 1 else 0.0
    # MCC multiclase (Gorodkin)
    c = tp.sum(); s = n
    den = np.sqrt((s ** 2 - (pred ** 2).sum()) * (s ** 2 - (real ** 2).sum()))
    mcc = (c * s - (pred * real).sum()) / den if den > 0 else 0.0
    return {"f1_macro": f1.mean(), "exactitud": acc, "precision_macro": prec.mean(), "recall_macro": rec.mean(),
            "f1_ponderado": (f1 * real).sum() / n, "kappa": kappa, "mcc": mcc}


def auc_macro(y, P):
    aucs = []
    for k in range(K):
        pos = y == k
        npos, nneg = pos.sum(), (~pos).sum()
        if npos == 0 or nneg == 0:
            continue
        r = rankdata(P[:, k])
        aucs.append((r[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))
    return float(np.mean(aucs))


def metricas_prob(y, P):
    orden = np.argsort(-P, axis=1)
    pos = np.argmax(orden == y[:, None], axis=1)          # 0 = la correcta es la mas probable
    pc = np.clip(P[np.arange(len(y)), y], 1e-12, 1)
    return {"auc_macro": auc_macro(y, P), "top2": float((pos < 2).mean()), "top3": float((pos < 3).mean()), "log_loss": float(-np.log(pc).mean())}


def todas(y, p, P=None):
    d = metricas_etiquetas(y, p)
    d.update(metricas_prob(y, P) if P is not None else {k: np.nan for k in METRICAS_PROB})
    return d


def prom_semillas(y, preds, probs, idx=None):
    """Media de las metricas sobre las semillas (varias en fine-tuning, una en el resto), sobre las oraciones `idx` (bootstrap)."""
    i = np.arange(len(y)) if idx is None else idx
    filas = [todas(y[i], p[i], None if probs is None else probs[i]) for p in preds]
    return {k: float(np.mean([f[k] for f in filas])) for k in TODAS}


def bootstrap(y, items, rng, reps):
    """items: {nombre: (preds (semillas x n), probs (n x K) o None)}. Devuelve {nombre: DataFrame (reps x metricas)} con los mismos remuestreos."""
    n = len(y)
    salida = {k: np.empty((reps, len(TODAS))) for k in items}
    for r in range(reps):
        idx = rng.integers(0, n, n)
        for k, (P, PR) in items.items():
            d = prom_semillas(y, P, PR, idx)
            salida[k][r] = [d[m] for m in TODAS]
    return {k: pd.DataFrame(v, columns=TODAS) for k, v in salida.items()}


def ic(boot, nombre, m):
    v = boot[nombre][m].to_numpy()
    if np.isnan(v).all():
        return np.nan, np.nan
    return float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))


def probs_de(df):
    return df.sort_values("pos")[COL_PROB].to_numpy()


# ------------------------------------------------------------------------------------------- principal
def main() -> int:
    rng = np.random.default_rng(SEMILLA)
    SALIDA.mkdir(parents=True, exist_ok=True)
    oe2, pool, ft = S.cargar_oe2(), S.cargar_pooling(), S.cargar_finetuning()
    y_txt = oe2[("mbert", "sin_aumento")]["real"].to_numpy(); y = a_indices(y_txt)
    P_NG = S.ngramas_de_letras(y_txt, False)
    T = pd.read_csv(S.SINTESIS_RESULTADOS / "tabla_36_combinaciones.csv")
    T36 = T[T.pooling != S.EXTRACCION_OE2]

    # ---------- enfoques (los mismos de enfoques_f1.csv)
    enf = {}      # nombre -> (grupo, preds (s x 700), probs)

    def agregar(nombre, grupo, preds, probs):
        enf[nombre] = (grupo, np.atleast_2d(preds), probs)

    agregar("N-gramas de letras (sin modelo de lenguaje)", "referencia", P_NG, None)
    for m in MODELOS:
        g = oe2[(m, "sin_aumento")]
        agregar(f"{NOM[m]} congelado (baseline OE2)", "congelado", a_indices(g.sort_values("pos")["prediccion"]), probs_de(g))
    for m in MODELOS:
        x = T36[(T36.modelo == m) & (T36.corpus == "sin aumento")]; r = x.loc[x.f1_macro.idxmax()]
        g = pool[(m, r.pooling, "sin aumento")]
        agregar(f"{NOM[m]} congelado, mejor pooling ({NOM_POOL[r.pooling]})", "congelado_mejor_pooling", a_indices(g["prediccion"]), probs_de(g))
    for m in MODELOS:
        x = T36[(T36.modelo == m) & (T36.corpus == "GtR")]; r = x.loc[x.f1_macro.idxmax()]
        g = pool[(m, r.pooling, "GtR")]
        agregar(f"{NOM[m]} congelado + GtR ({NOM_POOL[r.pooling]})", "congelado_gtr", a_indices(g["prediccion"]), probs_de(g))
    etq = {("mbert", "completo"): "mBERT fine-tuning completo", ("labse", "completo"): "LaBSE fine-tuning completo",
           ("xlmr", "completo"): "XLM-R fine-tuning (rejilla comun, no convergio)", ("xlmr_rejilla2", "completo"): "XLM-R fine-tuning (rejilla corregida)",
           ("mbert", "lora"): "mBERT LoRA (no concluyente)"}
    for clave, P in ft.items():
        agregar(etq.get(clave, f"{clave[0]} {clave[1]}"), "fine_tuning", P, None)

    print("bootstrap de los enfoques...")
    boot = bootstrap(y, {k: (v[1], v[2]) for k, v in enf.items()}, rng, REPS)
    filas = []
    for k, (grupo, P, PR) in enf.items():
        punto = prom_semillas(y, P, PR)
        fila = {"enfoque": k, "grupo": grupo, "semillas": len(P)}
        for m in TODAS:
            lo, hi = ic(boot, k, m)
            fila[m] = punto[m]; fila[m + "_ic95_bajo"] = lo; fila[m + "_ic95_alto"] = hi
        filas.append(fila)
    A = pd.DataFrame(filas)
    for m in TODAS:   # ranking (1 = mejor) solo entre enfoques que tienen la metrica
        A[m + "_ranking"] = A[m].rank(ascending=(m == "log_loss"), method="min")
    A.to_csv(SALIDA / "metricas_por_enfoque.csv", index=False, encoding="utf-8")

    # ---------- 36 combinaciones (+ Mixup, sin probabilidades: se calcula sobre las etiquetas de OE2)
    print("las 36 combinaciones...")
    filas36 = []
    for (m, p, c), g in pool.items():
        filas36.append({"modelo": m, "pooling": p, "corpus": c, **todas(y, a_indices(g["prediccion"]), probs_de(g))})
    for m in MODELOS:
        g = oe2[(m, "mixup")]
        filas36.append({"modelo": m, "pooling": S.EXTRACCION_OE2, "corpus": "Mixup", **todas(y, a_indices(g.sort_values("pos")["prediccion"]), probs_de(g))})
    B = pd.DataFrame(filas36)
    B.to_csv(SALIDA / "metricas_36_combinaciones.csv", index=False, encoding="utf-8")

    # ---------- ¿cambia la conclusion? 12 experimentos centrales frente a su baseline, en cada metrica
    print("diferencias pareadas de los 12 experimentos centrales...")
    tecnicas = [("mixup", "Mixup"), ("retrotraduccion", "Retrotraduccion"), ("generate_then_refine", "Generate-then-Refine")]
    base = {m: (np.atleast_2d(a_indices(oe2[(m, "sin_aumento")].sort_values("pos")["prediccion"])), probs_de(oe2[(m, "sin_aumento")])) for m in MODELOS}
    items = {}
    for m in MODELOS:
        items[(m, "base")] = base[m]
        for t, _ in tecnicas:
            g = oe2[(m, t)].sort_values("pos")
            items[(m, t)] = (np.atleast_2d(a_indices(g["prediccion"])), probs_de(g))
    boot12 = bootstrap(y, {f"{a}|{b}": v for (a, b), v in items.items()}, rng, REPS)
    filas_c = []
    for m in MODELOS:
        for t, nom in tecnicas:
            fila = {"modelo": NOM[m], "tecnica": nom}
            for met in TODAS:
                d = boot12[f"{m}|{t}"][met] - boot12[f"{m}|base"][met]
                punto = prom_semillas(y, *items[(m, t)])[met] - prom_semillas(y, *items[(m, "base")])[met]
                lo, hi = (np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5)) if not np.isnan(d).all() else (np.nan, np.nan)
                fila[f"{met}_dif"] = punto
                fila[f"{met}_dist"] = "" if np.isnan(lo) else ("mejora" if lo > 0 else ("empeora" if hi < 0 else "="))
            filas_c.append(fila)
    C = pd.DataFrame(filas_c)
    C.to_csv(SALIDA / "cambia_la_conclusion.csv", index=False, encoding="utf-8")

    # ---------- precision y recall por intencion de los enfoques principales
    filas_p = []
    for k, (grupo, P, PR) in enf.items():
        if grupo in ("fine_tuning",) and "completo" not in k and "corregida" not in k:
            continue
        cms = [S.matriz_confusion(y, p).astype(float) for p in P]
        cm = sum(cms) / len(cms)
        tp = np.diag(cm); pr = tp / np.maximum(cm.sum(0), 1e-12); rc = tp / np.maximum(cm.sum(1), 1e-12)
        for i, c in enumerate(CLASES):
            filas_p.append({"enfoque": k, "intencion": c, "precision": pr[i], "recall": rc[i], "f1": 2 * pr[i] * rc[i] / max(pr[i] + rc[i], 1e-12)})
    pd.DataFrame(filas_p).to_csv(SALIDA / "precision_recall_por_intencion.csv", index=False, encoding="utf-8")

    # ---------- concordancia entre metricas
    filas_k = []
    for nombre, tabla in (("15 enfoques", A), ("36 combinaciones + Mixup", B)):
        for m in TODAS:
            if m == "f1_macro":
                continue
            x = tabla[["f1_macro", m]].dropna()
            if len(x) > 3:
                filas_k.append({"ambito": nombre, "metrica": m, "n": len(x), "spearman_con_f1_macro": float(spearmanr(x.f1_macro, x[m])[0])})
    pd.DataFrame(filas_k).to_csv(SALIDA / "concordancia_entre_metricas.csv", index=False, encoding="utf-8")

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print("\n== Enfoques ==")
    print(A[["enfoque"] + TODAS].round(3).to_string(index=False))
    print("\n== Concordancia con F1 macro (Spearman) ==")
    print(pd.DataFrame(filas_k).round(3).to_string(index=False))
    print("\n== 12 experimentos: diferencia vs baseline ==")
    cols = ["modelo", "tecnica"] + [c for m in ("f1_macro", "exactitud", "kappa", "mcc", "auc_macro", "top2") for c in (f"{m}_dif", f"{m}_dist")]
    print(C[cols].round(3).to_string(index=False))
    print(f"\nSalidas en {SALIDA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
