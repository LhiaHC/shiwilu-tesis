"""
Auditoria de optimismo de los resultados (evidencia para interpretar el F1).

  A. Distribucion del F1 bajo 30 splits aleatorios 70/15/15 agrupados: muestra
     que el split unico (semilla 42) fue una tirada dificil y que la validacion
     cruzada NO esta inflada.
  B. F1 segun cuanto se parece cada oracion de test a alguna de train
     (solapamiento de palabras): cuanto depende del vocabulario ya visto.
  C. F1 por categoria (PRG ~0.97 delata un atajo).
  D. Ablacion: quitar la puntuacion antes de embeber (ver validacion_cruzada.py
     --sin-puntuacion para la matriz completa de 12 experimentos).

Requiere haber corrido antes validacion_cruzada.py. Solo imprime resultados.

Uso:
    python 3_baselines_y_aumento_datos/auditoria_optimismo.py
"""
import re, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

RAIZ = Path(__file__).resolve().parents[1]
for r in (RAIZ, RAIZ / "2_baselines", RAIZ / "3_baselines_y_aumento_datos"):
    sys.path.insert(0, str(r))
from comun import (MODELOS, SEMILLA, VALORES_C, cargar_corpus, cargar_folds, extraer_embeddings,
                   _normalizar_shiwilu, PROP_DEV_TEST)
from baseline_trivial import predecir_nn_palabras
from validacion_cruzada import ajustar_y_predecir

df = cargar_corpus(); n = len(df)
y = df["intencion"].to_numpy()
norm = df["shiwilu"].map(_normalizar_shiwilu)
folds = cargar_folds(df)
f1m = lambda a, b: float(f1_score(a, b, average="macro", zero_division=0))

def tokens(t): return set(re.findall(r"[a-z0-9\u00e1\u00e9\u00ed\u00f3\u00fa\u00f1\u2019']+", str(t).lower()))
toks = [tokens(t) for t in df["shiwilu"]]

def max_jacc(i, idx_train):
    a = toks[i]
    if not a: return 0.0
    return max((len(a & toks[j]) / len(a | toks[j]) for j in idx_train if toks[j] | a), default=0.0)

print("=== A. Distribucion del F1 bajo muchos splits aleatorios 70/15/15 (agrupados) ===", flush=True)
E = {m: extraer_embeddings(df["shiwilu"].astype(str).tolist(), m) for m in MODELOS}
grupos = pd.DataFrame({"k": norm, "intencion": y}).groupby("k", as_index=False)["intencion"].first()
res = {"nn_palabras": [], **{m: [] for m in MODELOS}}
for s in range(30):
    tg, rg = train_test_split(grupos, test_size=PROP_DEV_TEST, stratify=grupos["intencion"], random_state=s)
    dg, eg = train_test_split(rg, test_size=0.5, stratify=rg["intencion"], random_state=s)
    itr = np.where(norm.isin(tg["k"]))[0]; idv = np.where(norm.isin(dg["k"]))[0]; ite = np.where(norm.isin(eg["k"]))[0]
    res["nn_palabras"].append(f1m(y[ite], predecir_nn_palabras(df.iloc[itr], df.iloc[ite])))
    for m in MODELOS:
        pred, _ = ajustar_y_predecir(E[m][itr], y[itr], E[m][idv], y[idv], E[m][itr], y[itr], E[m][ite])
        res[m].append(f1m(y[ite], pred))
ref = {"nn_palabras": 0.3830, "labse": 0.6943, "mbert": 0.6977, "xlmr": 0.7223}
for k, v in res.items():
    v = np.array(v)
    print(f"{k:<12} media={v.mean():.4f} sd={v.std(ddof=1):.4f} min={v.min():.3f} max={v.max():.3f} | seed-42 (split unico)={ref[k]:.4f} -> percentil {100*(v<ref[k]).mean():.0f}", flush=True)

print("\n=== B. F1 segun cuanto se parece cada oracion de test a alguna de train (CV, sin aumento) ===", flush=True)
P = pd.read_csv(RAIZ / "3_baselines_y_aumento_datos" / "validacion_cruzada_predicciones.csv")
P = P[P["tecnica"] == "sin_aumento"].copy()
sim = {}
for i in range(n):
    sim[i] = max_jacc(i, np.where(folds != folds[i])[0])
P["sim"] = P["pos"].map(sim)
bins = [(-0.01, 0.0, "0 (ninguna palabra en comun)"), (0.0, 0.5, "0-0.5 (parcial)"), (0.5, 0.999, "0.5-1 (alta)"), (0.999, 1.01, "1 (mismas palabras)")]
uni = P[P["modelo"] == "labse"]
print("distribucion de oraciones de test por similitud maxima con train:")
for lo, hi, nombre in bins:
    print(f"  {nombre:<32} {((uni.sim>lo)&(uni.sim<=hi)).mean():6.1%}")
for m in MODELOS:
    g = P[P.modelo == m]
    linea = []
    for lo, hi, nombre in bins:
        s = g[(g.sim > lo) & (g.sim <= hi)]
        if len(s) >= 10:
            linea.append(f"{nombre.split(' ')[0]}: acc={(s.real == s.prediccion).mean():.2f} (n={len(s)})")
    print(f"  {m:<6}", " | ".join(linea), flush=True)

print("\n=== C. F1 por categoria (CV, sin aumento) ===")
for m in MODELOS:
    g = P[P.modelo == m]
    porcat = {c: f1_score(g.real == c, g.prediccion == c) for c in sorted(g.real.unique())}
    print(f"  {m:<6}", " ".join(f"{c}={v:.2f}" for c, v in porcat.items()), flush=True)

print("\n=== D. Ablacion: quitar signos de puntuacion antes de embeber (CV, sin aumento) ===", flush=True)
sin_punt = df["shiwilu"].map(lambda t: " ".join(re.findall(r"[^\W_]+(?:'[^\W_]+)*", str(t).lower())))
for m in MODELOS:
    Ep = extraer_embeddings(sin_punt.tolist(), m)
    preds = np.empty(n, dtype=object)
    for f in range(5):
        ite = np.where(folds == f)[0]; ipool = np.where(folds != f)[0]
        from sklearn.model_selection import StratifiedGroupKFold
        cv = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=SEMILLA + f)
        _, dv = next(cv.split(ipool, y[ipool], groups=norm.to_numpy()[ipool]))
        idv = ipool[dv]; icore = np.setdiff1d(ipool, idv)
        preds[ite], _ = ajustar_y_predecir(Ep[icore], y[icore], Ep[idv], y[idv], Ep[ipool], y[ipool], Ep[ite])
    g = P[P.modelo == m].sort_values("pos")
    porcat_con = {c: f1_score(g.real == c, g.prediccion == c) for c in sorted(set(y))}
    porcat_sin = {c: f1_score(y == c, preds == c) for c in sorted(set(y))}
    print(f"  {m:<6} F1 macro con puntuacion={f1m(g.real, g.prediccion):.4f} | sin puntuacion={f1m(y, preds):.4f} | PRG con={porcat_con['PRG']:.2f} sin={porcat_sin['PRG']:.2f}", flush=True)
