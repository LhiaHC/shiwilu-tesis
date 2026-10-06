"""
Por que el aumento de datos no mejora al baseline? Pruebas controladas con C FIJO (para que no intervenga la eleccion de C).

Todo usa lo ya generado (cache de GtR y Retrotraduccion, Mixup se recalcula); no llama a la API. Embeddings congelados de los 3 modelos,
texto normalizado, los 5 folds congelados. Los sinteticos son los de la etapa "pool" de cada fold (generados solo con el pool, sin el test).

A. TECHO / DATOS REALES. Curva de aprendizaje del baseline con 25/50/75/100% de las oraciones reales del pool.
   Si sigue subiendo al 100%, el modelo "tiene hambre" de datos reales y lo sintetico no la sacia; si se aplana, hay techo.
B. QUE TAN UTIL ES LO SINTETICO (para cada tecnica y modelo):
   B1  F1 entrenando SOLO con sinteticos y evaluando en el test real.
   B2  concordancia de etiqueta: un clasificador entrenado con las reales, acierta la etiqueta que se le quiso dar a cada sintetico?
   B3  novedad: similitud coseno maxima de cada sintetico con las reales del pool, frente a la de las oraciones reales del test.
       Si los sinteticos se parecen MAS a las reales de entrenamiento que las reales nuevas, son redundantes.
   B4  separabilidad: AUC de un clasificador que distingue "real" de "sintetico" (1.0 = se salen por completo de la distribucion real).
C. DOSIS Y ORIGEN DEL DAÑO (solo Generate-then-Refine): F1 al agregar 25/50/75/100% de lo sintetico; solo el sintetico de una categoria;
   todo menos una categoria (mBERT); y una version balanceada (mismo numero por categoria).

Salida: evaluacion/diagnostico_baseline/resultados/{A_curva_aprendizaje, B_utilidad_sintetico, C_dosis, C_por_categoria}.csv
Uso (desde la raiz del repositorio; ~20-30 min en CPU):
    python oe2_aumento_de_datos/evaluacion/diagnostico_baseline/por_que_no_mejora.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "oe2_aumento_de_datos" / "evaluacion"))

from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402

import validacion_cruzada as vc  # noqa: E402
import validacion_cruzada_cv_interna as cv  # noqa: E402
from shiwilu.clasificacion import (  # noqa: E402
    MODELOS, SEMILLA, _normalizar_estricto, cargar_corpus, cargar_corpus_normalizado, cargar_folds, extraer_embeddings,
)
from shiwilu.rutas import AUMENTO_SALIDA  # noqa: E402
from shiwilu.taxonomia import INTENCIONES  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)
C_FIJO = {"mbert": 0.3, "labse": 10.0, "xlmr": 10.0}   # el C que mas veces elige la CV interna en el baseline de cada modelo
TECNICAS = ["mixup", "retrotraduccion", "generate_then_refine"]
SALIDA = Path(__file__).resolve().parent / "resultados"


def ajustar(X, y, C):
    return LogisticRegression(C=C, max_iter=2000, random_state=SEMILLA).fit(X, y)


def f1m(y, p):
    return float(f1_score(y, p, average="macro", zero_division=0))


def coseno_max(A, B):
    A = A / np.linalg.norm(A, axis=1, keepdims=True); B = B / np.linalg.norm(B, axis=1, keepdims=True)
    return (A @ B.T).max(axis=1)


def main() -> int:
    SALIDA.mkdir(parents=True, exist_ok=True)
    corpus = cargar_corpus()
    y = corpus["intencion"].to_numpy()
    folds = cargar_folds(corpus)
    texto = cargar_corpus_normalizado()["shiwilu"].astype(str).tolist()
    claves = np.array([_normalizar_estricto(t) for t in texto])
    tx = vc.CONDICIONES["sin_puntuacion"]
    E = {m: extraer_embeddings([tx(t) for t in texto], m) for m in MODELOS}
    args = argparse.Namespace(cache=AUMENTO_SALIDA / "en_linea_marcadores_fuentes", cantidad=120, multiplicador=1, semilla_aumento=0,
                              checkpoint=None, repo_nmt=None, excluir_categorias=[])

    # ---- datos por fold: indices y sinteticos de cada tecnica (etapa pool)
    datos = {}
    for f in sorted(set(folds)):
        idx_test = np.where(folds == f)[0]; idx_pool = np.where(folds != f)[0]
        sint = {}
        for tec in TECNICAS:
            S = cv.Sinteticos(tec, args, corpus, y, claves, E, list(MODELOS), tx, f, None)
            if tec == "retrotraduccion":
                S.cargar_retro(idx_pool)
            sint[tec] = S.para(idx_pool, f"fold{f}_pool", set(claves[idx_test]), 0, list(MODELOS))
        datos[f] = (idx_test, idx_pool, sint)
        print(f"fold {f}: sinteticos", {t: len(next(iter(sint[t].values()))[1]) for t in TECNICAS}, flush=True)

    # ---------------- A. curva de aprendizaje con datos reales
    filas = []
    for m in MODELOS:
        for frac in [0.25, 0.5, 0.75, 1.0]:
            res = []
            for semilla in ([0] if frac == 1.0 else [0, 1, 2]):
                pred = np.empty(len(y), dtype=object)
                for f, (it, ip, _) in datos.items():
                    sub = ip if frac == 1.0 else train_test_split(ip, train_size=frac, stratify=y[ip], random_state=semilla)[0]
                    pred[it] = ajustar(E[m][sub], y[sub], C_FIJO[m]).predict(E[m][it])
                res.append(f1m(y, pred))
            filas.append({"modelo": m, "fraccion_de_reales": frac, "n_entrenamiento_aprox": int(frac * 560), "f1_macro": float(np.mean(res))})
        print("A", m, flush=True)
    pd.DataFrame(filas).to_csv(SALIDA / "A_curva_aprendizaje.csv", index=False)

    # ---------------- B. utilidad de lo sintetico
    filas = []
    for tec in TECNICAS:
        for m in MODELOS:
            pred_solo = np.empty(len(y), dtype=object)
            acc, n_acc, auc, sim_s, sim_t = [], [], [], [], []
            acc_cat = {c: [0, 0] for c in INTENCIONES}
            for f, (it, ip, sint) in datos.items():
                Xs, ys = sint[tec][m]
                if len(ys) == 0:
                    continue
                Xp, yp = E[m][ip], y[ip]
                # B1
                pred_solo[it] = ajustar(Xs, ys, C_FIJO[m]).predict(E[m][it])
                # B2
                p = ajustar(Xp, yp, C_FIJO[m]).predict(Xs)
                acc.append((p == ys).sum()); n_acc.append(len(ys))
                for c in INTENCIONES:
                    sel = ys == c
                    acc_cat[c][0] += int((p[sel] == c).sum()); acc_cat[c][1] += int(sel.sum())
                # B3
                sim_s.append(coseno_max(Xs, Xp).mean()); sim_t.append(coseno_max(E[m][it], Xp).mean())
                # B4
                Z = np.vstack([Xp, Xs]); d = np.r_[np.zeros(len(Xp)), np.ones(len(Xs))]
                pr = np.zeros(len(d))
                for a, b in StratifiedKFold(3, shuffle=True, random_state=SEMILLA).split(Z, d):
                    pr[b] = ajustar(Z[a], d[a], 1.0).predict_proba(Z[b])[:, 1]
                auc.append(roc_auc_score(d, pr))
            fila = {"tecnica": tec, "modelo": m,
                    "B1_f1_solo_sinteticos": f1m(y, pred_solo),
                    "B2_concordancia_etiqueta": float(np.sum(acc) / np.sum(n_acc)),
                    "B3_similitud_max_sintetico_con_reales": float(np.mean(sim_s)),
                    "B3_similitud_max_real_nuevo_con_reales": float(np.mean(sim_t)),
                    "B4_auc_real_vs_sintetico": float(np.mean(auc))}
            fila.update({f"B2_{c}": acc_cat[c][0] / max(acc_cat[c][1], 1) for c in INTENCIONES})
            filas.append(fila)
            print("B", tec, m, {k: round(v, 3) for k, v in fila.items() if isinstance(v, float)}, flush=True)
    pd.DataFrame(filas).to_csv(SALIDA / "B_utilidad_sintetico.csv", index=False)

    # ---------------- C. dosis y origen del daño (GtR)
    def evaluar(m, armar):
        """armar(Xp,yp,Xs,ys,fold,rng) -> (X,y) de entrenamiento; devuelve F1 macro y F1 por categoria (pooled)."""
        pred = np.empty(len(y), dtype=object)
        for f, (it, ip, sint) in datos.items():
            Xs, ys = sint["generate_then_refine"][m]
            X, yy = armar(E[m][ip], y[ip], Xs, ys, f)
            pred[it] = ajustar(X, yy, C_FIJO[m]).predict(E[m][it])
        return f1m(y, pred), f1_score(y, pred, average=None, labels=INTENCIONES, zero_division=0)

    dosis, porcat = [], []
    for m in MODELOS:
        base, base_cat = evaluar(m, lambda Xp, yp, Xs, ys, f: (Xp, yp))
        dosis.append({"modelo": m, "porcentaje_de_lo_sintetico": 0, "f1_macro": base})
        for p in [0.25, 0.5, 0.75, 1.0]:
            res = []
            for semilla in ([0] if p == 1.0 else [0, 1]):
                def armar(Xp, yp, Xs, ys, f, p=p, semilla=semilla):
                    rng = np.random.default_rng(SEMILLA + f + 31 * semilla)
                    k = rng.choice(len(ys), int(p * len(ys)), replace=False)
                    return np.vstack([Xp, Xs[k]]), np.r_[yp, ys[k]]
                res.append(evaluar(m, armar)[0])
            dosis.append({"modelo": m, "porcentaje_de_lo_sintetico": int(p * 100), "f1_macro": float(np.mean(res))})
        # balanceado: el mismo numero de sinteticos por categoria (el de la categoria con menos)
        def armar_bal(Xp, yp, Xs, ys, f):
            rng = np.random.default_rng(SEMILLA + f)
            n = min((ys == c).sum() for c in INTENCIONES)
            k = np.concatenate([rng.choice(np.where(ys == c)[0], n, replace=False) for c in INTENCIONES])
            return np.vstack([Xp, Xs[k]]), np.r_[yp, ys[k]]
        dosis.append({"modelo": m, "porcentaje_de_lo_sintetico": "balanceado", "f1_macro": evaluar(m, armar_bal)[0]})
        if m == "mbert":   # origen del daño: solo una categoria sintetica / todas menos una
            for c in INTENCIONES:
                solo, solo_cat = evaluar(m, lambda Xp, yp, Xs, ys, f, c=c: (np.vstack([Xp, Xs[ys == c]]), np.r_[yp, ys[ys == c]]))
                sin, sin_cat = evaluar(m, lambda Xp, yp, Xs, ys, f, c=c: (np.vstack([Xp, Xs[ys != c]]), np.r_[yp, ys[ys != c]]))
                porcat.append({"categoria": c, "f1_macro_baseline": base, "f1_macro_solo_sintetico_de_esta": solo,
                               "f1_macro_todo_menos_esta": sin, "f1_de_la_categoria_baseline": float(base_cat[INTENCIONES.index(c)]),
                               "f1_de_la_categoria_con_su_sintetico": float(solo_cat[INTENCIONES.index(c)])})
        print("C", m, flush=True)
    pd.DataFrame(dosis).to_csv(SALIDA / "C_dosis.csv", index=False)
    pd.DataFrame(porcat).to_csv(SALIDA / "C_por_categoria.csv", index=False)
    print("Guardado en", SALIDA)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
