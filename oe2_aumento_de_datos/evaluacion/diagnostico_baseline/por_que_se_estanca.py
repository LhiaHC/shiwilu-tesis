"""
Por que el F1 mejora con cierta cantidad de sintetico de GtR y luego se estanca? Es ruido? De que tipo?

Se agregan los lotes de GtR (20 oraciones por categoria cada uno) de a uno, de 1 a 6 (= 20 a 120 por categoria), al pool real de cada fold, y se mide el F1 (C fijo,
5 folds, test intacto) con cuatro versiones del sintetico:
  bruto            todo lo aprobado y unico
  sin_ruido_etiq   solo las filas cuya etiqueta coincide con la que predice un clasificador entrenado con las reales (quita el RUIDO DE ETIQUETA)
  sin_casi_dup     quita las que son casi duplicadas (coseno LaBSE >= 0.90) de otra ya conservada de su clase (quita la REDUNDANCIA)
  ambos            las dos limpiezas
Ademas, por lote: oraciones nuevas, novedad respecto de los lotes anteriores, acuerdo de la etiqueta y cuantos ejemplos reales distintos vio Claude.

Si limpiar el ruido de etiqueta hiciera que el F1 siguiera subiendo con mas lotes, el estancamiento seria ruido de etiqueta; si limpiar la redundancia lo hiciera, seria redundancia;
si ninguna cambia la curva, el limite es la informacion que entra al sintetico (los ejemplos reales que ve Claude y su conocimiento del shiwilu), no su cantidad.

No usa la API. Uso (desde la raiz del repositorio):  python oe2_aumento_de_datos/evaluacion/diagnostico_baseline/por_que_se_estanca.py
Salida: evaluacion/diagnostico_baseline/resultados/{E_curva_por_lotes, E_estadisticas_por_lote}.csv
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ)); sys.path.insert(0, str(RAIZ / "oe2_aumento_de_datos" / "evaluacion"))

from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

import validacion_cruzada as vc  # noqa: E402
from shiwilu.clasificacion import (  # noqa: E402
    MODELOS, SEMILLA, _normalizar_estricto, cargar_corpus, cargar_corpus_normalizado, cargar_folds, extraer_embeddings,
)
from shiwilu.rutas import AUMENTO_SALIDA  # noqa: E402
from shiwilu.taxonomia import INTENCIONES  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)
CACHE = AUMENTO_SALIDA / "en_linea_marcadores_fuentes" / "generate_then_refine"
SAL = Path(__file__).resolve().parent / "resultados"
C_FIJO = {"mbert": 0.3, "labse": 10.0, "xlmr": 10.0}
UMBRAL_DUP = 0.90
N_FEW_SHOT = 6   # ejemplos reales por llamada en generate_then_refine.py (N_EJEMPLOS_FEW_SHOT)


def lr(X, y, C):
    return LogisticRegression(C=C, max_iter=2000, random_state=SEMILLA).fit(X, y)


def norm(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def main() -> int:
    SAL.mkdir(parents=True, exist_ok=True)
    corpus = cargar_corpus(); y = corpus["intencion"].to_numpy(); folds = cargar_folds(corpus)
    texto = cargar_corpus_normalizado()["shiwilu"].astype(str).tolist()
    claves = np.array([_normalizar_estricto(t) for t in texto]); tx = vc.CONDICIONES["sin_puntuacion"]
    E = {m: extraer_embeddings([tx(t) for t in texto], m) for m in MODELOS}

    datos, estad = {}, []
    for f in sorted(set(folds)):
        it = np.where(folds == f)[0]; ip = np.where(folds != f)[0]
        partes = []
        for j, nombre in enumerate([f"fold{f}_pool.csv"] + [f"fold{f}_pool_l{k}.csv" for k in range(1, 6)]):
            d = pd.read_csv(CACHE / nombre); d["lote"] = j; partes.append(d)
        g = pd.concat(partes, ignore_index=True)
        g = g[g["estado_filtro"] == "aprobado"].copy()
        g["k"] = g["shiwilu"].map(_normalizar_estricto)
        g = g[~g["k"].isin(set(claves[ip]) | set(claves[it]))]                  # copias de reales (pool y test)
        g = g.drop_duplicates(["intencion", "k"], keep="first").reset_index(drop=True)   # repetidas: se conserva la del lote mas temprano
        t = [tx(s) for s in g["shiwilu"]]
        X = {m: extraer_embeddings(t, m) for m in MODELOS}
        L = norm(X["labse"])
        # acuerdo de etiqueta: clasificador entrenado con las reales del pool, con el C fijo de cada modelo
        acuerdo = {m: lr(E[m][ip], y[ip], C_FIJO[m]).predict(X[m]) == g["intencion"].to_numpy() for m in MODELOS}
        # casi-duplicados: en orden de lote, se descarta una fila muy parecida a otra ya conservada de su clase
        conserva = np.ones(len(g), bool)
        for c in INTENCIONES:
            idx = g.index[g["intencion"] == c].to_numpy()
            idx = idx[np.argsort(g.loc[idx, "lote"].to_numpy(), kind="stable")]
            kept = []
            for i in idx:
                if kept and (L[kept] @ L[i]).max() >= UMBRAL_DUP:
                    conserva[i] = False
                else:
                    kept.append(i)
        datos[f] = (it, ip, g, X, acuerdo, conserva)
        # estadisticas por lote
        real_L = norm(extraer_embeddings([tx(texto[i]) for i in ip], "labse"))
        for j in range(6):
            filas = g.index[g["lote"] == j].to_numpy()
            previas = g.index[g["lote"] < j].to_numpy()
            nov = []
            for i in filas:
                mismos = previas[g.loc[previas, "intencion"].to_numpy() == g.loc[i, "intencion"]] if len(previas) else np.array([], int)
                if len(mismos):
                    nov.append(1 - float((L[mismos] @ L[i]).max()))
            estad.append({"fold": f, "lote": j, "n_nuevas_unicas": len(filas), "novedad_media": float(np.mean(nov)) if nov else np.nan,
                          **{f"acuerdo_{m}": float(acuerdo[m][filas].mean()) if len(filas) else np.nan for m in MODELOS},
                          "pct_casi_duplicadas": float((~conserva[filas]).mean()) if len(filas) else np.nan,
                          "ejemplos_reales_vistos_por_clase": min((j + 1) * N_FEW_SHOT, int(len(ip) / 7))})
        print(f"fold {f}: {len(g)} sinteticas unicas aprobadas", flush=True)
    pd.DataFrame(estad).to_csv(SAL / "E_estadisticas_por_lote.csv", index=False)

    # ---------------- curva de F1 por numero de lotes
    filas = []
    for m in MODELOS:
        base = np.empty(len(y), dtype=object)
        for f, (it, ip, *_r) in datos.items():
            base[it] = lr(E[m][ip], y[ip], C_FIJO[m]).predict(E[m][it])
        filas.append({"modelo": m, "version": "sin_aumento", "lotes": 0, "f1_macro": vc.f1m(y, base), "n_sinteticas_medio": 0})
        for version in ["bruto", "sin_ruido_etiq", "sin_casi_dup", "ambos"]:
            for k in range(1, 7):
                pred = np.empty(len(y), dtype=object); n = []
                for f, (it, ip, g, X, acuerdo, conserva) in datos.items():
                    sel = (g["lote"].to_numpy() < k)
                    if version in ("sin_ruido_etiq", "ambos"):
                        sel &= acuerdo[m]
                    if version in ("sin_casi_dup", "ambos"):
                        sel &= conserva
                    Xt = np.vstack([E[m][ip], X[m][sel]]); yt = np.concatenate([y[ip], g["intencion"].to_numpy()[sel]])
                    pred[it] = lr(Xt, yt, C_FIJO[m]).predict(E[m][it]); n.append(int(sel.sum()))
                filas.append({"modelo": m, "version": version, "lotes": k, "f1_macro": vc.f1m(y, pred), "n_sinteticas_medio": float(np.mean(n))})
        print("curva", m, flush=True)
    R = pd.DataFrame(filas); R.to_csv(SAL / "E_curva_por_lotes.csv", index=False)
    pd.set_option("display.width", 200)
    for m in MODELOS:
        print(f"\n{m}: F1 macro por numero de lotes (C fijo)")
        print(R[R.modelo == m].pivot_table(index="version", columns="lotes", values="f1_macro").round(3).to_string())
    print("\nEstadisticas por lote (media de los 5 folds)")
    print(pd.DataFrame(estad).groupby("lote").mean(numeric_only=True).drop(columns="fold").round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
