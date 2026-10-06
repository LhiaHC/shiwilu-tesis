"""
Por que "mBERT sin aumento" es el mejor resultado? Dos diagnosticos sobre el texto shiwilu normalizado:

1. Fragmentacion: cuantas subpalabras produce el tokenizador de cada modelo por palabra y por oracion.
2. Clasificador que SOLO ve la forma de las letras: TF-IDF de n-gramas de caracteres (2-5) + Regresion Logistica, con los
   mismos folds congelados y el mismo protocolo de C (CV interna K=5). Sirve de referencia: si iguala o supera a los
   embeddings, la intencion se reconoce sobre todo por la forma superficial (sufijos, particulas) y no por el significado.

Salida: evaluacion/diagnostico_baseline/resultados/{fragmentacion.csv, ngramas_caracteres.csv}
Uso (desde la raiz del repositorio):  python oe2_aumento_de_datos/evaluacion/diagnostico_baseline/ngramas_caracteres.py
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ)); sys.path.insert(0, str(RAIZ / "oe2_aumento_de_datos" / "evaluacion"))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from shiwilu.clasificacion import MODELOS, SEMILLA, cargar_corpus, cargar_corpus_normalizado, cargar_folds, _normalizar_estricto
import validacion_cruzada_cv_interna as cv

corpus = cargar_corpus(); y = corpus["intencion"].to_numpy(); folds = cargar_folds(corpus)
txt = cargar_corpus_normalizado()["shiwilu"].astype(str).tolist()
claves = np.array([_normalizar_estricto(t) for t in txt])

# --- 1. fragmentacion del texto por tokenizador
from transformers import AutoTokenizer
print("== Subpalabras por palabra (fertilidad) y por oracion, texto shiwilu normalizado")
palabras = [len(t.split()) for t in txt]
FRAG = []
for nombre, (hf_id, tipo) in MODELOS.items():
    tok = AutoTokenizer.from_pretrained(hf_id)
    n = [len(tok.tokenize(t)) for t in txt]
    FRAG.append({"modelo": nombre, "tokens_por_oracion": float(np.mean(n)), "subpalabras_por_palabra": float(np.sum(n) / np.sum(palabras))})
    print(f"{nombre:6s} tokens/oracion {np.mean(n):5.1f} | subpalabras por palabra {np.sum(n)/np.sum(palabras):.2f}")
print(f"palabras por oracion: {np.mean(palabras):.2f}")

# --- 2. clasificador que solo ve la forma de las letras (n-gramas de caracteres)
GRILLA = [0.1, 0.3, 1, 3, 10, 30, 100]
def f1m(a, b): return f1_score(a, b, average="macro", zero_division=0)
def fit_pred(tr, te, C):
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
    Xtr = v.fit_transform([txt[i] for i in tr]); Xte = v.transform([txt[i] for i in te])
    return LogisticRegression(C=C, max_iter=3000, random_state=SEMILLA).fit(Xtr, y[tr]).predict(Xte)
pred = np.empty(len(y), dtype=object); elegidos = []
for f in sorted(set(folds)):
    te = np.where(folds == f)[0]; pool = np.where(folds != f)[0]
    parts = cv.particiones_internas(pool, y, claves, f, 5)
    puntaje = {}
    for C in GRILLA:
        p = np.empty(len(pool), dtype=object); pos = {int(i): n for n, i in enumerate(pool)}
        for tr, va in parts:
            p[[pos[int(i)] for i in va]] = fit_pred(tr, va, C)
        puntaje[C] = f1m(y[pool], p)
    mejor = max(GRILLA, key=lambda c: (puntaje[c], -c)); elegidos.append(mejor)
    pred[te] = fit_pred(pool, te, mejor)
cats = ["AFI", "DES", "EMO", "NEG", "PRG", "REQUEST", "SAL"]
print("\n== TF-IDF de n-gramas de caracteres (2-5) + Regresion Logistica, mismos folds y mismo protocolo de C")
print(f"F1 macro agrupado: {f1m(y, pred):.3f}  | C elegidos: {elegidos}")
print("por categoria:", {c: round(v, 2) for c, v in zip(cats, f1_score(y, pred, average=None, labels=cats))})

SAL = RAIZ / "oe2_aumento_de_datos" / "evaluacion" / "diagnostico_baseline" / "resultados"
SAL.mkdir(parents=True, exist_ok=True)
pd.DataFrame(FRAG).to_csv(SAL / "fragmentacion.csv", index=False)
fila = {"f1_macro_agrupado": f1m(y, pred), "C_elegidos_por_fold": " ".join(map(str, elegidos))}
fila.update({f"f1_{c}": float(v) for c, v in zip(cats, f1_score(y, pred, average=None, labels=cats))})
pd.DataFrame([fila]).to_csv(SAL / "ngramas_caracteres.csv", index=False)
