"""
Transformaciones de los embeddings antes de medir su agrupamiento por intención.

Con los embeddings crudos la silueta (coseno) sale cerca de 0. Los embeddings de BERT y modelos parecidos tienen dos
problemas conocidos que bajan esa métrica aunque la información esté: todos los vectores se concentran en un cono
(anisotropía) y unas pocas dimensiones dominan las distancias. Este módulo aplica siete transformaciones que no usan
las etiquetas, recalcula silueta, Davies-Bouldin y Calinski-Harabasz, y calcula la silueta que daría un reparto al
azar de las intenciones, para saber qué significa "cerca de 0" en cada espacio.

Transformaciones (definidas antes de ver los resultados)
  crudo               el embedding tal cual (igual que el OE3)
  centrado            se resta la media de todos los vectores
  estandarizado       centrado y dividido por la desviación de cada dimensión
  sin_1_componente    centrado y sin la primera componente principal
  sin_7_componentes   centrado y sin las 7 primeras (Mu y Viswanath, 2018)
  pca50               PCA a 50 componentes
  pca50_blanqueado    PCA a 50 componentes con varianza 1 en cada una
El blanqueo completo no se puede: hay menos oraciones que dimensiones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize

TRANSFORMACIONES = ["crudo", "centrado", "estandarizado", "sin_1_componente", "sin_7_componentes", "pca50", "pca50_blanqueado"]


def transformar(X, nombre: str) -> np.ndarray:
    """Aplica una transformacion no supervisada a la matriz de embeddings X (n oraciones x d)."""
    X = np.asarray(X, dtype=np.float64)
    if nombre == "crudo":
        return X
    Xc = X - X.mean(axis=0, keepdims=True)
    if nombre == "centrado":
        return Xc
    if nombre == "estandarizado":
        sd = Xc.std(axis=0, keepdims=True)
        sd[sd < 1e-12] = 1.0
        return Xc / sd
    if nombre.startswith("sin_"):
        k = int(nombre.split("_")[1])
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        V = Vt[:k]
        return Xc - (Xc @ V.T) @ V
    if nombre == "pca50":
        return PCA(n_components=50, random_state=0).fit_transform(X)
    if nombre == "pca50_blanqueado":
        return PCA(n_components=50, whiten=True, random_state=0).fit_transform(X)
    raise ValueError(f"transformacion desconocida: {nombre}")


def evaluar(X, etiquetas, n_perm: int = 50, semilla: int = 42) -> dict[str, float]:
    """Silueta (coseno), Davies-Bouldin y Calinski-Harabasz (estos dos sobre vectores de norma 1, como en OE3) y la silueta con etiquetas
    permutadas al azar (media y percentiles 2.5 y 97.5 de `n_perm` permutaciones)."""
    y = np.asarray(etiquetas)
    D = cosine_distances(X)
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    rng = np.random.default_rng(semilla)
    azar = np.array([silhouette_score(D, rng.permutation(y), metric="precomputed") for _ in range(n_perm)]) if n_perm > 0 else np.array([np.nan])
    Xn = normalize(np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype=np.float64))
    return {"silueta_coseno": float(silhouette_score(D, y, metric="precomputed")),
            "silueta_azar_media": float(np.mean(azar)),
            "silueta_azar_p2_5": float(np.percentile(azar, 2.5)),
            "silueta_azar_p97_5": float(np.percentile(azar, 97.5)),
            "davies_bouldin": float(davies_bouldin_score(Xn, y)),
            "calinski_harabasz": float(calinski_harabasz_score(Xn, y))}


def evaluar_todas(embeddings: dict, etiquetas, transformaciones=TRANSFORMACIONES, n_perm: int = 50, **extra) -> pd.DataFrame:
    """embeddings: {(modelo, pooling): matriz}. Devuelve una fila por modelo x pooling x transformacion."""
    filas = []
    for (modelo, pooling), X in embeddings.items():
        for t in transformaciones:
            filas.append({**extra, "modelo": modelo, "pooling": pooling, "transformacion": t, **evaluar(transformar(X, t), etiquetas, n_perm)})
    return pd.DataFrame(filas)


def control_tfidf(textos, etiquetas, n_perm: int = 50, **extra) -> pd.DataFrame:
    """Control SIN modelo de lenguaje: los mismos indicadores sobre el vector TF-IDF de n-gramas de letras (2 a 5), el mismo que usa el
    clasificador de referencia, en crudo y reducido a 50 dimensiones (SVD) para que sea comparable con `pca50`."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    M = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True).fit_transform(list(textos))
    filas = [{**extra, "modelo": "n-gramas (TF-IDF)", "pooling": "-", "transformacion": "crudo", **evaluar(M.astype(np.float32).toarray(), etiquetas, n_perm)}]
    Z = TruncatedSVD(n_components=50, random_state=0).fit_transform(M)
    filas.append({**extra, "modelo": "n-gramas (TF-IDF)", "pooling": "-", "transformacion": "svd50", **evaluar(Z, etiquetas, n_perm)})
    return pd.DataFrame(filas)
