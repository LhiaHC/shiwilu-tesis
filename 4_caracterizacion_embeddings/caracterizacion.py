"""
Objetivo 3 (R7-R8): caracterizacion de embeddings multilingues.

Extrae representaciones de las oraciones en SHIWILU del corpus con 4
estrategias de caracterizacion, sobre los 3 modelos ya usados como
baselines (LaBSE, mBERT, XLM-R), y las evalua de forma INTRINSECA (sin
clasificador): que tan bien separan las 7 categorias de intencion segun
metricas de calidad de agrupamiento.

Estrategias de caracterizacion (R7):
  cls               vector del token especial [CLS] / <s> de la ultima capa
  mean_pooling      promedio de todos los tokens de la oracion (ultima capa)
  max_pooling       maximo por dimension entre todos los tokens (ultima capa)
  combinacion_capas mean pooling sobre el promedio de las ultimas 4 capas
                     ocultas (Devlin et al., 2019, seccion 5.3)

A diferencia de comun.py (que usa la extraccion "de fabrica" de cada modelo,
mas simple), aqui se cargan los 3 modelos en su forma cruda (AutoModel) con
`output_hidden_states=True`, para poder aplicar las 4 estrategias sobre las
MISMAS activaciones internas en un solo forward pass por lote.

Metricas intrinsecas (R8), sobre TODAS las combinaciones modelo x estrategia,
usando las 7 categorias de intencion como particion de referencia:
  - Coeficiente de silueta (distancia coseno)
  - Indice de Davies-Bouldin       )  sobre vectores normalizados a norma 1,
  - Indice de Calinski-Harabasz    )  para que sean consistentes con coseno
                                       (scikit-learn no soporta metricas
                                       arbitrarias para estos dos indices)

No hay clasificador ni "metodo ganador": las 4 estrategias se REPORTAN y
comparan, no se filtran (mismo criterio que el resto de la tesis — ver R9,
donde se contrasta esta evaluacion intrinseca con la extrinseca).

Corpus (R7 pide correr esto "sobre el corpus original y los corpus aumentados
obtenidos en R6"), via --corpus:
  original              corpus/corpus_shiwilu_final.csv completo (default)
  retrotraduccion       original + filas aprobadas de retrotraduccion.csv
  generate_then_refine  original + filas aprobadas de generate_then_refine.csv

Mixup queda FUERA de este script: no genera oraciones en shiwilu, interpola
vectores ya extraidos de un modelo especifico (ver mixup.py), y las 4
estrategias de pooling de aqui requieren un forward pass sobre texto crudo
que Mixup no produce.

Entrada: corpus completo (sin dividir train/dev/test: R7/R8 son un analisis
         intrinseco, no una evaluacion de clasificador).
Salida:  4_caracterizacion_embeddings/resultados/<corpus>/
    metricas_intrinsecas.csv        una fila por combinacion modelo x estrategia
    proyeccion_2d/tsne_grid.png     grilla modelo x estrategia, proyeccion t-SNE
    proyeccion_2d/umap_grid.png     idem, proyeccion UMAP (si esta instalado)

Uso:
    python 4_caracterizacion_embeddings/caracterizacion.py
    python 4_caracterizacion_embeddings/caracterizacion.py --modelos labse xlmr
    python 4_caracterizacion_embeddings/caracterizacion.py --corpus retrotraduccion
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import normalize

import _ruta_raiz  # noqa: F401  (deja importable el paquete `shiwilu`)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2_baselines"))
from comun import MODELOS  # noqa: E402  (mismos 3 modelos que los baselines)

from shiwilu.rutas import CARACTERIZACION_RESULTADOS, CORPUS_CSV, FASE4  # noqa: E402
from shiwilu.taxonomia import COLOR_INT, INTENCIONES  # noqa: E402

ESTRATEGIAS = ["cls", "mean_pooling", "max_pooling", "combinacion_capas"]
N_CAPAS_COMBINACION = 4   # ultimas N capas para "combinacion de capas"

# Corpus de texto aumentados por tecnicas de R6 compatibles con este script
# (generan oraciones en shiwilu; Mixup no, ver docstring del modulo).
VARIANTES_TEXTO = {
    "retrotraduccion": FASE4 / "salidas" / "retrotraduccion.csv",
    "generate_then_refine": FASE4 / "salidas" / "generate_then_refine.csv",
}


def _pooling_todas_las_estrategias(salida_modelo, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Deriva las 4 estrategias a partir del mismo forward pass (hidden_states)."""
    ultima_capa = salida_modelo.hidden_states[-1]           # (batch, tokens, dim)
    mascara = attention_mask.unsqueeze(-1).expand(ultima_capa.size()).float()

    cls = ultima_capa[:, 0, :]

    suma = torch.sum(ultima_capa * mascara, dim=1)
    conteo = torch.clamp(mascara.sum(dim=1), min=1e-9)
    mean_pooling = suma / conteo

    enmascarado = ultima_capa.masked_fill(mascara == 0, float("-inf"))
    max_pooling, _ = torch.max(enmascarado, dim=1)

    capas = torch.stack(salida_modelo.hidden_states[-N_CAPAS_COMBINACION:], dim=0)   # (N, batch, tokens, dim)
    promedio_capas = capas.mean(dim=0)
    mascara_capas = attention_mask.unsqueeze(-1).expand(promedio_capas.size()).float()
    suma_c = torch.sum(promedio_capas * mascara_capas, dim=1)
    conteo_c = torch.clamp(mascara_capas.sum(dim=1), min=1e-9)
    combinacion_capas = suma_c / conteo_c

    return {"cls": cls, "mean_pooling": mean_pooling, "max_pooling": max_pooling,
            "combinacion_capas": combinacion_capas}


def extraer_todas_las_estrategias(oraciones: list[str], nombre_modelo: str, batch_size: int = 16) -> dict[str, np.ndarray]:
    """Extrae embeddings para las 4 estrategias en un solo recorrido del modelo."""
    from transformers import AutoModel, AutoTokenizer

    hf_id, _ = MODELOS[nombre_modelo]
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    modelo = AutoModel.from_pretrained(hf_id, output_hidden_states=True)
    modelo.eval()

    acumulado = {estrategia: [] for estrategia in ESTRATEGIAS}
    with torch.no_grad():
        for inicio in range(0, len(oraciones), batch_size):
            lote = oraciones[inicio: inicio + batch_size]
            enc = tokenizer(lote, padding=True, truncation=True, max_length=64, return_tensors="pt")
            salida = modelo(**enc)
            vectores = _pooling_todas_las_estrategias(salida, enc["attention_mask"])
            for estrategia, tensor in vectores.items():
                acumulado[estrategia].append(tensor.numpy())

    return {estrategia: np.concatenate(lotes, axis=0) for estrategia, lotes in acumulado.items()}


def cargar_corpus_variante(variante: str) -> pd.DataFrame:
    """Corpus original, o original + filas aprobadas de una tecnica de texto (R6)."""
    original = pd.read_csv(CORPUS_CSV)[["shiwilu", "intencion"]]
    if variante == "original":
        return original
    if variante == "mixup":
        raise SystemExit(
            "Mixup no se puede caracterizar con este script: sus vectores sinteticos "
            "se obtienen por interpolacion directa en el espacio de embeddings de UN "
            "modelo (ver mixup.py) y no son oraciones en shiwilu. Las 4 estrategias de "
            "pooling de aqui (CLS, mean, max, combinacion de capas) requieren un forward "
            "pass sobre texto crudo, que Mixup no produce."
        )
    ruta = VARIANTES_TEXTO[variante]
    if not ruta.exists():
        raise SystemExit(f"No se encontro {ruta}. Corre primero la tecnica correspondiente.")
    aumento = pd.read_csv(ruta)
    aprobado = aumento[aumento["estado_filtro"] == "aprobado"][["shiwilu", "intencion"]]
    print(f"{variante}: +{len(aprobado)}/{len(aumento)} filas aprobadas se agregan al corpus original")
    return pd.concat([original, aprobado], ignore_index=True)


def proyectar_2d(X: np.ndarray, metodo: str, semilla: int = 42) -> np.ndarray | None:
    """Proyecta a 2D con t-SNE o UMAP. Devuelve None si el metodo no esta disponible."""
    n = X.shape[0]
    if metodo == "tsne":
        from sklearn.manifold import TSNE

        perplejidad = max(5, min(30, (n - 1) // 3))
        return TSNE(n_components=2, perplexity=perplejidad, init="pca",
                    random_state=semilla).fit_transform(X)
    if metodo == "umap":
        try:
            import umap
        except ImportError:
            return None
        vecinos = max(2, min(15, n - 1))
        return umap.UMAP(n_components=2, n_neighbors=vecinos, random_state=semilla).fit_transform(X)
    raise ValueError(f"Metodo de proyeccion desconocido: {metodo!r}")


def graficar_grilla(
    proyecciones: dict[tuple[str, str], np.ndarray],
    etiquetas: np.ndarray,
    modelos: list[str],
    metodo: str,
    ruta_png: Path,
    titulo: str,
) -> None:
    """Una figura con una grilla (modelo x estrategia) de la proyeccion 2D dada."""
    fig, ejes = plt.subplots(len(modelos), len(ESTRATEGIAS),
                              figsize=(3.6 * len(ESTRATEGIAS), 3.6 * len(modelos)),
                              squeeze=False)
    colores = {intencion: f"#{COLOR_INT[intencion]}" for intencion in INTENCIONES}

    for i, modelo in enumerate(modelos):
        for j, estrategia in enumerate(ESTRATEGIAS):
            ax = ejes[i][j]
            coords = proyecciones[(modelo, estrategia)]
            for intencion in INTENCIONES:
                mascara = etiquetas == intencion
                ax.scatter(coords[mascara, 0], coords[mascara, 1],
                           s=10, alpha=0.75, color=colores[intencion], linewidths=0)
            ax.set_title(f"{modelo} / {estrategia}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    handles = [mpatches.Patch(color=colores[i], label=i) for i in INTENCIONES]
    fig.legend(handles=handles, loc="lower center", ncol=len(INTENCIONES), fontsize=8)
    fig.suptitle(titulo)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    ruta_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluar_interseco(X: np.ndarray, etiquetas: np.ndarray) -> dict[str, float]:
    """Silueta (coseno) + Davies-Bouldin y Calinski-Harabasz (sobre vectores
    normalizados a norma 1, para que sean consistentes con distancia coseno)."""
    X_norm = normalize(X)
    return {
        "silueta_coseno": float(silhouette_score(X, etiquetas, metric="cosine")),
        "davies_bouldin": float(davies_bouldin_score(X_norm, etiquetas)),
        "calinski_harabasz": float(calinski_harabasz_score(X_norm, etiquetas)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modelos", nargs="+", choices=list(MODELOS), default=list(MODELOS))
    ap.add_argument("--corpus", choices=["original", *VARIANTES_TEXTO, "mixup"], default="original",
                     help="Corpus original, o original + tecnica de aumento de texto de R6.")
    ap.add_argument("--sin-proyeccion", action="store_true",
                     help="Omite las grillas t-SNE/UMAP (solo calcula metricas intrinsecas).")
    args = ap.parse_args()

    carpeta_salida = CARACTERIZACION_RESULTADOS / args.corpus
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    print(f"Cargando corpus '{args.corpus}' (analisis intrinseco, sin dividir train/dev/test)...")
    df = cargar_corpus_variante(args.corpus)
    oraciones = df["shiwilu"].astype(str).tolist()
    etiquetas = df["intencion"].to_numpy()
    print(f"{len(df)} oraciones, {len(set(etiquetas))} categorias")

    filas = []
    proyecciones_tsne, proyecciones_umap = {}, {}
    for nombre_modelo in args.modelos:
        print(f"\n=== Modelo: {nombre_modelo} ===")
        print("Extrayendo las 4 estrategias de caracterizacion...")
        embeddings_por_estrategia = extraer_todas_las_estrategias(oraciones, nombre_modelo)

        for estrategia, X in embeddings_por_estrategia.items():
            metricas = evaluar_interseco(X, etiquetas)
            print(f"  {estrategia:<20} silueta={metricas['silueta_coseno']:.4f}  "
                  f"DB={metricas['davies_bouldin']:.4f}  CH={metricas['calinski_harabasz']:.2f}")
            filas.append({"modelo": nombre_modelo, "estrategia": estrategia, **metricas})

            if not args.sin_proyeccion:
                proyecciones_tsne[(nombre_modelo, estrategia)] = proyectar_2d(X, "tsne")
                umap_coords = proyectar_2d(X, "umap")
                if umap_coords is not None:
                    proyecciones_umap[(nombre_modelo, estrategia)] = umap_coords

    resultado = pd.DataFrame(filas)
    salida = carpeta_salida / "metricas_intrinsecas.csv"
    resultado.to_csv(salida, index=False, encoding="utf-8")
    print(f"\nResultados guardados en {salida}")

    print("\n=== Orden de separabilidad por silueta (coseno), de mayor a menor ===")
    print(resultado.sort_values("silueta_coseno", ascending=False)
                    [["modelo", "estrategia", "silueta_coseno", "davies_bouldin", "calinski_harabasz"]]
                    .to_string(index=False))

    if not args.sin_proyeccion:
        carpeta_2d = carpeta_salida / "proyeccion_2d"
        print(f"\nGenerando proyecciones 2D en {carpeta_2d}...")
        graficar_grilla(proyecciones_tsne, etiquetas, args.modelos, "tsne",
                         carpeta_2d / "tsne_grid.png",
                         f"Proyeccion t-SNE por modelo x estrategia — corpus: {args.corpus}")
        if proyecciones_umap:
            graficar_grilla(proyecciones_umap, etiquetas, args.modelos, "umap",
                             carpeta_2d / "umap_grid.png",
                             f"Proyeccion UMAP por modelo x estrategia — corpus: {args.corpus}")
        else:
            print("  umap-learn no esta instalado: se omitio umap_grid.png (`pip install umap-learn`).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
