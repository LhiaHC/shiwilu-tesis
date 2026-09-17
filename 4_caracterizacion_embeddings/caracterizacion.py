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

Entrada: corpus/corpus_shiwilu_final.csv (columna `shiwilu`, corpus completo
         sin dividir train/dev/test: R7/R8 son un analisis intrinseco, no
         una evaluacion de clasificador).
Salida:  4_caracterizacion_embeddings/resultados/
    metricas_intrinsecas.csv   una fila por combinacion modelo x estrategia
    proyeccion_2d/<modelo>_<estrategia>.png   t-SNE y UMAP (si esta instalado)

Uso:
    python 4_caracterizacion_embeddings/caracterizacion.py
    python 4_caracterizacion_embeddings/caracterizacion.py --modelos labse xlmr
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import normalize

import _ruta_raiz  # noqa: F401  (deja importable el paquete `shiwilu`)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2_baselines"))
from comun import MODELOS  # noqa: E402  (mismos 3 modelos que los baselines)

from shiwilu.rutas import CARACTERIZACION_RESULTADOS, CORPUS_CSV  # noqa: E402
from shiwilu.taxonomia import INTENCIONES  # noqa: E402

ESTRATEGIAS = ["cls", "mean_pooling", "max_pooling", "combinacion_capas"]
N_CAPAS_COMBINACION = 4   # ultimas N capas para "combinacion de capas"


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
    args = ap.parse_args()

    CARACTERIZACION_RESULTADOS.mkdir(parents=True, exist_ok=True)

    print("Cargando corpus completo (analisis intrinseco, sin dividir train/dev/test)...")
    df = pd.read_csv(CORPUS_CSV)
    oraciones = df["shiwilu"].astype(str).tolist()
    etiquetas = df["intencion"].to_numpy()
    print(f"{len(df)} oraciones, {len(set(etiquetas))} categorias")

    filas = []
    for nombre_modelo in args.modelos:
        print(f"\n=== Modelo: {nombre_modelo} ===")
        print("Extrayendo las 4 estrategias de caracterizacion...")
        embeddings_por_estrategia = extraer_todas_las_estrategias(oraciones, nombre_modelo)

        for estrategia, X in embeddings_por_estrategia.items():
            metricas = evaluar_interseco(X, etiquetas)
            print(f"  {estrategia:<20} silueta={metricas['silueta_coseno']:.4f}  "
                  f"DB={metricas['davies_bouldin']:.4f}  CH={metricas['calinski_harabasz']:.2f}")
            filas.append({"modelo": nombre_modelo, "estrategia": estrategia, **metricas})

    resultado = pd.DataFrame(filas)
    salida = CARACTERIZACION_RESULTADOS / "metricas_intrinsecas.csv"
    resultado.to_csv(salida, index=False, encoding="utf-8")
    print(f"\nResultados guardados en {salida}")

    print("\n=== Orden de separabilidad por silueta (coseno), de mayor a menor ===")
    print(resultado.sort_values("silueta_coseno", ascending=False)
                    [["modelo", "estrategia", "silueta_coseno", "davies_bouldin", "calinski_harabasz"]]
                    .to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
