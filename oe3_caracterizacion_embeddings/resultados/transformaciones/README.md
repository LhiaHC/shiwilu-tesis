# Silueta tras transformar los embeddings (complemento del OE3)

Salida del cuaderno [`colab_geometria_embeddings.ipynb`](../../colab_geometria_embeddings.ipynb) (módulo [`transformaciones_embeddings.py`](../../transformaciones_embeddings.py)).
3 modelos × 4 poolings × 3 corpus (original, + retrotraducción, + Generate-then-Refine) × 7 transformaciones no supervisadas fijadas de antemano, con 50 permutaciones de las etiquetas
como referencia del azar. Los embeddings se guardaron solo en Drive. No usa la API.

| Archivo | Contenido |
|---|---|
| `silueta_transformaciones.csv` | Una fila por corpus, modelo, pooling y transformación: silueta coseno, silueta con etiquetas al azar (media, p2.5, p97.5), Davies-Bouldin, Calinski-Harabasz; incluye el control de n-gramas (TF-IDF) |
| `correlacion_con_f1.csv` | Spearman entre la silueta (y Davies-Bouldin) de cada transformación y el F1 del OE4 |
| `fig_silueta_transformaciones.png` | Silueta por transformación (corpus original) con el percentil 97.5 del azar y el control de n-gramas |

## Resultado (corpus original, 12 combinaciones)

| Transformación | Silueta media | Máxima | Sobre el azar (p97.5) | Azar p97.5 |
|---|---|---|---|---|
| crudo (OE3) | −0.031 | −0.010 | 4 de 12 | −0.026 |
| centrado | −0.009 | +0.003 | 9 de 12 | −0.019 |
| estandarizado | −0.004 | +0.004 | 12 de 12 | −0.018 |
| sin 1 componente | −0.002 | +0.007 | 11 de 12 | −0.017 |
| sin 7 componentes | +0.001 | +0.004 | 12 de 12 | −0.014 |
| PCA 50 | −0.013 | 0.000 | 9 de 12 | −0.024 |
| PCA 50 blanqueado | +0.011 | +0.014 | 12 de 12 | −0.017 |

- **El valor del azar no es 0 y depende del espacio** (−0.037 de media en el espacio crudo, −0.015 a −0.029 en los transformados). Parte de lo «negativo» del OE3 era la geometría del espacio crudo.
- **Corregida la geometría, la silueta sube a ≈ 0 o ligeramente positiva y casi todas las combinaciones quedan por encima del azar, pero la magnitud sigue siendo mínima** (máximo +0.014 en el corpus original; +0.035 con sintéticos de GtR). Ninguna combinación queda significativamente por debajo del azar.
- **La ventaja de mBERT sobre LaBSE en silueta era en buena parte geométrica**: crudo, mBERT −0.013 y LaBSE −0.031; con estandarizado ambos ≈ +0.001. XLM-R queda por detrás en la mayoría de las transformaciones.
- **Control sin modelo de lenguaje** (TF-IDF de n-gramas, corpus original): silueta 0.024 en crudo y 0.045 reducido a 50 dimensiones (SVD), mayor que cualquier embedding en una transformación comparable (PCA 50: ≈ 0.000). Es coherente con que los n-gramas superen a los modelos congelados en F1.
- **Ninguna transformación mejora la relación con el F1**: Spearman silueta-F1 (36 combinaciones) 0.75 en crudo frente a 0.42 a 0.75 con las transformaciones.
- Davies-Bouldin (6.7 a 8.0) y Calinski-Harabasz (4.4 a 7.4) casi no cambian.

## Limitaciones

- Las transformaciones se ajustan con las mismas oraciones que se evalúan (sin usar etiquetas); en los corpus con sintéticos incluyen también las oraciones sintéticas.
- El blanqueo completo no es posible (menos oraciones que dimensiones): se usa PCA a 50 componentes.
- 50 permutaciones dan un percentil 97.5 aproximado del azar.
- La silueta es una medida global: un espacio puede separarse bien con un plano (lo que hace un clasificador) y tener silueta baja.
