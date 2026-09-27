# Caracterización de embeddings (OE3, R7-R8)

Caracteriza, para los mismos 3 modelos usados como baselines (LaBSE, mBERT,
XLM-R), **4 estrategias de extracción de representación** ("pooling") sobre
las mismas activaciones internas, y evalúa su calidad de agrupamiento de
forma **intrínseca** (sin clasificador): qué tan bien separan, por sí solas,
las 7 categorías de intención del corpus.

- **Estrategias (R7):**
  - `cls` — vector del token especial `[CLS]`/`<s>` de la última capa.
  - `mean_pooling` — promedio de todos los tokens de la oración (última capa).
  - `max_pooling` — máximo por dimensión entre todos los tokens (última capa).
  - `combinacion_capas` — mean pooling sobre el promedio de las últimas 4
    capas ocultas (Devlin et al., 2019, sección 5.3).

  Las 4 se derivan de un único forward pass por lote
  (`output_hidden_states=True`), a diferencia de `2_baselines/comun.py`
  (que usa la extracción "de fábrica" de cada modelo, más simple).

- **Métricas intrínsecas (R8):** coeficiente de silueta (distancia coseno),
  índice de Davies-Bouldin e índice de Calinski-Harabasz (estos dos últimos
  sobre vectores normalizados a norma 1, para ser consistentes con coseno).
  No hay clasificador ni "método ganador": las 4 estrategias se **reportan y
  comparan**, no se filtran — el contraste con la evaluación extrínseca
  (F1 de clasificación) se hace en [`5_sintesis_r9/`](../5_sintesis_r9/README.md).

- **Texto:** todas las oraciones se normalizan antes de extraer los embeddings —
  minúsculas, sin puntuación y sin distinguir tildes/ñ
  (`comun.quitar_puntuacion(..., quitar_tildes=True)`) — porque en el corpus los
  signos de pregunta y las mayúsculas delatan categorías (ver más abajo). Es la
  misma condición `sin_puntuacion` que se usa en los 12 experimentos extrínsecos
  (decisión 2026-09-27, ver
  [`../3_baselines_y_aumento_datos/tecnicas_aumento/README.md`](../3_baselines_y_aumento_datos/tecnicas_aumento/README.md)).

- **Corpus:** R7 pide correr esto "sobre el corpus original y los corpus
  aumentados obtenidos en R6". Este script soporta las 3 variantes de
  texto vía `--corpus`:
  - `original` (default) — las 700 oraciones del corpus.
  - `retrotraduccion` — original + filas `aprobado` de la retrotraducción.
  - `generate_then_refine` — original + filas `aprobado` de Generate-then-Refine.

  En ambas variantes se descartan las filas sintéticas idénticas a una oración real
  (ignorando mayúsculas, puntuación y tildes): el NMT de F. Prado memorizó parte del corpus.

  **Mixup queda fuera:** no genera oraciones en shiwilu, interpola vectores
  ya extraídos de un modelo específico (ver
  [`../3_baselines_y_aumento_datos/tecnicas_aumento/mixup.py`](../3_baselines_y_aumento_datos/tecnicas_aumento/mixup.py)),
  y las 4 estrategias de pooling de aquí requieren un forward pass sobre
  texto crudo, que Mixup no produce.

## Uso

```bash
python 4_caracterizacion_embeddings/caracterizacion.py                          # corpus original, 3 modelos
python 4_caracterizacion_embeddings/caracterizacion.py --modelos labse xlmr
python 4_caracterizacion_embeddings/caracterizacion.py --corpus retrotraduccion
python 4_caracterizacion_embeddings/caracterizacion.py --corpus generate_then_refine
python 4_caracterizacion_embeddings/caracterizacion.py --sin-proyeccion         # omite t-SNE/UMAP, solo metricas
```

No requiere GPU ni clave de API — corre en CPU (el corpus original tarda
pocos minutos; las variantes aumentadas tardan más por tener más oraciones).

## Salida

`4_caracterizacion_embeddings/resultados/<corpus>/` (uno por cada variante):
- `metricas_intrinsecas.csv` — una fila por combinación modelo × estrategia.
- `proyeccion_2d/tsne_grid.png` — grilla (modelo × estrategia) de la
  proyección t-SNE a 2D, coloreada por categoría de intención.
- `proyeccion_2d/umap_grid.png` — ídem, proyección UMAP (requiere
  `umap-learn`; se omite automáticamente si no está instalado).

## Resultado de referencia (ya ejecutado)

### Corpus original — orden de separabilidad por silueta (coseno)

| Modelo | Estrategia | Silueta | Davies-Bouldin | Calinski-Harabasz |
|---|---|---|---|---|
| mBERT | Mean pooling | -0.0096 | 6.74 | 6.21 |
| mBERT | Combinación de capas | -0.0113 | 6.89 | 6.03 |
| mBERT | Max pooling | -0.0124 | 6.97 | 5.62 |
| mBERT | CLS | -0.0183 | 6.94 | 6.05 |
| LaBSE | Combinación de capas | -0.0254 | 6.95 | 6.24 |
| LaBSE | Mean pooling | -0.0296 | 6.99 | 6.17 |
| LaBSE | CLS | -0.0317 | 7.14 | 5.82 |
| LaBSE | Max pooling | -0.0366 | 7.19 | 5.73 |
| XLM-R | Max pooling | -0.0387 | 7.84 | 5.26 |
| XLM-R | Mean pooling | -0.0510 | 7.90 | 6.08 |
| XLM-R | CLS | -0.0513 | 7.73 | 6.33 |
| XLM-R | Combinación de capas | -0.0567 | 7.70 | 6.87 |

**Sobre texto normalizado, ningún modelo separa las categorías de intención.** Todas
las siluetas son ≈ 0 o negativas (entre -0.057 y -0.010): los vecinos más
cercanos de una oración, en el espacio de embeddings crudos, no comparten su
categoría más que al azar. Por modelo, la mejor estrategia es
mBERT Mean pooling (-0.0096); LaBSE Combinación de capas (-0.0254); XLM-R Max pooling (-0.0387).
La Combinación de capas de XLM-R es la peor de las 12 combinaciones, y **mBERT domina
las 4 primeras posiciones**: sus 4 estrategias separan mejor que cualquier estrategia
de LaBSE o XLM-R — coincide con que mBERT también es el mejor en el eje extrínseco
(ver [`../5_sintesis_r9/README.md`](../5_sintesis_r9/README.md)).

**Por qué se calcula sobre texto normalizado.** Con el texto crudo, LaBSE + CLS daba
una silueta de +0.049 y parecía el mejor modelo. Esa ventaja venía de los atajos del
corpus, no de las palabras: los `¿?` y las MAYÚSCULAS (100% de DES, PRG y REQUEST)
separan esas categorías del resto. Al pasar el texto a minúsculas y quitar la
puntuación, la ventaja desaparece y todos los modelos quedan en ≈ 0. (Los resultados
con texto crudo están en el historial de git.) La normalización **preserva el
apóstrofo** (oclusiva glotal, un fonema real del shiwilu, ej. `pante'chek`) en
cualquier posición de la palabra — solo se quitan signos de puntuación reales
(`¿ ? ¡ ! . , ; :`), mayúsculas y tildes/ñ.

### Efecto del aumento de datos (R6) sobre la calidad intrínseca

Correr `caracterizacion.py` sobre los corpus aumentados por las 2 técnicas de texto
(se descartan las filas sintéticas que copian una oración real) casi no cambia la
silueta: las diferencias son de ≤ 0.01 en valor absoluto, o sea, prácticamente cero.

| Modelo | Técnica | Silueta original | Silueta aumentada | Delta |
|---|---|---|---|---|
| LaBSE | retrotraducción | -0.0254 | -0.0335 | -0.0081 |
| mBERT | retrotraducción | -0.0096 | -0.0126 | -0.0029 |
| XLM-R | retrotraducción | -0.0387 | -0.0295 | +0.0092 |
| LaBSE | generate_then_refine | -0.0254 | -0.0229 | +0.0025 |
| mBERT | generate_then_refine | -0.0096 | -0.0080 | +0.0017 |
| XLM-R | generate_then_refine | -0.0387 | -0.0352 | +0.0035 |

Como el punto de partida es ≈ 0 en todos los casos, no se puede afirmar que el
aumento mejore ni empeore la organización intrínseca de los embeddings. La
síntesis con el resultado extrínseco está en
[`5_sintesis_r9/README.md`](../5_sintesis_r9/README.md).
