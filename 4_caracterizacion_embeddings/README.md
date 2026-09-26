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

- **Texto:** las oraciones se pasan a minúsculas y se les quita la puntuación antes de
  extraer los embeddings (`comun.quitar_puntuacion`), porque en el corpus los signos de
  pregunta y las mayúsculas delatan categorías (ver más abajo).

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
| mBERT | Combinación de capas | -0.0054 | 6.62 | 7.58 |
| mBERT | Max pooling | -0.0079 | 6.88 | 6.63 |
| mBERT | CLS | -0.0086 | 6.78 | 7.36 |
| mBERT | Mean pooling | -0.0099 | 6.62 | 7.08 |
| LaBSE | Combinación de capas | -0.0202 | 7.07 | 6.73 |
| LaBSE | CLS | -0.0218 | 7.31 | 6.37 |
| LaBSE | Mean pooling | -0.0266 | 7.14 | 6.64 |
| XLM-R | Max pooling | -0.0350 | 7.66 | 8.12 |
| XLM-R | Combinación de capas | -0.0364 | 7.16 | 8.97 |
| LaBSE | Max pooling | -0.0366 | 7.27 | 6.20 |
| XLM-R | Mean pooling | -0.0368 | 7.45 | 9.55 |
| XLM-R | CLS | -0.1097 | 7.13 | 10.27 |

**Sobre texto normalizado, ningún modelo separa las categorías de intención.** Todas
las siluetas son ≈ 0 o negativas (entre -0.110 y -0.005): los vecinos más
cercanos de una oración, en el espacio de embeddings crudos, no comparten su
categoría más que al azar. Por modelo, la mejor estrategia es
mBERT Combinación de capas (-0.0054); LaBSE Combinación de capas (-0.0202); XLM-R Max pooling (-0.0350).
El CLS de XLM-R es el peor de las 12 combinaciones.

**Por qué se calcula sobre texto normalizado.** Con el texto crudo, LaBSE + CLS daba
una silueta de +0.049 y parecía el mejor modelo. Esa ventaja venía de los atajos del
corpus, no de las palabras: los `¿?` y las MAYÚSCULAS (100% de DES, PRG y REQUEST)
separan esas categorías del resto. Al pasar el texto a minúsculas y quitar la
puntuación, la ventaja desaparece y todos los modelos quedan en ≈ 0. (Los resultados
con texto crudo están en el historial de git.)

### Efecto del aumento de datos (R6) sobre la calidad intrínseca

Correr `caracterizacion.py` sobre los corpus aumentados por las 2 técnicas de texto
(se descartan las filas sintéticas que copian una oración real) casi no cambia la
silueta: las diferencias son de ≤ 0.01 en valor absoluto, o sea, prácticamente cero.

| Modelo | Técnica | Silueta original | Silueta aumentada | Delta |
|---|---|---|---|---|
| LaBSE | retrotraducción | -0.0202 | -0.0304 | -0.0103 |
| mBERT | retrotraducción | -0.0054 | -0.0095 | -0.0040 |
| XLM-R | retrotraducción | -0.0350 | -0.0365 | -0.0015 |
| LaBSE | generate_then_refine | -0.0202 | -0.0158 | +0.0044 |
| mBERT | generate_then_refine | -0.0054 | -0.0040 | +0.0014 |
| XLM-R | generate_then_refine | -0.0350 | -0.0307 | +0.0043 |

Como el punto de partida es ≈ 0 en todos los casos, no se puede afirmar que el
aumento mejore ni empeore la organización intrínseca de los embeddings. La
síntesis con el resultado extrínseco está en
[`5_sintesis_r9/README.md`](../5_sintesis_r9/README.md).
