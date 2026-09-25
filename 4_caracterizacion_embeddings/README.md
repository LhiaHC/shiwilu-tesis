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

- **Corpus:** R7 pide correr esto "sobre el corpus original y los corpus
  aumentados obtenidos en R6". Este script soporta las 3 variantes de
  texto vía `--corpus`:
  - `original` (default) — las 700 oraciones del corpus.
  - `retrotraduccion` — original + filas `aprobado` de la retrotraducción.
  - `generate_then_refine` — original + filas `aprobado` de Generate-then-Refine.

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
| LaBSE | cls | **0.0493** | 5.73 | 37.19 |
| LaBSE | mean_pooling | 0.0470 | 5.78 | 25.06 |
| LaBSE | combinacion_capas | 0.0455 | 5.67 | 30.04 |
| LaBSE | max_pooling | 0.0423 | 5.92 | 23.82 |
| mBERT | mean_pooling | 0.0303 | 5.51 | 50.42 |
| XLM-R | max_pooling | 0.0151 | 5.67 | 26.38 |
| mBERT | cls | 0.0001 | 5.15 | 73.21 |
| XLM-R | cls | -0.0868 | 5.32 | 72.28 |

LaBSE domina en todas sus 4 estrategias — es, con margen, el modelo cuyos
embeddings crudos separan mejor las 7 categorías sin necesidad de un
clasificador. Esto **no coincide** con el ganador extrínseco (XLM-R, ver
[`2_baselines/README.md`](../2_baselines/README.md) y
[`3_baselines_y_aumento_datos/`](../3_baselines_y_aumento_datos/)) — la
síntesis de este contraste está en
[`5_sintesis_r9/README.md`](../5_sintesis_r9/README.md).

### Efecto del aumento de datos (R6) sobre la calidad intrínseca

Correr `caracterizacion.py` sobre los corpus aumentados por las 2 técnicas
de texto (retrotraducción y Generate-then-Refine) muestra que **casi
siempre degradan la separabilidad intrínseca** (silueta más baja que en el
corpus original; mBERT y XLM-R llegan a silueta negativa con
retrotraducción) — la única excepción es LaBSE con Generate-then-Refine,
donde la silueta sube ligeramente (+0.0059, un cambio pequeño, no concluyente). Es decir, agregar las oraciones sintéticas casi siempre hace
más difuso el espacio de embeddings — aunque, paradójicamente, algunas de
esas mismas combinaciones mejoran el F1 de clasificación (ver
[`5_sintesis_r9/`](../5_sintesis_r9/README.md)): un clasificador supervisado
puede aprovechar datos más diversos/ruidosos para generalizar mejor, aun
cuando el agrupamiento no supervisado de esos mismos embeddings se vuelva
menos limpio.

| Modelo | Técnica | Silueta original | Silueta aumentada | Delta |
|---|---|---|---|---|
| LaBSE | retrotraducción | 0.0493 | 0.0206 | -0.0288 |
| mBERT | retrotraducción | 0.0303 | -0.0071 | -0.0374 |
| XLM-R | retrotraducción | 0.0151 | -0.0098 | -0.0249 |
| LaBSE | generate_then_refine | 0.0493 | 0.0553 | +0.0059 |
| mBERT | generate_then_refine | 0.0303 | 0.0222 | -0.0081 |
| XLM-R | generate_then_refine | 0.0151 | 0.0097 | -0.0054 |
