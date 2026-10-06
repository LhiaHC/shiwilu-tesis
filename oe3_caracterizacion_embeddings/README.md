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
  (`output_hidden_states=True`), a diferencia de
  [`shiwilu/clasificacion.py`](../shiwilu/clasificacion.py) (que usa la extracción
  "de fábrica" de cada modelo, más simple).

- **Métricas intrínsecas (R8):** coeficiente de silueta (distancia coseno),
  índice de Davies-Bouldin e índice de Calinski-Harabasz (estos dos últimos
  sobre vectores normalizados a norma 1, para ser consistentes con coseno).
  No hay clasificador ni "método ganador": las 4 estrategias se **reportan y
  comparan**, no se filtran — el contraste con la evaluación extrínseca
  (F1 de clasificación) se hace en [`oe4_sintesis/`](../oe4_sintesis/README.md).

- **Texto:** todas las oraciones se normalizan antes de extraer los embeddings —
  minúsculas, sin puntuación y sin distinguir tildes/ñ
  (`shiwilu.clasificacion.quitar_puntuacion(..., quitar_tildes=True)`) — porque en el corpus los
  signos de pregunta y las mayúsculas delatan categorías (ver más abajo). Es la
  misma condición `sin_puntuacion` que se usa en los 12 experimentos extrínsecos
  (decisión 2026-09-27, ver
  [`../oe2_aumento_de_datos/tecnicas_aumento/README.md`](../oe2_aumento_de_datos/tecnicas_aumento/README.md)).

- **Corpus:** R7 pide correr esto "sobre el corpus original y los corpus
  aumentados obtenidos en R6". Este script soporta las 3 variantes de
  texto vía `--corpus`:
  - `original` (default) — las 700 oraciones del corpus.
  - `retrotraduccion` — original + filas `aprobado` de la retrotraducción.
  - `generate_then_refine` — original + filas `aprobado` de Generate-then-Refine.

  En ambas variantes se descartan las filas sintéticas idénticas a una oración real
  (ignorando mayúsculas, puntuación y tildes) y las repetidas entre sí. Desde 2026-10-05 usan los **mismos sintéticos que la
  evaluación de OE2**:
  - `retrotraduccion`: el catálogo `tecnicas_aumento/salidas/retrotraduccion_pool.csv` (601 filas; +480 oraciones nuevas).
  - `generate_then_refine`: la caché de OE2 de 120 por categoría, con los marcadores de las fuentes, del **fold 0**
    (`salidas/en_linea_marcadores_fuentes/generate_then_refine/fold0_pool*.csv`; +493 oraciones únicas aprobadas). Se usa un solo fold
    para que el volumen sea parecido al de la retrotraducción (unir los 5 folds daría 2081 oraciones, el triple del corpus).
  - Las versiones anteriores (archivos del split único del 2026-09-24, antes de la limpieza del corpus; +336 y +62 oraciones) siguen disponibles como
    `--corpus retrotraduccion_split_unico` y `--corpus generate_then_refine_split_unico`; sus resultados están en `resultados/*_split_unico/`.
    Una copia completa de todos los resultados previos al cambio está en `resultados_anteriores/`.

  **Mixup queda fuera:** no genera oraciones en shiwilu, interpola vectores
  ya extraídos de un modelo específico (ver
  [`../oe2_aumento_de_datos/tecnicas_aumento/mixup.py`](../oe2_aumento_de_datos/tecnicas_aumento/mixup.py)),
  y las 4 estrategias de pooling de aquí requieren un forward pass sobre
  texto crudo, que Mixup no produce.

## Uso

```bash
python oe3_caracterizacion_embeddings/caracterizacion.py                          # corpus original, 3 modelos
python oe3_caracterizacion_embeddings/caracterizacion.py --modelos mbert xlmr --sin-proyeccion   # actualiza solo esos modelos
python oe3_caracterizacion_embeddings/caracterizacion.py --corpus retrotraduccion
python oe3_caracterizacion_embeddings/caracterizacion.py --corpus generate_then_refine
python oe3_caracterizacion_embeddings/caracterizacion.py --sin-proyeccion         # omite t-SNE/UMAP, solo metricas
```

No requiere GPU ni clave de API — corre en CPU (el corpus original tarda
pocos minutos; las variantes aumentadas tardan más por tener más oraciones).

## Salida

`oe3_caracterizacion_embeddings/resultados/<corpus>/` (uno por cada variante):
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
(ver [`../oe4_sintesis/README.md`](../oe4_sintesis/README.md)).

**Por qué se calcula sobre texto normalizado.** Con el texto crudo, LaBSE + CLS daba
una silueta de +0.049 y parecía el mejor modelo. Esa ventaja venía de los atajos del
corpus, no de las palabras: los `¿?` y las MAYÚSCULAS (100% de DES, PRG y REQUEST)
separan esas categorías del resto. Al pasar el texto a minúsculas y quitar la
puntuación, la ventaja desaparece y todos los modelos quedan en ≈ 0. (Los resultados
con texto crudo están en el historial de git.) La normalización **preserva el
apóstrofo** (oclusiva glotal, un fonema real del shiwilu, ej. `pante'chek`) en
cualquier posición de la palabra — solo se quitan signos de puntuación reales
(`¿ ? ¡ ! . , ; :`), mayúsculas y tildes/ñ.

### Las 36 evaluaciones (3 modelos × 3 corpus × 4 estrategias): silueta (coseno)

Corpus sin aumento / con retrotraducción actual / con GtR actual (resultados en `resultados/<corpus>/metricas_intrinsecas.csv`):

| Modelo | Estrategia | Sin aumento | Retrotraducción | GtR |
|---|---|---|---|---|
| LaBSE | CLS | -0.0317 | -0.0310 | -0.0017 |
| LaBSE | Mean pooling | -0.0296 | -0.0323 | +0.0012 |
| LaBSE | Max pooling | -0.0366 | -0.0413 | -0.0059 |
| LaBSE | Combinación de capas | -0.0254 | -0.0301 | +0.0051 |
| mBERT | CLS | -0.0183 | -0.0252 | +0.0027 |
| mBERT | Mean pooling | -0.0096 | -0.0126 | +0.0181 |
| mBERT | Max pooling | -0.0124 | -0.0161 | +0.0136 |
| mBERT | Combinación de capas | -0.0113 | -0.0106 | +0.0138 |
| XLM-R | CLS | -0.0513 | -0.0475 | -0.0253 |
| XLM-R | Mean pooling | -0.0510 | -0.0499 | -0.0220 |
| XLM-R | Max pooling | -0.0387 | -0.0356 | -0.0138 |
| XLM-R | Combinación de capas | -0.0567 | -0.0611 | -0.0269 |

- **Retrotraducción:** no cambia la silueta de forma apreciable (diferencias entre -0.007 y +0.004); Davies-Bouldin empeora de ~7.3 a ~8.3.
- **GtR:** la silueta sube entre +0.021 y +0.031 en las 12 combinaciones, Davies-Bouldin baja de ~7.3 a ~5.8 y Calinski-Harabasz sube de ~6 a ~15.
  mBERT pasa a valores positivos (hasta +0.018). **Pero no significa que las oraciones reales queden mejor organizadas:** al separar la silueta por tipo de punto
  (`resultados/silueta_real_vs_sintetico.csv`, mean pooling), las sintéticas de GtR tienen silueta positiva por sí solas y las reales casi no cambian:

| Modelo | Original (700) | Aumentado, solo reales | Aumentado, solo sintéticas GtR |
|---|---|---|---|
| LaBSE | -0.0296 | -0.0268 | **+0.0409** |
| mBERT | -0.0096 | -0.0045 | **+0.0503** |
| XLM-R | -0.0510 | -0.0566 | **+0.0272** |

  Es decir, lo que sube es la silueta de las propias oraciones sintéticas (muy "típicas" de su categoría, coherente con que en OE2 sus etiquetas coinciden con las de un clasificador real el 85% de las veces), no la de las reales.
  Con retrotraducción, las sintéticas están tan mal agrupadas como las reales (-0.017 a -0.055).
- **Ranking:** el orden de los modelos no cambia en ningún corpus (mBERT > LaBSE > XLM-R); la mejor estrategia por modelo es la misma (LaBSE: combinación de capas, XLM-R: max pooling; mBERT: mean pooling, o combinación de capas con retrotraducción).
- **Con los sintéticos antiguos** (split único) el efecto era ≤ 0.01 en todos los casos, porque GtR solo agregaba 62 oraciones; el cambio de ahora se debe al mayor volumen y a los filtros corregidos de GtR.
- Todas las siluetas siguen siendo ≈ 0 o negativas salvo las de GtR con sus propios puntos sintéticos: ningún modelo agrupa por intención las oraciones reales.

La síntesis con el resultado extrínseco está en [`oe4_sintesis/README.md`](../oe4_sintesis/README.md) (todavía usa el protocolo A de OE2 y las cifras antiguas de OE3).


### F1 de clasificación de las 36 combinaciones

Para poder comparar la silueta con el F1, `f1_por_pooling.py` entrena el clasificador del protocolo B de OE2 (Regresión Logística, `C` por CV interna K=5, 5 folds congelados, sintéticos generados solo con el entrenamiento de cada partición)
sobre los embeddings de cada estrategia de pooling. Resultados en `resultados/f1_por_pooling.csv` (con IC95%) y, por corrida, en `../oe2_aumento_de_datos/evaluacion/resultados/cv_interna_pooling/`. F1 macro (silueta entre paréntesis):

| Modelo | Pooling | Sin aumento | Retrotraducción | GtR 120 |
|---|---|---|---|---|
| mBERT | CLS | 0.617 (-0.018) | 0.601 (-0.025) | 0.636 (+0.003) |
| mBERT | Mean pooling | **0.667** (-0.010) | 0.616 (-0.013) | 0.657 (+0.018) |
| mBERT | Max pooling | 0.658 (-0.012) | 0.626 (-0.016) | 0.663 (+0.014) |
| mBERT | Combinación de capas | 0.646 (-0.011) | 0.638 (-0.011) | 0.660 (+0.014) |
| LaBSE | CLS | 0.591 (-0.032) | 0.585 (-0.031) | 0.588 (-0.002) |
| LaBSE | Mean pooling | 0.596 (-0.030) | 0.589 (-0.032) | 0.608 (+0.001) |
| LaBSE | Max pooling | 0.622 (-0.037) | 0.606 (-0.041) | 0.622 (-0.006) |
| LaBSE | Combinación de capas | 0.615 (-0.025) | 0.594 (-0.030) | 0.623 (+0.005) |
| XLM-R | CLS | 0.544 (-0.051) | 0.524 (-0.047) | 0.573 (-0.025) |
| XLM-R | Mean pooling | 0.573 (-0.051) | 0.550 (-0.050) | 0.586 (-0.022) |
| XLM-R | Max pooling | 0.592 (-0.039) | 0.557 (-0.036) | 0.608 (-0.014) |
| XLM-R | Combinación de capas | 0.600 (-0.057) | 0.563 (-0.061) | 0.625 (-0.027) |

- **El mejor de las 36 es mBERT + mean pooling + sin aumento (F1 0.667 [0.631, 0.699])**, que reproduce el 0.665 de OE2; los seis mejores son todos de mBERT.
- **Silueta y F1 se relacionan, pero no lo suficiente para elegir el pooling:** Spearman 0.75 entre las 36; 0.74 dentro de mBERT, pero solo 0.27 en LaBSE y 0.39 en XLM-R. El mejor pooling sin aumento por F1 es mean (mBERT), max (LaBSE) y combinación de capas (XLM-R);
  por silueta es mean (mBERT), combinación de capas (LaBSE) y max (XLM-R).
- **El pooling importa:** en LaBSE y XLM-R la extracción "de fábrica" de OE2 no es la mejor (LaBSE 0.587 contra 0.622 con max pooling; XLM-R 0.570 contra 0.600 con combinación de capas), lo que reduce la brecha con mBERT (0.667 a 0.045 y 0.067). Elegir el mejor de 4 poolings sobre el mismo test es optimista.
- **Aumento (media de los 4 poolings):** Retrotraducción baja el F1 en los tres modelos (-0.013 a -0.029); GtR lo deja igual o algo mejor (+0.007 mBERT, +0.004 LaBSE, +0.020 XLM-R), sin prueba de significancia.
