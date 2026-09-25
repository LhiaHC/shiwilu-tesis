# Baselines de clasificación (OE2, R4)

Tres baselines de referencia para la clasificación de intenciones en
shiwilu, todos con la misma estructura: **un embedding multilingüe
congelado (sin ajuste fino) + Regresión Logística, sin ningún aumento de
datos.** Sirven como línea base mínima contra la cual comparar tanto las
técnicas de aumento de datos (OE2) como los demás métodos de
caracterización de embeddings (OE3).

- **Embeddings (elige uno):**
  - `sentence-transformers/LaBSE` (Feng et al., 2022)
  - `bert-base-multilingual-cased` — mBERT (Devlin et al., 2019)
  - `xlm-roberta-base` — XLM-R (Conneau et al., 2020)

  Ninguno se ajusta (fine-tuning): solo se extraen representaciones de
  oración (LaBSE trae su propio pooling; mBERT/XLM-R usan mean pooling sobre
  la última capa, ponderado por la máscara de atención).
- **Clasificador:** `LogisticRegression` de scikit-learn, con `C` ajustado
  sobre el conjunto de desarrollo (grilla en `comun.py`).
- **Datos:** únicamente `corpus/corpus_shiwilu_final.csv` (700 pares),
  columna `shiwilu`, dividido 70/15/15 (train/dev/test) estratificado por
  categoría de intención — sin ninguna técnica de aumento aplicada. Misma
  división para los tres modelos, para que los resultados sean comparables.
  La división se hace agrupando por texto en shiwilu **normalizado**
  (`comun.dividir_train_dev_test`/`_normalizar_shiwilu`): el corpus tiene
  oraciones muy cortas que se repiten con distinta glosa en español,
  mayúsculas o puntuación (`"MUPALLI"`, `"¡PANTE'CHEK!"` vs. `"pante'chek"`),
  y antes de este ajuste una misma oración podía caer en train y en test a la vez. El split se calculó una sola vez y quedó **congelado** en `split_fijo.csv`: todos los scripts lo leen de ahí en vez de recalcularlo (así ningún cambio de código o de versión de librerías lo reordena y contamina las técnicas de aumento generadas a partir de train).

## Uso

```bash
# un modelo a la vez
python 2_baselines/baseline.py --modelo labse
python 2_baselines/baseline.py --modelo mbert
python 2_baselines/baseline.py --modelo xlmr

# los 3 de una vez, con tabla comparativa al final
python 2_baselines/correr_todos.py
```

No requiere GPU ni clave de API — corre en CPU en pocos minutos (mBERT y
XLM-R tardan más que LaBSE por ser más pesados de descargar/cargar, pero
igual corren bien sin GPU). Se puede correr localmente en VS Code sin
necesidad de Colab.

## Salida

`2_baselines/resultados/<modelo>/` (uno por cada modelo):
- `metricas.json` — F1 macro, F1 ponderado, exactitud (train/dev/test).
- `matriz_confusion.png` — matriz de confusión sobre el conjunto de prueba.
- `reporte_clasificacion.csv` — métricas desagregadas por categoría de intención.

## Resultado de referencia (ya ejecutado)

| Modelo | F1 macro | F1 ponderado | Exactitud |
|---|---|---|---|
| XLM-R | 0.7223 | 0.7241 | 0.7273 |
| mBERT | 0.6977 | 0.6948 | 0.6970 |
| LaBSE | 0.6943 | 0.6930 | 0.6970 |

Los 3 modelos quedan más cerca entre sí de lo que parecía antes de cerrar la
fuga de datos (ver más abajo) — XLM-R al frente, pero sin una distancia
grande. Con un conjunto de prueba de ~99 oraciones, el intervalo de
confianza (bootstrap, ver
[`../3_baselines_y_aumento_datos/bootstrap_ic.py`](../3_baselines_y_aumento_datos/bootstrap_ic.py))
de cada modelo es de ±0.04-0.05 en F1 y se solapan entre sí — ninguna
diferencia entre los 3 modelos es estadísticamente concluyente con esta
muestra. Interpretar con cautela antes de generalizar.

## Baselines triviales (piso de referencia, sin ningún embedding)

[`baseline_trivial.py`](baseline_trivial.py) corre dos baselines que **no
usan ningún modelo de lenguaje ni aprendizaje real**, para poder leer los
F1 de arriba con perspectiva: el corpus se construyó a partir de plantillas
y flashcards (no de lenguaje espontáneo), y varias oraciones repiten la
misma raíz shiwilu con variantes menores dentro de una misma categoría — así
que parte del F1 de los modelos reales puede reflejar esa repetición léxica
superficial, no comprensión semántica del shiwilu.

```bash
python 2_baselines/baseline_trivial.py
```

| Baseline | F1 macro | Qué mide |
|---|---|---|
| Mayoría (siempre predice `DES`) | 0.0309 | Piso absoluto — cualquier modelo real debe superarlo con margen. |
| 1-vecino-más-cercano por palabras compartidas (sin embeddings) | 0.3830 | Cuánto se puede clasificar solo por solapamiento léxico exacto, sin ninguna noción de significado. |

Los 3 modelos reales (0.69-0.72) superan claramente el 0.38 del vecino más
cercano por palabras — así que sí hay señal más allá de la simple
repetición léxica, pero la brecha (~0.3-0.35 puntos de F1) no es enorme para
un problema de 7 clases. Es una llamada a interpretar los resultados de OE2
con cautela, no evidencia de que estén invalidados.
