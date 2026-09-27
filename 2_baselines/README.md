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
  y antes de este ajuste una misma oración podía caer en train y en test a la vez. El split se calculó una sola vez y quedó **congelado** en `split_fijo.csv`: todos los scripts lo leen de ahí en vez de recalcularlo (así ningún cambio de código o de versión de librerías lo reordena y contamina las técnicas de aumento generadas a partir de train). El corpus mismo (`corpus/corpus_shiwilu_final.csv`, columna `shiwilu`) también se limpió una vez: 13 typos de comilla doble por apóstrofo, espacios dobles y mayúsculas inconsistentes — ver la nota de metodología en [`../3_baselines_y_aumento_datos/tecnicas_aumento/README.md`](../3_baselines_y_aumento_datos/tecnicas_aumento/README.md).

- **`corpus_normalizado.csv`** (`comun.cargar_corpus_normalizado()`): las mismas
  700 oraciones, con `shiwilu` ya normalizado — minúsculas, sin puntuación,
  sin tildes/ñ, apóstrofo (oclusiva glotal) siempre preservado. Es el archivo
  que se puede abrir para ver **exactamente** el texto que usan la validación
  cruzada y la caracterización de embeddings, en vez de confiar en que el
  código lo recalcula igual cada vez. Se regenera solo (automáticamente) si
  `corpus_shiwilu_final.csv` cambia.

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

## Resultado principal: validación cruzada de 5 folds, texto normalizado

Los scripts de esta carpeta (`baseline.py`, `correr_todos.py`) usan el split
único 70/15/15 y el texto crudo del corpus, cuyo test de ~99 oraciones da un F1
con ±0.09 de incertidumbre. La evaluación **vigente** de la tesis es la
validación cruzada de
[`../3_baselines_y_aumento_datos/validacion_cruzada.py`](../3_baselines_y_aumento_datos/validacion_cruzada.py)
(folds congelados en `folds_fijos.csv`, agrupados por texto normalizado),
**siempre sobre texto normalizado** (minúsculas, sin puntuación, sin
tildes/ñ — `--sin-puntuacion`): el corpus tiene atajos superficiales que
delatan la categoría (los signos `¿?` para PRG; las mayúsculas, 100% en
DES/PRG/REQUEST) y ya no se corre ni se mantiene la condición con texto crudo
más que como referencia histórica (tabla más abajo).

| Modelo (sin aumento) | F1 macro (normalizado) | IC95% |
|---|---|---|
| mBERT | 0.6575 | [0.622, 0.691] |
| LaBSE | 0.5873 | [0.549, 0.621] |
| XLM-R | 0.5481 | [0.510, 0.583] |

**mBERT es significativamente mejor que los otros dos**: mBERT − XLM-R = +0.109
(IC95% [+0.071, +0.148]) y mBERT − LaBSE = +0.070 (IC95% [+0.031, +0.108]).
XLM-R vs. LaBSE no es distinguible de cero (-0.039, IC95% [-0.081, +0.001]).

## Resultado con split único y texto crudo (histórico, congelado)

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
| Mayoría (siempre predice `DES`) | 0.0309 (CV: 0.0880) | Piso absoluto — cualquier modelo real debe superarlo con margen. |
| 1-vecino-más-cercano por palabras compartidas (sin embeddings) | 0.3830 (CV: 0.5176 con puntuación, 0.5048 sin ella) | Cuánto se puede clasificar solo por solapamiento léxico exacto, sin ninguna noción de significado. |

Los 3 modelos reales del split único con texto crudo (0.69-0.72) superan
claramente el 0.38 de ese mismo baseline — pero esa comparación mezclaba señal
real con los atajos de puntuación/mayúsculas. Con la validación cruzada y texto
normalizado (la comparación que vale), el vecino por palabras da 0.5048 y los
modelos reales 0.55-0.66: la brecha real es de solo ~0.04-0.15 puntos de F1, no
0.3-0.35. Es una llamada fuerte a interpretar los resultados de OE2 con cautela:
hay señal, pero es modesta, y en XLM-R es la más chica de los 3.
