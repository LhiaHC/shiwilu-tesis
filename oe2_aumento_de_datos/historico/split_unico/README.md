# Histórico: baselines con split único 70/15/15 y texto crudo

> **Esta carpeta es material histórico.** La evaluación vigente de los baselines
> (R4) y de las técnicas de aumento (R6) es la validación cruzada de
> [`../../evaluacion/`](../../evaluacion/) sobre texto normalizado; las filas
> `sin_aumento` de ahí son los baselines vigentes. Lo de aquí se conserva como
> evidencia (de dónde salió el diagnóstico de los atajos del corpus y de por qué
> se pasó a validación cruzada), pero **no se actualiza**.

Contenido: `baseline.py`, `correr_todos.py`, `evaluar_aumento_texto.py`,
`bootstrap_ic.py`, `resumen_experimentos.py`, `curva_aprendizaje.py` (todos sobre
el split único congelado en `split_fijo.csv`), y sus salidas
(`resultados_baselines/`, `resultados_tecnicas/`, `intervalos_confianza.csv`,
`resumen_experimentos.csv`, `curva_aprendizaje.{csv,png}`).

---

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
  sobre el conjunto de desarrollo (grilla en `shiwilu/clasificacion.py`).
- **Datos:** únicamente `corpus/corpus_shiwilu_final.csv` (700 pares),
  columna `shiwilu`, dividido 70/15/15 (train/dev/test) estratificado por
  categoría de intención — sin ninguna técnica de aumento aplicada. Misma
  división para los tres modelos, para que los resultados sean comparables.
  La división se hace agrupando por texto en shiwilu **normalizado**
  (`shiwilu.clasificacion.dividir_train_dev_test`/`_normalizar_shiwilu`): el corpus tiene
  oraciones muy cortas que se repiten con distinta glosa en español,
  mayúsculas o puntuación (`"MUPALLI"`, `"¡PANTE'CHEK!"` vs. `"pante'chek"`),
  y antes de este ajuste una misma oración podía caer en train y en test a la vez. El split se calculó una sola vez y quedó **congelado** en `split_fijo.csv`: todos los scripts lo leen de ahí en vez de recalcularlo (así ningún cambio de código o de versión de librerías lo reordena y contamina las técnicas de aumento generadas a partir de train). El corpus mismo (`corpus/corpus_shiwilu_final.csv`, columna `shiwilu`) también se limpió una vez: 13 typos de comilla doble por apóstrofo, espacios dobles y mayúsculas inconsistentes — ver la nota de metodología en [`../../tecnicas_aumento/README.md`](../../tecnicas_aumento/README.md).

- **`../../particiones/corpus_normalizado.csv`** (`shiwilu.clasificacion.cargar_corpus_normalizado()`): las mismas
  700 oraciones, con `shiwilu` ya normalizado — minúsculas, sin puntuación,
  sin tildes/ñ, apóstrofo (oclusiva glotal) siempre preservado. Es el archivo
  que se puede abrir para ver **exactamente** el texto que usan la validación
  cruzada y la caracterización de embeddings, en vez de confiar en que el
  código lo recalcula igual cada vez. Se regenera solo (automáticamente) si
  `corpus_shiwilu_final.csv` cambia.

## Uso

```bash
# un modelo a la vez
python oe2_aumento_de_datos/historico/split_unico/baseline.py --modelo labse
python oe2_aumento_de_datos/historico/split_unico/baseline.py --modelo mbert
python oe2_aumento_de_datos/historico/split_unico/baseline.py --modelo xlmr

# los 3 de una vez, con tabla comparativa al final
python oe2_aumento_de_datos/historico/split_unico/correr_todos.py
```

No requiere GPU ni clave de API — corre en CPU en pocos minutos (mBERT y
XLM-R tardan más que LaBSE por ser más pesados de descargar/cargar, pero
igual corren bien sin GPU). Se puede correr localmente en VS Code sin
necesidad de Colab.

## Salida

`resultados_baselines/<modelo>/` (uno por cada modelo):
- `metricas.json` — F1 macro, F1 ponderado, exactitud (train/dev/test).
- `matriz_confusion.png` — matriz de confusión sobre el conjunto de prueba.
- `reporte_clasificacion.csv` — métricas desagregadas por categoría de intención.

## Resultado principal: validación cruzada de 5 folds, texto normalizado

Los scripts de esta carpeta (`baseline.py`, `correr_todos.py`) usan el split
único 70/15/15 y el texto crudo del corpus, cuyo test de ~99 oraciones da un F1
con ±0.09 de incertidumbre. La evaluación **vigente** de la tesis es la
validación cruzada de
[`../../evaluacion/validacion_cruzada.py`](../../evaluacion/validacion_cruzada.py)
(folds congelados en [`particiones/folds_fijos.csv`](../../particiones/folds_fijos.csv), agrupados por texto normalizado),
**siempre sobre texto normalizado** (minúsculas, sin puntuación, sin
tildes/ñ): el corpus tiene atajos superficiales que
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
[`bootstrap_ic.py`](bootstrap_ic.py))
de cada modelo es de ±0.04-0.05 en F1 y se solapan entre sí — ninguna
diferencia entre los 3 modelos es estadísticamente concluyente con esta
muestra. Interpretar con cautela antes de generalizar.
