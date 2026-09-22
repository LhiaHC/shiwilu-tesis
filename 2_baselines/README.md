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
  La división se hace agrupando por texto en shiwilu (`comun.dividir_train_dev_test`):
  el corpus tiene 28 oraciones muy cortas que se repiten con distinta glosa
  en español, y antes de este ajuste un mismo texto podía caer en train y en
  test a la vez.

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
| XLM-R | 0.7756 | 0.7770 | 0.7788 |
| mBERT | 0.7681 | 0.7675 | 0.7692 |
| LaBSE | 0.6940 | 0.6961 | 0.7019 |

XLM-R y mBERT quedan cerca entre sí; LaBSE queda claramente atrás en este
baseline sin aumento, a pesar de estar optimizado para similitud semántica
entre idiomas. Con un conjunto de prueba de ~104 oraciones, el intervalo de
confianza (bootstrap, ver
[`../3_baselines_y_aumento_datos/bootstrap_ic.py`](../3_baselines_y_aumento_datos/bootstrap_ic.py))
de cada modelo es de ±0.04-0.05 en F1 — las diferencias entre XLM-R y mBERT
no son estadísticamente concluyentes con esta muestra; la de LaBSE sí es más
clara. Interpretar con cautela antes de generalizar.
