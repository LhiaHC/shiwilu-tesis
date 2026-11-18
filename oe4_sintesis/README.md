# OE4: comparación final y configuración óptima

Esta carpeta junta en tablas y figuras todo lo evaluado en el OE2 (aumento de datos), el OE3 (métodos de caracterización) y el fine-tuning, para responder cuál es la mejor configuración. No entrena modelos ni usa la API de Claude: `sintesis.py` lee predicciones que ya están guardadas.

Todo usa el protocolo B (el parámetro `C` de la Regresión Logística se elige con validación cruzada interna), el texto normalizado y las mismas 5 particiones (`oe2_aumento_de_datos/particiones/folds_fijos.csv`). La métrica es el F1 macro de las 700 predicciones juntas, con intervalo de confianza del 95% por bootstrap y diferencias pareadas.

## Cómo correrlo

Desde la raíz del repositorio:

```bash
python oe4_sintesis/sintesis.py
python oe4_sintesis/otras_metricas.py
```

Las entradas ya están en el repositorio (ver el encabezado de [`sintesis.py`](sintesis.py)), así que no hace falta correr nada antes. Si se agrega `--recalcular-ngramas` a la primera, se vuelve a calcular el clasificador de n-gramas en vez de leerlo de `resultados/ngramas_predicciones.csv`.

## Qué hay en la carpeta

| Ruta | Contenido |
|---|---|
| `sintesis.py` | Genera todas las tablas y figuras de `resultados/` |
| `otras_metricas.py` | Calcula otras métricas además del F1 macro (sección más abajo) |
| `finetuning/` | Cuadernos de Colab del fine-tuning y sus resultados ([README](finetuning/README.md)) |
| `resultados/` | Tablas y figuras (detalle abajo) |
| `resultados/historico_protocolo_A/` | Tablas de una versión anterior (protocolo A). Solo como historial, no usar para reportar |

Dentro de `resultados/`:

| Archivo | Contenido |
|---|---|
| `tabla_36_combinaciones.csv` | Las 36 combinaciones (3 modelos × 3 corpus × 4 poolings): F1, F1 ponderado, exactitud, F1 por intención, métricas intrínsecas y diferencias pareadas |
| `mejores_por_modelo.csv` | Mejor pooling de cada modelo según F1 y según silueta |
| `top_configuraciones_vs_mejor.csv` | Las mejores configuraciones congeladas comparadas con la mejor |
| `correlaciones_intrinseco_extrinseco.csv` | Correlación (Spearman) de cada métrica intrínseca con el F1 |
| `enfoques_f1.csv` | F1 de todos los enfoques: congelado, mejor pooling, con GtR, fine-tuning y n-gramas |
| `finetuning_vs_congelado.csv` | Primera tanda de fine-tuning frente a su baseline congelado y a los n-gramas |
| `finetuning_v2_comparaciones.csv` | Segunda tanda y XLM-R estabilizado: F1, diferencias pareadas, particiones colapsadas y un análisis sin la partición 0 |
| `f1_por_intencion_enfoques.csv` | F1 por intención de cada enfoque |
| `matriz_confusion_*.csv` | Matrices de confusión de la mejor configuración congelada y de mBERT con fine-tuning |
| `des_oraciones_dificiles.csv` | Las 100 oraciones de DES y qué métodos las aciertan; `fallan_todos` marca las que ninguno acierta |
| `otras_metricas/` | Salidas de `otras_metricas.py` |
| `figuras/` | Ocho figuras (la 7 resume el fine-tuning) |

## Resultados principales

| Enfoque | F1 macro |
|---|---|
| N-gramas de letras (sin modelo de lenguaje) | 0.751 [0.719, 0.780] |
| mBERT congelado (baseline) | 0.665 [0.630, 0.698] |
| mBERT congelado, mejor pooling (mean) | 0.667 [0.631, 0.700] |
| LaBSE congelado: baseline / mejor pooling (max) | 0.587 / 0.622 |
| XLM-R congelado: baseline / mejor pooling (combinación de capas) | 0.570 / 0.600 |
| mBERT con fine-tuning | 0.732 [0.703, 0.759] |
| LaBSE con fine-tuning | 0.726 [0.697, 0.753] |
| XLM-R con fine-tuning (rejilla corregida, inestable) | 0.627 [0.598, 0.654] |
| mBERT con LoRA (no concluyente) | 0.604 |
| Segunda tanda, mBERT (rejilla ampliada / + GtR / pooling) | 0.725 / 0.727 / 0.719 |
| Segunda tanda, LaBSE (rejilla ampliada / + GtR / pooling) | 0.733 / 0.732 / 0.740 |
| Segunda tanda, XLM-R (rejilla ampliada / + GtR / pooling) (*) | 0.680 / 0.714 / 0.721 |
| XLM-R estabilizado (rejilla propia / + GtR / pooling) | 0.722 / 0.746 / 0.728 |

(*) Incluye dos entrenamientos finales que colapsaron. Sin la partición 0, donde ocurrieron, quedan en 0.706 / 0.746 / 0.717.

### Lectura

- Con embeddings congelados, mBERT es el mejor modelo y ninguna técnica de aumento lo mejora. La mejor configuración (mBERT, mean pooling, sin aumento, 0.667) no se distingue de otras cinco de mBERT.
- Con el corpus completo, ninguna técnica de aumento mejora a su baseline. Mixup (mBERT −0.036, XLM-R −0.024) y Retrotraducción (mBERT −0.034) lo empeoran. Generate-then-Refine solo ayuda con 10 a 25 oraciones reales por intención (+0.029 a +0.054).
- Cambiar el pooling mejora a LaBSE (0.587 a 0.622) y a XLM-R (0.570 a 0.600), no a mBERT.
- Las siluetas son casi 0 o negativas en todas las combinaciones. Silueta y Davies-Bouldin se relacionan con el F1 (Spearman 0.75 y −0.55 en las 36), Calinski-Harabasz no (0.05).
- Reentrenar mejora a mBERT (+0.067) y a LaBSE (+0.139); después de reentrenar empatan entre sí y con los n-gramas.
- Ampliar la rejilla, agregar Generate-then-Refine o cambiar el pooling no mejora de forma distinguible a mBERT ni a LaBSE (diferencias entre −0.013 y +0.014): están en una meseta de 0.72 a 0.74. Solo XLM-R sube, y ahí Generate-then-Refine sí ayuda (+0.040 sobre la rejilla ampliada, también sin la partición 0).
- XLM-R estabilizado (tasas 2e-5 y 3e-5, 15 a 25 épocas, calentamiento del 20% y reinicio si el entrenamiento no aprende) no tuvo colapsos finales. Con Generate-then-Refine queda a −0.005 de los n-gramas, sin distinguirse. Dos cautelas: cambió varias cosas a la vez, y la regla de reinicio se activó en 65 de 180 entrenamientos de validación y en ninguno de los 45 finales. Los registros de pérdida sugieren que parte de esos reinicios fueron entrenamientos lentos y no colapsados, así que la estabilidad se debe sobre todo a los hiperparámetros. Además eligió el máximo de épocas (25) en 12 de 15 particiones.
- Los n-gramas de letras superan a los tres modelos congelados: en este corpus la intención se reconoce sobre todo por la forma de las palabras.
- DES rinde bajo con todos los métodos (cerca de 0.54). De sus 100 oraciones, 12 las falla todo método comparado.

### Configuración recomendada

| Situación | Configuración |
|---|---|
| Embeddings congelados | mBERT con mean pooling, sin aumento (0.667) |
| Se puede reentrenar | mBERT (0.732), LaBSE (0.726, hasta 0.740 con max pooling) o XLM-R estabilizado con Generate-then-Refine (0.746). No se distinguen entre sí ni de los n-gramas |
| Muy pocos datos reales (10 a 25 por intención) | mBERT con Generate-then-Refine |

## Otras métricas además del F1 macro

`otras_metricas.py` calcula con las mismas predicciones: exactitud, exactitud balanceada, precisión y recall macro, F1 ponderado, kappa y MCC; y, para los modelos congelados (los únicos que guardaron probabilidades), AUC macro, top-2, top-3 y log-loss. Salidas en `resultados/otras_metricas/`.

- El corpus está balanceado (100 oraciones por intención), así que exactitud, recall macro y F1 ponderado dan casi lo mismo que el F1 macro (diferencia de 0.005 o menos) y ordenan los enfoques igual. El F1 macro no castiga más que ellas.
- Kappa y MCC descuentan el azar (1/7) y dan valores más bajos, pero ordenan igual y no cambian ninguna conclusión de los 12 experimentos centrales.
- El AUC y el top-2 miden otra cosa: qué tan bien ordena las intenciones el modelo. Con mBERT congelado el AUC es 0.898 y la intención correcta está entre las dos más probables el 80% de las veces. Con AUC, Generate-then-Refine mejora a los tres modelos (+0.009 a +0.015), aunque no mejora el F1.
- El log-loss depende de la calibración y no sirve para comparar técnicas.

## Limitaciones

- Las oraciones sintéticas no están revisadas por un hablante de shiwilu.
- Muchas diferencias son menores que el margen de error. Se leen por su dirección y solo cuando el intervalo de la diferencia pareada excluye el cero; las menores a 0.01 no se interpretan.
- Se compararon muchas configuraciones sin corrección por comparaciones múltiples.
- El fine-tuning usó rejillas pequeñas y varias corridas eligieron el borde de la rejilla. LoRA y la primera corrida de XLM-R no son concluyentes (ver [`finetuning/resultados/README.md`](finetuning/resultados/README.md)).
- En XLM-R hubo dos entrenamientos finales colapsados en la segunda tanda (F1 de la partición de 0.129 y 0.038, con validación interna normal). Se reportan incluidos y aparte sin la partición 0, y no se repitieron.
- Las variantes con Generate-then-Refine y con otro pooling usan los hiperparámetros de la variante de rejilla ampliada, sin búsqueda propia.
