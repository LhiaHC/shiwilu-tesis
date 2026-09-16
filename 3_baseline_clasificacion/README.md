# Fase 3 — Baselines de clasificación (OE2/OE4, R4)

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

## Uso

```bash
# un modelo a la vez
python 3_baseline_clasificacion/baseline.py --modelo labse
python 3_baseline_clasificacion/baseline.py --modelo mbert
python 3_baseline_clasificacion/baseline.py --modelo xlmr

# los 3 de una vez, con tabla comparativa al final
python 3_baseline_clasificacion/correr_todos.py
```

No requiere GPU ni clave de API — corre en CPU en pocos minutos (mBERT y
XLM-R tardan más que LaBSE por ser más pesados de descargar/cargar, pero
igual corren bien sin GPU). Se puede correr localmente en VS Code sin
necesidad de Colab.

## Salida

`3_baseline_clasificacion/resultados/<modelo>/` (uno por cada modelo):
- `metricas.json` — F1 macro, F1 ponderado, exactitud (train/dev/test).
- `matriz_confusion.png` — matriz de confusión sobre el conjunto de prueba.
- `reporte_clasificacion.csv` — métricas desagregadas por categoría de intención.

## Resultado de referencia (ya ejecutado)

| Modelo | F1 macro | F1 ponderado | Exactitud |
|---|---|---|---|
| LaBSE | 0.7565 | 0.7565 | 0.7619 |
| mBERT | 0.7514 | 0.7514 | 0.7524 |
| XLM-R | 0.7828 | 0.7828 | 0.7810 |

XLM-R salió mejor de los tres en esta corrida, lo cual no era lo esperado a
priori (LaBSE está optimizado específicamente para similitud semántica
entre idiomas). Con un conjunto de prueba de solo 105 oraciones, esta
diferencia de ~3 puntos puede reflejar tanto una diferencia real como
variabilidad por tamaño de muestra — interpretar con cautela antes de
generalizar.
