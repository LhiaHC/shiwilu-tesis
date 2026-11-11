# Síntesis comparativa final (OE4, R9)

Reúne en tablas y figuras todo lo evaluado en OE2 (aumento de datos), OE3 (estrategias de pooling y métricas intrínsecas) y el fine-tuning, para responder la pregunta de R9: **cuál es la configuración óptima** y qué tan lejos queda de una referencia sin modelos de lenguaje (n-gramas de letras).

No entrena modelos ni usa la API de Claude: `sintesis.py` lee predicciones ya generadas. Lo único que calcula si no está en caché es el clasificador de n-gramas, que guarda en `resultados/ngramas_predicciones.csv`.

Todo usa el **protocolo B** (parámetro `C` de la Regresión Logística elegido por validación cruzada interna, K=5), texto normalizado, las 5 particiones congeladas
(`oe2_aumento_de_datos/particiones/folds_fijos.csv`) y la misma métrica en todos lados: **F1 macro agrupado** de las 700 predicciones, con IC 95% bootstrap y diferencias pareadas.

## Uso

Desde la raíz del repositorio:

```bash
python oe4_sintesis/sintesis.py
python oe4_sintesis/sintesis.py --recalcular-ngramas   # opcional: recalcula los n-gramas en vez de leer la caché
```

Requiere que existan las entradas listadas en el encabezado de [`sintesis.py`](sintesis.py): las predicciones de OE2 (`evaluacion/resultados/cv_interna/` y `cv_interna_pooling/`),
las métricas intrínsecas de OE3 y las predicciones de fine-tuning. Todas están en el repositorio, así que se puede ejecutar sin correr nada más.

## Contenido

| Carpeta | Qué hay |
|---|---|
| [`sintesis.py`](sintesis.py) | Script que genera todas las tablas y figuras de `resultados/` |
| [`resultados/`](resultados/) | Salidas de la síntesis (tabla siguiente) |
| [`finetuning/`](finetuning/README.md) | Cuadernos de Colab del fine-tuning de mBERT, LaBSE y XLM-R y sus resultados ya corridos |
| `resultados/historico_protocolo_A/` | Tablas de la versión anterior (protocolo A y cifras antiguas de OE3), conservadas solo como historial. **No usar para reportar** |

### Salidas de `resultados/`

| Archivo | Contenido |
|---|---|
| `tabla_36_combinaciones.csv` | Las 36 combinaciones (3 modelos × 3 corpus × 4 poolings): F1 macro con IC 95%, F1 ponderado, exactitud, F1 por intención, métricas intrínsecas y diferencias pareadas |
| `mejores_por_modelo.csv` | Mejor pooling por F1 y por silueta, para cada modelo sin aumento |
| `top_configuraciones_vs_mejor.csv` | Las mejores configuraciones congeladas frente a la mejor (diferencia pareada) |
| `correlaciones_intrinseco_extrinseco.csv` | Spearman entre cada métrica intrínseca (silueta, Davies-Bouldin, Calinski-Harabasz) y el F1 |
| `enfoques_f1.csv` | F1 con IC 95% de todos los enfoques: congelado, mejor pooling, + Generate-then-Refine, fine-tuning y n-gramas |
| `finetuning_vs_congelado.csv` | Fine-tuning de cada modelo frente a su baseline congelado y frente a los n-gramas |
| `f1_por_intencion_enfoques.csv` | F1 por intención de cada enfoque |
| `matriz_confusion_*.csv` | Matrices de confusión de la mejor configuración congelada y del fine-tuning de mBERT |
| `des_oraciones_dificiles.csv` | Las 100 oraciones de DES con el acierto de cada método; `fallan_todos` marca las que ninguno acierta |
| `ngramas_predicciones.csv` | Predicciones del clasificador de n-gramas (caché) |
| `figuras/fig_oe4_*.png` | Siete figuras de la síntesis |

## Resultados principales

| Enfoque | F1 macro [IC 95%] |
|---|---|
| N-gramas de letras (referencia, sin modelo de lenguaje) | 0.751 [0.719, 0.780] |
| mBERT congelado (baseline OE2) | 0.665 [0.630, 0.698] |
| mBERT congelado, mejor pooling (mean) | 0.667 [0.631, 0.700] |
| LaBSE congelado: baseline / mejor pooling (max) | 0.587 / 0.622 |
| XLM-R congelado: baseline / mejor pooling (combinación de capas) | 0.570 / 0.600 |
| mBERT fine-tuning completo | 0.732 [0.703, 0.759] |
| LaBSE fine-tuning completo | 0.726 [0.697, 0.753] |
| XLM-R fine-tuning (rejilla corregida; inestable) | 0.627 [0.598, 0.654] |
| mBERT LoRA (no concluyente) | 0.604 |

### Lectura

- **Con embeddings congelados, mBERT es el mejor modelo** y ninguna técnica de aumento lo mejora. La mejor configuración (mBERT + mean pooling, sin aumento, 0.667) no se distingue de otras cinco de mBERT.
- **Aumento de datos con el corpus completo:** ninguna técnica mejora su baseline; Mixup (mBERT −0.036, XLM-R −0.024) y Retrotraducción (mBERT −0.034) lo empeoran de forma distinguible. Generate-then-Refine ayuda solo con muy pocos datos reales (10 a 25 por intención).
- **Pooling:** cambiar la forma de extraer el vector mejora a LaBSE (0.587 → 0.622) y a XLM-R (0.570 → 0.600), pero no a mBERT.
- **Métricas intrínsecas:** las siluetas son ≈ 0 o negativas en todas las combinaciones; los embeddings crudos no agrupan por intención. Silueta y Davies-Bouldin se relacionan con el F1 (Spearman 0.75 y −0.55 sobre las 36), Calinski-Harabasz no (0.05).
- **Fine-tuning:** reentrenar mejora a mBERT (+0.067) y a LaBSE (+0.139); tras reentrenar empatan entre sí y no se distinguen de los n-gramas (0.751).
- **Referencia sin modelos:** los n-gramas de letras superan a los tres modelos congelados, lo que indica que en este corpus la intención se reconoce sobre todo por la forma de las palabras.
- **DES** rinde bajo con todos los métodos (≈ 0.54 en los mejores); de las 100 oraciones de DES, 12 las falla todo método comparado (columna `fallan_todos` de `des_oraciones_dificiles.csv`).

### Configuración óptima propuesta

| Situación | Configuración |
|---|---|
| Embeddings congelados (sin reentrenar) | mBERT + mean pooling, sin aumento (0.667) |
| Se permite reentrenar | mBERT con fine-tuning (0.732), equivalente a LaBSE (0.726) |
| Muy pocos datos reales (10 a 25 por intención) | mBERT + Generate-then-Refine |

## Limitaciones

- Las oraciones sintéticas **no están validadas por un hablante** de shiwilu.
- Muchas diferencias son menores que el margen de error: se interpretan por dirección y solo cuando el IC 95% de la diferencia pareada excluye el cero. Diferencias menores a ≈ 0.01 no se interpretan.
- El fine-tuning usó rejillas pequeñas y varias corridas eligieron el borde de la rejilla; LoRA y la primera corrida de XLM-R no son concluyentes (ver [`finetuning/resultados/README.md`](finetuning/resultados/README.md)).
- Se compararon muchas configuraciones sin corrección por comparaciones múltiples.
