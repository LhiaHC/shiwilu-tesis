# Resultados del fine-tuning (complemento del OE4)

Resultados de los cuadernos de Colab [`colab_finetuning_mbert.ipynb`](../colab_finetuning_mbert.ipynb) y [`colab_finetuning_modelos.ipynb`](../colab_finetuning_modelos.ipynb).
Mismo corpus (700 oraciones, 7 intenciones, texto normalizado), mismas 5 particiones congeladas, 3 semillas (0, 1 y 2); los hiperparámetros se eligieron
con validación interna del entrenamiento de cada partición, nunca con la prueba. Baseline de comparación: el de embeddings congelados del OE2 (protocolo B).

Se subieron tal como salieron de Colab, reorganizados con nombres uniformes (`<modelo>_<variante>`). **No se volvió a correr nada para generarlos.**

## Archivos

| Archivo | Contenido |
|---|---|
| `finetuning_vs_baseline_todos.csv` | Resumen: F1 macro agrupado de cada corrida (media de semillas, desviación, mínimo, máximo, IC 95%) y **diferencia pareada frente al baseline congelado del mismo modelo**, con la rejilla de hiperparámetros usada y una observación |
| `finetuning_comparaciones_pareadas.csv` | Diferencias pareadas **entre** los modelos reentrenados y contra el clasificador de n-gramas de letras (referencia, 0.751) |
| `f1_por_intencion_todos.csv` | F1 por intención (promedio de semillas) de cada modelo congelado y reentrenado, y de los n-gramas |
| `predicciones_<modelo>_<variante>.csv` | Predicciones de las 700 oraciones por semilla y partición (`pos` = posición en el corpus, igual que en los archivos del OE2) |
| `seleccion_hiperparametros_<modelo>_<variante>.csv` | Combinaciones de tasa de aprendizaje y épocas probadas y su F1 de validación interna |

## Resumen (F1 macro agrupado)

| Corrida | Baseline congelado | Fine-tuning | Diferencia pareada [IC 95%] | Entre semillas | Lectura |
|---|---|---|---|---|---|
| mBERT, completo | 0.665 | **0.732** | +0.067 [+0.038, +0.097] | 0.715–0.750 | mejora real y estable |
| LaBSE, completo | 0.587 | **0.726** | +0.139 [+0.108, +0.172] | 0.715–0.734 | mejora real, la mayor |
| XLM-R, completo (rejilla común) | 0.570 | 0.526 | −0.044 [−0.075, −0.011] | 0.499–0.580 | **no concluyente**: no convergió (a 1e-4 colapsó) |
| XLM-R, completo (rejilla corregida) | 0.570 | 0.627 | +0.057 [+0.026, +0.089] | 0.565–0.694 | mejora, pero inestable y sin saturar |
| mBERT, LoRA | 0.665 | 0.604 | −0.062 [−0.091, −0.032] | 0.509–0.669 | **no concluyente**: eligió el borde de la rejilla |

Comparaciones entre los reentrenados: mBERT − LaBSE = +0.006 [−0.014, +0.027] (empatan); ambos superan a XLM-R en ≈ 0.10; frente al clasificador de n-gramas (0.751)
mBERT queda en −0.019 [−0.043, +0.007] y LaBSE en −0.024 [−0.050, +0.003] (no distinguibles).

## Segunda tanda: rejilla ampliada, + Generate-then-Refine y pooling del OE3

Cuadernos [`colab_finetuning_v2_<modelo>.ipynb`](../). Misma metodología (corpus, 5 particiones, 3 semillas, F1 macro agrupado, IC 95% y diferencias pareadas). Las tres variantes por modelo:
`ampliada` (rejilla ampliada con los puntajes de validación previos reutilizados), `gtr` (reales + Generate-then-Refine del OE2, con los hiperparámetros de `ampliada`) y `pooling` (cabeza sobre el pooling del OE3, con los hiperparámetros de `ampliada`).
Archivos nuevos: `predicciones_<modelo>_<variante>.csv` y `seleccion_hiperparametros_<modelo>_ampliada.csv` (los puntajes marcados `origen = previo` son los reutilizados de la primera tanda). Las comparaciones se calculan en
[`../../resultados/finetuning_v2_comparaciones.csv`](../../resultados/finetuning_v2_comparaciones.csv).

| Modelo | Anterior | `ampliada` | `gtr` | `pooling` |
|---|---|---|---|---|
| mBERT | 0.732 | 0.725 | 0.727 | 0.719 |
| LaBSE | 0.726 | 0.733 | 0.732 | 0.740 |
| XLM-R (rejilla corregida) | 0.627 | 0.680 | 0.714 | 0.721 |

- **mBERT y LaBSE:** ninguna variante mejora de forma distinguible al fine-tuning anterior ni a `ampliada` (todas las diferencias con IC 95% que incluye el 0); meseta de 0.72 a 0.74. Frente a los n-gramas (0.751), ninguna de LaBSE se distingue; en mBERT, `ampliada` (−0.025) y `pooling` (−0.031) quedan por debajo, aunque `gtr` (−0.023) no.
- **XLM-R:** mejora frente al fine-tuning anterior en las tres variantes. Generate-then-Refine ayuda (+0.034 frente a `ampliada`; +0.040 sin la partición 0). La ventaja de `pooling` sobre `ampliada` (+0.040) no se sostiene sin la partición 0 (+0.011, no distinguible): proviene del colapso de `ampliada`.
- **Colapsos de XLM-R:** `ampliada` semilla 1 partición 0 (F1 de la partición 0.129, lr 5e-5, 20 épocas) y `gtr` semilla 2 partición 0 (0.038, lr 3e-5, 20 épocas), con validación interna normal (0.69 y 0.72). Se reportan incluidos; el análisis sin la partición 0 está en `finetuning_v2_comparaciones.csv`. No se repitieron: una regla de detección sin etiquetas («menos de 3 intenciones predichas») habría detectado solo uno de los dos.
- **Borde de la rejilla ampliada:** el máximo de épocas probado (20) se eligió en 13 de 15 particiones de XLM-R, el par (1e-4, 20 épocas) en 6 de 15 de LaBSE y el máximo de 20 épocas en 4 de 15 de mBERT. En mBERT hay meseta; XLM-R sigue sin techo establecido.
- Varias variantes se compararon sin corrección por comparaciones múltiples: se leen por dirección, no por centésimas. `gtr` y `pooling` no tuvieron búsqueda propia de hiperparámetros.

## Rejillas de hiperparámetros

- Común (mBERT y LaBSE completo, y XLM-R, rejilla 1): tasa de aprendizaje 2e-5, 5e-5 y 1e-4; épocas 5 y 10.
- LoRA (solo mBERT): tasa 1e-4, 3e-4 y 1e-3; épocas 5 y 10.
- XLM-R, rejilla corregida: tasa 1e-5, 2e-5 y 3e-5; épocas 10 y 15. **Se corrigió después de ver que la rejilla común no convergió** (ajuste declarado a posteriori; se eligió igualmente solo con validación interna).

## Limitaciones

- Varias corridas eligieron el **borde de la rejilla** (mBERT: 10 épocas en 14 de 15; LaBSE: tasa 1e-4 y 10 épocas en 10 de 15; XLM-R, rejilla corregida: tasa 3e-5 y 15 épocas en 13 de 15; LoRA: tasa 1e-3 y 10 épocas en 12 de 15). Las cifras son probablemente **conservadoras** y el techo de cada modelo no está establecido.
- LoRA y la primera corrida de XLM-R no son concluyentes (el entrenamiento no convergió); no se deben citar como «LoRA/XLM-R no ayudan».
- Se compararon varios modelos y rejillas sin corrección por comparaciones múltiples: leer por dirección, no por centésimas.
- Solo datos reales (sin texto sintético). Diferencias menores a ≈ 0.01 no se interpretan.
- `f1_por_intencion_todos.csv` no incluye LoRA (el resumen por intención de esa corrida se sobrescribió en Colab; sus predicciones sí están en `predicciones_mbert_lora.csv`).
- Las comparaciones pareadas se calcularon con 2000 remuestreos bootstrap de las 700 oraciones (semilla 42), promediando el F1 de las semillas; el clasificador de n-gramas es el de `oe2_aumento_de_datos/evaluacion/diagnostico_baseline/ngramas_caracteres.py`, recalculado con las mismas particiones.
