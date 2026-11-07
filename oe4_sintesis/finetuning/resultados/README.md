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
