# Fine-tuning de mBERT, LaBSE y XLM-R

Complemento del OE4. La pregunta es si reentrenar cada modelo con las oraciones reales del corpus mejora su baseline del OE2 (embeddings congelados más Regresión Logística, protocolo B: mBERT 0.665, LaBSE 0.587, XLM-R 0.570) y si mBERT sigue ganando después de reentrenar. No cambia ningún resultado del OE2 ni del OE3.

Todos los cuadernos se corren en Google Colab con GPU (T4), no usan la API de Claude y guardan su avance en Drive, así que si la sesión se corta se retoma volviendo a correr desde la celda de configuración. Con `PRUEBA = True` en esa celda se hace una comprobación rápida.

Lo que no cambia entre cuadernos: 700 oraciones con texto normalizado, las 5 particiones de `oe2_aumento_de_datos/particiones/folds_fijos.csv`, 3 semillas, hiperparámetros elegidos con validación interna del entrenamiento (nunca con la prueba), F1 macro de las 700 predicciones juntas con intervalo de confianza del 95% y diferencias pareadas.

## Cuadernos

| Cuaderno | Qué hace |
|---|---|
| `colab_finetuning_mbert.ipynb` | Primera versión, solo mBERT |
| `colab_finetuning_modelos.ipynb` | Primera tanda: mBERT, LaBSE y XLM-R (y LoRA en mBERT) con la misma rejilla (tasas 2e-5, 5e-5 y 1e-4; 5 y 10 épocas) |
| `colab_finetuning_v2_<modelo>.ipynb` | Segunda tanda, un cuaderno por modelo para correrlos en paralelo |
| `colab_finetuning_v3_xlmr_estable.ipynb` | Tercera tanda: XLM-R con un procedimiento estabilizado |

### Segunda tanda

Responde a que varias corridas de la primera tanda eligieron el borde de la rejilla y a la sugerencia de combinar el reentrenamiento con las técnicas del estudio. Cada cuaderno corre tres variantes:

| Variante | Qué cambia | Hiperparámetros |
|---|---|---|
| `ampliada` | Datos reales con más épocas y tasas. Reutiliza los puntajes de validación de las 6 combinaciones ya probadas | Validación interna |
| `gtr` | Se agrega Generate-then-Refine al entrenamiento, armado como en el OE2 (el cuaderno comprueba que el volumen coincide) | Los que eligió `ampliada` en esa semilla y partición |
| `pooling` | La capa de clasificación usa el pooling que mejor funcionó congelado (mBERT mean, LaBSE max, XLM-R combinación de capas) en lugar del vector `[CLS]` | Los que eligió `ampliada` |

Tomó unas 3 horas para mBERT y unas 6 para LaBSE. mBERT y LaBSE no mejoran de forma distinguible (meseta de 0.72 a 0.74); XLM-R sí mejora, y ahí Generate-then-Refine ayuda.

### Tercera tanda: XLM-R estabilizado

En la segunda tanda, XLM-R tuvo entrenamientos que colapsaron (el modelo terminó prediciendo una sola intención): 4 de 75 entrenamientos finales y 11 de 150 de validación, y mucho más con la tasa 5e-5. Este cuaderno repite XLM-R con cambios fijados antes de ver los resultados:

- tasas de aprendizaje 2e-5 y 3e-5, y 15, 20 y 25 épocas;
- calentamiento del 20% de los pasos (antes 10%);
- si al 30% de las épocas la pérdida de entrenamiento sigue en el nivel del azar (1.85 o más), se descarta ese entrenamiento y se repite con otra semilla, hasta 2 veces;
- se guarda la pérdida de cada época de cada entrenamiento;
- todas las combinaciones se recalculan, sin reutilizar puntajes anteriores.

Se reporta junto a la segunda tanda, no en su lugar. No hubo colapsos en los entrenamientos finales (XLM-R 0.722; 0.746 con Generate-then-Refine; 0.728 con pooling). La regla de reinicio se activó en 65 de 180 entrenamientos de validación y en ninguno de los 45 finales; como XLM-R tarda varias épocas en empezar a aprender, parte de esos reinicios fueron falsas alarmas, así que la estabilidad se debe sobre todo a los hiperparámetros. Más detalle en [`resultados/README.md`](resultados/README.md).

## Cómo leer los resultados

- Una diferencia es distinguible de cero solo si su intervalo de confianza no incluye el 0. Diferencias menores a 0.01 no se interpretan.
- Si la desviación entre semillas supera 0.02, el entrenamiento es inestable y el promedio se lee con cautela.
- Se copian a `resultados/` los archivos que descarga cada cuaderno (`predicciones_*.csv`, `seleccion_hiperparametros_*.csv`) y luego se corre `oe4_sintesis/sintesis.py`.

## Limitaciones

- Rejillas pequeñas; varias corridas eligieron su borde.
- Usa solo datos reales, salvo la variante `gtr`.
- Compara formas de usar los modelos sobre este corpus; no prueba que el modelo entienda el shiwilu.
