# Fine-tuning de mBERT, LaBSE y XLM-R (complemento del OE4)

Experimento **exploratorio y separado** que responde: ¿reentrenar cada modelo con las oraciones reales del corpus mejora a su baseline del OE2
(embeddings congelados + Regresión Logística, protocolo B; mBERT 0.665, LaBSE 0.587, XLM-R 0.570)? ¿Sigue ganando mBERT después de reentrenar?

No cambia ningún resultado del OE2 ni del OE3. Usa los mismos datos y se compara con las mismas reglas.

| | |
|---|---|
| Cuadernos | [`colab_finetuning_modelos.ipynb`](colab_finetuning_modelos.ipynb): **vigente**, corre mBERT, LaBSE y XLM-R con el mismo protocolo y compara los modelos reentrenados entre sí (reutiliza lo ya guardado de mBERT). [`colab_finetuning_mbert.ipynb`](colab_finetuning_mbert.ipynb): primera versión, solo mBERT. Ambos se corren en Google Colab con GPU T4 |
| Datos | Corpus de 700 oraciones, 7 intenciones, texto normalizado (el mismo de los 12 experimentos) |
| Particiones | Las 5 particiones congeladas de `oe2_aumento_de_datos/particiones/folds_fijos.csv` |
| Modelos | `bert-base-multilingual-cased` (mBERT), `sentence-transformers/LaBSE` y `xlm-roberta-base`, los mismos del OE2; variante `completo` (todos los pesos). LoRA se probó solo con mBERT y no fue concluyente (la rejilla quedó corta) |
| Hiperparámetros | Tasa de aprendizaje y épocas, elegidos con validación interna del entrenamiento de cada partición (nunca con la prueba) |
| Repeticiones | 3 semillas por defecto |
| Comparación | F1 macro agrupado de las 700 predicciones, IC 95% bootstrap y diferencia pareada contra el baseline |
| Costo | No usa la API de Claude. ~60 minutos en una T4 para la variante `completo` |

## Cómo correrlo

1. Sube el cuaderno a Colab (Archivo → Subir cuaderno) y elige GPU (Entorno de ejecución → Cambiar tipo de entorno).
2. Corre las celdas en orden. Con `PRUEBA = True` en la celda de configuración se hace una comprobación rápida.
3. El avance se guarda en tu Drive (`shiwilu_finetuning/`); si Colab se desconecta, vuelve a correr desde la celda de configuración.
4. Al final se descarga `finetuning_resultados.zip`. Cópialo a `oe4_sintesis/finetuning/resultados/` cuando lo hayas revisado.

## Resultados

Los resultados ya corridos están en [`resultados/`](resultados/README.md): reentrenar mejora a mBERT (+0.067) y a LaBSE (+0.139); tras reentrenar empatan y se acercan al clasificador de n-gramas de letras; XLM-R mejora con una rejilla corregida (+0.057) pero es inestable. Ver el resumen y las limitaciones (borde de la rejilla) en ese archivo.

## Salidas

- `finetuning_vs_baseline.csv`: F1 macro de cada variante (media de semillas, desviación, IC 95%) y diferencia pareada contra el baseline.
- `f1_por_intencion.csv`: F1 por intención (en particular DES, la más débil del baseline).
- `predicciones_<variante>.csv`: predicciones de las 700 oraciones por semilla y partición.
- `seleccion_hiperparametros_<variante>.csv`: combinaciones probadas y su F1 de validación interna.

## Cómo leerlo

- Una diferencia es *distinguible de cero* solo si su intervalo de confianza no incluye el 0.
- Diferencias menores a unos 0.01 no se interpretan, como en el resto de la tesis.
- Si la desviación entre semillas es grande (por ejemplo más de 0.02), el fine-tuning es inestable y el promedio debe leerse con cautela.
- Es un complemento: el resultado principal del OE2 sigue siendo el de embeddings congelados.

## Alcance y limitaciones

- Solo datos reales (sin texto sintético).
- Una sola rejilla pequeña de hiperparámetros.
- No concluye que el modelo «entienda» el shiwilu: compara dos formas de usar mBERT sobre este corpus.
