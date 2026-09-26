# Síntesis comparativa final (OE4, R9)

No entrena ni mide nada nuevo: **cruza dos evaluaciones que ya existen por
separado** para responder la pregunta de R9, "cuál es la configuración
óptima", desde dos ángulos que pueden no coincidir.

- **Extrínseco** (R4-R6,
  [`3_baselines_y_aumento_datos/validacion_cruzada_resumen_sin_puntuacion.csv`](../3_baselines_y_aumento_datos/validacion_cruzada_resumen_sin_puntuacion.csv)):
  qué tan bien clasifica un modelo de embeddings + Regresión Logística, con
  o sin cada técnica de aumento de datos. Métrica: F1 macro sobre las 700 predicciones de una validación cruzada de 5 folds.
- **Intrínseco** (R7-R8,
  [`4_caracterizacion_embeddings/resultados/`](../4_caracterizacion_embeddings/README.md)):
  qué tan bien separan las categorías los embeddings crudos, sin
  clasificador, con cada una de las 4 estrategias de pooling. Métrica:
  coeficiente de silueta (distancia coseno), sobre el corpus original en texto normalizado (minúsculas, sin puntuación).

## Uso

```bash
python 5_sintesis_r9/sintesis.py
```

Requiere haber corrido antes `3_baselines_y_aumento_datos/validacion_cruzada.py`
y `4_caracterizacion_embeddings/caracterizacion.py` (al menos con
`--corpus original`; si además se corrió con `--corpus retrotraduccion` y/o
`--corpus generate_then_refine`, la síntesis incluye también el efecto del
aumento de datos sobre la calidad intrínseca).

## Salida

`5_sintesis_r9/resultados/`
- `sintesis_extrinseco_vs_intrinseco.csv` — una fila por modelo, con su
  mejor configuración extrínseca (técnica + F1) y su mejor estrategia
  intrínseca (pooling + silueta), y el ranking de cada modelo en ambos ejes.
- `efecto_aumento_en_intrinseco.csv` — silueta antes/después de agregar
  cada técnica de aumento de texto, por modelo.

## Resultado (ya ejecutado)

*Actualizado 2026-09-26. El eje intrínseco se calcula siempre sobre **texto normalizado**
(minúsculas, sin puntuación), porque el corpus tiene dos atajos: los signos de
puntuación (`¿?` delatan PRG) y las mayúsculas (DES, PRG y REQUEST están 100% en
MAYÚSCULAS). Para comparar lo mismo con lo mismo, el eje extrínseco principal es la
validación cruzada de 5 folds **en esa misma condición**
([`../3_baselines_y_aumento_datos/validacion_cruzada.py`](../3_baselines_y_aumento_datos/validacion_cruzada.py)
`--sin-puntuacion`): cada una de las 700 oraciones es test una vez y el aumento de cada
fold se genera solo con su train.*

**Comparación principal (texto normalizado en ambos ejes):**

| Modelo | F1 sin aumento | Mejor config. | F1 mejor config | Rank extr. | Mejor estrategia intrínseca | Silueta | Rank intr. |
|---|---|---|---|---|---|---|---|
| mBERT | 0.6378 | generate_then_refine | 0.6417 | 1 | Combinación de capas | -0.0054 | 1 |
| XLM-R | 0.6080 | sin aumento | 0.6080 | 2 | Max pooling | -0.0350 | 3 |
| LaBSE | 0.5816 | generate_then_refine | 0.5984 | 3 | Combinación de capas | -0.0202 | 2 |

### Conclusión

- **Los dos ejes coinciden en el mejor modelo: mBERT.** Tiene el mejor F1
  (0.638 sin aumento) y la mejor silueta
  (-0.0054). Pero el margen es
  chico: en F1, mBERT − LaBSE = +0.056 (IC95% [+0.019, +0.094]) es la única diferencia entre modelos
  distinguible de cero; XLM-R − mBERT = -0.030 (IC95% [-0.064, +0.005]) no lo es. Y todas las siluetas son ≈ 0 o negativas.
- **Los embeddings crudos no organizan las categorías por sí solos.** Sobre texto normalizado la
  silueta de todas las combinaciones es ≈ 0 o negativa, y el clasificador (Regresión Logística
  con ~560 oraciones) solo llega a F1 ≈ 0.58-0.64, unos 0.10-0.16 por encima del baseline por
  palabras (0.48). Es una señal real pero débil.
- **La "divergencia" que aparecía antes era un artefacto.** Con el texto crudo, LaBSE + CLS
  ganaba en silueta (+0.049) y XLM-R en F1, lo que sugería que "un embedding bien
  organizado no clasifica mejor". Esa silueta de LaBSE venía de los `¿?` y las mayúsculas,
  no de las palabras; al quitarlos desaparece. Con esa cautela, la tabla de referencia
  (F1 con el texto crudo contra silueta normalizada, **no comparable como par**):

| Modelo | F1 sin aumento | Mejor config. | F1 mejor config | Rank extr. | Mejor estrategia intrínseca | Silueta | Rank intr. |
|---|---|---|---|---|---|---|---|
| XLM-R | 0.7583 | generate_then_refine | 0.7685 | 1 | Max pooling | -0.0350 | 3 |
| mBERT | 0.7577 | generate_then_refine | 0.7623 | 2 | Combinación de capas | -0.0054 | 1 |
| LaBSE | 0.7451 | sin aumento | 0.7451 | 3 | Combinación de capas | -0.0202 | 2 |

- **Ninguna técnica de aumento mejora el F1** de forma distinguible en ninguna condición
  ([`comparacion_pareada`](../3_baselines_y_aumento_datos/tecnicas_aumento/README.md)), y su efecto en la silueta es
  ≈ 0 (≤ 0.01). Retrotraducción y Mixup empeoran el F1 en varios casos.

**Recomendación práctica:** no afirmar un "mejor modelo" más allá de que mBERT queda
nominalmente primero y de forma significativa por encima de LaBSE sin los atajos; reportar
siempre las dos condiciones de texto con sus baselines triviales; y presentar el resultado
central como negativo y bien medido (ni las técnicas de aumento ni la elección de modelo
cambian de forma clara el desempeño con un corpus de 700 oraciones tan repetitivo).
