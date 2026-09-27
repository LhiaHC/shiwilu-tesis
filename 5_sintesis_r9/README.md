# Síntesis comparativa final (OE4, R9)

No entrena ni mide nada nuevo: **cruza dos evaluaciones que ya existen por
separado** para responder la pregunta de R9, "cuál es la configuración
óptima", desde dos ángulos que podrían no coincidir (y que, con el texto
crudo, parecían no coincidir — ver más abajo por qué era un artefacto).

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

*Actualizado 2026-09-27 — versión definitiva, sobre el corpus ya limpio (ver nota
de metodología en
[`../3_baselines_y_aumento_datos/tecnicas_aumento/README.md`](../3_baselines_y_aumento_datos/tecnicas_aumento/README.md)).
**Texto normalizado** (minúsculas, sin puntuación, sin distinguir tildes/ñ, con
el apóstrofo de oclusiva glotal siempre preservado) es la única condición que se
corre y se mantiene, en los dos ejes: el corpus tiene atajos superficiales (los
signos `¿?` delatan PRG; las mayúsculas, 100% en DES/PRG/REQUEST) que no son
señal real. El eje extrínseco es la validación cruzada de 5 folds en esa
condición
([`../3_baselines_y_aumento_datos/validacion_cruzada.py --sin-puntuacion`](../3_baselines_y_aumento_datos/validacion_cruzada.py));
el eje intrínseco, [`4_caracterizacion_embeddings/caracterizacion.py`](../4_caracterizacion_embeddings/caracterizacion.py),
ya normaliza el texto por defecto. La tabla con texto crudo (más abajo) queda
congelada solo como evidencia histórica de por qué se descartó esa condición.*

| Modelo | F1 sin aumento | Mejor config. | F1 mejor config | Rank extr. | Mejor estrategia intrínseca | Silueta | Rank intr. |
|---|---|---|---|---|---|---|---|
| mBERT | **0.6575** | sin aumento | 0.6575 | 1 | Mean pooling | **-0.0096** | 1 |
| LaBSE | 0.5873 | generate_then_refine | 0.5952 | 2 | Combinación de capas | -0.0254 | 2 |
| XLM-R | 0.5481 | generate_then_refine | 0.5798 | 3 | Max pooling | -0.0387 | 3 |

### Conclusión

- **Los dos ejes coinciden, sin excepción: mBERT gana en los dos.** Tiene el mejor
  F1 (0.6575 sin aumento — ninguna técnica lo mejora; Mixup y Retrotraducción lo
  **empeoran** de forma significativa) y la mejor silueta (-0.0096, la menos
  negativa de las 12 combinaciones). El margen es real: mBERT es
  significativamente mejor que **ambos** otros modelos en F1 (mBERT − XLM-R =
  +0.109, IC95% [+0.071, +0.148]; mBERT − LaBSE = +0.070, IC95% [+0.031,
  +0.108]); XLM-R vs. LaBSE no es distinguible de cero (-0.039, IC95% [-0.081,
  +0.001]). En intrínseco, mBERT ocupa las 4 primeras posiciones de las 12 (ver
  [`4_caracterizacion_embeddings/README.md`](../4_caracterizacion_embeddings/README.md)).
- **Los embeddings crudos no organizan las categorías por sí solos.** La silueta de
  las 12 combinaciones es ≈ 0 o negativa (entre -0.057 y -0.010), y el mejor
  clasificador (mBERT, ~560 oraciones de train por fold) solo llega a F1 = 0.6575,
  unos 0.15 por encima del baseline por palabras (0.50, sin ningún embedding). Es
  una señal real pero débil.
- **La "divergencia" que aparecía con el texto crudo era un artefacto de los
  atajos**, no un hallazgo sobre los embeddings. Con texto crudo, LaBSE + CLS
  ganaba en silueta (+0.049, por los `¿?`/mayúsculas) y XLM-R en F1 (por el mismo
  motivo), sugiriendo que "un embedding bien organizado no clasifica mejor". Al
  normalizar el texto, ambos ejes coinciden en el mismo modelo. Tabla histórica
  (texto crudo, **no se vuelve a correr**):

| Modelo | F1 sin aumento | Mejor config. | F1 mejor config | Rank extr. | Mejor estrategia intrínseca* | Silueta* | Rank intr.* |
|---|---|---|---|---|---|---|---|
| XLM-R | 0.7583 | generate_then_refine | 0.7685 | 1 | Max pooling | -0.0387 | 3 |
| mBERT | 0.7577 | generate_then_refine | 0.7623 | 2 | Mean pooling | -0.0096 | 1 |
| LaBSE | 0.7451 | sin aumento | 0.7451 | 3 | Combinación de capas | -0.0254 | 2 |

  \* La silueta de esta fila es la del texto normalizado (siempre se corrió así);
  se repite para que la tabla se lea de un vistazo, pero **no es un par válido**
  con el F1 de texto crudo de la misma fila.

- **Solo una técnica ayuda de forma distinguible: Generate-then-Refine en XLM-R**
  (+0.032, IC95% [+0.012, +0.052]) — ver
  [`comparacion_pareada`](../3_baselines_y_aumento_datos/tecnicas_aumento/README.md).
  No alcanza para que XLM-R le gane a mBERT. Mixup y Retrotraducción
  **empeoran** de forma significativa a mBERT (-0.023 y -0.036). El efecto de
  ambas técnicas en la silueta es pequeño (≤ 0.009 en valor absoluto).

**Recomendación práctica:** **mBERT** es la configuración recomendada en ambos
ejes, con diferencia estadísticamente significativa frente a los otros dos
modelos, y ninguna técnica de aumento la mejora ni la cambia (dos incluso la
empeoran). Presentar el resultado con cautela: la señal es real pero modesta
(F1 ≈ 0.66 sobre un piso de 0.50 sin ningún embedding), y solo se sostiene por
haber eliminado los atajos superficiales del corpus.
