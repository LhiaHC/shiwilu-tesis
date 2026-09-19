# Síntesis comparativa final (OE4, R9)

No entrena ni mide nada nuevo: **cruza dos evaluaciones que ya existen por
separado** para responder la pregunta de R9, "cuál es la configuración
óptima", desde dos ángulos que pueden no coincidir.

- **Extrínseco** (R4-R6,
  [`3_baselines_y_aumento_datos/resumen_experimentos.csv`](../3_baselines_y_aumento_datos/resumen_experimentos.csv)):
  qué tan bien clasifica un modelo de embeddings + Regresión Logística, con
  o sin cada técnica de aumento de datos. Métrica: F1 macro sobre el
  conjunto de prueba.
- **Intrínseco** (R7-R8,
  [`4_caracterizacion_embeddings/resultados/`](../4_caracterizacion_embeddings/README.md)):
  qué tan bien separan las categorías los embeddings crudos, sin
  clasificador, con cada una de las 4 estrategias de pooling. Métrica:
  coeficiente de silueta (distancia coseno), sobre el corpus original.

## Uso

```bash
python 5_sintesis_r9/sintesis.py
```

Requiere haber corrido antes `3_baselines_y_aumento_datos/resumen_experimentos.py`
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

| Modelo | F1 sin aumento | Mejor técnica | F1 mejor config | Rank extr. | Mejor estrategia intrínseca | Silueta | Rank intr. |
|---|---|---|---|---|---|---|---|
| XLM-R | 0.7828 | retrotraducción | **0.8206** | **1** | max_pooling | 0.0151 | 3 |
| mBERT | 0.7514 | generate_then_refine | 0.7720 | 2 | mean_pooling | 0.0303 | 2 |
| LaBSE | 0.7565 | mixup | 0.7582 | 3 | cls | **0.0493** | **1** |

### Conclusión

**Los dos criterios divergen.** XLM-R + retrotraducción es la configuración
óptima si el objetivo es clasificar intenciones (F1 macro = 0.8206, la más
alta de las 12 combinaciones evaluadas en R6). Pero si el criterio fuera la
calidad intrínseca de los embeddings — qué tan bien se agrupan las
categorías sin ningún clasificador encima —, LaBSE con el vector `[CLS]`
gana con claridad (silueta = 0.0493 frente a 0.0151 de XLM-R).

Esto es evidencia directa de que **la calidad intrínseca de un embedding no
garantiza el mejor desempeño en la tarea de clasificación final**: LaBSE
está optimizado para similitud semántica cross-lingüe (lo que favorece un
espacio bien organizado y separable), mientras que XLM-R parece beneficiarse
más de la capacidad de ajuste del clasificador (Regresión Logística) sobre
una representación menos organizada pero más informativa para esa frontera
de decisión especifica.

**Recomendación práctica:**
- Para clasificación de intenciones → **XLM-R + retrotraducción**.
- Para un uso no supervisado de los embeddings (agrupamiento, búsqueda por
  similitud semántica, exploración del corpus) → **LaBSE + `[CLS]`**.

### Efecto del aumento de datos: mejora el F1, pero no la separabilidad intrínseca

El caso de XLM-R + retrotraducción ilustra una segunda divergencia: esa
misma técnica que produce el **mejor** F1 de clasificación (0.7828 → 0.8206)
también **degrada** la silueta intrínseca de XLM-R (0.0151 → -0.0099, ver
[`4_caracterizacion_embeddings/README.md`](../4_caracterizacion_embeddings/README.md)).
Agregar oraciones sintéticas hace el espacio de embeddings menos "limpio"
para un agrupamiento no supervisado, pero le da al clasificador supervisado
más variedad para generalizar. Son preguntas distintas — "¿se organizan
mejor los embeddings?" vs. "¿clasifica mejor un modelo entrenado sobre
ellos?" — y R9 es precisamente el capítulo donde ese matiz se documenta en
vez de perderse.
