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

*Actualizado 2026-09-22 tras corregir una fuga de datos en la división
train/dev/test — primero por texto exacto, luego por texto normalizado (ver
[`../2_baselines/README.md`](../2_baselines/README.md)) — y en
Generate-then-Refine/Retrotraducción (ver
[`../3_baselines_y_aumento_datos/tecnicas_aumento/README.md`](../3_baselines_y_aumento_datos/tecnicas_aumento/README.md)).
Los números cambiaron respecto a versiones anteriores, pero la conclusión
central (divergencia extrínseco/intrínseco) se mantiene estable a través de
las 3 correcciones.*

| Modelo | F1 sin aumento | Mejor técnica | F1 mejor config | Rank extr. | Mejor estrategia intrínseca | Silueta | Rank intr. |
|---|---|---|---|---|---|---|---|
| XLM-R | 0.7223 | generate_then_refine | **0.7607** | **1** | max_pooling | 0.0151 | 3 |
| LaBSE | 0.6943 | generate_then_refine | 0.7046 | 2 | cls | **0.0493** | **1** |
| mBERT | 0.6977 | (ninguna mejora) | 0.6977 | 3 | mean_pooling | 0.0303 | 2 |

Con el bootstrap de intervalos de confianza
([`../3_baselines_y_aumento_datos/bootstrap_ic.py`](../3_baselines_y_aumento_datos/bootstrap_ic.py)),
el IC95% de XLM-R+Generate-then-Refine es [0.6695, 0.8378] y el de XLM-R sin
aumento es [0.6253, 0.8057] — muy solapados, así que ni siquiera dentro del propio XLM-R hay una "mejor técnica" estadísticamente concluyente. Los 3 modelos sí quedan
razonablemente cerca entre sí en el eje extrínseco tras la corrección (ver
también los [baselines triviales](../2_baselines/README.md) para
contextualizar qué tan altos son estos F1 en términos absolutos).

### Conclusión

**Los dos criterios divergen.** XLM-R es la configuración con mejor F1 de
clasificación (0.7607 con Generate-then-Refine, la más alta de las 12 combinaciones evaluadas en R6, aunque dentro del margen de incertidumbre frente a su propio 0.7223 sin aumento). Pero si el criterio fuera la calidad intrínseca de los
embeddings — qué tan bien se agrupan las categorías sin ningún clasificador
encima —, LaBSE con el vector `[CLS]` gana con claridad (silueta = 0.0493,
muy por encima de XLM-R en 0.0151, que de hecho queda último de los 3 en
este eje).

Esto es evidencia directa de que **la calidad intrínseca de un embedding no
garantiza el mejor desempeño en la tarea de clasificación final**: LaBSE
está optimizado para similitud semántica cross-lingüe (lo que favorece un
espacio bien organizado y separable), mientras que XLM-R parece beneficiarse
más de la capacidad de ajuste del clasificador (Regresión Logística) sobre
una representación menos organizada pero más informativa para esa frontera
de decisión específica.

**Recomendación práctica:**
- Para clasificación de intenciones → **XLM-R** (con o sin Generate-then-Refine,
  la diferencia no es concluyente con esta muestra).
- Para un uso no supervisado de los embeddings (agrupamiento, búsqueda por
  similitud semántica, exploración del corpus) → **LaBSE + `[CLS]`**.

### Efecto del aumento de datos: mejora el F1, pero no siempre la separabilidad intrínseca

El caso de XLM-R + Generate-then-Refine ilustra una segunda divergencia: esa
técnica mejora el F1 de clasificación de XLM-R (0.7223 → 0.7607) pero **degrada** su silueta intrínseca (0.0151 → 0.0097, ver
[`4_caracterizacion_embeddings/README.md`](../4_caracterizacion_embeddings/README.md)).
Agregar oraciones sintéticas suele hacer el espacio de embeddings menos
"limpio" para un agrupamiento no supervisado, pero le da al clasificador
supervisado algo más de variedad para generalizar. Son preguntas distintas —
"¿se organizan mejor los embeddings?" vs. "¿clasifica mejor un modelo
entrenado sobre ellos?" — y R9 es precisamente el capítulo donde ese matiz
se documenta en vez de perderse.
