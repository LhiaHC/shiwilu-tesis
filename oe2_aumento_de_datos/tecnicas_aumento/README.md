# Técnicas de aumento de datos (OE2, R6)

Tres técnicas para ampliar el corpus de entrenamiento en shiwilu, aplicadas
sobre los baselines de [`../evaluacion/`](../evaluacion/) y evaluadas
contra ellos.

| Técnica | Generador | Estado |
|---|---|---|
| **Mixup** | Ninguno — interpola embeddings de oraciones shiwilu existentes de la misma categoría | ✅ Probado — [`mixup.py`](mixup.py) |
| **Generate-then-Refine** | Claude (LLM), sin ajuste fino | ✅ Probado (138 oraciones, 81 aprobadas) — [`generate_then_refine.py`](generate_then_refine.py) |
| **Retrotraducción** | Helsinki-NLP (paráfrasis en español) + NMT de F. Prado (traduce a shiwilu) | ✅ Probado (429 oraciones, 421 aprobadas; checkpoint chrF++=43.19) — [`retrotraduccion.py`](retrotraduccion.py) |

### Contenido de esta carpeta

| Archivo | Para qué sirve |
|---|---|
| `mixup.py`, `generate_then_refine.py`, `retrotraduccion.py` | Generadores de datos sintéticos (código). |
| `colab/colab_entrenar_checkpoint.ipynb` | Paso de GPU de Retrotraducción: catálogo único de las 700 oraciones (checkpoint NLLB+LoRA de F. Prado, en Colab). |
| `colab/colab_retrotraduccion_en_linea.ipynb`, `colab/colab_generate_then_refine_en_linea.ipynb`, `colab/colab_mixup_en_linea.ipynb` | Un cuaderno por técnica: generan el aumento **dentro** de cada fold (core para elegir `C`, pool para el modelo final) y evalúan (ver más abajo). |
| `salidas/retrotraduccion_pool.csv` | Catálogo de las 700 oraciones retrotraducidas; **alimenta la validación cruzada vigente**. |
| `salidas/generate_then_refine_fold<0-4>.csv` | Un CSV por fold, generado solo con el train de ese fold; **alimenta la validación cruzada vigente**. |
| `salidas/retrotraduccion.csv`, `salidas/generate_then_refine.csv` | Versiones para el split único histórico; hoy solo las usa [`OE3`](../../oe3_caracterizacion_embeddings/) para construir los corpus aumentados. |

### Generación en línea dentro de cada fold (Colab)

> **Estado.** Los cuadernos de `colab/` corren esta generación con el protocolo de etapas core/pool y dev interno de ~90
> oraciones. Ese protocolo fue superado por la **CV interna** (K=5, ver [`../evaluacion/README.md`](../evaluacion/README.md)),
> y sus resultados (niveles 20-120 de Generate-then-Refine, Retrotraducción ×1) están archivados en
> [`../historico/en_linea_protocolo_dev/`](../historico/en_linea_protocolo_dev/). Cuando se corran los cuadernos, los CSV
> salen en `evaluacion/resultados/` (carpeta del clon de Colab); al traerlos al repositorio van a esa carpeta del histórico.
> Para Generate-then-Refine con CV interna (K=5, nivel 120) usa
> [`colab/colab_generate_then_refine_cv_interna.ipynb`](colab/colab_generate_then_refine_cv_interna.ipynb), que llama a
> `validacion_cruzada_cv_interna.py` (genera con `--solo-generar`, evalúa por nivel y entrega un `.zip`).

La evaluación vigente lee datos sintéticos ya generados. Los tres cuadernos `colab/colab_*_en_linea.ipynb` los **generan dentro del
propio entrenamiento de cada fold**, con la estrategia test / pool / core / dev, usando
[`../evaluacion/validacion_cruzada_en_linea.py`](../evaluacion/validacion_cruzada_en_linea.py):

- **Etapa core (elegir `C`)**: el aumento se genera y filtra solo con el core; se mide en el dev interno.
- **Etapa pool (modelo final)**: el aumento se genera y filtra con todo el pool; se mide en el test.
- Así el dev interno nunca influye en el aumento con el que se elige `C` (en la evaluación vigente, Generate-then-Refine se genera una vez
  con el pool y se usa en las dos etapas). El test no se toca nunca, y se descartan las filas sintéticas idénticas a una oración del test.

| Cuaderno | Qué se genera dentro de cada fold | Necesita |
|---|---|---|
| `colab_retrotraduccion_en_linea.ipynb` | Parafrasea y traduce el pool (Helsinki-NLP + NMT de F. Prado). Cada fila depende solo de su oración de origen, así que se genera una vez desde el pool y la etapa core usa las filas cuyo origen está en el core; los filtros usan el centroide de cada etapa | GPU, checkpoint en Drive, `git push` previo |
| `colab_generate_then_refine_en_linea.ipynb` | Dos generaciones con Claude por fold: una con el core y otra con el pool (70 llamadas a la API) | `ANTHROPIC_API_KEY` en los Secretos de Colab |
| `colab_mixup_en_linea.ipynb` | Vectores interpolados con el core y con el pool, repetidos con varias semillas | nada (CPU) |

**Mismos archivos que la evaluación vigente.** Con los 3 modelos y los 5 folds, cada corrida deja en `evaluacion/resultados/` (con el sufijo
`_en_linea_<técnica>_<etiqueta>`) los **12 experimentos**: la técnica corrida sale de esa corrida y las otras tres configuraciones, de los resultados
vigentes. Así los CSV tienen el mismo formato (incluidas las probabilidades por categoría) y los cuadernos generan, con los mismos scripts, todo lo
necesario para gráficos y curvas ROC:

| Archivo | Contenido |
|---|---|
| `validacion_cruzada_predicciones_sin_puntuacion_<TAG>.csv` | una fila por oración, modelo y técnica, con `prob_<categoría>` (insumo de las curvas ROC) |
| `validacion_cruzada_resumen_sin_puntuacion_<TAG>.csv` | F1 macro agrupado con IC95%, media y sd entre folds |
| `comparacion_pareada_sin_puntuacion_<TAG>.csv` | diferencias pareadas con IC95% (`comparacion_pareada.py --entrada ... --etiqueta <TAG>`) |
| `curvas_roc_<TAG>.csv`, `curvas_roc_<TAG>.png`, `curvas_roc_<TAG>/` | curvas ROC One-vs-Rest: datos, grilla 3 × 4 y una imagen por experimento (`curvas_roc.py --entrada ... --etiqueta <TAG>`) |

(`<TAG>` = `en_linea_<técnica>_<etiqueta>`; en Mixup, una por semilla.) En una corrida parcial (`--modelos` o `--folds`) los CSV traen solo la técnica corrida.

**Más datos sintéticos (niveles de volumen).** Hoy el aporte sintético medio es: Mixup 100% del pool real, Retrotraducción ~85% y Generate-then-Refine ~17%
(~95 aprobadas por etapa). `validacion_cruzada_en_linea.py` permite aumentarlo sin repetir lo ya generado:
- `--cantidad N` (Generate-then-Refine): oraciones por categoría y etapa, en lotes de 20 (una llamada por lote, con la ventana de ejemplos few-shot rotada).
  `fold<N>_<etapa>.csv` es el lote 0 (el de siempre) y `fold<N>_<etapa>_l<j>.csv` los siguientes; se reutilizan de la cache y solo se piden los que faltan. Se
  descartan las repetidas y las copias de oraciones reales. Medido: 40 → ~28% del pool, 80 → ~57%, 120 → ~85% (menos que n × 17% porque se descartan ~10-15% de repetidas y copias).
- `--multiplicador K` (Retrotraducción): K paráfrasis por oración (`fold<N>_pool.csv` y `fold<N>_pool_s<j>.csv`, con otra semilla del muestreo). K = 2 → ~170%.
  No se recomienda subir más sin antes filtrar la calidad de las paráfrasis.
- Cada corrida guarda además `volumen_sintetico_<TAG>.csv` (sintéticos usados por fold y etapa, y su % respecto de las oraciones reales).
Los cuadernos de `colab/` recorren varios niveles y entregan **un solo `.zip` por técnica** con lo generado y los resultados de todos los niveles.

Lo generado se guarda en una cache por (técnica, fold, etapa) —`salidas/en_linea/<técnica>/fold<N>_{core,pool}.csv`; en Colab, en Drive—, así que si
se corta la corrida o se repite, se reutiliza en vez de volver a gastar tokens o GPU. Los resultados se guardan con la etiqueta `*_en_linea_<técnica>_<etiqueta>`
y no pisan los vigentes; los cuadernos los comparan con `sin_aumento` y con lo vigente mediante diferencias pareadas con IC95%.
Control: con Mixup, la semilla 0 reproduce el resultado vigente dentro del ruido numérico (F1 ±0.003; la elección de `C` es sensible a diferencias de
~1e-6 en los embeddings).

Los resultados de cada técnica con el split único (histórico) están en
[`../historico/split_unico/resultados_tecnicas/`](../historico/split_unico/resultados_tecnicas/);
la evaluación vigente, en [`../evaluacion/`](../evaluacion/).

**Notas de metodología (2026-09-21/22):**
- Generate-then-Refine y Retrotraducción usaban ejemplos few-shot y el
  centroide del filtro semántico calculados sobre el corpus **completo**
  (train+dev+test), lo que dejaba que información de dev/test influyera en
  qué texto sintético se generaba y aprobaba. Se corrigió para usar solo
  train (`cargar_ejemplos_por_categoria` en `generate_then_refine.py` y el
  parámetro `corpus_train` en `refinar()` de `retrotraduccion.py`). Ambas
  técnicas ya se regeneraron con la corrección.
- La división train/dev/test (`shiwilu.clasificacion.dividir_train_dev_test`) agrupaba por
  texto shiwilu **exacto**, pero el corpus repite la misma raíz con variantes
  de mayúsculas/puntuación (`"PANTE'CHEK"` / `"¡pante'chek!"`). Ahora agrupa
  por texto **normalizado** (sin mayúsculas ni puntuación), cerrando esa fuga
  también. Ver [`../historico/split_unico/README.md`](../historico/split_unico/README.md)
  para el detalle y los números finales de los 12 experimentos.
- El split quedó **congelado** en `historico/split_unico/split_fijo.csv`. Ambas técnicas
  se regeneraron desde ese split final (Retrotraducción en Colab, con el mismo
  checkpoint), así que ninguna oración de dev/test participó en generar ni
  filtrar texto sintético. Si algún día se recalcula el split, hay que
  regenerar las dos técnicas.
- **Limpieza del corpus maestro (2026-09-27):** `corpus/corpus_shiwilu_final.csv`
  tenía 13 filas donde una comilla doble (`"`) reemplazaba por error un
  apóstrofo (oclusiva glotal, un fonema real del shiwilu — ej. `nanapi"la` debía
  ser `nanapi'la`); esas filas perdían ese fonema al normalizar el texto, porque
  el código no sabía que esa comilla debía leerse como apóstrofo. También tenía
  205 filas con espacios dobles y usaba mayúsculas/minúsculas sin criterio
  fijo (10-100% en mayúsculas según la categoría, ver más abajo). Se corrigieron
  los 3 problemas directamente en el corpus (columna `shiwilu` únicamente;
  `espanol` no se tocó). Esto cambió el agrupamiento por texto normalizado de 12
  oraciones, así que `split_fijo.csv` y `folds_fijos.csv` se recalcularon y
  ambas técnicas se regeneraron una vez más con el corpus ya limpio.


## Evaluación principal: validación cruzada de 5 folds

Con un solo split, el test tiene ~99 oraciones y el intervalo de confianza del
F1 es de ±0.09. Por eso la evaluación principal es una **validación cruzada de
5 folds** ([`../evaluacion/validacion_cruzada.py`](../evaluacion/validacion_cruzada.py)): cada una de
las 700 oraciones se usa como test una vez (folds agrupados por texto shiwilu
normalizado —mayúsculas, puntuación, espacios, tildes y ñ— y estratificados,
congelados en `particiones/folds_fijos.csv`), y el aumento de cada fold se genera
**solo a partir de su train**. El intervalo de confianza resulta ~2.8 veces más
angosto (±0.031 en promedio), y un análisis con 30 splits aleatorios
confirmó que no está inflado.

### Condición de texto: normalizado, sin excepción (decisión 2026-09-27)

El corpus tiene **atajos superficiales** que no son señal lingüística real: los
signos de puntuación (los `¿?` delatan la categoría PRG), las mayúsculas (las
oraciones de DES, PRG y REQUEST están 100% TODAS EN MAYÚSCULAS; las otras
categorías, 10-21%) y algunas tildes que solo distinguen variantes de una misma
oración. Por eso **desde ahora todos los experimentos —Mixup, Generate-then-Refine,
Retrotraducción, OE3, curvas, lo que siga— se corren únicamente sobre texto
normalizado**: minúsculas, sin puntuación y sin distinguir tildes/ñ
(`shiwilu.clasificacion.quitar_puntuacion(..., quitar_tildes=True)`, la condición `sin_puntuacion`
de [`../evaluacion/validacion_cruzada.py`](../evaluacion/validacion_cruzada.py)).

El texto **con puntuación** (tal cual está el corpus) ya no se vuelve a correr — la
tabla de abajo queda **congelada** como evidencia de por qué se descartó, no como
un resultado a mantener en paralelo.

**Con puntuación** (histórico, congelado — NO se actualiza), F1 macro sobre las 700 predicciones:

| Modelo | Sin aumento | Mixup | Generate-then-Refine | Retrotraducción |
|---|---|---|---|---|
| XLM-R | 0.7583 | 0.7388 | **0.7685** | 0.7383 |
| mBERT | 0.7577 | 0.7491 | **0.7623** | 0.7214 |
| LaBSE | **0.7451** | 0.7195 | 0.7333 | 0.7091 |

**Texto normalizado** (minúsculas, sin puntuación, sin tildes/ñ, apóstrofos de
oclusiva glotal preservados — la condición vigente y **definitiva**, sobre el
corpus ya limpio de typos):

| Modelo | Sin aumento | Mixup | Generate-then-Refine | Retrotraducción |
|---|---|---|---|---|
| mBERT | **0.6575** | 0.6346 | 0.6491 | 0.6214 |
| XLM-R | 0.5481 | 0.5580 | **0.5798** | 0.5574 |
| LaBSE | 0.5873 | 0.5927 | **0.5952** | 0.5683 |


**Cuánto pesaban los atajos** (sin aumento, F1 macro — diagnóstico que motivó la
decisión; se corrió antes de limpiar el corpus, sirve solo para dimensionar el
problema, no como número final):

| Modelo | Texto original | Solo minúsculas | Solo sin puntuación |
|---|---|---|---|
| XLM-R | 0.7583 | 0.6877 | 0.6693 |
| mBERT | 0.7577 | 0.7153 | 0.7076 |
| LaBSE | 0.7451 | 0.6862 | 0.6409 |

Los atajos pesaban de forma parecida (cada uno ~0.04-0.10 de F1) y se sumaban. La
categoría PRG pasa de F1 ≈ 0.97-0.99 a ≈ 0.50-0.54 al quitar la puntuación.

### Efecto de cada técnica frente a "sin aumento"

Diferencia pareada de F1 sobre las mismas 700 oraciones, texto normalizado
([`../evaluacion/comparacion_pareada.py`](../evaluacion/comparacion_pareada.py)); en
negrita, las diferencias cuyo IC95% no incluye 0.

| Técnica | LaBSE | mBERT | XLM-R |
|---|---|---|---|
| Mixup | +0.005 | **-0.023** | +0.010 |
| Generate-then-Refine | +0.008 | -0.008 | **+0.032** |
| Retrotraducción | -0.019 | **-0.036** | +0.009 |

**Conclusión:** la única técnica que ayuda de forma distinguible es
**Generate-then-Refine en XLM-R** (+0.032, IC95% [+0.012, +0.052]); Mixup y
Retrotraducción **empeoran** de forma distinguible a mBERT (-0.023 y -0.036); el
resto de combinaciones no es distinguible de cero. Entre modelos (sin aumento):
**mBERT es significativamente mejor que XLM-R** (+0.109, IC95% [+0.071, +0.148])
y **que LaBSE** (+0.070, IC95% [+0.031, +0.108]); XLM-R vs. LaBSE no es
distinguible (-0.039, IC95% [-0.081, +0.001]). La tabla "con puntuación" de
arriba y las tablas por técnica de más abajo (que usan el split único) quedan
solo como referencia histórica.

Cómo se genera el aumento para la validación cruzada:
- **Retrotraducción:** un único catálogo de las 700 oraciones
  (`salidas/retrotraduccion_pool.csv`, con `id_origen`); cada fold usa las filas
  cuya oración de origen está en su train y las filtra con el centroide de ese
  train (`retrotraduccion.py --etapa generar` en Colab, una sola vez).
- **Generate-then-Refine:** un CSV por fold
  (`salidas/generate_then_refine_fold<N>.csv`, `generate_then_refine.py --fold N`).
- **Deduplicación contra el test:** en cada fold se descartan las filas sintéticas
  cuyo texto normalizado (sin mayúsculas, puntuación ni tildes) coincide con una
  oración de su test. Motivo: los datos con los que F. Prado entrenó su NMT
  incluyen el 55% de las oraciones de este corpus (46% en su train; ambos parten de
  `flashcards2`) y el modelo las memorizó: el 15% de las traducciones del catálogo
  reproduce una oración real del corpus.
- **Limitación conocida:** los "patrones candidatos" que se le pasan a Claude en
  Generate-then-Refine vienen del análisis R5, calculado con el corpus completo
  (incluye oraciones de test). Es una fuga débil y agregada (8 patrones por
  categoría), no corregida.

### Otras comprobaciones ([`../historico/atajos_texto_crudo/auditoria_optimismo.py`](../historico/atajos_texto_crudo/auditoria_optimismo.py))
- 30 splits aleatorios 70/15/15 dan un F1 promedio de 0.751-0.760, igual que la
  validación cruzada con puntuación. El split único (semilla 42) fue una tirada
  difícil (percentil 7-20; su baseline por palabras, 0.383, quedó por debajo de los
  30 splits).
- Cuando una oración de test no comparte ninguna palabra con el train (37% de los
  casos), la exactitud baja a ~0.65 (0.79-0.81 con solapamiento parcial, ~0.95 con
  solapamiento alto).

## Mixup



Interpola los vectores de dos oraciones shiwilu existentes de la **misma**
categoría de intención (`x_sintetico = λx_i + (1-λ)x_j`, `λ ~ Beta(α, α)`),
generando un vector sintético que hereda la etiqueta compartida. No genera
texto nuevo — opera directamente sobre los embeddings ya extraídos, y por
eso no depende de ningún modelo generador externo.

### Uso

```bash
python oe2_aumento_de_datos/tecnicas_aumento/mixup.py --modelo labse
python oe2_aumento_de_datos/tecnicas_aumento/mixup.py --modelo labse --alpha 0.4 --multiplicador 1.0
```

### Resultado de referencia (ya ejecutado)

| Modelo | Sin aumento | Mixup | Delta |
|---|---|---|---|
| LaBSE | 0.6943 | 0.6632 | -0.0311 |
| mBERT | 0.6977 | 0.6937 | -0.0040 |
| XLM-R | 0.7223 | 0.7373 | +0.0150 |

Mejora pequeña en XLM-R, prácticamente sin cambio en mBERT, y una caída en
LaBSE — ver [`../historico/split_unico/resumen_experimentos.csv`](../historico/split_unico/resumen_experimentos.csv) para
la matriz completa y [`../historico/split_unico/bootstrap_ic.py`](../historico/split_unico/bootstrap_ic.py) para los
intervalos de confianza (se solapan bastante entre configuraciones, así que
ninguna de estas diferencias es concluyente con ~99 oraciones de prueba).

## Generate-then-Refine

**Generación:** Claude (`claude-sonnet-4-6`) genera oraciones nuevas en
shiwilu para una categoría de intención, condicionado por:
- la descripción de la categoría (`shiwilu/taxonomia.py`),
- ejemplos reales del corpus de esa categoría,
- los marcadores morfosintácticos que las fuentes (artículo de JIPA y el libro *Voces shiwilu*) afirman
  **explícitamente**, con su página
  ([`../analisis_intrinseco/marcadores_fuentes.csv`](../analisis_intrinseco/marcadores_fuentes.csv),
  solo las filas `usar = si`; ver «Marcadores explícitos» más abajo),
- patrones **candidatos** detectados estadísticamente en el corpus pero
  **sin validar** externamente (palabras características, secuencias
  iniciales/finales) — se le indica explícitamente al modelo que son pistas,
  no reglas confirmadas.

**Refinamiento:** cada oración generada pasa por tres filtros:
1. **Marcador** — si la categoría tiene marcadores explícitos en las fuentes, exige que
   la oración coincida con el patrón de al menos uno (los patrones están en
   `marcadores_fuentes.csv`; DES y SAL no tienen y siempre pasan).
2. **Idioma** — heurística simple contra "españolización" (rechaza si el
   shiwilu generado es idéntico al glosado en español).
3. **Semántico** — similitud coseno (LaBSE, congelado) entre la oración
   generada y el centroide de los ejemplos reales de su categoría; rechaza
   si está demasiado alejada (deriva de tema) o es casi idéntica a un
   ejemplo existente (duplicado).

### Marcadores explícitos (filtro de marcador)

El filtro y el prompt usan **solo** lo que las fuentes dicen de forma explícita. La tabla
[`marcadores_fuentes.csv`](../analisis_intrinseco/marcadores_fuentes.csv) lista cada forma con su función, página y nivel de evidencia:

| Nivel | Significado | ¿Se usa? |
|---|---|---|
| **A** | La fuente da la forma **y** su función | Sí |
| **B** | La fuente da la forma con una glosa, pero no la etiqueta como marcador de esa función (p. ej. `enchuku'` «vamos», palabras interrogativas léxicas) | No (activable cambiando `usar`) |
| **C** | La función descrita no corresponde a la categoría (`i'na` focalizador, `-sa'` delimitativo, `-sha` diminutivo) | No |

Usados (nivel A): NEG `i'n`, `-inpu'`; PRG `a'cha`, `a'ta'`; REQUEST imperativo `-(k)er'`; EMO `chi`, `ten` (palabras
independientes); AFI `ajá`, `ahã`, `untana`. DES y SAL no tienen marcador. Fuentes: Valenzuela & Gussenhoven (2013, JIPA) y *Voces
shiwilu* (Parte II). El capítulo de shiwilu de la Enciclopedia del Bicentenario no afirma ningún marcador de estos.

Si cambia la lista no hace falta volver a generar (ni gastar tokens): `refiltrar_marcadores.py` reaplica el filtro a las
candidatas ya guardadas (que incluyen las rechazadas) y escribe copias para evaluar.

Las oraciones que no pasan todos los filtros **no se descartan**: quedan
marcadas `revisar_hablante_nativo` en la columna `estado_filtro`, siguiendo
el protocolo descrito en la metodología (Cap. 2.2.8) — la decisión de
incluirlas o no en el corpus aumentado final es manual, no automática.

### Uso

```bash
python oe2_aumento_de_datos/tecnicas_aumento/generate_then_refine.py --cantidad 20
python oe2_aumento_de_datos/tecnicas_aumento/generate_then_refine.py --categorias DES NEG REQUEST --cantidad 15
```

Requiere `ANTHROPIC_API_KEY` en `.env` (ver `.env.example` en la raíz del
repositorio). Ya probado sobre las 7 categorías (20 por categoría): 138 oraciones generadas (2 candidatos del LLM llegaron mal formados y se descartaron), 81 aprobadas. Tasa de aprobación muy dispareja por
categoría — SAL y DES casi perfectas (19/19 y 20/20), PRG y REQUEST muy bajas (7/20 y 2/20), probablemente porque dependen de morfología verbal (conjugación
imperativa, partículas interrogativas) que un LLM sin ajuste fino reproduce
peor que los marcadores léxicos simples de SAL/DES.

| Modelo | Sin aumento | Generate-then-Refine | Delta |
|---|---|---|---|
| LaBSE | 0.6943 | 0.7046 | +0.0103 |
| mBERT | 0.6977 | 0.6652 | -0.0325 |
| XLM-R | 0.7223 | **0.7607** | +0.0384 |

XLM-R + Generate-then-Refine es la mejor de las 12 combinaciones evaluadas en toda la matriz. Su intervalo de confianza bootstrap ([0.6695, 0.8378]) se solapa bastante con el de XLM-R sin aumento ([0.6253, 0.8057]), así que la mejora es una tendencia, no una diferencia concluyente con ~99 oraciones de prueba. En mBERT la técnica empeora el F1 y en LaBSE lo mejora apenas.

### Salida

`oe2_aumento_de_datos/tecnicas_aumento/salidas/generate_then_refine.csv` — columnas: `id`,
`espanol`, `shiwilu`, `intencion`, `fuente`, `estado_filtro`,
`detalle_filtro` (qué filtro(s) falló, si aplica), `similitud_labse`.

## Retrotraducción

Opera sobre el componente en **español** del corpus (no existe
retrotraducción directa shiwilu→shiwilu, pues no hay sistemas de traducción
automática disponibles para shiwilu fuera del que aquí se reutiliza):

1. **Paráfrasis en español** (etapa `generar`, en Colab, sobre las 700 oraciones del corpus): cada oración se retrotraduce
   español→inglés→español con Helsinki-NLP (Opus-MT), obteniendo una
   paráfrasis nueva pero semánticamente equivalente. Se descartan las
   paráfrasis idénticas al original.
2. **Traducción a shiwilu:** la paráfrasis se traduce al shiwilu con el
   sistema NMT (NLLB-200 + LoRA) desarrollado por **F. Prado** en su propia
   tesis (comunicación personal, 14 de septiembre de 2026;
   <https://github.com/fapi19/Tesis_Spa-Jeb>).
3. **Refinamiento** (etapa `filtrar`, en local, solo con las filas cuya oración de origen está en train): mismo protocolo que Generate-then-Refine (filtro de
   idioma + filtro semántico vía LaBSE) — ambas técnicas comparten el riesgo
   de producir enunciados sintéticos erróneos, según la metodología (Cap. 2.2.8).

### Paso previo: obtener el checkpoint de F. Prado

Este script **no** entrena nada — necesita el checkpoint ya entrenado. Ver
[`colab/colab_entrenar_checkpoint.ipynb`](colab/colab_entrenar_checkpoint.ipynb) para el
notebook completo (clona su repo y reentrena su configuración campeona
`v2.1b LoRA+` en Colab, ya que no se distribuyen los pesos originales; ya
verificado: chrF++ promedio = 43.19, consistente con lo reportado por el
autor). El notebook asume que el checkpoint ya está guardado en Drive
(`shiwilu_checkpoint/`) y solo lo copia y corre `retrotraduccion.py`; el
reentrenamiento queda como paso opcional. Hay que volver a correrlo cada vez
que cambie el corpus, no el split: el notebook genera un catálogo de las 700 oraciones (`salidas/retrotraduccion_pool.csv`, con `id_origen`) y el filtro que depende del train se aplica después, en local, con `--etapa filtrar`. Resumen de los comandos para entrenar desde cero:

```bash
git clone https://github.com/fapi19/Tesis_Spa-Jeb.git oe2_aumento_de_datos/tecnicas_aumento/tesis_spa_jeb
cd oe2_aumento_de_datos/tecnicas_aumento/tesis_spa_jeb
pip install -r requirements/nmt.txt
python -m scripts.nmt.30_train_lora --variant xl --rank 32 --alpha 64 \
    --loraplus-lr-ratio 16 --output-dir models/nmt/nllb_bidi_lora_v2_1b_loraplus_xl
```

La carpeta `oe2_aumento_de_datos/tecnicas_aumento/tesis_spa_jeb/` queda ignorada por git (no se
vendoriza, ver `.gitignore`).

### Uso

```bash
python oe2_aumento_de_datos/tecnicas_aumento/retrotraduccion.py \
    --checkpoint oe2_aumento_de_datos/tecnicas_aumento/tesis_spa_jeb/models/nmt/nllb_bidi_lora_v2_1b_loraplus_xl
python oe2_aumento_de_datos/tecnicas_aumento/retrotraduccion.py --checkpoint ... --categorias DES NEG --limite 20   # prueba rapida
python oe2_aumento_de_datos/tecnicas_aumento/retrotraduccion.py --etapa filtrar --salida salidas/retrotraduccion_split.csv   # local, tras tener el catalogo
```

Ya probado de punta a punta: 429 oraciones generadas (de 497 de train — el
resto eran paráfrasis idénticas al original, descartadas), 421 aprobadas por los filtros y 8 marcadas `revisar_hablante_nativo`.

| Modelo | Sin aumento | Retrotraducción | Delta |
|---|---|---|---|
| LaBSE | 0.6943 | 0.6749 | -0.0194 |
| mBERT | 0.6977 | 0.6202 | -0.0775 |
| XLM-R | 0.7223 | 0.6546 | -0.0677 |

Retrotraducción empeora el F1 de los 3 modelos frente al baseline sin aumento (de -0.02 a -0.08), y es la técnica con peor resultado de las 3. No está claro por qué, pero es consistente con que el paso de parafraseo en español (Helsinki-NLP) introduce variación léxica que el NMT de F. Prado no siempre traduce de forma fiel al shiwilu (ver `similitud_labse` en `salidas/retrotraduccion.csv` para los casos límite). Las diferencias en LaBSE están dentro del margen de incertidumbre; en mBERT y XLM-R la caída es más marcada.

### Salida

`oe2_aumento_de_datos/tecnicas_aumento/salidas/retrotraduccion.csv` — mismas columnas que
`generate_then_refine.csv` (`id`, `espanol`, `shiwilu`, `intencion`,
`fuente`, `estado_filtro`, `detalle_filtro`, `similitud_labse`).
