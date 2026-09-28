# Técnicas de aumento de datos (OE2, R6)

Tres técnicas para ampliar el corpus de entrenamiento en shiwilu, aplicadas
sobre los baselines de [`../../2_baselines/`](../../2_baselines/) y evaluadas
contra ellos.

| Técnica | Generador | Estado |
|---|---|---|
| **Mixup** | Ninguno — interpola embeddings de oraciones shiwilu existentes de la misma categoría | ✅ Probado — [`mixup.py`](mixup.py) |
| **Generate-then-Refine** | Claude (LLM), sin ajuste fino | ✅ Probado (138 oraciones, 81 aprobadas) — [`generate_then_refine.py`](generate_then_refine.py) |
| **Retrotraducción** | Helsinki-NLP (paráfrasis en español) + NMT de F. Prado (traduce a shiwilu) | ✅ Probado (429 oraciones, 421 aprobadas; checkpoint chrF++=43.19) — [`retrotraduccion.py`](retrotraduccion.py) |

**Notas de metodología (2026-09-21/22):**
- Generate-then-Refine y Retrotraducción usaban ejemplos few-shot y el
  centroide del filtro semántico calculados sobre el corpus **completo**
  (train+dev+test), lo que dejaba que información de dev/test influyera en
  qué texto sintético se generaba y aprobaba. Se corrigió para usar solo
  train (`cargar_ejemplos_por_categoria` en `generate_then_refine.py` y el
  parámetro `corpus_train` en `refinar()` de `retrotraduccion.py`). Ambas
  técnicas ya se regeneraron con la corrección.
- La división train/dev/test (`comun.dividir_train_dev_test`) agrupaba por
  texto shiwilu **exacto**, pero el corpus repite la misma raíz con variantes
  de mayúsculas/puntuación (`"PANTE'CHEK"` / `"¡pante'chek!"`). Ahora agrupa
  por texto **normalizado** (sin mayúsculas ni puntuación), cerrando esa fuga
  también. Ver [`../../2_baselines/README.md`](../../2_baselines/README.md)
  para el detalle y los números finales de los 12 experimentos.
- El split quedó **congelado** en `2_baselines/split_fijo.csv`. Ambas técnicas
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
5 folds** ([`../validacion_cruzada.py`](../validacion_cruzada.py)): cada una de
las 700 oraciones se usa como test una vez (folds agrupados por texto shiwilu
normalizado —mayúsculas, puntuación, espacios, tildes y ñ— y estratificados,
congelados en `2_baselines/folds_fijos.csv`), y el aumento de cada fold se genera
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
(`comun.quitar_puntuacion(..., quitar_tildes=True)`, la condición `sin_puntuacion`
de [`../validacion_cruzada.py`](../validacion_cruzada.py)).

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
([`../comparacion_pareada.py --sin-puntuacion`](../comparacion_pareada.py)); en
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

### Otras comprobaciones ([`../auditoria_optimismo.py`](../auditoria_optimismo.py))
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
python 3_baselines_y_aumento_datos/tecnicas_aumento/mixup.py --modelo labse
python 3_baselines_y_aumento_datos/tecnicas_aumento/mixup.py --modelo labse --alpha 0.4 --multiplicador 1.0
```

### Resultado de referencia (ya ejecutado)

| Modelo | Sin aumento | Mixup | Delta |
|---|---|---|---|
| LaBSE | 0.6943 | 0.6632 | -0.0311 |
| mBERT | 0.6977 | 0.6937 | -0.0040 |
| XLM-R | 0.7223 | 0.7373 | +0.0150 |

Mejora pequeña en XLM-R, prácticamente sin cambio en mBERT, y una caída en
LaBSE — ver [`../resumen_experimentos.csv`](../resumen_experimentos.csv) para
la matriz completa y [`../bootstrap_ic.py`](../bootstrap_ic.py) para los
intervalos de confianza (se solapan bastante entre configuraciones, así que
ninguna de estas diferencias es concluyente con ~99 oraciones de prueba).

## Generate-then-Refine

**Generación:** Claude (`claude-sonnet-4-6`) genera oraciones nuevas en
shiwilu para una categoría de intención, condicionado por:
- la descripción de la categoría (`shiwilu/taxonomia.py`),
- ejemplos reales del corpus de esa categoría,
- los marcadores morfosintácticos ya **validados** contra la literatura
  lingüística en el análisis intrínseco
  (`3_baselines_y_aumento_datos/analisis_intrinseco/resultados/tablas/analisis_marcadores_documentados.csv`, R5),
- patrones **candidatos** detectados estadísticamente en el corpus pero
  **sin validar** externamente (palabras características, secuencias
  iniciales/finales) — se le indica explícitamente al modelo que son pistas,
  no reglas confirmadas.

**Refinamiento:** cada oración generada pasa por tres filtros:
1. **Marcador** — si la categoría tiene marcadores documentados, exige que
   al menos uno aparezca en la oración generada.
2. **Idioma** — heurística simple contra "españolización" (rechaza si el
   shiwilu generado es idéntico al glosado en español).
3. **Semántico** — similitud coseno (LaBSE, congelado) entre la oración
   generada y el centroide de los ejemplos reales de su categoría; rechaza
   si está demasiado alejada (deriva de tema) o es casi idéntica a un
   ejemplo existente (duplicado).

Las oraciones que no pasan todos los filtros **no se descartan**: quedan
marcadas `revisar_hablante_nativo` en la columna `estado_filtro`, siguiendo
el protocolo descrito en la metodología (Cap. 2.2.8) — la decisión de
incluirlas o no en el corpus aumentado final es manual, no automática.

### Uso

```bash
python 3_baselines_y_aumento_datos/tecnicas_aumento/generate_then_refine.py --cantidad 20
python 3_baselines_y_aumento_datos/tecnicas_aumento/generate_then_refine.py --categorias DES NEG REQUEST --cantidad 15
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

`3_baselines_y_aumento_datos/tecnicas_aumento/salidas/generate_then_refine.csv` — columnas: `id`,
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
[`colab_entrenar_checkpoint.ipynb`](colab_entrenar_checkpoint.ipynb) para el
notebook completo (clona su repo y reentrena su configuración campeona
`v2.1b LoRA+` en Colab, ya que no se distribuyen los pesos originales; ya
verificado: chrF++ promedio = 43.19, consistente con lo reportado por el
autor). El notebook asume que el checkpoint ya está guardado en Drive
(`shiwilu_checkpoint/`) y solo lo copia y corre `retrotraduccion.py`; el
reentrenamiento queda como paso opcional. Hay que volver a correrlo cada vez
que cambie el corpus, no el split: el notebook genera un catálogo de las 700 oraciones (`salidas/retrotraduccion_pool.csv`, con `id_origen`) y el filtro que depende del train se aplica después, en local, con `--etapa filtrar`. Resumen de los comandos para entrenar desde cero:

```bash
git clone https://github.com/fapi19/Tesis_Spa-Jeb.git 3_baselines_y_aumento_datos/tecnicas_aumento/tesis_spa_jeb
cd 3_baselines_y_aumento_datos/tecnicas_aumento/tesis_spa_jeb
pip install -r requirements/nmt.txt
python -m scripts.nmt.30_train_lora --variant xl --rank 32 --alpha 64 \
    --loraplus-lr-ratio 16 --output-dir models/nmt/nllb_bidi_lora_v2_1b_loraplus_xl
```

La carpeta `3_baselines_y_aumento_datos/tecnicas_aumento/tesis_spa_jeb/` queda ignorada por git (no se
vendoriza, ver `.gitignore`).

### Uso

```bash
python 3_baselines_y_aumento_datos/tecnicas_aumento/retrotraduccion.py \
    --checkpoint 3_baselines_y_aumento_datos/tecnicas_aumento/tesis_spa_jeb/models/nmt/nllb_bidi_lora_v2_1b_loraplus_xl
python 3_baselines_y_aumento_datos/tecnicas_aumento/retrotraduccion.py --checkpoint ... --categorias DES NEG --limite 20   # prueba rapida
python 3_baselines_y_aumento_datos/tecnicas_aumento/retrotraduccion.py --etapa filtrar --salida salidas/retrotraduccion_split.csv   # local, tras tener el catalogo
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

`3_baselines_y_aumento_datos/tecnicas_aumento/salidas/retrotraduccion.csv` — mismas columnas que
`generate_then_refine.csv` (`id`, `espanol`, `shiwilu`, `intencion`,
`fuente`, `estado_filtro`, `detalle_filtro`, `similitud_labse`).
