# Fase 4 — Técnicas de aumento de datos (OE2, R5/R6)

Tres técnicas para ampliar el corpus de entrenamiento en shiwilu, evaluadas
después contra los baselines de la Fase 3 (`3_baseline_clasificacion/`).

| Técnica | Generador | Estado |
|---|---|---|
| **Mixup** | Ninguno — interpola embeddings de oraciones shiwilu existentes de la misma categoría | Código listo y probado — [`mixup.py`](mixup.py) |
| **Generate-then-Refine** | Claude (LLM), sin ajuste fino | Código listo, sin probar (falta API key) — [`generate_then_refine.py`](generate_then_refine.py) |
| **Retrotraducción** | Helsinki-NLP (paráfrasis en español) + NMT de F. Prado (traduce a shiwilu) | Código listo; el paso 1 (Helsinki-NLP) ya probado, el paso 2 pendiente de que entrenes el checkpoint — [`retrotraduccion.py`](retrotraduccion.py) |

## Mixup

Interpola los vectores de dos oraciones shiwilu existentes de la **misma**
categoría de intención (`x_sintetico = λx_i + (1-λ)x_j`, `λ ~ Beta(α, α)`),
generando un vector sintético que hereda la etiqueta compartida. No genera
texto nuevo — opera directamente sobre los embeddings ya extraídos, y por
eso no depende de ningún modelo generador externo.

### Uso

```bash
python 4_aumento_datos/mixup.py --modelo labse
python 4_aumento_datos/mixup.py --modelo labse --alpha 0.4 --multiplicador 1.0
```

### Resultado de referencia (LaBSE, ya ejecutado)

| | F1 macro | F1 ponderado | Exactitud |
|---|---|---|---|
| Baseline (sin aumento) | 0.7565 | 0.7565 | 0.7619 |
| Mixup | 0.7582 | 0.7582 | 0.7619 |

Mejora mínima (+0.0017) — dentro del margen de ruido esperable con un
conjunto de prueba de 105 oraciones.

## Generate-then-Refine

**Generación:** Claude (`claude-sonnet-4-6`) genera oraciones nuevas en
shiwilu para una categoría de intención, condicionado por:
- la descripción de la categoría (`shiwilu/taxonomia.py`),
- ejemplos reales del corpus de esa categoría,
- los marcadores morfosintácticos ya **validados** contra la literatura
  lingüística en el análisis intrínseco
  (`2_analisis_corpus/resultados/tablas/analisis_marcadores_documentados.csv`, R5),
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
python 4_aumento_datos/generate_then_refine.py --cantidad 20
python 4_aumento_datos/generate_then_refine.py --categorias DES NEG REQUEST --cantidad 15
```

Requiere `ANTHROPIC_API_KEY` en `.env` (ver `.env.example` en la raíz del
repositorio) — **el código está listo pero no se ha podido probar todavía**
por falta de créditos de API. Antes de correrlo sobre las 7 categorías,
pruébalo primero con `--categorias DES --cantidad 3` para revisar el formato
de salida y el costo real por oración.

### Salida

`4_aumento_datos/salidas/generate_then_refine.csv` — columnas: `id`,
`espanol`, `shiwilu`, `intencion`, `fuente`, `estado_filtro`,
`detalle_filtro` (qué filtro(s) falló, si aplica), `similitud_labse`.

## Retrotraducción

Opera sobre el componente en **español** del corpus (no existe
retrotraducción directa shiwilu→shiwilu, pues no hay sistemas de traducción
automática disponibles para shiwilu fuera del que aquí se reutiliza):

1. **Paráfrasis en español:** cada oración de entrenamiento se retrotraduce
   español→inglés→español con Helsinki-NLP (Opus-MT), obteniendo una
   paráfrasis nueva pero semánticamente equivalente. Se descartan las
   paráfrasis idénticas al original.
2. **Traducción a shiwilu:** la paráfrasis se traduce al shiwilu con el
   sistema NMT (NLLB-200 + LoRA) desarrollado por **F. Prado** en su propia
   tesis (comunicación personal, 14 de septiembre de 2026;
   <https://github.com/fapi19/Tesis_Spa-Jeb>).
3. **Refinamiento:** mismo protocolo que Generate-then-Refine (filtro de
   idioma + filtro semántico vía LaBSE) — ambas técnicas comparten el riesgo
   de producir enunciados sintéticos erróneos, según la metodología (Cap. 2.2.8).

### Paso previo obligatorio: obtener el checkpoint de F. Prado

Este script **no** entrena nada — necesita el checkpoint ya entrenado. Sigue
los mismos pasos que ya usamos para el Baseline de traducción (clonar su
repo y reentrenar su configuración campeona `v2.1b LoRA+` en Colab, ya que
no se distribuyen los pesos originales):

```bash
git clone https://github.com/fapi19/Tesis_Spa-Jeb.git 4_aumento_datos/tesis_spa_jeb
cd 4_aumento_datos/tesis_spa_jeb
pip install -r requirements/nmt.txt
python -m scripts.nmt.30_train_lora --variant xl --rank 32 --alpha 64 \
    --loraplus-lr-ratio 16 --output-dir models/nmt/nllb_bidi_lora_v2_1b_loraplus_xl
```

La carpeta `4_aumento_datos/tesis_spa_jeb/` queda ignorada por git (no se
vendoriza, ver `.gitignore`).

### Uso

```bash
python 4_aumento_datos/retrotraduccion.py \
    --checkpoint 4_aumento_datos/tesis_spa_jeb/models/nmt/nllb_bidi_lora_v2_1b_loraplus_xl
python 4_aumento_datos/retrotraduccion.py --checkpoint ... --categorias DES NEG --limite 20
```

El paso 1 (paráfrasis en español, Helsinki-NLP) ya se probó de forma
aislada y funciona correctamente. El paso 2 (traducción a shiwilu) todavía
no se ha podido probar de punta a punta porque falta entrenar el checkpoint.

### Salida

`4_aumento_datos/salidas/retrotraduccion.csv` — mismas columnas que
`generate_then_refine.csv` (`id`, `espanol`, `shiwilu`, `intencion`,
`fuente`, `estado_filtro`, `detalle_filtro`, `similitud_labse`).
