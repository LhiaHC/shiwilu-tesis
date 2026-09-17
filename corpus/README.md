# Corpus — frontera entre las dos fases

Esta carpeta contiene el **producto de la Fase 1** y el **insumo de la Fase 2**.

> **Regla:** solo la Etapa 3 de `1_objetivo1_corpus/pipeline/` escribe aquí.
> La Fase 2 lee este archivo pero nunca lo modifica; sus salidas van a
> `3_baselines_y_aumento_datos/analisis_intrinseco/resultados/`.

## `corpus_shiwilu_final.csv`

700 pares bilingües español–shiwilu etiquetados con 7 categorías de intención
comunicativa (100 por categoría).

| Columna | Tipo | Descripción |
|---|---|---|
| `id` | texto | Identificador único del par |
| `espanol` | texto | Enunciado en español |
| `shiwilu` | texto | Traducción al shiwilu |
| `intencion` | categoría | Una de las 7 intenciones (ver abajo) |
| `fuente` | categoría | `flashcards` o `api_generada` |

### Taxonomía de intenciones

Definida en `shiwilu/taxonomia.py`, que es la única fuente de verdad para ambas
fases.

| Código | Intención | Tipo Searle |
|---|---|---|
| `SAL` | Saludos y despedidas | Expresivo |
| `EMO` | Expresiones emocionales | Expresivo |
| `PRG` | Preguntas informativas | Directivo |
| `REQUEST` | Solicitudes y mandatos | Directivo |
| `AFI` | Afirmaciones y confirmaciones | Asertivo |
| `NEG` | Negaciones y rechazos | Asertivo |
| `DES` | Descripciones de estados o eventos | Asertivo |

### Distribución por fuente

| Intención | Flashcards | API | Total |
|---|---|---|---|
| SAL | 15 | 85 | 100 |
| EMO | 21 | 79 | 100 |
| PRG | 100 | 0 | 100 |
| REQUEST | 100 | 0 | 100 |
| AFI | 9 | 91 | 100 |
| NEG | 14 | 86 | 100 |
| DES | 100 | 0 | 100 |
| **Total** | **359** | **341** | **700** |

## Versión

Versión 1.0.0 del corpus. Si se corrige o amplía, conviene etiquetar el commit
correspondiente (`git tag corpus-v1.1`) para que los resultados de la Fase 2
puedan referirse sin ambigüedad a la versión sobre la que se calcularon.

## Licencia y citación

CC BY 4.0 — ver [`LICENSE-DATOS`](../LICENSE-DATOS) y [`CITATION.cff`](../CITATION.cff).
