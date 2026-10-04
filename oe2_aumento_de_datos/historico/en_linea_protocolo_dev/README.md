# Generación en línea con dev (etapa cerrada)

Resultados de la corrida en Colab que genera el aumento dentro de cada fold con la estrategia
**core / pool / dev**: `C` se elegía en un dev interno de ~90 oraciones, con el aumento generado solo con el core.
Esa forma de elegir `C` es ruidosa y fue reemplazada por la **CV interna** (K=5) de
[`../../evaluacion/`](../../evaluacion/). **No se corre ni se actualiza**; se conserva porque documenta cómo se llegó a
las decisiones de la tesis y porque `diagnostico_c/` y el README de OE2 citan estas cifras. No usar para reportar resultados finales.

Cada archivo contiene los 12 experimentos: la técnica corrida sale de esa corrida y las otras tres, de la evaluación vigente (protocolo A).

| Etiqueta del archivo | Qué se evaluó |
|---|---|
| `..._en_linea_generate_then_refine_colab` | GtR 20 por categoría, con el filtro de marcador original (lista R5, coincidencia por subcadena) |
| `..._en_linea_generate_then_refine_marcadores_fuentes` | las mismas candidatas de GtR 20, refiltradas con los marcadores que las fuentes afirman explícitamente |
| `..._en_linea_generate_then_refine_colab_c40`, `_c80`, `_c120` | GtR con 40, 80 y 120 por categoría (~28%, ~57% y ~85% de las reales) |
| `..._en_linea_retrotraduccion_colab` | Retrotraducción con 1 paráfrasis por oración (~83%) |

Prefijos: `validacion_cruzada_predicciones_sin_puntuacion` / `validacion_cruzada_resumen_sin_puntuacion` (F1 y probabilidades),
`comparacion_pareada_sin_puntuacion` (diferencias pareadas), `curvas_roc` (ROC) y `volumen_sintetico_sin_puntuacion` (sintéticos usados).
El F1 con `C` fijo de estas configuraciones está en [`../../evaluacion/diagnostico_c/resultados/`](../../evaluacion/diagnostico_c/resultados/).

Lo que se concluyó con estos datos: GtR no mejora a mBERT sin aumento de forma distinguible, ayuda de forma moderada a XLM-R, y más
volumen (20 → 120) no da una tendencia sostenida; Retrotraducción empeora a mBERT. Esas conclusiones se revisan con la CV interna.
El zip original de Colab está en [`../colab_zips/`](../colab_zips/).
