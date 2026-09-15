# Fase 3 — Baselines de generación español-shiwilu (OE2 / R4)

Esta fase implementa los dos baselines de generación español→shiwilu definidos
como punto de comparación para las técnicas de aumento de datos (OE2, R4 del
Entregable 1). Ninguno de los dos "compite" por ser mejor entre sí: son dos
formas distintas de producir nuevas oraciones en shiwilu a partir de español,
que luego se combinan con cada técnica de aumento (R5/R6) y cada método de
caracterización de embeddings (R7/R8) en la matriz experimental de OE4.

| | Baseline 1 | Baseline 2 |
|---|---|---|
| Enfoque | NMT especializado, entrenado | LLM de propósito general, sin entrenar |
| Arquitectura | NLLB-200-distilled-600M + LoRA (r=32, α=64, LoRA+) | Claude (`claude-sonnet-4-6`), few-shot prompting |
| Requiere entrenamiento | Sí (GPU, horas) | No |
| Carpeta | [`baseline1_nmt_lora/`](baseline1_nmt_lora/) | [`baseline2_llm_claude/`](baseline2_llm_claude/) |

## Baseline 1 — NMT NLLB-200 + LoRA

Reentrenamiento del sistema desarrollado por **Fabian Prado** para su propia
tesis (F. Prado, comunicación personal, 14 de septiembre de 2026; código en
<https://github.com/fapi19/Tesis_Spa-Jeb>). No se vendoriza su código ni sus
datos en este repositorio — se clona su repositorio como dependencia externa
y se reentrena la configuración campeona (`v2.1b LoRA+`) siguiendo sus propios
scripts. Ver [`baseline1_nmt_lora/README.md`](baseline1_nmt_lora/README.md)
para los pasos exactos.

## Baseline 2 — LLM few-shot (Claude)

Generación mediante `claude-sonnet-4-6` vía API, sin ajuste fino, condicionada
por ejemplos reales del corpus (`corpus/corpus_shiwilu_final.csv`) y por los
patrones lingüísticos y marcadores morfosintácticos documentados en el
análisis intrínseco de la Fase 2
(`2_analisis_corpus/resultados/tablas/analisis_marcadores_documentados.csv`).
Ver [`baseline2_llm_claude/README.md`](baseline2_llm_claude/README.md).

## Quién escribe dónde

Ambos baselines solo **leen** de `corpus/` y de la Fase 2 (`2_analisis_corpus/resultados/`);
escriben sus traducciones únicamente en su propia carpeta `salidas/`. No
modifican el corpus final ni los resultados del análisis intrínseco.
