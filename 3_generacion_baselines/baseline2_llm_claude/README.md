# Baseline 2 — LLM few-shot (Claude, sin ajuste fino)

Genera traducciones español→shiwilu con `claude-sonnet-4-6` vía API,
condicionado por *few-shot prompting*, sin ningún ajuste fino del modelo. Es
el contraste metodológico frente al Baseline 1 (NMT especializado y
entrenado): aquí toda la adaptación al shiwilu ocurre en el prompt, no en los
pesos del modelo.

## Qué usa como contexto

- **Ejemplos reales del corpus** (`corpus/corpus_shiwilu_final.csv`), muestreados
  de la misma categoría de intención que la oración a traducir.
- **Marcadores morfosintácticos documentados** en el análisis intrínseco de la
  Fase 2 (`2_analisis_corpus/resultados/tablas/analisis_marcadores_documentados.csv`,
  producto de R5): formas como el marcador de negación `i'n`, la partícula
  interrogativa `a'cha`, o el sufijo imperativo `-ker'/-e'r/-r'`, con su
  función y su fuente lingüística (JIPA, *Voces*).

## Uso

```bash
python 3_generacion_baselines/baseline2_llm_claude/generar.py \
    --entrada ruta/a/oraciones_nuevas.csv \
    --salida 3_generacion_baselines/baseline2_llm_claude/salidas/baseline2_generado.csv
```

`oraciones_nuevas.csv` necesita las columnas `id`, `espanol` e `intencion`
(mismo esquema que `corpus/corpus_shiwilu_final.csv`, sin la columna
`shiwilu`).

Requiere `ANTHROPIC_API_KEY` en `.env` (ver `.env.example` en la raíz del
repositorio). Se hace una llamada a la API por oración — para lotes grandes,
estima el consumo antes de correrlo (ver RI07 en la matriz de riesgos del
Entregable 1).
