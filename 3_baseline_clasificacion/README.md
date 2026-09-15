# Fase 3 — Baseline de clasificación (OE2/OE4, R4)

Baseline de referencia para la clasificación de intenciones en shiwilu:
**LaBSE congelado (sin ajuste fino) + Regresión Logística, sin ningún
aumento de datos.** Sirve como línea base mínima contra la cual comparar
tanto las técnicas de aumento de datos (OE2) como los demás métodos de
caracterización de embeddings (OE3).

- **Embeddings:** `sentence-transformers/LaBSE` (Feng et al., 2022), usado
  tal cual, sin fine-tuning — solo se extraen representaciones de oración.
- **Clasificador:** `LogisticRegression` de scikit-learn, sobre esas
  representaciones.
- **Datos:** únicamente `corpus/corpus_shiwilu_final.csv` (700 pares),
  columna `shiwilu`, dividido 70/15/15 (train/dev/test) estratificado por
  categoría de intención — sin ninguna técnica de aumento aplicada.

## Uso

```bash
python 3_baseline_clasificacion/baseline.py
```

No requiere GPU ni clave de API — corre en CPU en menos de un minuto (LaBSE
sobre 700 oraciones cortas es rápido incluso sin GPU). Se puede correr
localmente en VS Code sin necesidad de Colab.

## Salida

`3_baseline_clasificacion/resultados/`:
- `metricas.json` — F1 macro, F1 ponderado, exactitud (train/dev/test).
- `matriz_confusion.png` — matriz de confusión sobre el conjunto de prueba.
- `reporte_clasificacion.csv` — métricas desagregadas por categoría de intención.
