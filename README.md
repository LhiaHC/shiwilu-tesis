# Corpus de intenciones para el shiwilu

Corpus digital de la lengua **shiwilu** (jebero, ISO 639-3: `jeb`) anotado con
categorías de intención comunicativa, construido como parte del proyecto de
tesis *Evaluación de embeddings multilingües en la clasificación de intenciones
para la lengua shiwilu*.

El shiwilu es una lengua amazónica peruana hablada en el distrito de Jeberos
(Loreto), en situación de peligro crítico de extinción. Este es el primer corpus
digital anotado para esta lengua.

- **Autora:** Lhía Antonella Hurtado Claros
- **Asesor:** Erasmo Gómez Montoya
- **Institución:** Pontificia Universidad Católica del Perú — Facultad de Ciencias e Ingeniería

---

## Organización del repositorio

El repositorio está organizado por **objetivo específico de la tesis**
(ver Entregable 1), con el corpus como frontera explícita entre la
construcción del corpus (OE1) y todo lo que se construye sobre él (OE2).

```
├── shiwilu/                       ◆ Núcleo compartido — lo que cruza la frontera
│   ├── taxonomia.py                 Intenciones, tipos de Searle, descripciones
│   ├── anotacion.py                 Reglas de anotación (baseline de OE1)
│   ├── dominios.py                  Dominios semánticos y vocabulario semilla
│   ├── excel.py                     Formato de los reportes Excel
│   └── rutas.py                     Rutas ancladas a la raíz del repositorio
│
├── 1_objetivo1_corpus/            ◆ OE1 (R1-R3) — construir el corpus
│   ├── datos/                       Materiales fuente
│   ├── pipeline/                    Etapas 0 a 3
│   └── intermedios/                 Productos intermedios y logs de la API
│
├── corpus/                        ★ FRONTERA — salida de OE1, entrada de OE2
│   └── corpus_shiwilu_final.csv
│
├── 2_baselines/                   ◆ OE2 (R4) — baselines de clasificación
│   ├── baseline.py                  LaBSE / mBERT / XLM-R congelados + Regresión Logística
│   ├── correr_todos.py              corre los 3 y compara
│   ├── baseline_trivial.py          piso sin embeddings (mayoria, solapamiento de palabras)
│   ├── split_fijo.csv               split train/dev/test congelado (no recalcular)
│   └── folds_fijos.csv              folds de la validación cruzada, congelados
│
├── 3_baselines_y_aumento_datos/   ◆ OE2 (R5-R6) — análisis + aumento, sobre los baselines
│   ├── analisis_intrinseco/         Notebooks y tablas del análisis intrínseco (R5)
│   ├── tecnicas_aumento/            Mixup, Generate-then-Refine, Retrotraducción (R6)
│   ├── resumen_experimentos.py      Consolida los 12 experimentos (3 modelos x 4 config.)
│   ├── validacion_cruzada.py        EVALUACIÓN PRINCIPAL: 12 experimentos en 5 folds
│   ├── comparacion_pareada.py       Diferencias pareadas (técnica vs. sin aumento, modelo vs. modelo)
│   ├── curva_aprendizaje.py         F1 según cuánto train se usa
│   ├── auditoria_optimismo.py       Auditoría: splits aleatorios, solapamiento léxico, puntuación
│   └── bootstrap_ic.py              Intervalo de confianza del F1 del split único
│
├── 4_caracterizacion_embeddings/  ◆ OE3 (R7-R8) — caracterización de embeddings
│   └── caracterizacion.py           4 estrategias de pooling x 3 modelos, métricas
│                                     intrínsecas (silueta, DB, CH) y proyecciones t-SNE/UMAP,
│                                     sobre el corpus original y los aumentados por R6
│
├── 5_sintesis_r9/                 ◆ OE4 (R9) — síntesis comparativa final
│   └── sintesis.py                  Cruza lo extrínseco (R4-R6) con lo intrínseco (R7-R8)
│
├── docs/                          Metodología de construcción del corpus
└── tests/                         Pruebas de las reglas de anotación
```

### Quién escribe dónde

| Zona | Lee de | Escribe en |
|---|---|---|
| `1_objetivo1_corpus/` | `datos/`, `shiwilu/` | `intermedios/` y `corpus/` |
| `corpus/` | — | *solo la Etapa 3 de OE1* |
| `2_baselines/` | `corpus/`, `shiwilu/` | `resultados/` únicamente |
| `3_baselines_y_aumento_datos/analisis_intrinseco/` | `corpus/`, `shiwilu/` | `resultados/` únicamente |
| `3_baselines_y_aumento_datos/tecnicas_aumento/` | `corpus/`, `analisis_intrinseco/resultados/`, `2_baselines/`, `shiwilu/` | `salidas/` únicamente |
| `4_caracterizacion_embeddings/` | `corpus/`, `tecnicas_aumento/salidas/`, `2_baselines/`, `shiwilu/` | `resultados/` únicamente |
| `5_sintesis_r9/` | `validacion_cruzada_resumen.csv`, `4_caracterizacion_embeddings/resultados/` | `resultados/` únicamente (no entrena ni mide nada nuevo) |
| `shiwilu/` | — | nada (es solo código) |

Esa separación mantiene el corpus estable y citable, y permite borrar y
regenerar `3_baselines_y_aumento_datos/analisis_intrinseco/resultados/` sin
tocar nada más.

---

## El corpus

`corpus/corpus_shiwilu_final.csv` contiene **700 pares bilingües
español–shiwilu** etiquetados con 7 categorías de intención (100 por categoría).
El esquema completo, la taxonomía y la distribución por fuente están en
[`corpus/README.md`](corpus/README.md).

La taxonomía se definió a partir de la teoría de actos ilocucionarios de Searle
(1975) y se validó contra el benchmark multilingüe MASSIVE
(FitzGerald et al., 2022).

| Código | Intención | Tipo Searle |
|---|---|---|
| `SAL` | Saludos y despedidas | Expresivo |
| `EMO` | Expresiones emocionales | Expresivo |
| `PRG` | Preguntas informativas | Directivo |
| `REQUEST` | Solicitudes y mandatos | Directivo |
| `AFI` | Afirmaciones y confirmaciones | Asertivo |
| `NEG` | Negaciones y rechazos | Asertivo |
| `DES` | Descripciones de estados o eventos | Asertivo |

---

## Instalación

```bash
git clone <url-del-repositorio>
cd <nombre-del-repositorio>

pip install -e .                        # deja importable el paquete `shiwilu`
pip install -r requirements.txt          # o solo requirements/fase1.txt / fase2.txt / fase3.txt
```

`pip install -e .` es opcional: los scripts y notebooks localizan la raíz del
repositorio por su cuenta. Instalarlo hace los imports más limpios.

**Solo para reproducir OE1 o Generate-then-Refine** hace falta una clave de API:

```bash
cp .env.example .env      # y colocar la clave real
```

Los baselines (OE2/R4) y el análisis intrínseco no requieren clave ni GPU —
corren en CPU en minutos. Retrotraducción sí requiere GPU para entrenar el
checkpoint NMT una vez (ver su propio README).

## Uso

- Reproducir la construcción del corpus (OE1) → [`1_objetivo1_corpus/README.md`](1_objetivo1_corpus/README.md)
- Correr los baselines de clasificación (OE2/R4) → [`2_baselines/README.md`](2_baselines/README.md)
- Reproducir el análisis intrínseco (OE2/R5) → [`3_baselines_y_aumento_datos/analisis_intrinseco/README.md`](3_baselines_y_aumento_datos/analisis_intrinseco/README.md)
- Generar datos aumentados (OE2/R6) → [`3_baselines_y_aumento_datos/tecnicas_aumento/README.md`](3_baselines_y_aumento_datos/tecnicas_aumento/README.md)
- Caracterizar los embeddings (OE3/R7-R8) → [`4_caracterizacion_embeddings/README.md`](4_caracterizacion_embeddings/README.md)
- Ver la síntesis comparativa final (OE4/R9) → [`5_sintesis_r9/README.md`](5_sintesis_r9/README.md)

---

## Materiales fuente

`1_objetivo1_corpus/datos/II_TEXTOS_SHIWILU.pdf`, usado en la Etapa 0 para la
extracción de dominios semánticos, **no se incluye** en este repositorio por
tratarse de material de terceros. El vocabulario extraído de ese material sí está
disponible en `1_objetivo1_corpus/intermedios/vocabulario_dominios.json`.

---

## Licencia

- **Código** (`shiwilu/`, pipeline, notebooks): MIT — ver [`LICENSE`](LICENSE)
- **Corpus y datos**: CC BY 4.0 — ver [`LICENSE-DATOS`](LICENSE-DATOS)

---

## Referencias

- Searle, J. R. (1975). A taxonomy of illocutionary acts. En K. Gunderson (Ed.), *Language, mind and knowledge* (pp. 344–369). University of Minnesota Press.
- FitzGerald, J., et al. (2022). MASSIVE: A 1M-example multilingual natural language understanding dataset with 51 typologically diverse languages. *arXiv*. https://arxiv.org/abs/2204.08582
- Bocklisch, T., Faulkner, J., Pawlowski, N., & Nichol, A. (2017). Rasa: Open source language understanding and dialogue management. *arXiv*. https://arxiv.org/abs/1712.05181
- Coucke, A., et al. (2018). Snips voice platform: An embedded spoken language understanding system for private-by-design voice interfaces. *arXiv*. https://arxiv.org/abs/1805.10190
- Valenzuela, P., & Gussenhoven, C. (2013). Shiwilu (Jebero). *Journal of the International Phonetic Association*, 43(1), 97–106.

---

## Cómo citar

```
Hurtado Claros, L. A. (2026). Corpus de intenciones para el shiwilu
(jebero, ISO 639-3: jeb) [Conjunto de datos]. Pontificia Universidad
Católica del Perú.
```

Ver también [`CITATION.cff`](CITATION.cff).
