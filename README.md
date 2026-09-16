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

El repositorio está organizado en **dos fases**, que corresponden a los dos
trabajos de tesis, con el corpus como frontera explícita entre ambas.

```
├── shiwilu/                  ◆ Núcleo compartido — lo que cruza la frontera
│   ├── taxonomia.py            Intenciones, tipos de Searle, descripciones
│   ├── anotacion.py            Reglas de anotación (baseline de la Fase 2)
│   ├── dominios.py             Dominios semánticos y vocabulario semilla
│   ├── excel.py                Formato de los reportes Excel
│   └── rutas.py                Rutas ancladas a la raíz del repositorio
│
├── 1_construccion_corpus/    ◆ FASE 1 — construir el corpus
│   ├── datos/                  Materiales fuente
│   ├── pipeline/               Etapas 0 a 3
│   └── intermedios/            Productos intermedios y logs de la API
│
├── corpus/                   ★ FRONTERA — salida de la Fase 1, entrada de la Fase 2
│   └── corpus_shiwilu_final.csv
│
├── 2_analisis_corpus/        ◆ FASE 2 — analizar el corpus
│   ├── notebooks/
│   └── resultados/             tablas/ y figuras/
│
├── 3_baseline_clasificacion/ ◆ FASE 3 — baselines de clasificación (R4)
│   ├── baseline.py             LaBSE / mBERT / XLM-R congelados + Regresión Logística
│   └── correr_todos.py         corre los 3 y compara
│
├── 4_aumento_datos/          ◆ FASE 4 — técnicas de aumento de datos (R5/R6)
│   ├── mixup.py                Interpolación de embeddings (misma categoría)
│   ├── generate_then_refine.py Generación con LLM + filtros de calidad
│   └── retrotraduccion.py      Helsinki-NLP + NMT de F. Prado + filtros de calidad
│
├── docs/                     Metodología de construcción del corpus
└── tests/                    Pruebas de las reglas de anotación
```

### Quién escribe dónde

| Zona | Lee de | Escribe en |
|---|---|---|
| `1_construccion_corpus/` | `datos/`, `shiwilu/` | `intermedios/` y `corpus/` |
| `corpus/` | — | *solo la Etapa 3 de la Fase 1* |
| `2_analisis_corpus/` | `corpus/`, `shiwilu/` | `resultados/` únicamente |
| `3_baseline_clasificacion/` | `corpus/`, `shiwilu/` | `resultados/` únicamente |
| `4_aumento_datos/` | `corpus/`, `2_analisis_corpus/resultados/`, `shiwilu/` | `salidas/` únicamente |
| `shiwilu/` | — | nada (es solo código) |

Esa separación mantiene el corpus estable y citable, y permite borrar y
regenerar `2_analisis_corpus/resultados/` sin tocar nada más.

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

**Solo para reproducir la Fase 1** hace falta una clave de API:

```bash
cp .env.example .env      # y colocar la clave real
```

Las Fases 2 y 3 no requieren clave ni GPU: la Fase 3 corre en CPU en menos de
un minuto.

## Uso

- Reproducir la construcción del corpus → [`1_construccion_corpus/README.md`](1_construccion_corpus/README.md)
- Reproducir el análisis → [`2_analisis_corpus/README.md`](2_analisis_corpus/README.md)
- Correr los baselines de clasificación → [`3_baseline_clasificacion/README.md`](3_baseline_clasificacion/README.md)
- Generar datos aumentados → [`4_aumento_datos/README.md`](4_aumento_datos/README.md)

---

## Materiales fuente

`1_construccion_corpus/datos/II_TEXTOS_SHIWILU.pdf`, usado en la Etapa 0 para la
extracción de dominios semánticos, **no se incluye** en este repositorio por
tratarse de material de terceros. El vocabulario extraído de ese material sí está
disponible en `1_construccion_corpus/intermedios/vocabulario_dominios.json`.

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
