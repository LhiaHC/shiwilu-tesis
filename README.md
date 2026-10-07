# Corpus de intenciones para el shiwilu

Corpus digital de la lengua **shiwilu** (jebero, ISO 639-3: `jeb`) anotado con
categorías de intención comunicativa, construido como parte del proyecto de
tesis *Evaluación de técnicas de aumento de datos y caracterización de embeddings en
la clasificación de intenciones para la lengua shiwilu*.

El shiwilu es una lengua amazónica peruana hablada en el distrito de Jeberos
(Loreto), en situación de peligro crítico de extinción. Este es el primer corpus
digital anotado para esta lengua.

- **Autora:** Lhía Antonella Hurtado Claros
- **Asesor:** Erasmo Gómez Montoya
- **Institución:** Pontificia Universidad Católica del Perú — Facultad de Ciencias e Ingeniería

---

## Organización del repositorio

El repositorio está organizado por **objetivo específico (OE) de la tesis**: el
número de cada carpeta `oeN_*` es el del objetivo. El corpus (`corpus/`) es la
frontera explícita entre la construcción del corpus (OE1) y todo lo que se
construye sobre él (OE2-OE4). Cada carpeta tiene su propio `README.md`.

```
├── corpus/                          ★ FRONTERA — salida de OE1, entrada del resto
│   └── corpus_shiwilu_final.csv
│
├── oe1_corpus/                      ◆ OE1 (R1-R3) — construir el corpus
│   ├── datos/                         Materiales fuente
│   ├── pipeline/                      Etapas 0 a 3
│   └── intermedios/                   Productos intermedios y logs de la API
│
├── oe2_aumento_de_datos/            ◆ OE2 (R4-R6) — baselines, análisis y aumento de datos
│   ├── particiones/                   Folds congelados + corpus normalizado
│   ├── analisis_intrinseco/           R5: notebooks y tablas del análisis del corpus; auditoría de la categoría DES
│   ├── tecnicas_aumento/              R6: Mixup, Retrotraducción, Generate-then-Refine, sus salidas y cachés (permiten reproducir sin API)
│   ├── evaluacion/                    R4+R6: validación cruzada (protocolo A y B), pareadas, ROC, regímenes de pocos datos,
│   │                                  variantes de selección, diagnósticos de causa y sensibilidad al hiperparámetro C
│   ├── analisis_resultados/           Notebook que analiza los resultados y regenera las 10 figuras (figuras/)
│   ├── reproducir_oe2.py              Plan y ejecución ordenada de TODOS los experimentos del OE2
│   ├── verificar_resultados.py        Compara las salidas con valores_de_referencia.csv (258 cifras, incluye R5)
│   └── historico/                     Etapas cerradas: split único, texto crudo, generación en línea con dev, zips de Colab
│
├── oe3_caracterizacion_embeddings/  ◆ OE3 (R7-R8) — 4 estrategias de pooling x 3 modelos,
│                                      métricas intrínsecas y proyecciones t-SNE/UMAP
│
├── oe4_sintesis/                    ◆ OE4 (R9) — cruza lo extrínseco (OE2) con lo intrínseco (OE3)
│
├── shiwilu/                         Núcleo compartido (código)
│   ├── taxonomia.py, anotacion.py, dominios.py, excel.py   OE1
│   ├── clasificacion.py               Corpus normalizado, folds, embeddings y Regresión Logística (OE2-OE3)
│   └── rutas.py                       Todas las rutas del proyecto, ancladas a la raíz
│
├── docs/                            Metodología de construcción del corpus
├── requirements/                    Dependencias por fase y versiones usadas (versiones_usadas.txt)
├── REPRODUCIBILIDAD.md              ★ Cómo repetir todos los experimentos y verificar los resultados
└── tests/                           Pruebas de las reglas de anotación
```

### Quién escribe dónde

| Zona | Lee de | Escribe en |
|---|---|---|
| `oe1_corpus/` | `datos/`, `shiwilu/` | `intermedios/` y `corpus/` |
| `corpus/` | — | *solo la Etapa 3 de OE1* |
| `oe2_aumento_de_datos/analisis_intrinseco/` | `corpus/`, `shiwilu/` | `resultados/` únicamente |
| `oe2_aumento_de_datos/tecnicas_aumento/` | `corpus/`, `analisis_intrinseco/resultados/`, `particiones/`, `shiwilu/` | `salidas/` únicamente |
| `oe2_aumento_de_datos/evaluacion/` | `corpus/`, `particiones/`, `tecnicas_aumento/salidas/`, `shiwilu/` | `resultados/` únicamente |
| `oe3_caracterizacion_embeddings/` | `corpus/`, `tecnicas_aumento/salidas/`, `shiwilu/` | `resultados/` únicamente |
| `oe4_sintesis/` | `evaluacion/resultados/`, `oe3_.../resultados/` | `resultados/` únicamente (no entrena ni mide nada nuevo) |
| `shiwilu/` | — | nada (es solo código) |

Esa separación mantiene el corpus estable y citable, y permite borrar y
regenerar cualquier carpeta `resultados/` sin tocar nada más. **Nunca** se
recalculan a mano los archivos congelados de `oe2_aumento_de_datos/particiones/`:
hacerlo invalida todos los datos sintéticos y resultados generados a partir de ellos.

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

Los experimentos del OE2 (baselines, técnicas de aumento, análisis de causas) **no requieren clave ni GPU**: lo generado con Claude y con el traductor de
F. Prado está guardado en cachés dentro del repositorio (ver [`REPRODUCIBILIDAD.md`](REPRODUCIBILIDAD.md)). Solo volver a *generar* texto sintético desde cero
requiere la clave de API (Generate-then-Refine) o una GPU (checkpoint NMT de Retrotraducción).

## Uso

- **Reproducir los experimentos del OE2 y comprobar sus resultados → [`REPRODUCIBILIDAD.md`](REPRODUCIBILIDAD.md)**
  (comandos, entradas, salidas, valores esperados y conclusiones con su evidencia; no requiere API ni GPU)
- Reproducir la construcción del corpus (OE1) → [`oe1_corpus/README.md`](oe1_corpus/README.md)
- Ver el protocolo, los resultados y cómo correr OE2 (baselines, aumento de datos, evaluación) → [`oe2_aumento_de_datos/README.md`](oe2_aumento_de_datos/README.md)
  - Análisis intrínseco (R5) → [`analisis_intrinseco/README.md`](oe2_aumento_de_datos/analisis_intrinseco/README.md)
  - Generar datos aumentados (R6) → [`tecnicas_aumento/README.md`](oe2_aumento_de_datos/tecnicas_aumento/README.md)
- Caracterizar los embeddings (OE3/R7-R8) → [`oe3_caracterizacion_embeddings/README.md`](oe3_caracterizacion_embeddings/README.md)
- Ver la síntesis comparativa final (OE4/R9) → [`oe4_sintesis/README.md`](oe4_sintesis/README.md)

---

## Materiales fuente

`oe1_corpus/datos/II_TEXTOS_SHIWILU.pdf`, usado en la Etapa 0 para la
extracción de dominios semánticos, **no se incluye** en este repositorio por
tratarse de material de terceros. El vocabulario extraído de ese material sí está
disponible en `oe1_corpus/intermedios/vocabulario_dominios.json`.

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
