# OE2 — Evaluación de técnicas de aumento de datos (R4-R6)

Responde a la pregunta central del objetivo: **¿qué técnicas de aumento de datos
mejoran la clasificación de intenciones en shiwilu**, con embeddings congelados
(LaBSE, mBERT, XLM-R) + Regresión Logística, en un escenario de bajos recursos
(700 oraciones, 7 categorías)?

| Resultado | Qué es | Dónde |
|---|---|---|
| R4 | Baselines de clasificación (sin aumento) | filas `sin_aumento` de [`evaluacion/`](evaluacion/) |
| R5 | Análisis intrínseco del corpus y preselección de técnicas | [`analisis_intrinseco/`](analisis_intrinseco/) |
| R6 | Evaluación comparativa de las técnicas de aumento | [`tecnicas_aumento/`](tecnicas_aumento/) (generación) + [`evaluacion/`](evaluacion/) (resultados) |

## Mapa de la carpeta

```
oe2_aumento_de_datos/
├── particiones/            Datos congelados: NO recalcular a mano
│   ├── folds_fijos.csv        fold (0-4) de cada oración, agrupado por texto normalizado
│   └── corpus_normalizado.csv las 700 oraciones tal como las ven los modelos (autogenerado)
├── analisis_intrinseco/    R5: notebooks y tablas (perfil lingüístico, marcadores, patrones)
├── tecnicas_aumento/       R6: Mixup, Retrotraducción, Generate-then-Refine y sus salidas
│   └── colab/                 cuadernos de Colab (checkpoint NMT y generación en línea, uno por técnica)
├── evaluacion/             R4+R6 (ver evaluacion/README.md)
│   ├── validacion_cruzada.py            PROTOCOLO A: 12 experimentos, C elegido en un dev interno de ~90 oraciones
│   ├── validacion_cruzada_cv_interna.py PROTOCOLO B: C elegido por CV interna (K=5); el que cerrará el OE2
│   ├── validacion_cruzada_en_linea.py   apoyo: genera el aumento dentro de cada fold y lo guarda en caché (lo importa el B)
│   ├── unir_predicciones.py, comparacion_pareada.py, curvas_roc.py
│   ├── diagnostico_c/                   sensibilidad del F1 al hiperparámetro C (scripts + resultados)
│   └── resultados/                      protocolo A (raíz) y protocolo B (cv_interna/)
└── historico/              Etapas cerradas, no se tocan (ver su README): split único, texto crudo,
                            generación en línea con dev (niveles 20-120) y zips de Colab
```

El núcleo de código compartido (carga del corpus, normalización, folds, embeddings,
Regresión Logística) está en [`../shiwilu/clasificacion.py`](../shiwilu/clasificacion.py).

## Protocolo vigente

- **Datos.** `corpus/corpus_shiwilu_final.csv`, columna `shiwilu`, **normalizada**
  (minúsculas, sin puntuación, sin tildes/ñ, apóstrofo de oclusiva glotal preservado):
  el corpus tiene atajos superficiales (los `¿?` delatan PRG; DES/PRG/REQUEST están
  100% en mayúsculas) que no son señal lingüística. Es la única condición vigente.
- **Validación cruzada de 5 folds**, agrupada por texto normalizado (una oración
  repetida con variantes de mayúsculas/puntuación cae siempre en el mismo fold) y
  estratificada por categoría. Los folds están congelados en
  [`particiones/folds_fijos.csv`](particiones/folds_fijos.csv): los datos sintéticos
  se generan a partir del train de cada fold, así que si el reparto cambiara, una
  oración de test podría haber inspirado su propio entrenamiento.
- **Protocolo por fold** (el test no se toca hasta el final): pool = los otros 4
  folds (~560 oraciones). Hay dos formas de elegir el hiperparámetro `C`:
  - **Protocolo A** (`validacion_cruzada.py`): ~1/6 del pool se aparta como *dev interno* (~90 oraciones)
    solo para elegir `C ∈ {0.01, 0.1, 1, 3, 10}`; con ese `C` se reentrena sobre todo el pool (+ su
    aumento) y se predice el fold. La elección es ruidosa.
  - **Protocolo B** (`validacion_cruzada_cv_interna.py`): el pool se parte en K=5 particiones;
    en cada una el aumento se genera **solo con su entrenamiento** y se evalúa en la parte retenida;
    se elige el `C ∈ {0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30}` con mejor F1 sobre las ~560 predicciones
    retenidas. Cada configuración (incluida "sin aumento") elige su propio `C` con la misma regla.
- **Aumento, siempre solo con el pool del fold**; se descartan las filas sintéticas
  idénticas a una oración del test (el NMT de F. Prado memorizó parte del corpus).
- **Métrica.** F1 macro sobre las 700 predicciones, con IC95% bootstrap (2000
  remuestreos) y diferencias pareadas contra `sin_aumento` sobre las mismas oraciones.

## Cómo correr

Desde la raíz del repositorio, en este orden (los pasos 1-2 ya están hechos y sus
salidas versionadas):

```bash
# 1. Datos sintéticos (R6): ver tecnicas_aumento/README.md (Claude para Generate-then-Refine, Colab/GPU para Retrotraducción)
# 2. Particiones y corpus normalizado: se crean solos la primera vez que se necesitan
# Protocolo A (12 experimentos, ~10 min en CPU)
python oe2_aumento_de_datos/evaluacion/validacion_cruzada.py
python oe2_aumento_de_datos/evaluacion/comparacion_pareada.py    # diferencias pareadas
python oe2_aumento_de_datos/evaluacion/curvas_roc.py             # curvas ROC (usa las probabilidades guardadas)
# Protocolo B (una técnica por corrida; resultados en evaluacion/resultados/cv_interna/)
python oe2_aumento_de_datos/evaluacion/validacion_cruzada_cv_interna.py --tecnica sin_aumento
python oe2_aumento_de_datos/evaluacion/validacion_cruzada_cv_interna.py --tecnica mixup
python oe2_aumento_de_datos/evaluacion/validacion_cruzada_cv_interna.py --tecnica retrotraduccion --checkpoint x   # usa la caché
# Diagnóstico opcional del hiperparámetro C (~10-15 min)
python oe2_aumento_de_datos/evaluacion/diagnostico_c/sensibilidad_c.py
```

## Resultados con el protocolo A (dev interno de ~90 oraciones)

F1 macro sobre las 700 predicciones (texto normalizado, validación cruzada de 5 folds);
[`evaluacion/resultados/validacion_cruzada_resumen_sin_puntuacion.csv`](evaluacion/resultados/validacion_cruzada_resumen_sin_puntuacion.csv):

| Modelo | Sin aumento | Mixup | Retrotraducción | Generate-then-Refine |
|---|---|---|---|---|
| mBERT | **0.6575** [0.622, 0.691] | 0.6346 | 0.6214 | 0.6491 |
| LaBSE | 0.5873 [0.549, 0.621] | 0.5927 | 0.5683 | 0.5952 |
| XLM-R | 0.5481 [0.510, 0.583] | 0.5580 | 0.5574 | 0.5798 |

Diferencia pareada de F1 frente a `sin_aumento`
([`comparacion_pareada_sin_puntuacion.csv`](evaluacion/resultados/comparacion_pareada_sin_puntuacion.csv));
en negrita, las que tienen IC95% que no incluye 0:

| Técnica | LaBSE | mBERT | XLM-R |
|---|---|---|---|
| Mixup | +0.005 | **-0.023** | +0.010 |
| Retrotraducción | -0.019 | **-0.036** | +0.009 |
| Generate-then-Refine | +0.008 | -0.008 | **+0.032** |

- **mBERT sin aumento es el mejor de los 12** y supera de forma distinguible a XLM-R
  (+0.109) y a LaBSE (+0.070); XLM-R y LaBSE no se distinguen entre sí.
- Ninguna técnica mejora a mBERT: Mixup y Retrotraducción lo empeoran de forma
  distinguible; Generate-then-Refine queda a la par.
- La única mejora distinguible de cualquier técnica es Generate-then-Refine en XLM-R
  (+0.032), insuficiente para alcanzar a mBERT.
- **Curvas ROC** ([`evaluacion/resultados/curvas_roc/`](evaluacion/resultados/curvas_roc/),
  una imagen por experimento): el AUC macro de las 12 combinaciones va de 0.848 a
  0.894 y las curvas de cada modelo casi se superponen entre técnicas; el aumento no
  mejora la capacidad de separar las categorías. mBERT sin aumento tiene el AUC más
  alto (0.894); Mixup lo reduce a 0.882.

## Resultados con el protocolo B (CV interna K=5): los 12 experimentos

F1 macro sobre las 700 predicciones, con `C` elegido por validación cruzada interna (5 particiones del pool, ~560 oraciones retenidas, grilla 0.01-30), corpus original (7 intenciones)
y texto normalizado sin signos. Resultados en [`evaluacion/resultados/cv_interna/`](evaluacion/resultados/cv_interna/). Generate-then-Refine (GtR) se generó con Claude dentro de cada
partición interna (120 por categoría; los niveles 80 y 40 usan los primeros lotes).

| Modelo | Sin aumento | Mixup (100%) | Retrotraducción (~83%) | **GtR 120** (~89%) | GtR 80 (~59%) | GtR 40 (~30%) |
|---|---|---|---|---|---|---|
| mBERT | **0.665** [0.629, 0.698] | 0.629 | 0.631 | 0.652 [0.616, 0.684] | 0.650 | 0.650 |
| LaBSE | 0.587 [0.547, 0.623] | 0.587 | 0.581 | 0.593 [0.558, 0.626] | 0.583 | 0.595 |
| XLM-R | 0.570 [0.534, 0.602] | 0.546 | 0.551 | 0.589 [0.550, 0.624] | 0.583 | 0.595 |

Diferencia pareada de F1 frente a `sin_aumento` (en negrita, IC95% que no incluye 0):

| Técnica | LaBSE | mBERT | XLM-R |
|---|---|---|---|
| Mixup | 0.000 | **-0.036** | **-0.024** |
| Retrotraducción | -0.006 | **-0.034** | -0.019 |
| GtR 120 | +0.006 | -0.013 | +0.018 |
| GtR 80 | -0.004 | -0.015 | +0.012 |
| GtR 40 | +0.008 | -0.015 | **+0.025** |

- **Ninguna técnica mejora al baseline de forma consistente.** La única diferencia positiva distinguible (GtR 40 en XLM-R, +0.025) es 1 de 9 comparaciones de GtR y no se repite en 80 ni en 120.
- **GtR es la técnica menos mala:** no empeora a ningún modelo de forma distinguible, mientras Mixup y Retrotraducción empeoran a mBERT (y Mixup a XLM-R). Su AUC macro es el más alto en los tres modelos
  (mBERT 0.907 contra 0.899 sin aumento; LaBSE 0.878 contra 0.868; XLM-R 0.868 contra 0.853), sin prueba de significancia.
- **Más volumen no ayuda:** de 40 a 120 por categoría el F1 de GtR queda plano (mBERT 0.650/0.650/0.652).
- **mBERT sin aumento es el mejor de los 18 resultados** y supera a LaBSE (+0.078) y XLM-R (+0.095) de forma distinguible. Ver [`evaluacion/diagnostico_baseline/`](evaluacion/diagnostico_baseline/) para el análisis de por qué.
- **Lo sintético de GtR queda desbalanceado**: por fold, aprobadas únicas DES 118, SAL 106, NEG 100, PRG 77, REQUEST 51, AFI 44, EMO 31 (frente a ~80 reales por categoría): los filtros de marcador dejan pasar casi
  todo DES y SAL y rechazan mucho EMO, AFI y REQUEST. Su calidad no fue verificada por un hablante.
- Respecto del protocolo A, la ventaja aparente de GtR en XLM-R (+0.037) se reduce a +0.018 y deja de ser distinguible: parte venía de un baseline con `C` mal elegido.

Versión anterior de estos resultados (9 de 12 experimentos, sin GtR): [`evaluacion/resultados/cv_interna/_previo_sin_gtr/`](evaluacion/resultados/cv_interna/_previo_sin_gtr/). El lote 0 del pool de GtR que había
antes de regenerarlo con el prompt actual está en `tecnicas_aumento/salidas/historico/lote0_pool_prompt_anterior/`.

## Verificaciones y limitaciones conocidas

Verificado al reorganizar el repositorio (2026-10):

- **Reproducibilidad.** Volver a correr `validacion_cruzada.py` desde cero reproduce
  las 8400 predicciones versionadas (diferencia numérica ~1e-21; etiquetas idénticas).
  Los folds no parten ningún grupo de oraciones duplicadas (0 de 646 grupos cruzan
  folds) y las 12 combinaciones se evalúan sobre las mismas 700 oraciones.
- **Tablas derivadas.** Resumen, comparaciones pareadas y curvas ROC se regeneran
  idénticas a partir de las predicciones.

Limitaciones (no invalidan el resultado, pero conviene declararlas):

- **Elección de `C` en un dev pequeño (protocolo A).** Cada fold elige `C` con ~90-104 oraciones,
  una elección ruidosa; el protocolo B la corrige. Ver [`evaluacion/diagnostico_c/`](evaluacion/diagnostico_c/)
  y su salida `resultados/sensibilidad_c.csv` (F1 con `C` fijo, sin elegirlo en ningún dev):
  - La grilla de la validación cruzada (hasta `C=10`) **no se queda corta**: el mejor `C`
    fijo es 0.1 en mBERT y 10-100 en LaBSE/XLM-R, y ampliar la grilla a 1000 no cambia el
    ranking de modelos (con el mejor `C` fijo: mBERT 0.658 > LaBSE 0.586 ≈ XLM-R 0.570).
  - En mBERT, Retrotraducción queda por debajo de `sin_aumento` con los 8 valores de `C`
    probados y Mixup con 5 de 8; Generate-then-Refine queda a la par o por encima con los 8
    (0.662 vs. 0.658 en `C=0.1`). Por eso "mBERT sin aumento es el mejor" es robusto frente
    a Retrotraducción y, en menor medida, a Mixup, pero **no frente a Generate-then-Refine**:
    con `C` fijo son indistinguibles.
- **Generate-then-Refine aporta pocos sintéticos y muy desbalanceados** (~80 por fold,
  frente a ~560 reales, ~15%): con el filtro de marcador original (cadenas literales como
  `-ker'/-e'r/-r'` que nunca aparecen en una oración) REQUEST recibía 1-2 oraciones por fold y
  DES y SAL ~20. Con el filtro corregido a los marcadores que las fuentes afirman explícitamente
  (`analisis_intrinseco/marcadores_fuentes.csv`) REQUEST sube a ~12 y NEG a ~19, pero AFI y EMO
  bajan (se dejan de aceptar `i'na`, `-sa'` y coincidencias por subcadena), y el F1 no cambia de
  forma distinguible (ver `tecnicas_aumento/README.md`). Mixup añade 100% del pool y Retrotraducción ~85% (antes de
  los filtros), así que las tres técnicas perturban el entrenamiento en grados muy distintos.
- **Patrones candidatos de R5.** Los patrones estadísticos que se le pasan a Claude en
  Generate-then-Refine se calcularon con el corpus completo (fuga débil y agregada;
  ver [`tecnicas_aumento/README.md`](tecnicas_aumento/README.md)). Favorecería a esa
  técnica, y aun así no supera a mBERT sin aumento.
- **Bootstrap por oración.** El IC95% remuestrea las 700 oraciones como independientes,
  pero 91 (13%) pertenecen a 37 grupos de oraciones repetidas: los intervalos son
  ligeramente optimistas.
- **Texto normalizado ambiguo.** Al quitar la puntuación, 3 grupos de texto idéntico
  quedan con intenciones distintas (p. ej. `tekinchi` AFI vs. `¿tekinchi?` PRG): error
  irreducible para un clasificador que solo ve el texto.
- **Los 12 experimentos usan un solo corpus de 700 oraciones**; las diferencias
  entre técnicas son del orden de los intervalos de confianza.
