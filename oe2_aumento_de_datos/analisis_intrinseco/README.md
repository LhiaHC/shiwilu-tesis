# Análisis intrínseco del corpus (OE2, R5)

Análisis del corpus shiwilu **en función de los filtros de calidad** que aprueban o rechazan las oraciones sintéticas, y justificación de las técnicas de aumento que se
aplican después en [`../tecnicas_aumento/`](../tecnicas_aumento/).

**Consume:** [`corpus/corpus_shiwilu_final.csv`](../../corpus/) — el producto de OE1 ([`../../oe1_corpus/`](../../oe1_corpus/)), los 5 folds congelados de
[`../particiones/folds_fijos.csv`](../particiones/folds_fijos.csv), la ficha `marcadores_fuentes.csv` y las cachés de Generate-then-Refine.

> **Regla:** esta fase nunca escribe en `corpus/`. Todas sus salidas van a `resultados/`, que puede borrarse y regenerarse por completo
> (salvo `resultados/tablas/`, ver abajo).

## Cuaderno principal: `notebooks/analisis_corpus_y_filtros.ipynb`

Cada sección analiza lo que usa un filtro, con **el mismo código** (`tecnicas_aumento/generate_then_refine.py`):

| Filtro | Qué exige | De dónde sale | Secciones |
|---|---|---|---|
| **Marcador** | Al menos un marcador de su intención (si la intención tiene) | `marcadores_fuentes.csv`, con **3 fuentes**: Valenzuela & Gussenhoven (2013, *JIPA*), Valenzuela (2012, *Voces shiwilu*) y Valenzuela, Vásquez & Chota (2024, *Enciclopedia*) | 2, 3, 3b, 4 |
| **Idioma** | Que el «shiwilu» no sea idéntico al español | Una regla simple | 5 |
| **Semántico** | Similitud LaBSE con el centro de su intención entre 0.20 y 0.995 | Las oraciones reales de entrenamiento del fold | 6 |

Además: perfil del corpus (sección 1), qué hicieron los filtros con lo generado (7), atajos superficiales (8), pistas estadísticas del prompt (9) y la preselección de técnicas (10).

**No hay partición entrenamiento/desarrollo/prueba en este análisis.** La versión anterior separaba un 70 % (con 15 % de desarrollo y 15 % de prueba) que no se usaba en ningún experimento (la evaluación
usa los 5 folds congelados). Los marcadores salen de la literatura, no del corpus, así que no hay nada que «filtrar» hacia una prueba; lo que sí depende del entrenamiento (el filtro semántico) se calibra con los folds reales.
Detalle en la sección 1 del cuaderno. No requiere API ni GPU; calcula embeddings de LaBSE para las 700 oraciones.

```
notebooks/
  analisis_corpus_y_filtros.ipynb   VIGENTE: análisis en función de los filtros y de las 3 fuentes
  metricas_corpus.ipynb             Métricas descriptivas del corpus (sin partición)
  historico/
    analisis_patrones_por_intencion_split70.ipynb   versión anterior (partición 70/15/15); no se usa más
resultados/
  analisis_filtros/                 salidas del cuaderno vigente (tablas CSV y figuras/)
  tablas/                           salidas de la versión anterior (calculadas con el 70 %)
```

`marcadores_fuentes.csv` (en esta carpeta): marcadores que las fuentes afirman explícitamente, con página y nivel de evidencia (A usada, B y C no usadas). Es una tabla curada a mano, no una salida
regenerable; la usa Generate-then-Refine para el filtro de marcador y el prompt.

### Por qué `resultados/tablas/` se conserva

El prompt de Generate-then-Refine incluye, como **pistas no validadas**, las 8 palabras características y las 8 terminaciones más asociadas a cada intención. Esas listas se leen de
`tablas/analisis_palabras_caracteristicas.csv` y `tablas/analisis_secuencias_finales.csv` (rutas de `shiwilu/rutas.py`), que se calcularon con la partición 70/15/15 anterior. Se conservan sin cambios porque
son la entrada con la que se generó el texto de las cachés. La sección 9 del cuaderno las recalcula con las 700 oraciones y muestra cuánto cambian (la mayoría coincide). Es una limitación declarada:
estrictamente, esas pistas se habrían calculado con el entrenamiento de cada fold.

## Ejecución

```bash
pip install -r requirements/fase2.txt -r requirements/fase3.txt -r requirements/fase4.txt   # desde la raíz del repositorio (anthropic y LaBSE se importan, no se llama a la API)
python -m nbconvert --to notebook --execute --inplace oe2_aumento_de_datos/analisis_intrinseco/notebooks/analisis_corpus_y_filtros.ipynb
# o: jupyter lab oe2_aumento_de_datos/analisis_intrinseco/notebooks/
```

Los notebooks localizan la raíz del repositorio buscando `pyproject.toml` hacia arriba.

## Hallazgos principales (resumen)

- **Solo 5 de las 7 intenciones tienen marcadores** utilizables (NEG, PRG, REQUEST, EMO, AFI). SAL y DES no tienen: el filtro de marcador aprueba todo lo generado para ellas.
- **Marcadores presentes y concentrados:** `i'n` (62 % de las NEG), `-inpu'` (12 %), `a'cha` (10 % de las PRG), `-(k)er'` (30 % de las REQUEST), `ajá` (28 % de las AFI). **Ausentes:** `chi`, `ten` (EMO), `ahã`, `untana` (AFI).
- **El filtro exige más de lo que cumple el corpus real:** solo el 74 % de las NEG, 30 % de las REQUEST, 28 % de las AFI, 11 % de las PRG y **0 % de las EMO** reales llevarían un marcador. En lo generado se aprueban SAL y DES al 100 %,
  NEG 85 %, PRG 69 %, AFI 57 %, REQUEST 47 % y EMO 28 %: eso explica el desbalance del texto sintético aprobado.
- **El filtro de idioma y el semántico casi no rechazan nada** (0 % de lo generado; las 700 reales tienen similitud 0.32-0.93 con su centro, muy por encima del umbral de 0.20).
- **El patrón `a'cha` no captura la variante fusionada `…'cha`** (19 oraciones de PRG con la forma fusionada frente a 10 con `a'cha`).
- `¿?` cubre el 100 % de PRG (atajo ALTO). El corpus actual no tiene mayúsculas.
- **Preselección:** Retrotraducción, Generate-then-Refine y Mixup (XL-LoRA se descartó por no ser una técnica de aumento aplicable a este diseño).

## Dependencia con OE1

El cuaderno histórico importaba `shiwilu.anotacion.anotar_intencion` (el sistema de reglas con el que se construyó el corpus) como baseline; el cuaderno vigente no lo necesita.
