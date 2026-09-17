# Análisis intrínseco del corpus (OE2, R5)

Análisis intrínseco del corpus shiwilu: perfil lingüístico, marcadores
morfosintácticos e insumos que justifican las técnicas de aumento de datos
aplicadas después en [`../tecnicas_aumento/`](../tecnicas_aumento/).

**Consume:** [`corpus/corpus_shiwilu_final.csv`](../../corpus/) — el producto de
OE1 ([`../../1_objetivo1_corpus/`](../../1_objetivo1_corpus/)).

> **Regla:** esta fase nunca escribe en `corpus/`. Todas sus salidas van a
> `resultados/`, que puede borrarse y regenerarse por completo.

## Contenido

```
notebooks/
  metricas_corpus.ipynb                 Métricas descriptivas del corpus
  analisis_patrones_por_intencion.ipynb Patrones morfosintácticos por intención
resultados/
  tablas/     Salidas en CSV de los notebooks
  figuras/    Gráficos exportados
```

## Ejecución

```bash
pip install -r requirements/fase2.txt   # desde la raíz del repositorio
jupyter lab 3_baselines_y_aumento_datos/analisis_intrinseco/notebooks/
```

No requiere clave de API: parte del corpus ya construido.

Los notebooks localizan la raíz del repositorio buscando `pyproject.toml` hacia
arriba, así que funcionan sin importar desde dónde se lance Jupyter.

## Dependencia con la Fase 1

`analisis_patrones_por_intencion.ipynb` importa del núcleo compartido:

```python
from shiwilu.anotacion import anotar_intencion
from shiwilu.taxonomia import DESC_INT, TIPO_SEARLE
```

`anotar_intencion()` es el sistema de reglas con el que se construyó el corpus,
y aquí se usa como **baseline** contra el que contrastar los clasificadores
basados en embeddings. Por eso el import es directo y sin respaldo: si falla,
detener el análisis es preferible a ejecutarlo contra una taxonomía distinta de
la que se usó para anotar el corpus.

## Tablas generadas

Los 11 archivos de `resultados/tablas/` cubren longitud, riqueza léxica,
n-gramas, secuencias iniciales y finales, marcadores documentados, uso del
apóstrofo (oclusiva glotal) y contraste con la literatura descriptiva de la
lengua.
