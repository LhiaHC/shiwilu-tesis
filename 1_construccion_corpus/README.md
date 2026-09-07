# Fase 1 — Construcción del corpus

Corresponde al **primer trabajo de tesis**: la construcción del corpus digital
anotado de la lengua shiwilu.

**Produce:** [`corpus/corpus_shiwilu_final.csv`](../corpus/) — 700 pares
bilingües español–shiwilu etiquetados con intención comunicativa.

## Contenido

```
datos/          Materiales fuente
  flashcards2.csv           Flashcards bilingües español–shiwilu
  II_TEXTOS_SHIWILU.pdf     Textos narrativos (no versionado, ver datos/README.md)
pipeline/       Las cuatro etapas del pipeline
intermedios/    Productos intermedios y logs de las llamadas a la API
```

## Pipeline

| Etapa | Script | Descripción | API |
|---|---|---|---|
| 0 | `0_extraer_dominios.py` | Extrae vocabulario por dominio semántico desde los textos shiwilu | Sí |
| 1 | `1_generar_pares_anotados.py` | Anota los pares de las flashcards mediante reglas + revisión manual | No |
| 2 | `2_generar_oraciones.py` | Genera oraciones en español para las categorías con déficit | Sí |
| 3 | `3_consolidar_corpus.py` | Une ambas fuentes en el CSV final | No |

Entre las etapas 2 y 3 hay un **paso manual**: las oraciones generadas se
exportan como `intermedios/2_oraciones_para_traducir.xlsx`, un hablante nativo
especialista completa la columna en shiwilu, y el archivo resultante se guarda
como `intermedios/2_oraciones_traducidas.xlsx`, que es lo que consume la Etapa 3.

## Ejecución

```bash
pip install -r requirements/fase1.txt      # desde la raíz del repositorio
cp .env.example .env                        # y colocar la clave real

python 1_construccion_corpus/pipeline/0_extraer_dominios.py
python 1_construccion_corpus/pipeline/1_generar_pares_anotados.py
python 1_construccion_corpus/pipeline/2_generar_oraciones.py
# ── paso manual de traducción ──
python 1_construccion_corpus/pipeline/3_consolidar_corpus.py
```

Las rutas se resuelven desde la raíz del repositorio, así que los scripts
funcionan desde cualquier directorio de trabajo.

Para regenerar solo el CSV final a partir de los intermedios ya existentes basta
con la Etapa 3, que no requiere clave de API.

## Nota sobre la Etapa 0

La Etapa 0 escribe su resultado en `intermedios/vocabulario_dominios.json` y
**no modifica el código**. Ampliar las listas de `semillas` de
`shiwilu/dominios.py` con ese vocabulario es un paso manual y deliberado: así el
código versionado no cambia como efecto secundario de una ejecución, y el
comportamiento del pipeline no depende de si alguien corrió o no la Etapa 0.

## Metodología

Detalle completo de las técnicas de prompting, la taxonomía y los criterios de
anotación en [`docs/metodologia_construccion_corpus.md`](../docs/metodologia_construccion_corpus.md).
