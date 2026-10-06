# Evaluación (OE2: R4 y R6)

Valida de forma cruzada (5 folds) los 12 experimentos: 3 modelos de embeddings (LaBSE, mBERT, XLM-R) ×
4 configuraciones (sin aumento, Mixup, Retrotraducción, Generate-then-Refine). El protocolo completo y los
resultados están en [`../README.md`](../README.md); aquí se explica qué hace cada archivo.

## Qué es vigente y qué es apoyo

```
evaluacion/
├── validacion_cruzada.py              PROTOCOLO A (12 experimentos). C elegido en un dev de ~90 oraciones
├── validacion_cruzada_cv_interna.py   PROTOCOLO B. C elegido por CV interna (K=5). Una técnica por corrida
├── validacion_cruzada_en_linea.py     Apoyo: genera el aumento dentro de cada fold y lo guarda en caché.
│                                      Lo importa el protocolo B y lo usan los cuadernos de Colab
├── unir_predicciones.py               Junta las predicciones de varias corridas del protocolo B en un CSV
├── comparacion_pareada.py             Diferencias de F1 con IC95% (técnica vs. sin aumento, bootstrap pareado)
├── curvas_roc.py                      Curvas ROC One-vs-Rest (usa las columnas prob_<categoria>)
├── diagnostico_c/                     Sensibilidad del F1 al hiperparámetro C (no cambia ningún resultado)
│   ├── sensibilidad_c.py                 F1 con C fijo para la evaluación vigente (protocolo A)
│   ├── sensibilidad_c_en_linea.py        lo mismo para GtR (20-120) y Retrotraducción desde la caché
│   ├── comparacion_pareada_c_fijo.py     diferencias pareadas con el MISMO C fijo en ambas configuraciones
│   └── resultados/
├── variantes_gtr.py                   Variantes de selección del sintético de GtR (balanceo, filtros de similitud/centroides, sin DES); no usa la API
├── notebooks/variantes_gtr.ipynb      Notebook que lo ejecuta igual en local y en Colab (resultados en resultados/variantes_gtr/)
├── diagnostico_baseline/              Por qué mBERT sin aumento es el mejor: fragmentación del texto y clasificador de n-gramas de caracteres
└── resultados/
    ├── (raíz)                         Protocolo A: resumen, predicciones, comparación pareada y curvas ROC
    └── cv_interna/                    Protocolo B: lo mismo, por técnica y unido, más el F1 de la CV por C
```

Las corridas anteriores de generación en línea con dev (niveles 20, 40, 80 y 120 de Generate-then-Refine,
Retrotraducción ×1 de Colab) están en [`../historico/en_linea_protocolo_dev/`](../historico/en_linea_protocolo_dev/).

## Los dos protocolos

| | Protocolo A | Protocolo B |
|---|---|---|
| Elección de `C` | un dev de ~90 oraciones (1/6 del pool) | CV interna: 5 particiones del pool, ~560 oraciones retenidas |
| Grilla de `C` | 0.01, 0.1, 1, 3, 10 | 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30 |
| Aumento al elegir `C` | generado una vez con todo el pool | generado dentro de cada partición, solo con su entrenamiento |
| Estado | 12 experimentos completos | 12 experimentos completos (GtR con 120, 80 y 40 por categoría) |

## Variante con signos de interrogación

`validacion_cruzada_cv_interna.py --condicion con_interrogacion` repite el protocolo B conservando `¿` y `?` (como tokens aparte; el resto de la
puntuación y las tildes se quitan igual). Escribe en `resultados/cv_interna_con_interrogacion/` y **no** reemplaza los resultados vigentes. Con esos signos el F1
macro sube ~0.07-0.10, casi todo por PRG (F1 ≈ 1.00, porque el 100% de PRG lleva `¿?`) y AFI (se deja de confundir con PRG). Está calculada para sin aumento, Mixup y
Retrotraducción; Generate-then-Refine no necesita regenerarse (sus oraciones de PRG ya traen `¿?`), solo reevaluarse con esta condición.

## Cómo correr el protocolo B completo

Desde la raíz del repositorio:

```bash
E=oe2_aumento_de_datos/evaluacion
python $E/validacion_cruzada_cv_interna.py --tecnica sin_aumento
python $E/validacion_cruzada_cv_interna.py --tecnica mixup
python $E/validacion_cruzada_cv_interna.py --tecnica retrotraduccion --checkpoint x     # usa la caché; el checkpoint no se necesita
python $E/validacion_cruzada_cv_interna.py --tecnica generate_then_refine --cantidad 120 --solo-generar   # genera con Claude (ANTHROPIC_API_KEY); lo hizo el cuaderno de Colab
python $E/validacion_cruzada_cv_interna.py --tecnica generate_then_refine --cantidad 120 --etiqueta c120   # evalua desde la cache

python $E/unir_predicciones.py --etiqueta cv_interna --entradas \
  validacion_cruzada_predicciones_sin_puntuacion_cv_interna_sin_aumento.csv \
  validacion_cruzada_predicciones_sin_puntuacion_cv_interna_mixup.csv \
  validacion_cruzada_predicciones_sin_puntuacion_cv_interna_retrotraduccion.csv \
  validacion_cruzada_predicciones_sin_puntuacion_cv_interna_generate_then_refine.csv

R=$E/resultados/cv_interna
python $E/comparacion_pareada.py --entrada $R/validacion_cruzada_predicciones_sin_puntuacion_cv_interna.csv --etiqueta cv_interna --carpeta $R
python $E/curvas_roc.py --entrada $R/validacion_cruzada_predicciones_sin_puntuacion_cv_interna.csv --etiqueta cv_interna --carpeta $R
```

`comparacion_pareada.py` y `curvas_roc.py` toleran que falte una técnica (hoy, Generate-then-Refine) y trabajan
con las que haya.

## Cachés que usa el protocolo B

`--cache` (por defecto, [`../tecnicas_aumento/salidas/en_linea_marcadores_fuentes/`](../tecnicas_aumento/salidas/en_linea_marcadores_fuentes/)):

- `generate_then_refine/fold<N>_pool[_l<j>].csv`: lo generado con todo el pool de cada fold (lotes 0-5), ya refiltrado con
  los marcadores de las fuentes. **Faltan** los `fold<N>_in<k>[_l<j>].csv` (uno por partición interna) que se generan en Colab.
- `retrotraduccion/fold<N>_pool.csv`: catálogo de paráfrasis y traducciones; cada partición interna usa las filas cuyo origen
  está en su entrenamiento, así que no hay que regenerar nada.

Los archivos originales (con el filtro de marcador anterior en el lote 0) siguen en `../tecnicas_aumento/salidas/en_linea/`; no se tocan.
