# Reproducibilidad del OE2 (y de los análisis de OE3 que sostienen sus conclusiones)

Esta guía permite que otra persona **repita todos los experimentos** con los que se obtuvieron las conclusiones del OE2 y compruebe que
obtiene los mismos números. Nada de lo hecho queda fuera: cada experimento, diagnóstico y escenario tiene aquí su comando, sus
entradas, sus salidas y el valor que debe dar.

- [1. Ruta rápida](#1-ruta-rápida)
- [2. Requisitos](#2-requisitos)
- [3. Qué trae el repositorio para no depender de la API ni de GPU](#3-qué-trae-el-repositorio-para-no-depender-de-la-api-ni-de-gpu)
- [4. Los experimentos, uno por uno](#4-los-experimentos-uno-por-uno)
- [5. Conclusiones y la evidencia que las sostiene](#5-conclusiones-y-la-evidencia-que-las-sostiene)
- [6. De dónde sale cada carpeta de resultados](#6-de-dónde-sale-cada-carpeta-de-resultados)
- [7. Qué esperar al reproducir](#7-qué-esperar-al-reproducir)
- [8. Regenerar desde cero lo que usa la API o GPU (opcional)](#8-regenerar-desde-cero-lo-que-usa-la-api-o-gpu-opcional)
- [9. Material histórico (no se usa para reportar)](#9-material-histórico-no-se-usa-para-reportar)
- [10. Pendientes y limitaciones](#10-pendientes-y-limitaciones)

---

## 1. Ruta rápida

Desde la raíz del repositorio, con el entorno instalado (sección 2):

```bash
# 1. Ver el plan completo (no ejecuta nada): 51 pasos agrupados, con su comando exacto
python oe2_aumento_de_datos/reproducir_oe2.py

# 2. Comprobar que los resultados que trae el repositorio son los de las conclusiones (segundos, sin modelos)
python oe2_aumento_de_datos/verificar_resultados.py          # cifras de referencia (OE2 y R5)

# 3. Regenerar las 10 figuras con el notebook (≈1 min)
python -m nbconvert --to notebook --execute --inplace oe2_aumento_de_datos/analisis_resultados/analisis_resultados_oe2.ipynb

# 4. Repetir los experimentos (sobrescribe las salidas; luego vuelva a correr el paso 2)
python oe2_aumento_de_datos/reproducir_oe2.py --grupos base --ejecutar      # los 12 experimentos del protocolo B
python oe2_aumento_de_datos/reproducir_oe2.py --grupos diag regimen variantes des --ejecutar
python oe2_aumento_de_datos/reproducir_oe2.py --grupos verificar --ejecutar
```

`reproducir_oe2.py` ejecuta, en orden, los mismos comandos que se listan en la sección 4. Si se prefiere, se pueden copiar y correr a mano.
**Ningún paso llama a la API de Claude**: lo generado por Claude está guardado en cachés dentro del repositorio (sección 3).

---

## 2. Requisitos

| | |
|---|---|
| Sistema | Cualquiera con Python. Los resultados vigentes se obtuvieron en Windows 11, CPU, sin GPU |
| Python | 3.10 o superior (se usó 3.13) |
| Memoria | 8 GB de RAM alcanzan; los modelos se descargan de Hugging Face la primera vez (mBERT, XLM-R y LaBSE, ≈ 3-4 GB en total) |
| API / GPU | **No** se necesitan para reproducir (sección 3). Solo para regenerar desde cero (sección 8) |

Instalación:

```bash
pip install -e .
pip install -r requirements.txt            # entorno completo
# o solo lo necesario para el OE2/OE3:
pip install -r requirements/fase2.txt -r requirements/fase3.txt -r requirements/fase5.txt
pip install nbconvert ipykernel            # para ejecutar los notebooks por línea de comandos
```

Las **versiones exactas** con las que se obtuvieron los resultados están en [`requirements/versiones_usadas.txt`](requirements/versiones_usadas.txt)
(numpy 2.3.4, scikit-learn 1.7.2, torch 2.9.1, transformers 4.57.1, sentence-transformers 5.1.2, entre otras). Con otras versiones el resultado
puede variar en las milésimas (sección 7).

Los scripts localizan la raíz del repositorio por su cuenta; se ejecutan desde la raíz. En Windows, si aparecen errores de codificación,
anteponer `PYTHONIOENCODING=utf8` (en PowerShell: `$env:PYTHONIOENCODING="utf8"`).

---

## 3. Qué trae el repositorio para no depender de la API ni de GPU

Todo lo que se genera con Claude o con el traductor de F. Prado está **guardado** y es la entrada de los experimentos. Así, repetirlos solo requiere CPU.

| Dato | Ruta | Para qué sirve |
|---|---|---|
| Corpus | `corpus/corpus_shiwilu_final.csv` | 700 oraciones, 7 intenciones (100 c/u). No se modifica |
| Folds congelados | `oe2_aumento_de_datos/particiones/folds_fijos.csv` | Asignación de cada oración a uno de los 5 folds. **No recalcular a mano** (invalidaría todo lo sintético) |
| Texto normalizado | `oe2_aumento_de_datos/particiones/corpus_normalizado.csv` | Lo que ven los modelos (minúsculas, sin puntuación, sin tildes). Se regenera solo |
| GtR del protocolo B | `oe2_aumento_de_datos/tecnicas_aumento/salidas/en_linea_marcadores_fuentes/generate_then_refine/` | 210 archivos: `fold<N>_pool[_l<j>].csv` (generado con todo el pool del fold), `fold<N>_in<k>[_l<j>].csv` (con cada partición interna) y `fold<N>_core*.csv` (etapa del protocolo A). Ya refiltrado con los marcadores de las fuentes |
| Retrotraducción | `.../en_linea_marcadores_fuentes/retrotraduccion/fold<N>_pool.csv` y `salidas/retrotraduccion_pool.csv` | Catálogo de paráfrasis y traducciones al shiwilu; cada partición usa las filas cuyo origen está en su entrenamiento |
| GtR de los regímenes | `.../salidas/regimenes/generate_then_refine/` | 35 archivos: GtR generado solo con las oraciones de cada régimen (10, 25 y 50 por categoría); el de 80 usa la caché del pool |
| Caché de la versión anterior | `.../salidas/en_linea/` y `.../salidas/historico/lote0_pool_prompt_anterior/` | Respaldo del filtro de marcador anterior; no se usa en los resultados vigentes |
| Marcadores lingüísticos | `oe2_aumento_de_datos/analisis_intrinseco/marcadores_fuentes.csv` | Condicionan el prompt y el filtro de GtR (resultado R5) |

---

## 4. Los experimentos, uno por uno

Los 10 experimentos coinciden con la tabla de la sección 5.1 del capítulo de la tesis. Cada fila indica **qué comando lo repite**, **qué produce** y **qué valor debe dar**.
`E` = `oe2_aumento_de_datos/evaluacion`, `CV` = `E/validacion_cruzada_cv_interna.py`.

### Marco común (se aplica a todo)

- 5 folds congelados; el test de cada fold no se toca; el aumento se genera solo con el entrenamiento del fold; se descartan las sintéticas idénticas a una oración de prueba.
- Texto normalizado; embeddings congelados de LaBSE, mBERT y XLM-R + Regresión Logística; métrica F1 macro agrupado (700 predicciones) con IC 95 % bootstrap y diferencias pareadas.
- **Protocolo B (vigente):** C elegido por CV interna (K=5) con grilla {0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30}; el aumento de cada partición interna se genera solo con su entrenamiento.
  **Protocolo A (referencia):** C elegido en un dev de ~90 oraciones, grilla {0.01, 0.1, 1, 3, 10}.
- El núcleo está en `shiwilu/clasificacion.py` (corpus, normalización, folds, embeddings, clasificador).

### Experimento 1. Baselines (R4)

| | |
|---|---|
| Comando | `python $CV --tecnica sin_aumento` (parte de los pasos 1 de `reproducir_oe2.py`) |
| Salida | `E/resultados/cv_interna/validacion_cruzada_resumen_sin_puntuacion_cv_interna_sin_aumento.csv` |
| Valor esperado | F1 macro: mBERT **0.665**, LaBSE **0.587**, XLM-R **0.570** |

### Experimento 2. Referencias simples

| | |
|---|---|
| Comandos | `python E/diagnostico_baseline/referencias_simples.py` (mayoría y vecino léxico) · `python E/diagnostico_baseline/ngramas_caracteres.py` (n-gramas de caracteres y fragmentación) |
| Salidas | `E/diagnostico_baseline/resultados/referencias_simples.csv`, `ngramas_caracteres.csv`, `fragmentacion.csv` |
| Valores esperados | mayoría 0.088 · vecino léxico (palabras compartidas) 0.503 · n-gramas de caracteres **0.751** · subpalabras por palabra: mBERT 3.69, XLM-R 3.52, LaBSE 3.29 |

### Experimento 3. Decisiones de evaluación (partición, texto, elección de C)

| Decisión | Cómo se repite | Dónde está |
|---|---|---|
| Partición única frente a validación cruzada; 30 particiones aleatorias | `python oe2_aumento_de_datos/historico/atajos_texto_crudo/auditoria_optimismo.py` (necesita las predicciones con texto crudo de esa carpeta) | `oe2_aumento_de_datos/historico/` (ver su README) |
| Texto original o normalizado | `python E/validacion_cruzada.py --condicion original` (también `minusculas`, `sin_puntuacion_mayusculas`) | `historico/atajos_texto_crudo/` |
| Protocolo A (C en ~90 oraciones) | `python E/validacion_cruzada.py` + `comparacion_pareada.py` + `curvas_roc.py` | `E/resultados/` (raíz). Baselines A: mBERT 0.657, LaBSE 0.587, XLM-R 0.548 |
| Protocolo B (CV interna) | Experimentos 1 y 5 | `E/resultados/cv_interna/` |
| Sensibilidad al valor de C | `python E/diagnostico_c/sensibilidad_c.py`, `sensibilidad_c_en_linea.py`, `comparacion_pareada_c_fijo.py` | `E/diagnostico_c/resultados/` |

### Experimento 4. Análisis intrínseco del corpus y preselección de técnicas (R5)

| | |
|---|---|
| Notebooks | `oe2_aumento_de_datos/analisis_intrinseco/notebooks/analisis_corpus_y_filtros.ipynb` (vigente) y `metricas_corpus.ipynb`. Comando: `python -m nbconvert --to notebook --execute --inplace <notebook>` o `python oe2_aumento_de_datos/reproducir_oe2.py --grupos r5 --ejecutar` |
| Salidas | `oe2_aumento_de_datos/analisis_intrinseco/resultados/analisis_filtros/` (tablas y figuras) y la ficha `analisis_intrinseco/marcadores_fuentes.csv` (curada a mano, 3 fuentes) |
| Contenido | Perfil del corpus (700 oraciones, **sin partición**); las 3 fuentes y la ficha de marcadores; presencia de los marcadores en el corpus; qué porcentaje de oraciones reales cumpliría el filtro de marcador; calibración de los filtros de idioma y semántico con los folds reales; qué hicieron los filtros con lo generado; atajos (`¿?`); pistas estadísticas del prompt; preselección de técnicas |
| Valores esperados | Oraciones reales que cumplirían el filtro de marcador: SAL 100 %, DES 100 %, NEG 74 %, REQUEST 30 %, AFI 28 %, PRG 11 %, EMO 0 % · lo generado se aprueba: SAL y DES 100 %, NEG 85 %, PRG 69 %, AFI 57 %, REQUEST 47 %, EMO 28 % · filtros de idioma y semántico: 0 % de rechazo en lo generado |
| Versión anterior | `analisis_intrinseco/notebooks/historico/analisis_patrones_por_intencion_split70.ipynb` (usaba una partición 70/15/15 que no se empleaba en ningún experimento); sus tablas están en `resultados/tablas/` porque el prompt de GtR las leyó como pistas |

### Experimento 5. Los 12 experimentos (R6) y el volumen de GtR

```bash
python $CV --tecnica sin_aumento
python $CV --tecnica mixup
python $CV --tecnica retrotraduccion --checkpoint x            # usa la caché del catálogo; el checkpoint no se necesita
python $CV --tecnica generate_then_refine --cantidad 120 --etiqueta c120     # desde la caché (también 80 y 40)
python E/unir_predicciones.py --etiqueta cv_interna_c120 --entradas \
   validacion_cruzada_predicciones_sin_puntuacion_cv_interna_sin_aumento.csv \
   validacion_cruzada_predicciones_sin_puntuacion_cv_interna_mixup.csv \
   validacion_cruzada_predicciones_sin_puntuacion_cv_interna_retrotraduccion.csv \
   validacion_cruzada_predicciones_sin_puntuacion_cv_interna_generate_then_refine_c120.csv
python E/comparacion_pareada.py --entrada E/resultados/cv_interna/validacion_cruzada_predicciones_sin_puntuacion_cv_interna_c120.csv --etiqueta cv_interna_c120 --carpeta E/resultados/cv_interna
python E/curvas_roc.py          --entrada E/resultados/cv_interna/validacion_cruzada_predicciones_sin_puntuacion_cv_interna_c120.csv --etiqueta cv_interna_c120 --carpeta E/resultados/cv_interna
```

| | |
|---|---|
| Salidas | `E/resultados/cv_interna/*_c120.csv` (resumen, predicciones, pareadas, ROC, volumen sintético y F1 por C), y lo mismo con `_c80` y `_c40` |
| Valores esperados (F1 macro) | mBERT: sin aumento **0.665**, Mixup 0.629, Retrotraducción 0.631, GtR 0.652 · LaBSE: 0.587, 0.587, 0.581, 0.593 · XLM-R: 0.570, 0.546, 0.551, 0.589 |
| Diferencias pareadas | Distinguibles de cero (todas negativas): Mixup en mBERT (−0.036) y XLM-R (−0.024), Retrotraducción en mBERT (−0.034); GtR: −0.013, +0.006, +0.018 (no distinguibles) |
| Volumen de GtR | mBERT con 40, 80 y 120 por categoría: 0.650, 0.650, 0.652 |

### Experimento 6. Cantidad generada por GtR

Mismo comando que arriba con `--cantidad 40` y `--cantidad 80` (etiquetas `c40` y `c80`). Se evalúan desde la misma caché, que guarda lotes de 20 por categoría y usa los primeros 2, 4 o 6 según la cantidad.

### Experimento 7. Pruebas de causa (con C fijo: mBERT 0.3, LaBSE 10, XLM-R 10)

| Prueba | Comando | Salida | Valor esperado |
|---|---|---|---|
| Curva de aprendizaje con datos reales, utilidad del sintético, dosis y composición | `python E/diagnostico_baseline/por_que_no_mejora.py` | `A_curva_aprendizaje.csv`, `B_utilidad_sintetico.csv`, `C_dosis.csv`, `C_por_categoria.csv` | mBERT al 25/50/75/100 %: 0.508, 0.575, 0.630, 0.667 · solo sintético (GtR/Retro/Mixup): 0.484, 0.452, 0.594 · dosis 25-100 %: 0.660 → 0.655; balanceado 0.669 |
| Estancamiento por lotes y tipos de ruido | `python E/diagnostico_baseline/por_que_se_estanca.py` | `E_curva_por_lotes.csv`, `E_estadisticas_por_lote.csv` | mBERT 0.667 → 0.670 (20) → 0.655 (120); sin ruido de etiqueta 0.668 con 120 |
| Variantes de selección del sintético (8 + 2 referencias, mBERT) | `python E/variantes_gtr.py` o el notebook `E/notebooks/variantes_gtr.ipynb` (también en Colab) | `E/resultados/variantes_gtr/` | Ninguna supera 0.665; la mejor, v4, 0.656; GtR completo 0.654 |

### Experimento 8. Pocos datos reales (10, 25, 50 y 80 por categoría)

| | |
|---|---|
| Comando | `python E/curva_regimen.py` (usa la caché de regímenes; **nunca** llama a la API sin `--permitir-api`) |
| Salida | `E/resultados/curva_regimen/{predicciones,resumen_por_regimen,comparacion_pareada}.csv` y `curva_regimen.png` |
| Valores esperados | Ganancia de GtR frente a sin aumento con 10 / 25 / 50 / 80 por categoría: mBERT +0.054, +0.042, +0.022, −0.009 · LaBSE +0.038, +0.040, +0.016, −0.007 · XLM-R +0.044, +0.029, +0.024, +0.018 |

### Experimento 9. La categoría DES

| Parte | Comando | Salida |
|---|---|---|
| Auditoría de etiquetas | `python oe2_aumento_de_datos/analisis_intrinseco/auditoria_etiquetas_des.py` | `analisis_intrinseco/resultados/auditoria_des/` |
| Propuesta de reetiquetado (copia aparte; no toca el corpus) | `python oe2_aumento_de_datos/analisis_intrinseco/proponer_correccion_des.py` | `auditoria_des/corpus_shiwilu_propuesta_des.csv` |
| Sin la categoría DES | `python $CV --tecnica {sin_aumento,mixup,retrotraduccion} --excluir-categorias DES` | `E/resultados/cv_interna_sin_DES/` (baseline: mBERT 0.702, LaBSE 0.659, XLM-R 0.663) |
| Corpus reetiquetado | `python $CV --tecnica {sin_aumento,mixup} --corpus <corpus_shiwilu_propuesta_des.csv>` | `E/resultados/cv_interna_corpus_shiwilu_propuesta_des/` (mBERT 0.638, LaBSE 0.581, XLM-R 0.561) |

GtR no se repitió en estos escenarios (costo de la API); solo sin aumento, Mixup y Retrotraducción.

### Experimento 10. Sensibilidad al protocolo

| Parte | Comando | Salida y valor esperado |
|---|---|---|
| Con signos de interrogación | `python $CV --tecnica {sin_aumento,mixup,retrotraduccion} --condicion con_interrogacion` | `cv_interna_con_interrogacion/`: baselines mBERT 0.735, LaBSE 0.684, XLM-R 0.648 |
| Con `¿?` y sin DES / con DES reetiquetado | Las mismas opciones combinadas (`--excluir-categorias DES`, `--corpus ...`) | `cv_interna_con_interrogacion_sin_DES/` (mBERT 0.807) y `cv_interna_con_interrogacion_corpus_shiwilu_propuesta_des/` (mBERT 0.717) |
| Protocolo A frente a B | Experimento 3 | GtR sobre XLM-R: +0.032 (distinguible) con A, +0.018 (no) con B |
| Reproducibilidad entre equipos | Ver sección 7 | GtR en mBERT: 0.652 (referencia) frente a 0.654 en una recomputación local |

### Análisis intrínseco de embeddings usado en las conclusiones (OE3)

| | |
|---|---|
| Comandos | `python oe3_caracterizacion_embeddings/caracterizacion.py` (también `--corpus retrotraduccion` y `--corpus generate_then_refine`) · `python oe3_caracterizacion_embeddings/silueta_real_vs_sintetico.py` · `python oe3_caracterizacion_embeddings/f1_por_pooling.py` |
| Salidas | `oe3_caracterizacion_embeddings/resultados/` (`metricas_intrinsecas.csv`, `proyeccion_2d/`, `silueta_real_vs_sintetico.csv`, `f1_por_pooling.csv`) y `E/resultados/cv_interna_pooling/` |
| Valores esperados | Silueta con mean pooling: mBERT −0.010, LaBSE −0.030, XLM-R −0.051 · F1 por pooling sin aumento: mBERT 0.617-0.667, LaBSE 0.591-0.622, XLM-R 0.544-0.600 |

### Figuras

`oe2_aumento_de_datos/analisis_resultados/analisis_resultados_oe2.ipynb` lee los CSV anteriores, regenera las 10 figuras
(`analisis_resultados/figuras/fig1…fig10`) y explica cada una (qué pregunta responde, cómo leerla y qué concluir).

---

## 5. Conclusiones y la evidencia que las sostiene

Las rutas son desde la raíz. «Firmeza»: **fuerte** = diferencia pareada distinguible o consistente en los tres modelos; **media** = coherente con varias pruebas, no probada
causalmente; **media-baja** = diferencias menores a ~0.02 sin prueba de significancia.

| # | Conclusión | Evidencia (archivos) | Firmeza |
|---|---|---|---|
| C1 | Con el corpus completo ninguna técnica de aumento supera al baseline | `oe2_aumento_de_datos/evaluacion/resultados/cv_interna/comparacion_pareada_sin_puntuacion_cv_interna_c120.csv` y `validacion_cruzada_resumen_...c120.csv`; figuras 1 y 2 | Fuerte |
| C2 | mBERT sin aumento (0.665) es el mejor y supera a LaBSE y XLM-R (+0.078 y +0.095) | mismo resumen y pareadas (`sin_aumento: mbert - labse`, `mbert - xlmr`) | Fuerte |
| C3 | La ventaja de mBERT es coherente con que la intención se reconoce por la forma de las letras | `diagnostico_baseline/resultados/ngramas_caracteres.csv` (0.751), `fragmentacion.csv`, `referencias_simples.csv` | Media |
| C4 | El aumento no ayuda porque lo sintético vale menos que lo real (no es un techo de datos) | `A_curva_aprendizaje.csv`, `B_utilidad_sintetico.csv`, `C_dosis.csv`, `C_por_categoria.csv` | Media |
| C5 | GtR ayuda con 10-25 ejemplos reales por categoría (+0.03 a +0.05) y el efecto desaparece con 80 | `evaluacion/resultados/curva_regimen/` y la caché `tecnicas_aumento/salidas/regimenes/` | Fuerte en dirección (una muestra por régimen) |
| C6 | Mixup y Retrotraducción no ayudan en ningún régimen; Retrotraducción empeora | `curva_regimen/comparacion_pareada.csv`; `B_utilidad_sintetico.csv` (52 % de etiquetas coherentes) | Fuerte / media |
| C7 | El F1 se estanca por un límite de información del sintético más que por ruido; el ruido de etiqueta explica la leve caída de mBERT | `E_curva_por_lotes.csv`, `E_estadisticas_por_lote.csv` | Media-baja |
| C8 | Balancear, filtrar o quitar DES no basta para que GtR supere al baseline con todos los datos | `evaluacion/resultados/variantes_gtr/resumen_variantes.csv` | Media |
| C9 | DES es la clase más débil por ser la clase residual; reetiquetar no mejora el F1 | `analisis_intrinseco/resultados/auditoria_des/`, `cv_interna_corpus_shiwilu_propuesta_des/`, `cv_interna_sin_DES/` | Media |
| C10 | Los resultados dependen del protocolo (elección de C, signos `¿?`); diferencias < 0.01 no son interpretables | `evaluacion/resultados/` (raíz, protocolo A) frente a `cv_interna/`; `cv_interna_con_interrogacion/`; `diagnostico_c/resultados/` | Fuerte |
| C11 | Ningún modelo agrupa por intención (silueta ≈ 0 o negativa); el orden intrínseco coincide con el extrínseco; el pooling mueve el F1 hasta 0.05 pero no cambia el ganador | `oe3_caracterizacion_embeddings/resultados/original/metricas_intrinsecas.csv`, `f1_por_pooling.csv` | Media |
| — | Las oraciones sintéticas son shiwilu gramatical | **No evaluado**: requiere un hablante | — |

---

## 6. De dónde sale cada carpeta de resultados

| Carpeta | La produce |
|---|---|
| `oe2_aumento_de_datos/evaluacion/resultados/` (raíz) | Protocolo A: `validacion_cruzada.py`, `comparacion_pareada.py`, `curvas_roc.py` |
| `.../resultados/cv_interna/` | Protocolo B: `validacion_cruzada_cv_interna.py` + `unir_predicciones.py` + pareadas + ROC. La subcarpeta `_previo_sin_gtr/` es una versión anterior de 9 experimentos (histórico) |
| `.../resultados/cv_interna_sin_DES/`, `cv_interna_corpus_shiwilu_propuesta_des/`, `cv_interna_con_interrogacion*/` | `validacion_cruzada_cv_interna.py` con `--excluir-categorias`, `--corpus` o `--condicion` |
| `.../resultados/cv_interna_pooling/` | `oe3_caracterizacion_embeddings/f1_por_pooling.py` |
| `.../resultados/curva_regimen/` | `curva_regimen.py` |
| `.../resultados/variantes_gtr/` | `variantes_gtr.py` o el notebook `variantes_gtr.ipynb` |
| `.../diagnostico_baseline/resultados/` | `referencias_simples.py`, `ngramas_caracteres.py`, `por_que_no_mejora.py`, `por_que_se_estanca.py` |
| `.../diagnostico_c/resultados/` | `sensibilidad_c.py`, `sensibilidad_c_en_linea.py`, `comparacion_pareada_c_fijo.py` |
| `oe2_aumento_de_datos/analisis_intrinseco/resultados/` | `analisis_filtros/`: el cuaderno `analisis_corpus_y_filtros.ipynb`; `tablas/`: la versión anterior (70/15/15); `auditoria_des/`: `auditoria_etiquetas_des.py` y `proponer_correccion_des.py` |
| `oe2_aumento_de_datos/tecnicas_aumento/salidas/` | `generate_then_refine.py`, `retrotraduccion.py`, `refiltrar_marcadores.py`, los cuadernos de `tecnicas_aumento/colab/` y `curva_regimen.py --solo-generar` |
| `oe3_caracterizacion_embeddings/resultados/` | `caracterizacion.py`, `silueta_real_vs_sintetico.py`, `f1_por_pooling.py` |
| `oe2_aumento_de_datos/analisis_resultados/figuras/` | El notebook `analisis_resultados_oe2.ipynb` |
| `oe2_aumento_de_datos/valores_de_referencia.csv` | `verificar_resultados.py --generar-referencia` (solo mantenedores) |

---

## 7. Qué esperar al reproducir

- **Verificado:** al recalcular un fold (mBERT, sin aumento, fold 0) las 137 predicciones y todas las probabilidades coinciden exactamente con las guardadas,
  y las diferencias pareadas, los resúmenes y las curvas ROC se regeneran con el mismo contenido (solo cambian los bytes de los PNG, que dependen de la versión de matplotlib;
  los CSV son la referencia).
- **Qué cambia entre equipos:** el valor de C que elige la validación cruzada interna es sensible a diferencias numéricas mínimas (librerías, sistema operativo, CPU/GPU).
  Una recomputación local de GtR en mBERT dio 0.654 en lugar de 0.652, porque en uno de los cinco folds se eligió otro C (1.0 en vez de 0.3); el 97 % de las predicciones fue idéntico.
  Por eso **las diferencias menores a ~0.005-0.01 en F1 macro no son interpretables**, y `verificar_resultados.py` acepta por defecto una tolerancia de 0.006 por cifra
  (`--tolerancia` la cambia).
- **Aleatoriedad controlada:** la semilla es 42; Mixup usa `--semilla-aumento` (0 por defecto); las variantes de balanceo de `variantes_gtr.py` usan semillas 0 y 1;
  la curva por régimen usa la muestra con semilla 0 (GtR solo se generó para esa muestra).
- **Los intervalos de confianza** son bootstrap (2000 remuestreos, semilla fija), así que se reproducen.
- **Tiempos:** solo hay mediciones puntuales (p. ej. un fold de un modelo sin aumento ≈ 2 minutos; `f1_por_pooling.py` ≈ 1-2 h). El resto no se ha medido de forma fiable; ver `reproducir_oe2.py`.

---

## 8. Regenerar desde cero lo que usa la API o GPU (opcional)

Solo hace falta si se quiere **volver a generar** el texto sintético (en lugar de usar las cachés). El texto generado nuevo no será idéntico: Claude no es determinista,
por lo que los resultados cambiarán dentro de la variabilidad descrita en la sección 7.

| Qué | Cómo | Costo / requisito |
|---|---|---|
| GtR del protocolo B (K=5, 120 por categoría) | Cuaderno `oe2_aumento_de_datos/tecnicas_aumento/colab/colab_generate_then_refine_cv_interna.ipynb` (Colab) o `python $CV --tecnica generate_then_refine --cantidad 120 --solo-generar` | `ANTHROPIC_API_KEY` en `.env` (ver `.env.example`); ≈ 1 085 llamadas, unos US$ 8-10 con `claude-sonnet-4-6`. Guarda en `salidas/en_linea_marcadores_fuentes/generate_then_refine/`; es reanudable (omite lo que ya existe) |
| GtR de los regímenes (10, 25, 50 por categoría) | `python E/curva_regimen.py --solo-generar --permitir-api` | ≈ 245 llamadas, unos US$ 2. **No** llama a la API sin `--permitir-api` |
| Retrotraducción (catálogo) | Entrenar el checkpoint NLLB+LoRA de F. Prado (repositorio externo, ver `tecnicas_aumento/README.md`; cuaderno `colab_entrenar_checkpoint.ipynb`) y correr `tecnicas_aumento/retrotraduccion.py` o el cuaderno `colab_retrotraduccion_en_linea.ipynb` | GPU; el repositorio externo se clona en `tecnicas_aumento/tesis_spa_jeb/` (ignorado por git) |
| Reconstruir el corpus (OE1) | `oe1_corpus/README.md` | API para las etapas 0 y 2; el PDF fuente de terceros no se incluye |

Antes de gastar créditos conviene estimar el consumo (cada llamada cuesta ≈ US$ 0.007-0.009).

---

## 9. Material histórico (no se usa para reportar)

`oe2_aumento_de_datos/historico/` conserva, **sin borrar nada**, las etapas anteriores que llevaron al protocolo vigente (ver su `README.md`):
la partición única 70/15/15 y el texto crudo (con signos y mayúsculas, F1 inflado ~0.75), la generación en línea con C elegido en un dev (niveles 20, 40, 80 y 120 de GtR),
los zips tal cual salieron de Colab y la auditoría de optimismo (30 particiones aleatorias). Los scripts siguen ejecutables, pero sus salidas escriben ahí mismo.
También se conservan `oe3_caracterizacion_embeddings/resultados_anteriores/` (OE3 con las fuentes sintéticas previas) y
`evaluacion/resultados/cv_interna/_previo_sin_gtr/`. No usar estos números para reportar resultados.

---

## 10. Pendientes y limitaciones

- **OE4 (`oe4_sintesis/`)** aún usa el protocolo A y los resultados antiguos de OE3; se actualizará con los resultados del protocolo B en su etapa.
- Las oraciones sintéticas **no están validadas por un hablante** de shiwilu.
- Cada régimen de pocos datos se evaluó con **una sola muestra** de oraciones reales (GtR solo se generó para la semilla 0).
- Las pruebas de causa y las variantes de selección se hicieron con **C fijo** y, en el caso de las variantes, solo con mBERT.
- GtR no se repitió en los escenarios de DES ni con `¿?` (costo de la API).
- El 0.503 del vecino léxico (palabras compartidas; con Jaccard, 0.511) reemplaza al 0.505 de una versión anterior no reproducible de ese cálculo.
- Las pruebas automáticas (`tests/`) cubren las reglas de anotación del OE1 (`pytest tests/`); los experimentos se verifican con `verificar_resultados.py`.
