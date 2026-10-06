# Índice de evidencias del informe de los 12 experimentos

Capa **documental**: no contiene resultados ni mueve ningún archivo. Sirve para ir de cada afirmación del informe
([`oe2_aumento_de_datos/informe_12_experimentos/INFORME.md`](../../oe2_aumento_de_datos/informe_12_experimentos/INFORME.md), PDF en la misma carpeta)
al archivo que la sostiene. Todas las rutas son desde la raíz del repositorio.

**Nivel de evidencia:** *Fuerte* = diferencia pareada o intervalo que excluye el cero, 3 modelos, protocolo B · *Media* = coherente con varias pruebas de apoyo, pero no probada causalmente o con diferencias pequeñas ·
*Media-baja* = diferencias < ~0.02 sin prueba de significancia · *No evaluado* = requiere un hablante.

## 1. Qué es vigente y qué es histórico

| Estado | Qué | Dónde |
|---|---|---|
| **Vigente (resultado principal)** | Protocolo B: C por CV interna (K=5), 7 clases, corpus original, texto normalizado **sin** `¿?` | `oe2_aumento_de_datos/evaluacion/resultados/cv_interna/` (archivos `*_c120.csv`) |
| Vigente, escenario aparte | Con `¿?` | `oe2_aumento_de_datos/evaluacion/resultados/cv_interna_con_interrogacion/` |
| Vigente, escenario aparte | Sin DES / corpus con DES reetiquetado | `.../resultados/cv_interna_sin_DES/`, `.../cv_interna_con_interrogacion_sin_DES/`, `.../cv_interna_corpus_shiwilu_propuesta_des/`, `.../cv_interna_con_interrogacion_corpus_shiwilu_propuesta_des/` |
| Anterior (protocolo A: C elegido en ~90 oraciones) | Resultados de la primera corrida; se conservan para comparar | `oe2_aumento_de_datos/evaluacion/resultados/validacion_cruzada_resumen_sin_puntuacion.csv` y archivos hermanos (`validacion_cruzada_predicciones_*`, `comparacion_pareada_*`, `curvas_roc*`) |
| Anterior | Versión de 9 experimentos (sin GtR de protocolo B) | `oe2_aumento_de_datos/evaluacion/resultados/cv_interna/_previo_sin_gtr/` |
| Histórico (cerrado) | Split único, atajos de texto crudo, zips de Colab, protocolo dev en línea | `oe2_aumento_de_datos/historico/` (ver su `README.md`) |
| Histórico OE3 | Caracterización con las fuentes anteriores | `oe3_caracterizacion_embeddings/resultados_anteriores/` |
| Apoyo (no son resultados principales) | Sensibilidad al C | `oe2_aumento_de_datos/evaluacion/diagnostico_c/` |

## 2. Resultados principales (los que van en el cuerpo de la tesis)

| Resultado | Archivo | Figura |
|---|---|---|
| F1 macro de los 12 experimentos con IC 95% | `oe2_aumento_de_datos/evaluacion/resultados/cv_interna/validacion_cruzada_resumen_sin_puntuacion_cv_interna_c120.csv` | `oe2_aumento_de_datos/informe_12_experimentos/figuras/fig1_f1_12_experimentos.png` |
| Diferencias pareadas frente a sin aumento | `oe2_aumento_de_datos/evaluacion/resultados/cv_interna/comparacion_pareada_sin_puntuacion_cv_interna_c120.csv` | `.../figuras/fig2_diferencias_pareadas.png` |
| Curva por régimen de datos reales (10/25/50/80 por clase) | `oe2_aumento_de_datos/evaluacion/resultados/curva_regimen/resumen_por_regimen.csv` y `comparacion_pareada.csv` | `.../figuras/fig5_curva_por_regimen.png` |
| Métricas intrínsecas (silueta, DB, CH) | `oe3_caracterizacion_embeddings/resultados/original/metricas_intrinsecas.csv` (y carpetas `retrotraduccion/`, `generate_then_refine/`) | — |
| F1 por pooling (36 combinaciones) | `oe3_caracterizacion_embeddings/resultados/f1_por_pooling.csv` | — |

## 3. Evidencia por conclusión

| # | Afirmación | Archivo(s) fuente | Figura / tabla | Sección del informe | Nivel |
|---|---|---|---|---|---|
| 1 | Con todos los datos, ninguna técnica supera al baseline | `oe2_aumento_de_datos/evaluacion/resultados/cv_interna/comparacion_pareada_sin_puntuacion_cv_interna_c120.csv`; `.../cv_interna/validacion_cruzada_predicciones_sin_puntuacion_cv_interna_c120.csv` | fig1, fig2 | 2, 3 | Fuerte |
| 1b | GtR con 80 y 40 por categoría da lo mismo que con 120 | `.../cv_interna/validacion_cruzada_resumen_sin_puntuacion_cv_interna_c80.csv` y `_c40.csv` | tabla sec. 3 | 3 | Media |
| 1c | GtR tiene el AUC macro más alto | `oe2_aumento_de_datos/evaluacion/resultados/cv_interna/curvas_roc_cv_interna_c120.csv` | `curvas_roc_cv_interna_c120.png` | 3 | Media (sin prueba) |
| 1d | Cambios por clase pequeños; el mayor perjuicio es sobre DES | `oe2_aumento_de_datos/evaluacion/diagnostico_baseline/resultados/C_por_categoria.csv` | fig8 | 3 | Media |
| 2 | mBERT sin aumento es el mejor resultado | `.../cv_interna/validacion_cruzada_resumen_sin_puntuacion_cv_interna_c120.csv` | fig1 | 2, 4 | Fuerte |
| 2b | mBERT gana porque conserva la forma de las letras | `oe2_aumento_de_datos/evaluacion/diagnostico_baseline/resultados/ngramas_caracteres.csv` (n-gramas de caracteres, F1 0.751); `.../fragmentacion.csv`; `oe3_caracterizacion_embeddings/resultados/f1_por_pooling.csv` | fig3 | 4 | Media |
| 3a | mBERT aún mejora con más datos reales (no hay techo de datos) | `oe2_aumento_de_datos/evaluacion/diagnostico_baseline/resultados/A_curva_aprendizaje.csv` (script `por_que_no_mejora.py`) | fig4 | 5.1 | Media |
| 3b | Lo sintético vale menos que lo real | `.../diagnostico_baseline/resultados/B_utilidad_sintetico.csv` | fig6 | 5.2 | Media |
| 3c | El sintético está desbalanceado y parte perjudica a su clase | `.../diagnostico_baseline/resultados/C_dosis.csv`, `C_por_categoria.csv`; `oe2_aumento_de_datos/evaluacion/resultados/cv_interna/volumen_sintetico_sin_puntuacion_cv_interna_generate_then_refine_c120.csv` | fig7 | 5.3 | Media / Media-baja |
| 4 | GtR ayuda con 10–25 reales por clase | `oe2_aumento_de_datos/evaluacion/resultados/curva_regimen/{predicciones,resumen_por_regimen,comparacion_pareada}.csv`; caché de generación: `oe2_aumento_de_datos/tecnicas_aumento/salidas/regimenes/generate_then_refine/` | fig5 | 6 | Fuerte en dirección (una muestra de reales por régimen) |
| 5a | El F1 se estanca por límite de información y no por ruido | `oe2_aumento_de_datos/evaluacion/diagnostico_baseline/resultados/E_curva_por_lotes.csv`, `E_estadisticas_por_lote.csv` (script `por_que_se_estanca.py`) | fig9 | 5.5 | Media-baja |
| 5b | El ruido de etiqueta (10–29% por lote) explica la leve caída de mBERT | `.../diagnostico_baseline/resultados/E_curva_por_lotes.csv` (versión `sin_ruido_etiq`) | fig9 | 5.5 | Media |
| 5c | Balancear, filtrar o quitar DES no hace que GtR supere al baseline | `oe2_aumento_de_datos/evaluacion/resultados/variantes_gtr/resumen_variantes.csv`, `f1_por_clase.csv`, `conteos_por_clase.csv`, `pred_*.csv`; código `oe2_aumento_de_datos/evaluacion/variantes_gtr.py` y `oe2_aumento_de_datos/evaluacion/notebooks/variantes_gtr.ipynb` | fig10 | 5.6 | Media |
| 5d | La caída de DES con GtR persiste sin su propio sintético | `.../variantes_gtr/f1_por_clase.csv` (v3 y v8) | fig10 | 5.6 | Media |
| 6a | DES es la clase débil (clase residual por regex) | `oe2_aumento_de_datos/analisis_intrinseco/resultados/auditoria_des/README.md`, `des_oraciones_con_errores.csv`, `des_vs_request_indicios.csv`, `request_predichas_como_des.csv`; `docs/metodologia_construccion_corpus.md` | fig8 | 7 | Media |
| 6b | Reetiquetar DES no mejora el F1 | `oe2_aumento_de_datos/evaluacion/resultados/cv_interna_corpus_shiwilu_propuesta_des/`; propuesta: `oe2_aumento_de_datos/analisis_intrinseco/resultados/auditoria_des/cambios_propuestos_des.csv`, `corpus_shiwilu_propuesta_des.csv` | tabla sec. 7 | 7 | Media |
| 6c | Quitar DES sube el F1 (problema más fácil) | `oe2_aumento_de_datos/evaluacion/resultados/cv_interna_sin_DES/` | tabla sec. 7 | 7 | Media |
| 7a | El protocolo de elegir C cambia la lectura de GtR | `oe2_aumento_de_datos/evaluacion/resultados/cv_interna/cv_interna_f1_por_c_*.csv` frente a `oe2_aumento_de_datos/evaluacion/resultados/validacion_cruzada_resumen_sin_puntuacion.csv` (protocolo A); `oe2_aumento_de_datos/evaluacion/diagnostico_c/resultados/` | sec. 3 (precaución), 8 | 3, 8 | Fuerte |
| 7b | Conservar `¿?` sube el F1 (casi todo PRG) | `oe2_aumento_de_datos/evaluacion/resultados/cv_interna_con_interrogacion/` | tabla sec. 8 | 8 | Fuerte |
| 8 | Las oraciones sintéticas son shiwilu gramatical | — (requiere un hablante) | — | 9 | No evaluado |

## 4. Cómo reproducir (sin API salvo donde se indica)

| Qué | Comando (desde la raíz) | Usa API |
|---|---|---|
| Los 12 experimentos, protocolo B | `python oe2_aumento_de_datos/evaluacion/validacion_cruzada_cv_interna.py --tecnica <sin_aumento\|mixup\|retrotraduccion\|generate_then_refine>` | Solo GtR si falta la caché |
| Curva por régimen | `python oe2_aumento_de_datos/evaluacion/curva_regimen.py` (para generar GtR nuevo: `--permitir-api`) | Solo con `--permitir-api` |
| Diagnósticos de por qué no mejora / se estanca | `python oe2_aumento_de_datos/evaluacion/diagnostico_baseline/por_que_no_mejora.py` y `por_que_se_estanca.py` | No |
| Variantes de selección de GtR | `python oe2_aumento_de_datos/evaluacion/variantes_gtr.py` | No |
| Figuras y PDF del informe | `python oe2_aumento_de_datos/informe_12_experimentos/generar_figuras.py` y `generar_pdf.py` | No |
