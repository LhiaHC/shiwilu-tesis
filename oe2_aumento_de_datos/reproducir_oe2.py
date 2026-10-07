"""
Reproduce, en orden, TODOS los experimentos del OE2 (y los analisis de OE3 que se usan en sus conclusiones) a partir de los
datos y las caches que ya estan en el repositorio. Cada paso es un comando de los que se documentan en REPRODUCIBILIDAD.md.

  - Por defecto NO ejecuta nada: solo imprime el plan (comandos, tiempo aproximado en CPU, si usa la API de Claude).
  - Con --ejecutar corre los pasos elegidos, en orden, desde la raiz del repositorio, y se detiene si uno falla.
  - Ningun paso llama a la API de Claude con las caches incluidas. Los pasos que generarian texto nuevo con Claude estan marcados
    con [API] y solo se pueden ejecutar con --permitir-api (gasta creditos; ver REPRODUCIBILIDAD.md).
  - ATENCION: ejecutar un paso SOBREESCRIBE los resultados de ese paso en las carpetas `resultados/`. Para comprobar que se
    reproducen, corra `verificar_resultados.py` despues (compara con los valores de referencia).

Grupos de pasos (--grupos):
  r5        analisis del corpus en funcion de los filtros y de las 3 fuentes (cuadernos)
  base      los 12 experimentos del protocolo B (GtR 120, 80 y 40), diferencias pareadas y curvas ROC
  proto_a   protocolo A: C elegido en un dev de ~90 oraciones (referencia para ver el efecto del protocolo)
  diag      referencias simples, n-gramas, por que no mejora / se estanca, diagnostico de C
  regimen   curva por regimen de datos reales (10, 25, 50 y 80 por categoria)
  variantes variantes de seleccion del sintetico de GtR (mBERT)
  des       auditoria de DES, corpus reetiquetado, sin DES y con signos de interrogacion
  oe3       caracterizacion de embeddings, silueta real vs sintetico y F1 por pooling (usados en el analisis)
  figuras   regenera las 10 figuras con el notebook de analisis de resultados (~1 min)
  verificar compara los resultados actuales con los valores de referencia (segundos)

Tiempos: solo se indican los que estan documentados en los scripts (CPU de escritorio, sin GPU). Para el resto no hay una medicion
fiable: como referencia, un fold de la validacion cruzada con un modelo y sin aumento tarda ~2 minutos (incluye cargar el modelo), y los
pasos con aumento tardan mas porque hay que obtener los vectores de todo el texto sintetico. El primer uso descarga los modelos de Hugging Face.

Uso (desde la raiz del repositorio):
    python oe2_aumento_de_datos/reproducir_oe2.py                           # imprime el plan completo
    python oe2_aumento_de_datos/reproducir_oe2.py --grupos base --ejecutar  # corre solo los 12 experimentos
    python oe2_aumento_de_datos/reproducir_oe2.py --grupos verificar --ejecutar
    python oe2_aumento_de_datos/reproducir_oe2.py --paso 12 --ejecutar      # un solo paso (numeros del plan)
"""

import argparse
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
E = "oe2_aumento_de_datos/evaluacion"
CV = f"{E}/validacion_cruzada_cv_interna.py"
R = f"{E}/resultados/cv_interna"
DES = "oe2_aumento_de_datos/analisis_intrinseco/resultados/auditoria_des/corpus_shiwilu_propuesta_des.csv"
PRED = "validacion_cruzada_predicciones_sin_puntuacion_cv_interna"

PASOS = []   # (grupo, descripcion, comando, tiempo, usa_api)


def paso(grupo, desc, cmd, tiempo, api=False):
    PASOS.append((grupo, desc, cmd, tiempo, api))


# ----------------------------------------------------------------------------------------------- R5: analisis del corpus
paso("r5", "R5: análisis del corpus en función de los filtros y las 3 fuentes (cuaderno; calcula embeddings de LaBSE)",
     "python -m nbconvert --to notebook --execute --inplace oe2_aumento_de_datos/analisis_intrinseco/notebooks/analisis_corpus_y_filtros.ipynb", "no medido")
paso("r5", "R5: métricas descriptivas del corpus (cuaderno)",
     "python -m nbconvert --to notebook --execute --inplace oe2_aumento_de_datos/analisis_intrinseco/notebooks/metricas_corpus.ipynb", "~1 min")

# ----------------------------------------------------------------------------------------------- base: protocolo B
paso("base", "Baselines (sin aumento), protocolo B", f"python {CV} --tecnica sin_aumento", "no medido")
paso("base", "Mixup, protocolo B", f"python {CV} --tecnica mixup", "no medido")
paso("base", "Retrotraducción, protocolo B (usa la caché del catálogo; el checkpoint no se necesita)",
     f"python {CV} --tecnica retrotraduccion --checkpoint x", "no medido")
for n in (120, 80, 40):
    paso("base", f"Generate-then-Refine con {n} por categoría, protocolo B (desde la caché)",
         f"python {CV} --tecnica generate_then_refine --cantidad {n} --etiqueta c{n}", "no medido")
for n in (120, 80, 40):
    entradas = " ".join([f"{PRED}_sin_aumento.csv", f"{PRED}_mixup.csv", f"{PRED}_retrotraduccion.csv", f"{PRED}_generate_then_refine_c{n}.csv"])
    paso("base", f"Unir las 4 técnicas (GtR {n}) en un solo archivo",
         f"python {E}/unir_predicciones.py --etiqueta cv_interna_c{n} --entradas {entradas}", "segundos")
    paso("base", f"Diferencias pareadas (GtR {n})",
         f"python {E}/comparacion_pareada.py --entrada {R}/validacion_cruzada_predicciones_sin_puntuacion_cv_interna_c{n}.csv --etiqueta cv_interna_c{n} --carpeta {R}", "no medido")
    paso("base", f"Curvas ROC (GtR {n})",
         f"python {E}/curvas_roc.py --entrada {R}/validacion_cruzada_predicciones_sin_puntuacion_cv_interna_c{n}.csv --etiqueta cv_interna_c{n} --carpeta {R}", "no medido")

# ----------------------------------------------------------------------------------------------- protocolo A
paso("proto_a", "Protocolo A: 12 experimentos con C elegido en un dev de ~90 oraciones", f"python {E}/validacion_cruzada.py", "~10 min (documentado)")
paso("proto_a", "Protocolo A: diferencias pareadas", f"python {E}/comparacion_pareada.py", "no medido")
paso("proto_a", "Protocolo A: curvas ROC", f"python {E}/curvas_roc.py", "no medido")

# ----------------------------------------------------------------------------------------------- diagnosticos
D = f"{E}/diagnostico_baseline"
paso("diag", "Referencias simples (mayoría y vecino léxico)", f"python {D}/referencias_simples.py", "segundos")
paso("diag", "Fragmentación del texto y clasificador de n-gramas de caracteres", f"python {D}/ngramas_caracteres.py", "no medido")
paso("diag", "Por qué no mejora: curva de aprendizaje, calidad de lo sintético, dosis y composición", f"python {D}/por_que_no_mejora.py", "~20-30 min (documentado)")
paso("diag", "Por qué se estanca: curva por lotes y tipos de ruido", f"python {D}/por_que_se_estanca.py", "no medido")
paso("diag", "Sensibilidad del F1 al hiperparámetro C (protocolo A)", f"python {E}/diagnostico_c/sensibilidad_c.py", "~10-15 min")
paso("diag", "Sensibilidad al C para GtR y Retrotraducción desde la caché", f"python {E}/diagnostico_c/sensibilidad_c_en_linea.py", "no medido")
paso("diag", "Diferencias pareadas con el mismo C fijo en ambas configuraciones", f"python {E}/diagnostico_c/comparacion_pareada_c_fijo.py", "~30-40 min")

# ----------------------------------------------------------------------------------------------- regimenes
paso("regimen", "Curva por régimen de datos reales (10, 25, 50, 80 por categoría; GtR desde la caché de regímenes)", f"python {E}/curva_regimen.py", "no medido")

# ----------------------------------------------------------------------------------------------- variantes
paso("variantes", "Variantes de selección del sintético de GtR (mBERT, 2 muestras)", f"python {E}/variantes_gtr.py", "no medido")

# ----------------------------------------------------------------------------------------------- DES / interrogacion
paso("des", "Auditoría de las etiquetas de DES", "python oe2_aumento_de_datos/analisis_intrinseco/auditoria_etiquetas_des.py", "segundos")
paso("des", "Propuesta de corrección de DES (corpus reetiquetado, copia aparte)", "python oe2_aumento_de_datos/analisis_intrinseco/proponer_correccion_des.py", "segundos")
for t, extra in (("sin_aumento", ""), ("mixup", ""), ("retrotraduccion", " --checkpoint x")):
    paso("des", f"Sin la categoría DES: {t}", f"python {CV} --tecnica {t}{extra} --excluir-categorias DES", "no medido")
for t in ("sin_aumento", "mixup"):
    paso("des", f"Corpus con 25 oraciones de DES reetiquetadas: {t}", f"python {CV} --tecnica {t} --corpus {DES}", "no medido")
for t, extra in (("sin_aumento", ""), ("mixup", ""), ("retrotraduccion", " --checkpoint x")):
    paso("des", f"Con signos de interrogación: {t}", f"python {CV} --tecnica {t}{extra} --condicion con_interrogacion", "no medido")
for t, extra in (("sin_aumento", ""), ("mixup", ""), ("retrotraduccion", " --checkpoint x")):
    paso("des", f"Con signos de interrogación y sin DES: {t}", f"python {CV} --tecnica {t}{extra} --condicion con_interrogacion --excluir-categorias DES", "no medido")
for t in ("sin_aumento", "mixup"):
    paso("des", f"Con signos de interrogación y DES reetiquetado: {t}", f"python {CV} --tecnica {t} --condicion con_interrogacion --corpus {DES}", "no medido")

# ----------------------------------------------------------------------------------------------- OE3
O3 = "oe3_caracterizacion_embeddings"
paso("oe3", "Caracterización del corpus original (4 poolings × 3 modelos, silueta/DB/CH, t-SNE y UMAP)", f"python {O3}/caracterizacion.py", "no medido")
paso("oe3", "Caracterización del corpus aumentado con Retrotraducción", f"python {O3}/caracterizacion.py --corpus retrotraduccion", "no medido")
paso("oe3", "Caracterización del corpus aumentado con Generate-then-Refine", f"python {O3}/caracterizacion.py --corpus generate_then_refine", "no medido")
paso("oe3", "Silueta de las oraciones reales vs las sintéticas", f"python {O3}/silueta_real_vs_sintetico.py", "no medido")
paso("oe3", "F1 por estrategia de pooling (36 combinaciones)", f"python {O3}/f1_por_pooling.py", "~1-2 h")

# ----------------------------------------------------------------------------------------------- figuras y verificacion
paso("figuras", "Regenerar las 10 figuras (notebook de análisis de resultados)",
     "python -m nbconvert --to notebook --execute --inplace oe2_aumento_de_datos/analisis_resultados/analisis_resultados_oe2.ipynb", "no medido")
paso("verificar", "Comparar los resultados actuales con los valores de referencia", "python oe2_aumento_de_datos/verificar_resultados.py", "segundos")

GRUPOS = ["r5", "base", "proto_a", "diag", "regimen", "variantes", "des", "oe3", "figuras", "verificar"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grupos", nargs="+", choices=GRUPOS, default=GRUPOS)
    ap.add_argument("--paso", type=int, nargs="+", default=None, help="Numeros de paso del plan (anula --grupos).")
    ap.add_argument("--ejecutar", action="store_true", help="Corre los pasos (por defecto solo los imprime).")
    ap.add_argument("--permitir-api", action="store_true", help="Permite pasos que llaman a la API de Claude (ninguno de los actuales lo hace).")
    ap.add_argument("--continuar", action="store_true", help="No detenerse si un paso falla.")
    args = ap.parse_args()

    elegidos = [(i, p) for i, p in enumerate(PASOS, 1) if (args.paso and i in args.paso) or (not args.paso and p[0] in args.grupos)]
    print(f"{len(elegidos)} paso(s) de {len(PASOS)}. Raíz del repositorio: {RAIZ}\n")
    for i, (g, desc, cmd, t, api) in elegidos:
        print(f"[{i:02d}] ({g}) {desc}{'  [API]' if api else ''}\n      {cmd}\n      tiempo: {t}")
    if not args.ejecutar:
        print("\n(No se ejecutó nada: agregue --ejecutar para correr estos pasos.)")
        return 0
    for i, (g, desc, cmd, t, api) in elegidos:
        if api and not args.permitir_api:
            print(f"\n[{i:02d}] OMITIDO (usa la API de Claude; agregue --permitir-api): {desc}")
            continue
        print(f"\n===== [{i:02d}] {desc} =====\n$ {cmd}", flush=True)
        r = subprocess.run(cmd, shell=True, cwd=RAIZ)
        if r.returncode != 0:
            print(f"El paso {i} terminó con error ({r.returncode}).")
            if not args.continuar:
                return r.returncode
    print("\nListo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
