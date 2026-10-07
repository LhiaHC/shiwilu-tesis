"""
Verifica que los resultados del OE2 (y los de OE3 usados en su analisis) coinciden con los valores de referencia con los que se
sacaron las conclusiones. Sirve para comprobar una reproduccion: despues de volver a correr un experimento (o de clonar el
repositorio en otra maquina), este script lee las salidas actuales y las compara con `valores_de_referencia.csv`.

  - Cada valor de referencia es una cifra concreta (p. ej. "F1 macro de mBERT sin aumento con el protocolo B = 0.665").
  - Se acepta una diferencia de hasta --tolerancia (0.006 por defecto): el valor de C elegido por validacion cruzada interna es
    sensible a diferencias numericas minimas entre equipos y librerias, asi que diferencias menores a ~0.005-0.01 no son
    interpretables (ver REPRODUCIBILIDAD.md, seccion "Que esperar al reproducir").
  - No entrena nada, no usa la API: solo lee CSV. Corre en segundos.

Uso (desde la raiz del repositorio):
    python oe2_aumento_de_datos/verificar_resultados.py                        # compara contra los valores de referencia
    python oe2_aumento_de_datos/verificar_resultados.py --tolerancia 0.01
    python oe2_aumento_de_datos/verificar_resultados.py --generar-referencia   # (solo mantenedores) vuelve a escribir la referencia
Codigo de salida: 0 si todo coincide; 1 si alguna cifra difiere mas que la tolerancia o falta un archivo.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
OE2 = RAIZ / "oe2_aumento_de_datos"
EV = OE2 / "evaluacion" / "resultados"
DIAG = OE2 / "evaluacion" / "diagnostico_baseline" / "resultados"
OE3 = RAIZ / "oe3_caracterizacion_embeddings" / "resultados"
REFERENCIA = OE2 / "valores_de_referencia.csv"

MODELOS = ["mbert", "labse", "xlmr"]
TECNICAS = ["sin_aumento", "mixup", "retrotraduccion", "generate_then_refine"]
PRE = "validacion_cruzada_resumen_sin_puntuacion_cv_interna_sin_aumento.csv"


def especificaciones():
    """Lista de (id, archivo, filtros, columna_del_valor)."""
    S = []

    def cifra(id_, archivo, filtros, col):
        S.append((id_, archivo, filtros, col))

    # 1. Los 12 experimentos con el protocolo B (GtR 120) y sus diferencias pareadas
    res = EV / "cv_interna" / "validacion_cruzada_resumen_sin_puntuacion_cv_interna_c120.csv"
    par = EV / "cv_interna" / "comparacion_pareada_sin_puntuacion_cv_interna_c120.csv"
    for m in MODELOS:
        for t in TECNICAS:
            cifra(f"B12|f1|{m}|{t}", res, {"modelo": m, "tecnica": t}, "f1_macro_agrupado")
        for t in TECNICAS[1:]:
            cifra(f"B12|dif|{m}|{t}", par, {"comparacion": f"{m}: {t} - sin_aumento"}, "diferencia_f1")
    for cant in (40, 80):   # volumen de GtR
        r = EV / "cv_interna" / f"validacion_cruzada_resumen_sin_puntuacion_cv_interna_c{cant}.csv"
        for m in MODELOS:
            cifra(f"B12|gtr{cant}|{m}", r, {"modelo": m, "tecnica": "generate_then_refine"}, "f1_macro_agrupado")
    # 2. Protocolo A (referencia)
    pa = EV / "validacion_cruzada_resumen_sin_puntuacion.csv"
    for m in MODELOS:
        for t in TECNICAS:
            cifra(f"A12|f1|{m}|{t}", pa, {"modelo": m, "tecnica": t}, "f1_macro_agrupado")
    # 3. Referencias simples y n-gramas
    for r in ("mayoria", "vecino_lexico", "vecino_lexico_jaccard"):
        cifra(f"ref|{r}", DIAG / "referencias_simples.csv", {"referencia": r}, "f1_macro_agrupado")
    S.append(("ref|ngramas", DIAG / "ngramas_caracteres.csv", {}, "f1_macro_agrupado"))
    for m in MODELOS:
        cifra(f"diag|fragmentacion|{m}", DIAG / "fragmentacion.csv", {"modelo": m}, "subpalabras_por_palabra")
    # 4. Escenarios DES e interrogacion (baseline sin aumento de cada uno)
    for esc in ("cv_interna_sin_DES", "cv_interna_corpus_shiwilu_propuesta_des", "cv_interna_con_interrogacion",
                "cv_interna_con_interrogacion_sin_DES", "cv_interna_con_interrogacion_corpus_shiwilu_propuesta_des"):
        for m in MODELOS:
            cifra(f"escenario|{esc}|{m}", EV / esc / PRE, {"modelo": m}, "f1_macro_agrupado")
    # 5. Curva por regimen de datos reales
    rr = EV / "curva_regimen" / "resumen_por_regimen.csv"
    rd = EV / "curva_regimen" / "comparacion_pareada.csv"
    for reg in (10, 25, 50, 80):
        for m in MODELOS:
            for t in TECNICAS:
                cifra(f"regimen|f1|{reg}|{m}|{t}", rr, {"regimen": reg, "modelo": m, "tecnica": t}, "f1_macro")
            cifra(f"regimen|dif_gtr|{reg}|{m}", rd, {"regimen": reg, "modelo": m, "tecnica": "generate_then_refine"}, "dif_vs_sin_aumento")
    # 6. Diagnosticos de por que no mejora / se estanca
    for m in MODELOS:
        for fr in (0.25, 0.5, 0.75, 1.0):
            cifra(f"diag|aprendizaje|{m}|{fr}", DIAG / "A_curva_aprendizaje.csv", {"modelo": m, "fraccion_de_reales": fr}, "f1_macro")
        for t in ("generate_then_refine", "retrotraduccion", "mixup"):
            cifra(f"diag|utilidad_B1|{m}|{t}", DIAG / "B_utilidad_sintetico.csv", {"modelo": m, "tecnica": t}, "B1_f1_solo_sinteticos")
            cifra(f"diag|utilidad_B2|{m}|{t}", DIAG / "B_utilidad_sintetico.csv", {"modelo": m, "tecnica": t}, "B2_concordancia_etiqueta")
        for ver in ("bruto", "sin_ruido_etiq", "sin_casi_dup", "ambos"):
            for lotes in (1, 3, 6):
                cifra(f"diag|estancamiento|{m}|{ver}|{lotes}", DIAG / "E_curva_por_lotes.csv", {"modelo": m, "version": ver, "lotes": lotes}, "f1_macro")
    # 7. Variantes de seleccion del sintetico de GtR (mBERT)
    rv = EV / "variantes_gtr" / "resumen_variantes.csv"
    for v in ("ref_sin_aumento", "ref_gtr_completo", "v1_c40_bal", "v2_c40_bal_sim060", "v3_c40_bal_sim060_sinDES", "v4_c20_bal_sim060",
              "v5_c20_AFI_PRG_REQUEST_sim060", "v6_c40_bal_margen", "v7_c40_bal_margen_borde", "v8_c40_bal_sinDES"):
        cifra(f"variantes|{v}", rv, {"variante": v}, "f1_macro")
    # 8. OE3: F1 por pooling y silueta
    fp = OE3 / "f1_por_pooling.csv"
    for m in MODELOS:
        for est in ("cls", "mean_pooling", "max_pooling", "combinacion_capas"):
            cifra(f"oe3|f1_pooling|{m}|{est}", fp, {"modelo": m, "corpus": "sin aumento", "estrategia": est}, "f1_macro")
            cifra(f"oe3|silueta|{m}|{est}", OE3 / "original" / "metricas_intrinsecas.csv", {"modelo": m, "estrategia": est}, "silueta_coseno")
    # 9. R5: analisis del corpus en funcion de los filtros (cuaderno analisis_corpus_y_filtros.ipynb)
    AF = OE2 / "analisis_intrinseco" / "resultados" / "analisis_filtros"
    for cat in ("SAL", "EMO", "PRG", "REQUEST", "AFI", "NEG", "DES"):
        cifra(f"r5|perfil_TTR|{cat}", AF / "perfil_corpus.csv", {"intencion": cat}, "TTR")
        cifra(f"r5|reales_cumplen_filtro_marcador|{cat}", AF / "filtro_marcador_sobre_reales.csv", {"intencion": cat}, "reales_que_cumplirian_el_filtro")
        cifra(f"r5|similitud_mediana_reales|{cat}", AF / "filtro_semantico_sobre_reales.csv", {"intencion": cat}, "mediana")
        cifra(f"r5|generadas_aprobadas_%|{cat}", AF / "filtros_sobre_lo_generado_fuentes.csv", {"intencion": cat}, "aprobadas_%")
    for id_ in ("NEG-01", "NEG-02", "PRG-01", "REQ-01", "EMO-01", "EMO-02", "AFI-01", "AFI-02", "AFI-03"):
        cifra(f"r5|cobertura_marcador|{id_}", AF / "marcadores_en_el_corpus.csv", {"id": id_}, "cobertura_en_su_intencion")
    return S


def leer(archivo, filtros, col):
    if not Path(archivo).exists():
        return None, "falta el archivo"
    d = pd.read_csv(archivo)
    for k, v in filtros.items():
        if k not in d.columns:
            return None, f"falta la columna {k}"
        if d[k].dtype.kind in "fi":
            d = d[(d[k] - float(v)).abs() < 1e-9]
        else:
            d = d[d[k].astype(str) == str(v)]
    if len(d) != 1:
        return None, f"{len(d)} filas coinciden (se esperaba 1)"
    return float(d[col].iloc[0]), ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tolerancia", type=float, default=0.006, help="Diferencia maxima aceptada por cifra (por defecto 0.006).")
    ap.add_argument("--generar-referencia", action="store_true", help="Escribe valores_de_referencia.csv con lo que hay hoy en las salidas.")
    args = ap.parse_args()
    spec = especificaciones()
    actuales = {}
    problemas = []
    for id_, archivo, filtros, col in spec:
        v, err = leer(archivo, filtros, col)
        if v is None:
            problemas.append((id_, err))
        actuales[id_] = v
    if args.generar_referencia:
        if problemas:
            for id_, err in problemas:
                print(f"  NO SE PUDO LEER {id_}: {err}")
            return 1
        pd.DataFrame({"id": list(actuales), "valor": [round(v, 6) for v in actuales.values()]}).to_csv(REFERENCIA, index=False)
        print(f"{len(actuales)} cifras escritas en {REFERENCIA.relative_to(RAIZ)}")
        return 0
    if not REFERENCIA.exists():
        print(f"Falta {REFERENCIA.relative_to(RAIZ)}")
        return 1
    ref = pd.read_csv(REFERENCIA).set_index("id")["valor"].to_dict()
    exactas = cercanas = 0
    distintas = []
    for id_, v in actuales.items():
        if v is None or id_ not in ref:
            continue
        dif = abs(v - ref[id_])
        if dif < 5e-4:
            exactas += 1
        elif dif <= args.tolerancia:
            cercanas += 1
        else:
            distintas.append((id_, ref[id_], v))
    faltan = [id_ for id_ in ref if id_ not in actuales]
    print(f"Cifras de referencia: {len(ref)}")
    print(f"  coinciden (diferencia < 0.0005):        {exactas}")
    print(f"  dentro de la tolerancia (<= {args.tolerancia}):   {cercanas}")
    print(f"  fuera de la tolerancia:                 {len(distintas)}")
    print(f"  no se pudieron leer:                    {len(problemas)}")
    for id_, r, v in distintas:
        print(f"    DIFERENTE {id_}: referencia {r:.4f}, actual {v:.4f}")
    for id_, err in problemas:
        print(f"    NO SE PUDO LEER {id_}: {err}")
    for id_ in faltan:
        print(f"    SIN ESPECIFICACION {id_}")
    ok = not distintas and not problemas and not faltan
    print("RESULTADO:", "REPRODUCCION CORRECTA" if ok else "HAY DIFERENCIAS (ver arriba)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
