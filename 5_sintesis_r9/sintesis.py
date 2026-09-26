"""
Objetivo 4 (R9): sintesis comparativa final.

No entrena ni mide nada nuevo: cruza dos evaluaciones que ya existen por
separado para responder la pregunta de R9, "cual es la configuracion
optima", desde dos angulos que pueden no coincidir:

  EXTRINSECO (R4-R6, 3_baselines_y_aumento_datos/validacion_cruzada_resumen[_sin_puntuacion].csv)
      Que tan bien clasifica un modelo de embeddings + Regresion Logistica,
      con o sin cada tecnica de aumento de datos. Metrica: F1 macro sobre las 700
      predicciones de una validacion cruzada de 5 folds.

  INTRINSECO (R7-R8, 4_caracterizacion_embeddings/resultados/<corpus>/
      metricas_intrinsecas.csv)
      Que tan bien separan las categorias los embeddings crudos, sin
      clasificador, con cada una de las 4 estrategias de pooling. Metrica:
      coeficiente de silueta (distancia coseno), sobre el corpus original.

Un modelo puede ganar en uno y perder en el otro (embeddings "bien
organizados" internamente no garantizan ser la mejor entrada para un
clasificador). Este script deja esa comparacion explicita en una tabla y,
si ya se corrio caracterizacion.py --corpus <tecnica> (ver ese script),
tambien reporta si el aumento de datos de R6 mejora o empeora la calidad
intrinseca de los embeddings (pregunta distinta de si mejora el F1).

Entrada:
    3_baselines_y_aumento_datos/validacion_cruzada_resumen.csv
    4_caracterizacion_embeddings/resultados/original/metricas_intrinsecas.csv
    4_caracterizacion_embeddings/resultados/<retrotraduccion|generate_then_refine>/
        metricas_intrinsecas.csv   (opcional, si ya se corrieron)
Salida:
    5_sintesis_r9/resultados/sintesis_extrinseco_vs_intrinseco.csv
    5_sintesis_r9/resultados/efecto_aumento_en_intrinseco.csv   (si aplica)

Uso:
    python 5_sintesis_r9/sintesis.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import _ruta_raiz  # noqa: F401  (deja importable el paquete `shiwilu`)

from shiwilu.rutas import CARACTERIZACION_RESULTADOS, FASE_AUMENTO, SINTESIS_RESULTADOS  # noqa: E402

# El analisis intrinseco (OE3) se calcula SIEMPRE sobre texto normalizado (minusculas, sin
# puntuacion); por eso la comparacion principal cruza contra el F1 de la validacion
# cruzada en esa misma condicion. El F1 con el texto crudo se reporta solo como referencia.
CONDICIONES_EXTRINSECAS = {
    "sin_puntuacion": ("validacion_cruzada_resumen_sin_puntuacion.csv", "sintesis_extrinseco_vs_intrinseco.csv"),
    "con_puntuacion": ("validacion_cruzada_resumen.csv", "sintesis_extrinseco_vs_intrinseco_con_puntuacion.csv"),
}
TECNICAS_TEXTO = ["retrotraduccion", "generate_then_refine"]


def cargar_extrinseco(nombre_csv: str) -> pd.DataFrame:
    """F1 macro agrupado (700 predicciones) de la validacion cruzada de 5 folds."""
    ruta = FASE_AUMENTO / nombre_csv
    if not ruta.exists():
        raise SystemExit(f"No se encontro {ruta}. Corre primero "
                          "3_baselines_y_aumento_datos/validacion_cruzada.py [--sin-puntuacion]")
    cv = pd.read_csv(ruta)
    cv = cv[cv["modelo"] != "-"]   # quita los baselines triviales
    return cv.rename(columns={"f1_macro_agrupado": "f1_macro"})[["modelo", "tecnica", "f1_macro"]]


def cargar_intrinseco(corpus: str) -> pd.DataFrame | None:
    ruta = CARACTERIZACION_RESULTADOS / corpus / "metricas_intrinsecas.csv"
    if not ruta.exists():
        return None
    return pd.read_csv(ruta)


def construir_sintesis(extrinseco: pd.DataFrame, intrinseco: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for modelo in sorted(extrinseco["modelo"].unique()):
        exp_modelo = extrinseco[extrinseco["modelo"] == modelo]
        mejor_exp = exp_modelo.loc[exp_modelo["f1_macro"].idxmax()]
        sin_aumento = exp_modelo[exp_modelo["tecnica"] == "sin_aumento"]
        f1_sin_aumento = float(sin_aumento["f1_macro"].iloc[0]) if len(sin_aumento) else float("nan")

        int_modelo = intrinseco[intrinseco["modelo"] == modelo]
        mejor_int = int_modelo.loc[int_modelo["silueta_coseno"].idxmax()]

        filas.append({
            "modelo": modelo,
            "f1_macro_sin_aumento": f1_sin_aumento,
            "mejor_tecnica_extrinseca": mejor_exp["tecnica"],
            "f1_macro_mejor_config": float(mejor_exp["f1_macro"]),
            "mejor_estrategia_intrinseca": mejor_int["estrategia"],
            "silueta_coseno_mejor_estrategia": float(mejor_int["silueta_coseno"]),
        })

    tabla = pd.DataFrame(filas)
    tabla["rank_extrinseco"] = tabla["f1_macro_mejor_config"].rank(ascending=False, method="min").astype(int)
    tabla["rank_intrinseco"] = tabla["silueta_coseno_mejor_estrategia"].rank(ascending=False, method="min").astype(int)
    return tabla.sort_values("rank_extrinseco")


def construir_efecto_aumento(modelos: list[str]) -> pd.DataFrame:
    """Compara la silueta (mejor estrategia) del corpus original vs. cada
    corpus aumentado por tecnicas de TEXTO, por modelo. Mixup queda fuera:
    no genera oraciones (ver docstring de caracterizacion.py)."""
    original = cargar_intrinseco("original")
    filas = []
    for tecnica in TECNICAS_TEXTO:
        aumentado = cargar_intrinseco(tecnica)
        if aumentado is None:
            continue
        for modelo in modelos:
            silueta_orig = original[original["modelo"] == modelo]["silueta_coseno"].max()
            silueta_aum = aumentado[aumentado["modelo"] == modelo]["silueta_coseno"].max()
            filas.append({
                "modelo": modelo,
                "tecnica_aumento": tecnica,
                "silueta_original": silueta_orig,
                "silueta_aumentado": silueta_aum,
                "delta": silueta_aum - silueta_orig,
            })
    return pd.DataFrame(filas)


def informe(extrinseco: pd.DataFrame, intrinseco: pd.DataFrame, nombre_salida: str, titulo: str) -> None:
    tabla = construir_sintesis(extrinseco, intrinseco)
    salida = SINTESIS_RESULTADOS / nombre_salida
    tabla.to_csv(salida, index=False, encoding="utf-8")

    print(f"=== R9 ({titulo}): extrinseco (clasificacion) vs. intrinseco (agrupamiento), por modelo ===")
    print(tabla.to_string(index=False))
    print(f"Guardado en {salida}")

    ganador_extrinseco = tabla.loc[tabla["rank_extrinseco"] == 1, "modelo"].iloc[0]
    ganador_intrinseco = tabla.loc[tabla["rank_intrinseco"] == 1, "modelo"].iloc[0]
    if ganador_extrinseco == ganador_intrinseco:
        print(f"  '{ganador_extrinseco}' gana en AMBOS criterios (mejor F1 y mejor silueta).\n")
    else:
        print(f"  Los criterios DIVERGEN: extrinseco '{ganador_extrinseco}', intrinseco '{ganador_intrinseco}'.\n")


def main() -> int:
    SINTESIS_RESULTADOS.mkdir(parents=True, exist_ok=True)

    intrinseco = cargar_intrinseco("original")
    if intrinseco is None:
        raise SystemExit(
            "No se encontro metricas_intrinsecas.csv para el corpus original. "
            "Corre primero: python 4_caracterizacion_embeddings/caracterizacion.py"
        )

    modelos = None
    for condicion, (csv_extrinseco, csv_salida) in CONDICIONES_EXTRINSECAS.items():
        extrinseco = cargar_extrinseco(csv_extrinseco)
        modelos = sorted(extrinseco["modelo"].unique())
        informe(extrinseco, intrinseco, csv_salida, condicion.replace("_", " "))

    efecto = construir_efecto_aumento(modelos)
    if not efecto.empty:
        salida_efecto = SINTESIS_RESULTADOS / "efecto_aumento_en_intrinseco.csv"
        efecto.to_csv(salida_efecto, index=False, encoding="utf-8")
        print("=== Efecto del aumento de datos (R6) sobre la calidad intrinseca (R7-R8) ===")
        print("(silueta coseno, mejor estrategia de pooling por modelo; delta = aumentado - original)")
        print(efecto.to_string(index=False))
        print(f"\nGuardado en {salida_efecto}")
    else:
        print("(No se encontraron corpus aumentados ya caracterizados: corre "
              "'caracterizacion.py --corpus retrotraduccion' y/o '--corpus generate_then_refine'.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
