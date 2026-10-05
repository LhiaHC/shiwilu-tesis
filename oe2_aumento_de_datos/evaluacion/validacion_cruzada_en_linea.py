"""
Validacion cruzada con generacion de datos sinteticos EN LINEA (dentro de cada fold).

`validacion_cruzada.py` (la evaluacion vigente) lee datos sinteticos ya generados. Este
script hace lo mismo que ese pero GENERA el aumento dentro del propio fold, con la misma
estrategia pool / test / core / dev que el entrenamiento:

  test  = el fold; no se toca hasta el final (ni para generar ni para filtrar).
  pool  = los otros 4 folds; se parte en dev interno (~1/6) y core.
  ETAPA CORE  (elegir C): el aumento se genera/filtra SOLO con el core, y se mide en el dev.
  ETAPA POOL  (modelo final): el aumento se genera/filtra con TODO el pool y se mide en el test.

Asi el dev interno nunca influye en el aumento con el que se elige C (en la evaluacion
vigente, Generate-then-Refine se genera una vez con el pool y se usa en las dos etapas).

  mixup                 vectores interpolados con el core (etapa core) y con el pool (etapa pool)
  retrotraduccion       parafrasis (Helsinki-NLP) + traduccion (NMT de F. Prado) de las
                        oraciones del pool, una vez por fold; como cada fila depende solo de
                        su oracion de origen, la etapa core usa las filas cuyo origen esta en
                        el core (equivale a generar solo desde el core). Los filtros
                        (idioma + semantico con el centroide LaBSE) se aplican en cada etapa
                        con el centroide de SU conjunto de entrenamiento.
  generate_then_refine  dos generaciones con Claude por fold: una con el core y otra con el
                        pool (cada una con sus ejemplos few-shot y su centroide)

En cualquier tecnica se descartan las filas sinteticas identicas a una oracion del test.
Mas datos sinteticos: --cantidad N (Generate-then-Refine, en lotes de 20 por categoria y etapa) y
--multiplicador K (Retrotraduccion, K parafrasis por oracion). Cada lote/sorteo se guarda aparte en la cache y
el 0 es el de siempre, asi que subir de nivel solo genera lo que falta; se quitan repetidas y copias. Cada corrida
tambien guarda volumen_sintetico_<...>.csv (sinteticos usados y su % respecto de las oraciones reales).
Lo generado se guarda en una cache por (tecnica, fold, etapa): si se corta la corrida (o se
repite), se reutiliza en vez de volver a generar (y a gastar tokens/GPU).

Entrada: corpus, particiones/folds_fijos.csv, evaluacion/resultados/validacion_cruzada_predicciones_sin_puntuacion.csv
Salida:  evaluacion/resultados/validacion_cruzada_{predicciones,resumen}_sin_puntuacion_en_linea_<tecnica>[_<etiqueta>].csv
         Con los 3 modelos y los 5 folds, estos CSV traen LOS 12 EXPERIMENTOS con el mismo formato que los vigentes
         (con columnas prob_<categoria>): la tecnica corrida sale de esta corrida y las otras tres, de los resultados
         vigentes. Asi se pueden pasar directo a comparacion_pareada.py y curvas_roc.py (--entrada / --etiqueta).
         En una corrida parcial (--modelos o --folds) solo traen la tecnica corrida.
         tecnicas_aumento/salidas/en_linea/<tecnica>/fold<N>_{core,pool}.csv   (cache de lo generado)

Uso (desde la raiz del repositorio):
    python oe2_aumento_de_datos/evaluacion/validacion_cruzada_en_linea.py --tecnica mixup
    python oe2_aumento_de_datos/evaluacion/validacion_cruzada_en_linea.py --tecnica generate_then_refine   # requiere ANTHROPIC_API_KEY
    python oe2_aumento_de_datos/evaluacion/validacion_cruzada_en_linea.py --tecnica retrotraduccion --checkpoint ruta/al/checkpoint   # GPU y repo de F. Prado
Los cuadernos de tecnicas_aumento/colab/ lo corren en Colab.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _ruta_raiz  # noqa: F401  (deja importable el paquete `shiwilu`)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tecnicas_aumento"))

import validacion_cruzada as vc  # noqa: E402
from mixup import generar_sinteticos_mixup  # noqa: E402

from shiwilu.clasificacion import (  # noqa: E402
    MODELOS,
    SEMILLA,
    _normalizar_estricto,
    cargar_corpus,
    cargar_corpus_normalizado,
    cargar_folds,
    extraer_embeddings,
)
from shiwilu.rutas import AUMENTO_SALIDA, EVALUACION_RESULTADOS, NMT_REPO_EXTERNO  # noqa: E402

TECNICAS = ["mixup", "retrotraduccion", "generate_then_refine"]
POR_LLAMADA = 20   # oraciones que se le piden a Claude por categoria en cada llamada (un "lote")


def generar_retro(corpus: pd.DataFrame, idx: np.ndarray, fold: int, checkpoint: Path, repo_nmt: Path,
                  sorteo: int = 0) -> pd.DataFrame:
    """Parafrasea (Helsinki-NLP) y traduce al shiwilu (NMT de F. Prado) las oraciones `idx` del corpus.
    `sorteo` k >= 1 repite la parafrasis con otra semilla (otra muestra del muestreo); 0 es la original."""
    import retrotraduccion as rt

    sub = corpus.iloc[idx]
    parafrasis = rt.parafrasear_espanol(sub["espanol"].astype(str).tolist(), semilla=SEMILLA + fold + 1000 * sorteo)
    df = pd.DataFrame({
        "id_origen": sub.index.to_numpy(),
        "espanol_original": sub["espanol"].astype(str).to_numpy(),
        "espanol": parafrasis,
        "intencion": sub["intencion"].to_numpy(),
    })
    df = df[df["espanol"].str.strip().str.lower() != df["espanol_original"].str.strip().str.lower()].copy()
    df["shiwilu"] = rt.traducir_a_shiwilu(df["espanol"].tolist(), checkpoint, repo_nmt)
    df["fuente"] = "retrotraduccion"
    return df.reset_index(drop=True)


def generar_gtr(corpus: pd.DataFrame, idx: np.ndarray, cantidad: int, lote: int = 0, categorias: list[str] | None = None) -> pd.DataFrame:
    """Claude genera y refina `cantidad` oraciones por categoria usando como train SOLO las oraciones `idx`
    del corpus. `lote` >= 1 rota los ejemplos few-shot para obtener oraciones distintas a las del lote 0."""
    import anthropic
    import generate_then_refine as gtr

    clave = os.environ.get("ANTHROPIC_API_KEY", "")
    if not clave:
        raise SystemExit("Falta ANTHROPIC_API_KEY en el entorno (en Colab: Secretos).")
    extra = {"categorias": categorias} if categorias else {}   # por defecto, todas las intenciones
    return gtr.generar_para_train(corpus.iloc[idx], cantidad, anthropic.Anthropic(api_key=clave), lote=lote, **extra)


def combinar_con_vigente(P_nuevo: pd.DataFrame, tecnica: str) -> pd.DataFrame:
    """Los 12 experimentos: los de la evaluacion vigente, con las filas de `tecnica` reemplazadas por las nuevas."""
    vigente = pd.read_csv(EVALUACION_RESULTADOS / "validacion_cruzada_predicciones_sin_puntuacion.csv")
    return pd.concat([vigente[vigente["tecnica"] != tecnica], P_nuevo], ignore_index=True)


def combinar_sin_duplicados(partes: list[pd.DataFrame], claves_excluir: set) -> pd.DataFrame:
    """Une varios lotes/sorteos y quita las filas repetidas (misma categoria y mismo texto normalizado) y las
    que copian una oracion de `claves_excluir`. Solo se usa cuando hay mas de un lote o sorteo."""
    g = pd.concat(partes, ignore_index=True)
    claves = g["shiwilu"].map(_normalizar_estricto)
    repetida = pd.Series(list(zip(g["intencion"], claves))).duplicated().to_numpy()
    copia = claves.isin(claves_excluir).to_numpy()
    print(f"  al unir {len(partes)} partes ({len(g)} filas) se descartan {int(repetida.sum())} repetidas y {int((copia & ~repetida).sum())} copias de oraciones reales")
    g = g[~(repetida | copia)].reset_index(drop=True)
    if "id" in g.columns:
        g["id"] = [f"GTR_{i:04d}" for i in range(len(g))]
    return g


def con_cache(ruta: Path, generar) -> pd.DataFrame:
    if ruta.exists():
        print(f"  [cache] se reutiliza {ruta.name}")
        return pd.read_csv(ruta)
    df = generar()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=False, encoding="utf-8")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tecnica", choices=TECNICAS, required=True)
    ap.add_argument("--modelos", nargs="+", choices=list(MODELOS), default=list(MODELOS))
    ap.add_argument("--folds", nargs="+", type=int, default=None, help="Subconjunto de folds (por defecto, los 5).")
    ap.add_argument("--checkpoint", type=Path, default=None, help="(retrotraduccion) checkpoint NLLB+LoRA de F. Prado.")
    ap.add_argument("--repo-nmt", type=Path, default=NMT_REPO_EXTERNO, help="(retrotraduccion) repo clonado de F. Prado.")
    ap.add_argument("--cantidad", type=int, default=20,
                     help="(generate_then_refine) oraciones por categoria y etapa. Se piden en lotes de 20 (una llamada "
                          "por lote, con ejemplos few-shot rotados); 40 = 2 lotes, 80 = 4... El lote 0 es el de siempre "
                          "(fold<N>_<etapa>.csv) y se reutiliza de la cache; los demas, fold<N>_<etapa>_l<j>.csv.")
    ap.add_argument("--multiplicador", type=int, default=1,
                     help="(retrotraduccion) parafrasis por oracion de origen: 1 = la de siempre (fold<N>_pool.csv); "
                          "k >= 2 agrega sorteos extra (fold<N>_pool_s<j>.csv) con otra semilla del muestreo.")
    ap.add_argument("--semilla-aumento", type=int, default=0, help="Se suma a la semilla de Mixup (0 = la vigente).")
    ap.add_argument("--cache", type=Path, default=AUMENTO_SALIDA / "en_linea", help="Carpeta de la cache de lo generado.")
    ap.add_argument("--etiqueta", default="", help="Texto que se agrega al nombre de los CSV de salida.")
    args = ap.parse_args()
    tecnica = args.tecnica
    if tecnica == "retrotraduccion" and args.checkpoint is None:
        raise SystemExit("--checkpoint es obligatorio para retrotraduccion.")
    if args.cantidad < 1 or args.multiplicador < 1:
        raise SystemExit("--cantidad y --multiplicador deben ser >= 1.")

    corpus = cargar_corpus()
    assert corpus.index.equals(pd.RangeIndex(len(corpus))), "se asume indice 0..n-1 (id_origen = posicion)"
    n = len(corpus)
    y = corpus["intencion"].to_numpy()
    folds = cargar_folds(corpus)
    df = corpus.copy()
    df["shiwilu"] = cargar_corpus_normalizado()["shiwilu"].to_numpy()   # texto normalizado: la condicion vigente
    claves = df["shiwilu"].map(_normalizar_estricto).to_numpy()
    tx = vc.CONDICIONES["sin_puntuacion"]
    lista_folds = args.folds if args.folds is not None else sorted(set(folds.tolist()))

    # LaBSE hace falta siempre que haya texto sintetico: es el modelo del filtro semantico
    modelos_emb = list(args.modelos) if tecnica == "mixup" else ["labse"] + [m for m in args.modelos if m != "labse"]
    textos_orig = [tx(t) for t in df["shiwilu"]]
    E_orig = {m: extraer_embeddings(textos_orig, m) for m in modelos_emb}
    print(f"{n} oraciones; folds {lista_folds}; tecnica {tecnica}; modelos {args.modelos}")

    predicciones, volumen = [], []
    for f in lista_folds:
        idx_test, idx_pool, idx_dev, idx_core = vc.particionar_fold(folds, y, claves, f)
        claves_test = np.unique(claves[idx_test])
        print(f"\n=== fold {f}: test={len(idx_test)} pool={len(idx_pool)} (core={len(idx_core)}, dev interno={len(idx_dev)}) ===")
        carpeta_cache = args.cache / tecnica

        if tecnica == "retrotraduccion":
            sorteos = []
            for k in range(args.multiplicador):
                nombre = f"fold{f}_pool.csv" if k == 0 else f"fold{f}_pool_s{k}.csv"
                sorteos.append(con_cache(carpeta_cache / nombre,
                                         lambda k=k: generar_retro(corpus, idx_pool, f, args.checkpoint, args.repo_nmt, k)))
            gen = sorteos[0] if len(sorteos) == 1 else combinar_sin_duplicados(sorteos, set())
            E_gen = {m: extraer_embeddings([tx(t) for t in gen["shiwilu"]], m) for m in modelos_emb}
            copia = np.isin(gen["shiwilu"].map(_normalizar_estricto).to_numpy(), claves_test)
            ok_core = vc.filtrar_catalogo_retro(gen, E_orig["labse"], E_gen["labse"], y, idx_core) & ~copia
            ok_pool = vc.filtrar_catalogo_retro(gen, E_orig["labse"], E_gen["labse"], y, idx_pool) & ~copia
            y_gen = gen["intencion"].to_numpy()
            volumen += [{"fold": f, "etapa": "core", "reales": len(idx_core), "sinteticos": int(ok_core.sum())},
                        {"fold": f, "etapa": "pool", "reales": len(idx_pool), "sinteticos": int(ok_pool.sum())}]
            print(f"  retrotraduccion: {len(gen)} generadas; etapa core usa {int(ok_core.sum())}, etapa pool {int(ok_pool.sum())} "
                  f"({int((copia & (gen['id_origen'].isin(set(idx_pool)).to_numpy())).sum())} descartadas por copiar el test)")
        elif tecnica == "generate_then_refine":
            etapas = {}
            for etapa, idx_etapa in (("core", idx_core), ("pool", idx_pool)):
                por_llamada = min(POR_LLAMADA, args.cantidad)
                partes = []
                for j in range(math.ceil(args.cantidad / por_llamada)):
                    nombre = f"fold{f}_{etapa}.csv" if j == 0 else f"fold{f}_{etapa}_l{j}.csv"
                    partes.append(con_cache(carpeta_cache / nombre,
                                            lambda j=j, idx_etapa=idx_etapa: generar_gtr(corpus, idx_etapa, por_llamada, j)))
                g = partes[0] if len(partes) == 1 else combinar_sin_duplicados(partes, set(claves[idx_etapa]))
                aprobado = g[g["estado_filtro"] == "aprobado"].reset_index(drop=True)
                copia = np.isin(aprobado["shiwilu"].map(_normalizar_estricto).to_numpy(), claves_test)
                aprobado = aprobado[~copia].reset_index(drop=True)
                etapas[etapa] = (aprobado, {m: extraer_embeddings([tx(t) for t in aprobado["shiwilu"]], m) for m in modelos_emb})
                volumen.append({"fold": f, "etapa": etapa, "reales": len(idx_etapa), "sinteticos": len(aprobado)})
                print(f"  generate_then_refine etapa {etapa}: {len(g)} generadas, {len(aprobado)} aprobadas y sin copiar el test")

        for modelo in args.modelos:
            E = E_orig[modelo]
            rng = np.random.default_rng(SEMILLA + f + args.semilla_aumento)
            if tecnica == "mixup":
                aug_core = generar_sinteticos_mixup(E[idx_core], y[idx_core], vc.ALPHA_MIXUP, vc.MULT_MIXUP, rng)
                aug_pool = generar_sinteticos_mixup(E[idx_pool], y[idx_pool], vc.ALPHA_MIXUP, vc.MULT_MIXUP, rng)
            elif tecnica == "retrotraduccion":
                aug_core = (E_gen[modelo][ok_core], y_gen[ok_core])
                aug_pool = (E_gen[modelo][ok_pool], y_gen[ok_pool])
            else:
                (g_core, E_core), (g_pool, E_pool) = etapas["core"], etapas["pool"]
                aug_core = (E_core[modelo], g_core["intencion"].to_numpy())
                aug_pool = (E_pool[modelo], g_pool["intencion"].to_numpy())

            X_core = np.concatenate([E[idx_core], aug_core[0]]); y_core = np.concatenate([y[idx_core], aug_core[1]])
            X_pool = np.concatenate([E[idx_pool], aug_pool[0]]); y_pool = np.concatenate([y[idx_pool], aug_pool[1]])
            pred, proba, clases, c = vc.ajustar_y_predecir(X_core, y_core, E[idx_dev], y[idx_dev], X_pool, y_pool, E[idx_test])
            for pos_i, real_i, pred_i, proba_i in zip(idx_test, y[idx_test], pred, proba):
                fila = {"modelo": modelo, "tecnica": tecnica, "fold": f, "pos": int(pos_i),
                        "real": real_i, "prediccion": pred_i, "C": c}
                fila.update({f"prob_{clase}": p for clase, p in zip(clases, proba_i)})
                predicciones.append(fila)
            print(f"  {modelo}: C elegido = {c}")

    P = pd.DataFrame(predicciones)
    if set(args.modelos) == set(MODELOS) and set(lista_folds) == set(folds.tolist()):
        P = combinar_con_vigente(P, tecnica)
        print(f"\nSe guardan los 12 experimentos: {tecnica} de esta corrida + las otras tres configuraciones vigentes.")
    else:
        print("\nAviso: corrida parcial (--modelos/--folds): los CSV traen solo la tecnica corrida, no los 12 experimentos.")
    sufijo = f"_sin_puntuacion_en_linea_{tecnica}" + (f"_{args.etiqueta}" if args.etiqueta else "")
    EVALUACION_RESULTADOS.mkdir(parents=True, exist_ok=True)
    P.to_csv(EVALUACION_RESULTADOS / f"validacion_cruzada_predicciones{sufijo}.csv", index=False, encoding="utf-8")
    if volumen:
        V = pd.DataFrame(volumen)
        V["pct_de_los_reales"] = (100 * V["sinteticos"] / V["reales"]).round(1)
        V.to_csv(EVALUACION_RESULTADOS / f"volumen_sintetico{sufijo}.csv", index=False, encoding="utf-8")
        print("\nVolumen sintetico medio (% de las oraciones reales de la etapa):",
              V.groupby("etapa")["pct_de_los_reales"].mean().round(1).to_dict())
    R = vc.resumir_predicciones(P)
    R.to_csv(EVALUACION_RESULTADOS / f"validacion_cruzada_resumen{sufijo}.csv", index=False, encoding="utf-8")
    print("\n" + R.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
