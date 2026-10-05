"""
Validacion cruzada con C elegido por VALIDACION CRUZADA INTERNA (en lugar de un dev de ~90 oraciones).

En `validacion_cruzada.py` / `validacion_cruzada_en_linea.py` cada configuracion elige C en un dev
interno de ~90 oraciones (1/6 del pool). Con tan pocas oraciones la eleccion es ruidosa y puede
inclinar la comparacion entre tecnicas unas centesimas. Aqui, dentro de cada fold externo:

  test  = el fold; no se toca hasta el final.
  pool  = los otros 4 folds.
  CV interna (elegir C): el pool se parte en K particiones (por defecto 5; agrupadas por texto
        normalizado y estratificadas). Para cada particion k, el aumento se genera/filtra SOLO con las
        oraciones de entrenamiento de esa particion y se evalua en las oraciones retenidas. Se elige el
        C con mayor F1 macro de las predicciones retenidas de todo el pool (~560 oraciones, no ~90).
  modelo final: se entrena con el pool + su aumento (generado con TODO el pool) y el C elegido, y
        predice el test.

Asi las oraciones retenidas nunca influyen en el aumento con el que se elige C (igual que la etapa
"core" de la evaluacion en linea, pero con K particiones en vez de una). Cada configuracion
(incluido "sin aumento") elige su propio C con la misma regla; la grilla es tambien la misma:
C in {0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30}.

  sin_aumento           solo oraciones reales
  mixup                 interpolacion de embeddings dentro de cada categoria (100% de las reales)
  retrotraduccion       catalogo de paráfrasis+traduccion (cache fold<N>_pool.csv); en cada particion
                        se usan las filas cuyo origen esta en el entrenamiento de la particion y los
                        filtros (idioma + semantico) con el centroide LaBSE de ese entrenamiento
  generate_then_refine  Claude genera con las oraciones de entrenamiento de cada particion
                        (cache fold<N>_in<k>[_l<j>].csv) y con todo el pool (fold<N>_pool[_l<j>].csv)

Se descartan siempre las filas sinteticas identicas a una oracion del test o de las retenidas.

Con --solo-generar solo se produce la cache de lo generado (la parte que gasta tokens); sin esa opcion se evalua.

Cache: --cache (como validacion_cruzada_en_linea.py). Si falta un archivo de GtR se genera con la API
(requiere ANTHROPIC_API_KEY, p. ej. en Colab); si no, se reutiliza.

Salida (evaluacion/resultados/cv_interna/):
  validacion_cruzada_predicciones_sin_puntuacion_cv_interna_<tecnica>[_<etiqueta>].csv  (con prob_<categoria>)
  validacion_cruzada_resumen_sin_puntuacion_cv_interna_<tecnica>[_<etiqueta>].csv
  cv_interna_f1_por_c_<tecnica>[_<etiqueta>].csv   (F1 de la CV interna por modelo, fold y C)
  volumen_sintetico_sin_puntuacion_cv_interna_<tecnica>[_<etiqueta>].csv
`unir_predicciones.py` junta las cuatro tecnicas en un solo archivo para comparacion_pareada.py y curvas_roc.py.

Uso (desde la raiz del repositorio):
    python oe2_aumento_de_datos/evaluacion/validacion_cruzada_cv_interna.py --tecnica sin_aumento
    python oe2_aumento_de_datos/evaluacion/validacion_cruzada_cv_interna.py --tecnica mixup
    python oe2_aumento_de_datos/evaluacion/validacion_cruzada_cv_interna.py --tecnica retrotraduccion --checkpoint x --cache <carpeta>
    python oe2_aumento_de_datos/evaluacion/validacion_cruzada_cv_interna.py --tecnica generate_then_refine --cantidad 120 --cache <carpeta>
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

import _ruta_raiz  # noqa: F401  (deja importable el paquete `shiwilu`)

import validacion_cruzada as vc  # noqa: E402
import validacion_cruzada_en_linea as ven  # noqa: E402   (ademas deja importable tecnicas_aumento/)
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
from shiwilu.taxonomia import INTENCIONES  # noqa: E402
from shiwilu.rutas import AUMENTO_SALIDA, CV_INTERNA_RESULTADOS as CV_RESULTADOS_BASE, NMT_REPO_EXTERNO  # noqa: E402

TECNICAS = ["sin_aumento", "mixup", "retrotraduccion", "generate_then_refine"]
VALORES_C_CV = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
K_INTERNO = 5
f1m = vc.f1m


def particiones_internas(idx_pool: np.ndarray, y: np.ndarray, claves: np.ndarray, f: int, k: int):
    """K particiones (entrenamiento, retenidas) del pool, agrupadas por texto normalizado y estratificadas."""
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SEMILLA + f + 500)
    return [(idx_pool[a], idx_pool[b]) for a, b in cv.split(idx_pool, y[idx_pool], groups=claves[idx_pool])]


class Sinteticos:
    """Datos sinteticos de un fold para cualquier subconjunto de entrenamiento (`para`)."""

    def __init__(self, tecnica, args, corpus, y, claves, E_orig, modelos_emb, tx, f, mapa_pos=None):
        self.tecnica, self.args, self.corpus, self.y, self.claves = tecnica, args, corpus, y, claves
        self.mapa_pos = mapa_pos
        self.E_orig, self.modelos_emb, self.tx, self.f = E_orig, modelos_emb, tx, f
        self.carpeta = args.cache / tecnica
        self.retro = None

    # -- retrotraduccion: catalogo del fold, una sola vez
    def cargar_retro(self, idx_pool):
        sorteos = []
        for k in range(self.args.multiplicador):
            nombre = f"fold{self.f}_pool.csv" if k == 0 else f"fold{self.f}_pool_s{k}.csv"
            sorteos.append(ven.con_cache(self.carpeta / nombre,
                                         lambda k=k: ven.generar_retro(self.corpus, idx_pool, self.f, self.args.checkpoint,
                                                                       self.args.repo_nmt, k)))
        gen = sorteos[0] if len(sorteos) == 1 else ven.combinar_sin_duplicados(sorteos, set())
        if self.mapa_pos is not None:   # corpus sin algunas categorias: se descartan sus filas y se re-indexa el origen
            gen = gen[~gen["intencion"].isin(self.args.excluir_categorias)].copy()
            gen["id_origen"] = gen["id_origen"].map(self.mapa_pos)
            assert gen["id_origen"].notna().all(), "filas del catalogo con origen excluido"
            gen["id_origen"] = gen["id_origen"].astype(int)
            gen = gen.reset_index(drop=True)
        E_gen = {m: extraer_embeddings([self.tx(t) for t in gen["shiwilu"]], m) for m in self.modelos_emb}
        self.retro = (gen, E_gen, gen["shiwilu"].map(_normalizar_estricto).to_numpy(), gen["intencion"].to_numpy())

    def para(self, idx_train: np.ndarray, nombre: str, excluir: set, orden: int, modelos: list[str]):
        """{modelo: (X_sintetico, y_sintetico)} generado con las oraciones `idx_train`; se descartan las filas
        cuyo texto normalizado esta en `excluir` (test y/o retenidas)."""
        y = self.y
        if self.tecnica == "sin_aumento":
            return {m: (np.zeros((0, self.E_orig[m].shape[1])), np.array([], dtype=object)) for m in modelos}
        if self.tecnica == "mixup":
            salida = {}
            for m in modelos:
                rng = np.random.default_rng(SEMILLA + self.f + self.args.semilla_aumento + 7919 * orden)
                salida[m] = generar_sinteticos_mixup(self.E_orig[m][idx_train], y[idx_train], vc.ALPHA_MIXUP, vc.MULT_MIXUP, rng)
            return salida
        if self.tecnica == "retrotraduccion":
            gen, E_gen, claves_gen, y_gen = self.retro
            ok = vc.filtrar_catalogo_retro(gen, self.E_orig["labse"], E_gen["labse"], y, idx_train) & ~np.isin(claves_gen, list(excluir))
            return {m: (E_gen[m][ok], y_gen[ok]) for m in modelos}
        # generate_then_refine
        por_llamada = min(ven.POR_LLAMADA, self.args.cantidad)
        categorias = [c for c in INTENCIONES if c in set(self.y)]   # sin las categorias excluidas
        partes = []
        for j in range(math.ceil(self.args.cantidad / por_llamada)):
            archivo = f"{nombre}.csv" if j == 0 else f"{nombre}_l{j}.csv"
            partes.append(ven.con_cache(self.carpeta / archivo, lambda j=j: ven.generar_gtr(self.corpus, idx_train, por_llamada, j, categorias)))
        partes = [x[x["intencion"].isin(categorias)].reset_index(drop=True) for x in partes]   # la pool sembrada puede traer categorias excluidas
        g = partes[0] if len(partes) == 1 else ven.combinar_sin_duplicados(partes, set(self.claves[idx_train]))
        aprobado = g[g["estado_filtro"] == "aprobado"].reset_index(drop=True)
        copia = np.isin(aprobado["shiwilu"].map(_normalizar_estricto).to_numpy(), list(excluir))
        aprobado = aprobado[~copia].reset_index(drop=True)
        return {m: (extraer_embeddings([self.tx(t) for t in aprobado["shiwilu"]], m), aprobado["intencion"].to_numpy())
                for m in modelos}


def elegir_c_por_cv(E, y, idx_pool, particiones, sint_inner, modelo):
    """F1 macro (predicciones retenidas de todo el pool) por C, para una configuracion y un modelo."""
    pos = {int(i): n for n, i in enumerate(idx_pool)}
    pred = {c: np.empty(len(idx_pool), dtype=object) for c in VALORES_C_CV}
    for (idx_tr, idx_va), aug in zip(particiones, sint_inner):
        X_aug, y_aug = aug[modelo]
        X = np.concatenate([E[idx_tr], X_aug]) if len(X_aug) else E[idx_tr]
        yy = np.concatenate([y[idx_tr], y_aug]) if len(y_aug) else y[idx_tr]
        for c in VALORES_C_CV:
            clf = LogisticRegression(C=c, max_iter=2000, random_state=SEMILLA).fit(X, yy)
            pred[c][[pos[int(i)] for i in idx_va]] = clf.predict(E[idx_va])
    f1 = {c: f1m(y[idx_pool], pred[c]) for c in VALORES_C_CV}
    mejor = max(VALORES_C_CV, key=lambda c: (f1[c], -c))   # empate: el C mas pequeño (mas regularizado)
    return mejor, f1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tecnica", choices=TECNICAS, required=True)
    ap.add_argument("--modelos", nargs="+", choices=list(MODELOS), default=list(MODELOS))
    ap.add_argument("--folds", nargs="+", type=int, default=None, help="Subconjunto de folds (por defecto, los 5).")
    ap.add_argument("--k-interno", type=int, default=K_INTERNO, help="Particiones de la CV interna.")
    ap.add_argument("--checkpoint", type=Path, default=None, help="(retrotraduccion) checkpoint NLLB+LoRA de F. Prado.")
    ap.add_argument("--repo-nmt", type=Path, default=NMT_REPO_EXTERNO)
    ap.add_argument("--cantidad", type=int, default=20, help="(generate_then_refine) oraciones por categoria (lotes de 20).")
    ap.add_argument("--multiplicador", type=int, default=1, help="(retrotraduccion) parafrasis por oracion.")
    ap.add_argument("--semilla-aumento", type=int, default=0, help="Se suma a la semilla de Mixup.")
    ap.add_argument("--cache", type=Path, default=AUMENTO_SALIDA / "en_linea_marcadores_fuentes",
                    help="Carpeta de la cache de lo generado (por defecto, la de GtR ya refiltrada con los marcadores de las fuentes).")
    ap.add_argument("--etiqueta", default="", help="Texto que se agrega al nombre de los CSV de salida.")
    ap.add_argument("--condicion", choices=["sin_puntuacion", "con_interrogacion"], default="sin_puntuacion",
                    help="Texto que ven los modelos: `sin_puntuacion` (vigente) o `con_interrogacion` (conserva ¿ ?). "
                         "La segunda escribe en evaluacion/resultados/cv_interna_con_interrogacion/.")
    ap.add_argument("--corpus", type=Path, default=None,
                    help="CSV de un corpus alternativo con los MISMOS textos y orden (p. ej. con etiquetas corregidas). "
                         "Escribe en una carpeta aparte (cv_interna_<condicion>_<nombre del archivo>). Solo sin_aumento y mixup: "
                         "las caches de Retrotraduccion y GtR llevan las etiquetas del corpus original.")
    ap.add_argument("--excluir-categorias", nargs="+", default=[], metavar="CAT",
                    help="Quita del corpus todas las oraciones de esas categorias (p. ej. DES) antes de evaluar. Sirve con sin_aumento, mixup y "
                         "retrotraduccion (el catalogo se filtra y se re-indexa) y con generate_then_refine (solo se generan las demas categorias; "
                         "las particiones internas se calculan con el corpus ya reducido, asi que la cache es PROPIA de esta configuracion).")
    ap.add_argument("--solo-generar", action="store_true",
                    help="(generate_then_refine) solo genera y guarda en la cache lo que falta (particiones internas y pool), "
                         "sin entrenar ni escribir resultados. Sirve para separar la parte que gasta tokens y retomarla si se corta.")
    args = ap.parse_args()
    tecnica = args.tecnica
    if args.solo_generar:
        if tecnica != "generate_then_refine":
            raise SystemExit("--solo-generar solo aplica a generate_then_refine.")
        args.modelos = ["labse"]   # el filtro semantico usa LaBSE; no hace falta cargar los otros dos modelos
    if tecnica == "retrotraduccion" and args.checkpoint is None:
        raise SystemExit("--checkpoint es obligatorio para retrotraduccion (no se usa si el catalogo ya esta en la cache).")
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    if args.corpus and tecnica in ("retrotraduccion", "generate_then_refine"):
        raise SystemExit("--corpus solo se puede usar con sin_aumento y mixup: las caches de retrotraduccion y GtR llevan las etiquetas originales.")
    corpus = cargar_corpus(args.corpus) if args.corpus else cargar_corpus()
    mantener = ~corpus["intencion"].isin(args.excluir_categorias).to_numpy()
    mapa_pos = {int(a): n for n, a in enumerate(np.where(mantener)[0])} if args.excluir_categorias else None   # posicion original -> nueva
    corpus = corpus[mantener].reset_index(drop=True)
    assert corpus.index.equals(pd.RangeIndex(len(corpus))), "se asume indice 0..n-1 (id_origen = posicion)"
    y = corpus["intencion"].to_numpy()
    folds = cargar_folds(corpus)
    df = corpus.copy()
    if args.condicion == "sin_puntuacion":
        df["shiwilu"] = cargar_corpus_normalizado()["shiwilu"].to_numpy()[mantener]
    # con_interrogacion parte del texto del corpus tal cual (minusculas, con signos): la condicion decide que se conserva
    claves = df["shiwilu"].map(_normalizar_estricto).to_numpy()
    tx = vc.CONDICIONES[args.condicion]
    nombre_carpeta = "cv_interna" + ("" if args.condicion == "sin_puntuacion" else f"_{args.condicion}") \
        + (f"_{args.corpus.stem}" if args.corpus else "")         + (f"_sin_{'_'.join(args.excluir_categorias)}" if args.excluir_categorias else "")
    CV_INTERNA_RESULTADOS = CV_RESULTADOS_BASE if nombre_carpeta == "cv_interna" else CV_RESULTADOS_BASE.parent / nombre_carpeta
    lista_folds = args.folds if args.folds is not None else sorted(set(folds.tolist()))
    modelos_emb = list(args.modelos) if tecnica in ("sin_aumento", "mixup") else ["labse"] + [m for m in args.modelos if m != "labse"]
    E_orig = {m: extraer_embeddings([tx(t) for t in df["shiwilu"]], m) for m in modelos_emb}
    print(f"{len(df)} oraciones; folds {lista_folds}; tecnica {tecnica}; modelos {args.modelos}; CV interna K={args.k_interno}")

    predicciones, f1_cv, volumen = [], [], []
    for f in lista_folds:
        idx_test = np.where(folds == f)[0]
        idx_pool = np.where(folds != f)[0]
        claves_test = set(claves[idx_test])
        particiones = particiones_internas(idx_pool, y, claves, f, args.k_interno)
        print(f"\n=== fold {f}: test={len(idx_test)} pool={len(idx_pool)}; particiones internas {[len(b) for _, b in particiones]} ===")
        S = Sinteticos(tecnica, args, corpus, y, claves, E_orig, modelos_emb, tx, f, mapa_pos)
        if tecnica == "retrotraduccion":
            S.cargar_retro(idx_pool)
            gen = S.retro[0]
            print(f"  retrotraduccion: catalogo de {len(gen)} filas")

        sint_inner = [S.para(idx_tr, f"fold{f}_in{k}", claves_test | set(claves[idx_va]), k + 1, args.modelos)
                      for k, (idx_tr, idx_va) in enumerate(particiones)]
        sint_pool = S.para(idx_pool, f"fold{f}_pool", claves_test, 0, args.modelos)
        n_pool = len(next(iter(sint_pool.values()))[1])
        volumen.append({"fold": f, "reales": len(idx_pool), "sinteticos": n_pool})
        print(f"  sinteticos con todo el pool: {n_pool} ({100 * n_pool / len(idx_pool):.1f}% de las reales); "
              f"por particion interna: {[len(next(iter(a.values()))[1]) for a in sint_inner]}")
        if args.solo_generar:
            continue

        for modelo in args.modelos:
            E = E_orig[modelo]
            c, f1 = elegir_c_por_cv(E, y, idx_pool, particiones, sint_inner, modelo)
            for cc, v in f1.items():
                f1_cv.append({"modelo": modelo, "fold": f, "C": cc, "f1_cv_interna": v, "elegido": cc == c})
            X_aug, y_aug = sint_pool[modelo]
            X_pool = np.concatenate([E[idx_pool], X_aug]) if len(X_aug) else E[idx_pool]
            y_pool = np.concatenate([y[idx_pool], y_aug]) if len(y_aug) else y[idx_pool]
            clf = LogisticRegression(C=c, max_iter=2000, random_state=SEMILLA).fit(X_pool, y_pool)
            pred, proba = clf.predict(E[idx_test]), clf.predict_proba(E[idx_test])
            for pos_i, real_i, pred_i, proba_i in zip(idx_test, y[idx_test], pred, proba):
                fila = {"modelo": modelo, "tecnica": tecnica, "fold": f, "pos": int(pos_i), "real": real_i,
                        "prediccion": pred_i, "C": c}
                fila.update({f"prob_{clase}": p for clase, p in zip(clf.classes_, proba_i)})
                predicciones.append(fila)
            print(f"  {modelo}: C elegido = {c} (F1 CV interna {f1[c]:.3f})")

    if args.solo_generar:
        print("\nGeneracion terminada: todo lo necesario esta en la cache (--cache). Corre ahora la evaluacion sin --solo-generar.")
        return 0
    P = pd.DataFrame(predicciones)
    sufijo = f"_{tecnica}" + (f"_{args.etiqueta}" if args.etiqueta else "")
    CV_INTERNA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    P.to_csv(CV_INTERNA_RESULTADOS / f"validacion_cruzada_predicciones_sin_puntuacion_cv_interna{sufijo}.csv", index=False, encoding="utf-8")
    pd.DataFrame(f1_cv).to_csv(CV_INTERNA_RESULTADOS / f"cv_interna_f1_por_c{sufijo}.csv", index=False, encoding="utf-8")
    V = pd.DataFrame(volumen)
    V["pct_de_los_reales"] = (100 * V["sinteticos"] / V["reales"]).round(1)
    V.to_csv(CV_INTERNA_RESULTADOS / f"volumen_sintetico_sin_puntuacion_cv_interna{sufijo}.csv", index=False, encoding="utf-8")
    R = vc.resumir_predicciones(P)
    R.to_csv(CV_INTERNA_RESULTADOS / f"validacion_cruzada_resumen_sin_puntuacion_cv_interna{sufijo}.csv", index=False, encoding="utf-8")
    if set(lista_folds) != set(folds.tolist()):
        print("\nAviso: corrida parcial (--folds): los resultados no son comparables con los de los 5 folds.")
    print("\n" + R.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
