"""
Funciones compartidas por los baselines de clasificacion (congelado + Regresion
Logistica) y, mas adelante, por los experimentos con tecnicas de aumento.

Cada "baseline" es una combinacion (modelo de embeddings, sin ajuste fino) +
(Regresion Logistica). Todos comparten el mismo corpus, la misma division
train/dev/test y el mismo procedimiento de ajuste/evaluacion — lo unico que
cambia es como se extraen los embeddings.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from shiwilu.rutas import CORPUS_CSV
from shiwilu.taxonomia import INTENCIONES

SEMILLA = 42
PROP_DEV_TEST = 0.30   # 70% train, 15% dev, 15% test
VALORES_C = [0.01, 0.1, 1.0, 3.0, 10.0]   # grilla de ajuste sobre el dev
SPLIT_FIJO_CSV = Path(__file__).resolve().parent / "split_fijo.csv"
FOLDS_FIJOS_CSV = Path(__file__).resolve().parent / "folds_fijos.csv"
N_FOLDS = 5

# Modelos de embeddings soportados: nombre corto -> (id de HuggingFace, tipo)
#   "sentence_transformers": el modelo ya trae su propio pooling de oracion.
#   "mean_pooling":          encoder generico (mBERT, XLM-R); se aplica mean
#                            pooling sobre la ultima capa oculta, ponderado
#                            por la mascara de atencion.
MODELOS = {
    "labse": ("sentence-transformers/LaBSE", "sentence_transformers"),
    "mbert": ("bert-base-multilingual-cased", "mean_pooling"),
    "xlmr":  ("xlm-roberta-base", "mean_pooling"),
}


def cargar_corpus(corpus_csv=CORPUS_CSV) -> pd.DataFrame:
    df = pd.read_csv(corpus_csv)
    faltantes = {"shiwilu", "intencion"} - set(df.columns)
    if faltantes:
        raise SystemExit(f"Al corpus le faltan columnas: {faltantes}")
    return df


_RE_NO_ALFANUMERICO = re.compile(r"[^a-z0-9áéíóúñ’']")


def _normalizar_shiwilu(texto: str) -> str:
    """Normaliza may/min, puntuacion y espacios para detectar cuasi-duplicados.

    El corpus repite la misma raiz shiwilu con variantes de puntuacion o
    espaciado ("PANTE'CHEK" / "�PANTE'CHEK!" / "pante'chek"): son la MISMA
    oracion para efectos de fuga de datos, aunque el texto crudo no coincida
    caracter por caracter.
    """
    return _RE_NO_ALFANUMERICO.sub("", str(texto).lower())


def _normalizar_estricto(texto: str) -> str:
    """Como `_normalizar_shiwilu` pero tambien ignora tildes y la n con tilde
    ("ipa' ñinchitulek" == "ipa' ninchitulek"). Se usa para deduplicar texto sintetico
    contra el test y como criterio de agrupacion de los folds (ver `cargar_folds`).
    """
    sin_marcas = "".join(c for c in unicodedata.normalize("NFD", str(texto).lower())
                         if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9’']", "", sin_marcas)


def quitar_puntuacion(texto: str, minusculas: bool = True) -> str:
    """Deja solo las palabras (con sus apostrofes), separadas por un espacio.

    Con `minusculas=True` (por defecto) tambien pasa todo a minusculas. Ambas
    cosas importan porque son atajos del corpus: los signos `¿?` delatan PRG y
    las oraciones TODAS EN MAYUSCULAS son el 100% de DES, PRG y REQUEST (contra
    10-21% en las demas categorias).
    """
    t = str(texto)
    if minusculas:
        t = t.lower()
    return " ".join(re.findall(r"[^\W_]+(?:'[^\W_]+)*", t))


def _asignar_split(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula el split 70/15/15 por GRUPO de texto shiwilu normalizado.

    Devuelve un DataFrame (shiwilu_norm, split) con una fila por grupo.
    """
    claves = df["shiwilu"].map(_normalizar_shiwilu)
    grupos = (
        pd.DataFrame({"shiwilu_norm": claves, "intencion": df["intencion"]})
        .groupby("shiwilu_norm", as_index=False)["intencion"].first()
    )
    train_g, resto_g = train_test_split(
        grupos, test_size=PROP_DEV_TEST, stratify=grupos["intencion"], random_state=SEMILLA,
    )
    dev_g, test_g = train_test_split(
        resto_g, test_size=0.5, stratify=resto_g["intencion"], random_state=SEMILLA,
    )
    return pd.concat([
        pd.DataFrame({"shiwilu_norm": train_g["shiwilu_norm"], "split": "train"}),
        pd.DataFrame({"shiwilu_norm": dev_g["shiwilu_norm"], "split": "dev"}),
        pd.DataFrame({"shiwilu_norm": test_g["shiwilu_norm"], "split": "test"}),
    ], ignore_index=True)


def dividir_train_dev_test(df: pd.DataFrame):
    """Devuelve (train, dev, test) leyendo el split CONGELADO de `split_fijo.csv`.

    El split se calcula una sola vez (agrupando por texto shiwilu normalizado,
    para que una misma oracion con variantes de mayusculas/puntuacion no caiga
    en dos splits) y se guarda en `split_fijo.csv`. A partir de ahi todos los
    scripts leen ese archivo en vez de recalcularlo.

    Esto importa porque retrotraduccion y Generate-then-Refine generan texto
    sintetico a partir de las oraciones de train: si el split cambiara despues
    (por un cambio de codigo, de version de sklearn, etc.), oraciones que
    fueron train al generar podrian pasar a ser test, y el aumento quedaria
    contaminado. Para recalcular el split a proposito, borrar `split_fijo.csv`
    — y entonces hay que REGENERAR las tecnicas de aumento y re-correr todo.
    """
    if SPLIT_FIJO_CSV.exists():
        asignacion = pd.read_csv(SPLIT_FIJO_CSV, encoding="utf-8", dtype=str, keep_default_na=False)
    else:
        asignacion = _asignar_split(df)
        asignacion.to_csv(SPLIT_FIJO_CSV, index=False, encoding="utf-8")
        print(f"[split] calculado y guardado en {SPLIT_FIJO_CSV}")

    mapa = dict(zip(asignacion["shiwilu_norm"], asignacion["split"]))
    claves = df["shiwilu"].map(_normalizar_shiwilu)
    sin_asignar = claves[~claves.isin(mapa)]
    if len(sin_asignar):
        raise SystemExit(
            f"{len(sin_asignar)} oraciones del corpus no estan en {SPLIT_FIJO_CSV.name} "
            "(el corpus cambio despues de congelar el split). Si fue a proposito, borra "
            "ese archivo, regenera las tecnicas de aumento y re-corre los experimentos."
        )
    split = claves.map(mapa)
    return df[split == "train"], df[split == "dev"], df[split == "test"]


def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


# Los modelos se cargan una sola vez por proceso: recargar LaBSE en cada llamada
# (Generate-then-Refine lo hace varias veces por categoria) acumula memoria y
# termino en un "segmentation fault".
_CACHE_MODELOS: dict = {}


def extraer_embeddings(oraciones: list[str], nombre_modelo: str) -> np.ndarray:
    """Extrae embeddings de oracion, congelados (sin ajuste fino)."""
    if nombre_modelo not in MODELOS:
        raise ValueError(f"Modelo desconocido {nombre_modelo!r}. Opciones: {list(MODELOS)}")
    hf_id, tipo = MODELOS[nombre_modelo]

    if tipo == "sentence_transformers":
        from sentence_transformers import SentenceTransformer

        if hf_id not in _CACHE_MODELOS:
            _CACHE_MODELOS[hf_id] = SentenceTransformer(hf_id)
        return _CACHE_MODELOS[hf_id].encode(oraciones, convert_to_numpy=True, show_progress_bar=True)

    # tipo == "mean_pooling": mBERT / XLM-R, sin cabeza de embedding de oracion
    from transformers import AutoModel, AutoTokenizer

    if hf_id not in _CACHE_MODELOS:
        modelo_hf = AutoModel.from_pretrained(hf_id)
        modelo_hf.eval()
        _CACHE_MODELOS[hf_id] = (AutoTokenizer.from_pretrained(hf_id), modelo_hf)
    tokenizer, modelo = _CACHE_MODELOS[hf_id]

    vectores = []
    batch_size = 16
    with torch.no_grad():
        for inicio in range(0, len(oraciones), batch_size):
            lote = oraciones[inicio : inicio + batch_size]
            enc = tokenizer(lote, padding=True, truncation=True, max_length=64, return_tensors="pt")
            salida = modelo(**enc)
            pooled = _mean_pooling(salida, enc["attention_mask"])
            vectores.append(pooled.numpy())
    return np.concatenate(vectores, axis=0)


def ajustar_clasificador(X_train, y_train, X_dev, y_dev) -> LogisticRegression:
    """Ajusta el hiperparametro C sobre el conjunto de desarrollo (F1 macro)."""
    mejor_c, mejor_f1, mejor_clf = None, -1.0, None
    for c in VALORES_C:
        clf = LogisticRegression(C=c, max_iter=2000, random_state=SEMILLA)
        clf.fit(X_train, y_train)
        f1_dev = f1_score(y_dev, clf.predict(X_dev), average="macro", zero_division=0)
        print(f"  C={c:<6} F1-macro (dev) = {f1_dev:.4f}")
        if f1_dev > mejor_f1:
            mejor_c, mejor_f1, mejor_clf = c, f1_dev, clf
    print(f"Mejor C sobre dev: {mejor_c} (F1-macro={mejor_f1:.4f})")
    return mejor_clf


def evaluar(clf: LogisticRegression, X_test, y_test) -> dict:
    y_pred = clf.predict(X_test)
    metricas = {
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_ponderado": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "exactitud": accuracy_score(y_test, y_pred),
    }
    reporte = classification_report(
        y_test, y_pred, labels=INTENCIONES, output_dict=True, zero_division=0,
    )
    matriz = confusion_matrix(y_test, y_pred, labels=INTENCIONES)
    return {
        "metricas": metricas, "reporte": reporte, "matriz": matriz,
        "y_test": np.asarray(y_test), "y_pred": y_pred,
    }


def guardar_resultados(carpeta, titulo: str, info_extra: dict, resultado: dict) -> None:
    carpeta.mkdir(parents=True, exist_ok=True)
    m = resultado["metricas"]

    with (carpeta / "metricas.json").open("w", encoding="utf-8") as f:
        json.dump({**info_extra, **m}, f, ensure_ascii=False, indent=2)

    pd.DataFrame(resultado["reporte"]).T.to_csv(
        carpeta / "reporte_clasificacion.csv", encoding="utf-8"
    )

    # una fila por oracion de test: insumo para el bootstrap de intervalos
    # de confianza (ver bootstrap_ic.py), sin tener que reentrenar nada.
    pd.DataFrame({"real": resultado["y_test"], "prediccion": resultado["y_pred"]}).to_csv(
        carpeta / "predicciones.csv", index=False, encoding="utf-8"
    )

    disp = ConfusionMatrixDisplay(resultado["matriz"], display_labels=INTENCIONES)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title(titulo)
    fig.tight_layout()
    fig.savefig(carpeta / "matriz_confusion.png", dpi=150)
    plt.close(fig)

    print(f"F1 macro={m['f1_macro']:.4f}  F1 ponderado={m['f1_ponderado']:.4f}  "
          f"Exactitud={m['exactitud']:.4f}")
    print(f"Resultados guardados en {carpeta}")


def cargar_folds(df: pd.DataFrame) -> np.ndarray:
    """Fold (0..N_FOLDS-1) de cada fila del corpus, para validacion cruzada.

    Igual que el split unico: se agrupa por texto shiwilu NORMALIZADO (una misma
    oracion con variantes de mayusculas/puntuacion cae siempre en el mismo fold)
    y se estratifica por categoria. Se calcula una sola vez y queda congelado en
    `folds_fijos.csv`; para recalcularlo a proposito hay que borrar ese archivo
    y regenerar las tecnicas de aumento que dependen del train de cada fold.
    """
    claves = df["shiwilu"].map(_normalizar_shiwilu)
    claves_estrictas = df["shiwilu"].map(_normalizar_estricto)
    if FOLDS_FIJOS_CSV.exists():
        asignacion = pd.read_csv(FOLDS_FIJOS_CSV, encoding="utf-8", dtype=str, keep_default_na=False)
    else:
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMILLA)
        fold = np.full(len(df), -1)
        for k, (_, idx_test) in enumerate(cv.split(df, df["intencion"], groups=claves_estrictas)):
            fold[idx_test] = k
        asignacion = pd.DataFrame({"shiwilu_norm": claves.to_numpy(), "fold": fold}).drop_duplicates("shiwilu_norm")
        asignacion.to_csv(FOLDS_FIJOS_CSV, index=False, encoding="utf-8")
        print(f"[folds] calculados y guardados en {FOLDS_FIJOS_CSV}")
    mapa = dict(zip(asignacion["shiwilu_norm"], asignacion["fold"].astype(int)))
    faltan = int((~claves.isin(mapa)).sum())
    if faltan:
        raise SystemExit(f"{faltan} oraciones no estan en {FOLDS_FIJOS_CSV.name}: el corpus cambio; "
                         "borra ese archivo y regenera las tecnicas de aumento.")
    return claves.map(mapa).to_numpy()
