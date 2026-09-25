"""
Tecnica de aumento de datos: retrotraduccion (back-translation).

Opera sobre el componente en ESPAÑOL del corpus (no existe retrotraduccion
directa shiwilu-shiwilu, pues no hay sistemas de traduccion automatica
disponibles para shiwilu fuera del que aqui se reutiliza):

  1. Parafraseo: cada oracion en espanol se retrotraduce
     español -> ingles -> español con Helsinki-NLP (Opus-MT), obteniendo una
     paráfrasis nueva pero semanticamente equivalente.
  2. Traduccion a shiwilu: la paráfrasis nueva en español se traduce al
     shiwilu usando el sistema NMT (NLLB-200 + LoRA) desarrollado por
     F. Prado en su propia tesis (comunicacion personal, 14 de septiembre de
     2026; https://github.com/fapi19/Tesis_Spa-Jeb) — ver README.md de esta
     carpeta para clonarlo y entrenarlo localmente antes de usar este script.
  3. Refinamiento: mismo protocolo de calidad que Generate-then-Refine
     (filtro de idioma + filtro semantico via LaBSE), pues ambas tecnicas
     comparten el riesgo de producir enunciados sinteticos erroneos.

Se corre en DOS etapas, porque solo la ultima depende del split train/dev/test:

  --etapa generar  (pasos 1 y 2; requiere GPU y el checkpoint, se corre en Colab)
      Parafrasea y traduce las 700 oraciones del corpus y guarda un CATALOGO
      (`retrotraduccion_pool.csv`) con una fila por oracion, indicando de cual
      viene (`id_origen`). Cada fila depende solo de su oracion de origen, asi
      que el catalogo sirve para cualquier reparto train/dev/test (o folds de
      validacion cruzada) sin volver a correr el Colab.

  --etapa filtrar  (paso 3; local, sin GPU)
      Toma del catalogo solo las filas cuya oracion de origen esta en TRAIN
      (split congelado) y aplica los filtros con el centroide del train. Dev y
      test nunca participan. Exige --salida para no pisar por accidente el
      retrotraduccion.csv que ya se uso en los experimentos.

Solo se aumenta el conjunto de ENTRENAMIENTO; dev y test se mantienen sin
modificar.

Entrada: corpus/corpus_shiwilu_final.csv
Salida:  salidas/retrotraduccion_pool.csv (generar) / --salida (filtrar)

Uso:
    python 3_baselines_y_aumento_datos/tecnicas_aumento/retrotraduccion.py         --etapa generar --checkpoint ruta/al/checkpoint/nllb_bidi_lora_v2_1b_loraplus_xl
    python 3_baselines_y_aumento_datos/tecnicas_aumento/retrotraduccion.py         --etapa filtrar --salida salidas/retrotraduccion_split.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import _ruta_raiz  # noqa: F401  (deja importable el paquete `shiwilu`)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "2_baselines"))
from comun import extraer_embeddings  # noqa: E402  (reutiliza la extraccion LaBSE)

from shiwilu.rutas import AUMENTO_SALIDA, NMT_REPO_EXTERNO, preparar_directorios  # noqa: E402

MODELO_ES_EN = "Helsinki-NLP/opus-mt-es-en"
MODELO_EN_ES = "Helsinki-NLP/opus-mt-en-es"
BASE_NLLB = "facebook/nllb-200-distilled-600M"
SRC_LANG_CODE = "spa_Latn"
TGT_LANG_CODE = "shw_Latn"

# Filtro semantico: igual que en generate_then_refine.py
UMBRAL_SIMILITUD_MIN = 0.20
UMBRAL_SIMILITUD_MAX = 0.995


# ---------------------------------------------------------------------------
# Paso 1: retrotraduccion en espanol (Helsinki-NLP, publico, no requiere el
# repo de F. Prado ni GPU especial)
# ---------------------------------------------------------------------------

def _cargar_modelo_marian(nombre_modelo: str):
    from transformers import MarianMTModel, MarianTokenizer

    tokenizer = MarianTokenizer.from_pretrained(nombre_modelo)
    modelo = MarianMTModel.from_pretrained(nombre_modelo)
    modelo.eval()
    return tokenizer, modelo


def _traducir_lote(
    oraciones: list[str], tokenizer, modelo, batch_size: int = 16,
    muestreo: bool = False, temperatura: float = 1.0, top_k: int = 50, semilla: int | None = None,
) -> list[str]:
    if semilla is not None:
        torch.manual_seed(semilla)
    salidas = []
    with torch.no_grad():
        for inicio in range(0, len(oraciones), batch_size):
            lote = oraciones[inicio : inicio + batch_size]
            enc = tokenizer(lote, return_tensors="pt", padding=True, truncation=True, max_length=64)
            if muestreo:
                generados = modelo.generate(
                    **enc, do_sample=True, temperature=temperatura, top_k=top_k,
                    num_beams=1, max_new_tokens=64,
                )
            else:
                generados = modelo.generate(**enc, num_beams=4, max_new_tokens=64)
            salidas.extend(tokenizer.batch_decode(generados, skip_special_tokens=True))
    return salidas


def parafrasear_espanol(oraciones: list[str], semilla: int | None = 42) -> list[str]:
    """Retrotraduccion espanol -> ingles -> espanol (Helsinki-NLP/Opus-MT).

    IMPORTANTE: con beam search determinista, oraciones cortas casi siempre
    vuelven identicas al original (verificado empiricamente) — no aportan
    nada como aumento de datos. Por eso el tramo ingles -> espanol usa
    muestreo (do_sample) con temperatura, para forzar variacion real en la
    parafrasis final. El tramo espanol -> ingles se mantiene determinista
    (beam search), pues ahi solo interesa una traduccion fiel de entrada.
    """
    tok_es_en, mod_es_en = _cargar_modelo_marian(MODELO_ES_EN)
    ingles = _traducir_lote(oraciones, tok_es_en, mod_es_en)
    del tok_es_en, mod_es_en

    tok_en_es, mod_en_es = _cargar_modelo_marian(MODELO_EN_ES)
    parafrasis = _traducir_lote(
        ingles, tok_en_es, mod_en_es,
        muestreo=True, temperatura=1.3, top_k=50, semilla=semilla,
    )
    return parafrasis


# ---------------------------------------------------------------------------
# Paso 2: traduccion espanol -> shiwilu con el checkpoint de F. Prado
# ---------------------------------------------------------------------------

def _importar_modulo_de_prado(repo_dir: Path):
    if not repo_dir.exists():
        raise SystemExit(
            f"No se encontro el repositorio clonado en {repo_dir}.\n"
            "Ver README.md de esta carpeta: "
            "git clone https://github.com/fapi19/Tesis_Spa-Jeb.git "
            f"{repo_dir}"
        )
    sys.path.insert(0, str(repo_dir))
    from src.nmt.inference.generate import load_checkpoint  # type: ignore

    return load_checkpoint


def traducir_a_shiwilu(
    oraciones: list[str],
    checkpoint_dir: Path,
    repo_dir: Path = NMT_REPO_EXTERNO,
) -> list[str]:
    load_checkpoint = _importar_modulo_de_prado(repo_dir)
    modelo, tokenizer, device = load_checkpoint(checkpoint_dir, base_model=BASE_NLLB)
    tokenizer.src_lang = SRC_LANG_CODE
    if TGT_LANG_CODE not in tokenizer.lang_code_to_id:
        raise RuntimeError(f"{TGT_LANG_CODE!r} no esta registrado en el tokenizer del checkpoint.")
    forced_bos = tokenizer.lang_code_to_id[TGT_LANG_CODE]

    salidas = []
    with torch.no_grad():
        for inicio in range(0, len(oraciones), 8):
            lote = oraciones[inicio : inicio + 8]
            enc = tokenizer(lote, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            generados = modelo.generate(
                **enc, forced_bos_token_id=forced_bos,
                num_beams=5, length_penalty=1.0, max_new_tokens=128,
                no_repeat_ngram_size=0, early_stopping=True, num_return_sequences=1,
            )
            salidas.extend(tokenizer.batch_decode(generados, skip_special_tokens=True))
    return salidas


# ---------------------------------------------------------------------------
# Paso 3: refinamiento (mismo protocolo que generate_then_refine.py)
# ---------------------------------------------------------------------------

def filtro_idioma(texto_espanol: str, texto_shiwilu: str) -> bool:
    return texto_shiwilu.strip().lower() != texto_espanol.strip().lower()


def filtro_semantico(embedding_generado: np.ndarray, embeddings_categoria: np.ndarray) -> tuple[bool, float]:
    centroide = embeddings_categoria.mean(axis=0)
    sim = float(
        np.dot(embedding_generado, centroide)
        / (np.linalg.norm(embedding_generado) * np.linalg.norm(centroide) + 1e-9)
    )
    return UMBRAL_SIMILITUD_MIN <= sim <= UMBRAL_SIMILITUD_MAX, sim


def refinar(df: pd.DataFrame, corpus_train: pd.DataFrame) -> pd.DataFrame:
    """corpus_train debe ser SOLO el split de entrenamiento: el centroide del
    filtro semantico no debe calcularse con oraciones de dev/test, o el
    filtro terminaria "sabiendo" cosas del conjunto de prueba."""
    filas = []
    for categoria, grupo in df.groupby("intencion"):
        ejemplos_reales = corpus_train.loc[corpus_train["intencion"] == categoria, "shiwilu"].astype(str).tolist()
        embeddings_categoria = extraer_embeddings(ejemplos_reales, "labse")
        embeddings_generados = extraer_embeddings(grupo["shiwilu"].astype(str).tolist(), "labse")

        for (_, fila), emb in zip(grupo.iterrows(), embeddings_generados):
            paso_idioma = filtro_idioma(fila["espanol"], fila["shiwilu"])
            paso_semantico, sim = filtro_semantico(emb, embeddings_categoria)
            fallas = []
            if not paso_idioma:
                fallas.append("idioma")
            if not paso_semantico:
                fallas.append(f"semantico(sim={sim:.3f})")
            filas.append({
                **fila.to_dict(),
                "estado_filtro": "aprobado" if not fallas else "revisar_hablante_nativo",
                "detalle_filtro": ";".join(fallas),
                "similitud_labse": round(sim, 4),
            })
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generar(args) -> int:
    """Etapa 1 (Colab): parafrasea y traduce las 700 oraciones. Sin filtros."""
    if args.checkpoint is None:
        raise SystemExit("--checkpoint es obligatorio en la etapa 'generar'.")
    preparar_directorios()

    print("Cargando corpus...")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "2_baselines"))
    from comun import cargar_corpus
    corpus = cargar_corpus()
    if args.categorias:
        corpus = corpus[corpus["intencion"].isin(args.categorias)]
    if args.limite:
        corpus = corpus.groupby("intencion", group_keys=False).head(args.limite)
    print(f"Retrotraduciendo {len(corpus)} oraciones (todo el corpus, sin filtrar por split)...")

    print("Paso 1/2: parafraseando en espanol (Helsinki-NLP, es->en->es)...")
    parafrasis = parafrasear_espanol(corpus["espanol"].astype(str).tolist())

    df = pd.DataFrame({
        "id_origen": corpus.index.to_numpy(),
        "espanol_original": corpus["espanol"].astype(str).to_numpy(),
        "espanol": parafrasis,
        "intencion": corpus["intencion"].to_numpy(),
    })
    # descarta paráfrasis identicas al original (no aportan nada nuevo)
    df = df[df["espanol"].str.strip().str.lower() != df["espanol_original"].str.strip().str.lower()].copy()
    print(f"  {len(df)}/{len(corpus)} paráfrasis distintas del original")

    print("Paso 2/2: traduciendo la paráfrasis al shiwilu (NLLB+LoRA de F. Prado)...")
    df["shiwilu"] = traducir_a_shiwilu(df["espanol"].tolist(), args.checkpoint, args.repo)
    df["fuente"] = "retrotraduccion"
    df = df[["id_origen", "espanol_original", "espanol", "shiwilu", "intencion", "fuente"]]

    salida = args.salida or (AUMENTO_SALIDA / "retrotraduccion_pool.csv")
    df.to_csv(salida, index=False, encoding="utf-8")
    print(f"\nCatalogo: {len(df)} filas -> {salida}")
    print(df.groupby("intencion").size().to_string())
    return 0


def filtrar(args) -> int:
    """Etapa 2 (local): filtra el catalogo con el train del split congelado."""
    if args.salida is None:
        raise SystemExit("--salida es obligatorio en la etapa 'filtrar' (para no pisar el "
                         "retrotraduccion.csv ya usado en los experimentos).")
    ruta_pool = args.pool or (AUMENTO_SALIDA / "retrotraduccion_pool.csv")
    if not ruta_pool.exists():
        raise SystemExit(f"No se encontro el catalogo {ruta_pool}. Corre primero --etapa generar.")

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "2_baselines"))
    from comun import cargar_corpus, dividir_train_dev_test
    corpus = cargar_corpus()
    train, _dev, _test = dividir_train_dev_test(corpus)

    pool = pd.read_csv(ruta_pool)
    desajuste = pool["espanol_original"].to_numpy() != corpus.loc[pool["id_origen"], "espanol"].astype(str).to_numpy()
    if desajuste.any():
        raise SystemExit(f"{desajuste.sum()} filas del catalogo no coinciden con el corpus actual "
                         "(cambio el corpus despues de generar el catalogo): regeneralo.")

    en_train = pool["id_origen"].isin(train.index)
    print(f"Catalogo: {len(pool)} filas; {en_train.sum()} vienen de oraciones de train (se usan), "
          f"{(~en_train).sum()} de dev/test (se descartan).")
    pool = pool[en_train].copy()

    print("Refinando (filtro de idioma + filtro semantico LaBSE, centroide del train)...")
    resultado = refinar(pool, train)
    resultado.insert(0, "id", [f"RT_{i:04d}" for i in range(len(resultado))])
    resultado.to_csv(args.salida, index=False, encoding="utf-8")
    print(f"\nTotal: {len(resultado)} filas -> {args.salida}")
    print(resultado["estado_filtro"].value_counts().to_string())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--etapa", choices=["generar", "filtrar"], default="generar")
    ap.add_argument("--checkpoint", type=Path, default=None,
                     help="Carpeta del checkpoint NLLB+LoRA entrenado (etapa generar; ver README.md).")
    ap.add_argument("--categorias", nargs="+", default=None,
                     help="(generar) Categorias a procesar (por defecto: todas).")
    ap.add_argument("--limite", type=int, default=None,
                     help="(generar) Maximo de oraciones por categoria, para pruebas rapidas.")
    ap.add_argument("--repo", type=Path, default=NMT_REPO_EXTERNO)
    ap.add_argument("--pool", type=Path, default=None,
                     help="(filtrar) Catalogo a filtrar (por defecto: salidas/retrotraduccion_pool.csv).")
    ap.add_argument("--salida", type=Path, default=None)
    args = ap.parse_args()
    return generar(args) if args.etapa == "generar" else filtrar(args)


if __name__ == "__main__":
    raise SystemExit(main())
