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

Solo se aumenta el conjunto de ENTRENAMIENTO; dev y test se mantienen sin
modificar.

Entrada: corpus/corpus_shiwilu_final.csv
Salida:  3_baselines_y_aumento_datos/tecnicas_aumento/salidas/retrotraduccion.csv

Uso:
    python 3_baselines_y_aumento_datos/tecnicas_aumento/retrotraduccion.py \
        --checkpoint ruta/al/checkpoint/nllb_bidi_lora_v2_1b_loraplus_xl
    python 3_baselines_y_aumento_datos/tecnicas_aumento/retrotraduccion.py --checkpoint ... --categorias DES NEG --limite 20
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

from shiwilu.rutas import AUMENTO_SALIDA, CORPUS_CSV, NMT_REPO_EXTERNO, preparar_directorios  # noqa: E402

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


def refinar(df: pd.DataFrame, corpus: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for categoria, grupo in df.groupby("intencion"):
        ejemplos_reales = corpus.loc[corpus["intencion"] == categoria, "shiwilu"].astype(str).tolist()
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

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path,
                     help="Carpeta del checkpoint NLLB+LoRA entrenado (ver README.md).")
    ap.add_argument("--categorias", nargs="+", default=None,
                     help="Categorias a aumentar (por defecto: todas).")
    ap.add_argument("--limite", type=int, default=None,
                     help="Maximo de oraciones de train a retrotraducir por categoria (por defecto: todas).")
    ap.add_argument("--repo", type=Path, default=NMT_REPO_EXTERNO)
    ap.add_argument("--salida", type=Path, default=None)
    args = ap.parse_args()

    preparar_directorios()

    print("Cargando corpus...")
    corpus = pd.read_csv(CORPUS_CSV)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "2_baselines"))
    from comun import cargar_corpus, dividir_train_dev_test
    train, _dev, _test = dividir_train_dev_test(cargar_corpus())

    if args.categorias:
        train = train[train["intencion"].isin(args.categorias)]
    if args.limite:
        train = train.groupby("intencion", group_keys=False).head(args.limite)
    print(f"Retrotraduciendo {len(train)} oraciones de entrenamiento...")

    print("Paso 1/3: parafraseando en espanol (Helsinki-NLP, es->en->es)...")
    parafrasis = parafrasear_espanol(train["espanol"].astype(str).tolist())

    # descarta paráfrasis identicas al original (no aportan nada nuevo)
    df = train[["espanol", "intencion"]].copy()
    df["espanol_parafraseado"] = parafrasis
    df = df[df["espanol_parafraseado"].str.strip().str.lower() != df["espanol"].str.strip().str.lower()]
    print(f"  {len(df)}/{len(train)} paráfrasis distintas del original")

    print("Paso 2/3: traduciendo la paráfrasis al shiwilu (NLLB+LoRA de F. Prado)...")
    df["shiwilu"] = traducir_a_shiwilu(df["espanol_parafraseado"].tolist(), args.checkpoint, args.repo)
    df["espanol"] = df["espanol_parafraseado"]
    df = df.drop(columns=["espanol_parafraseado"])
    df["fuente"] = "retrotraduccion"

    print("Paso 3/3: refinando (filtro de idioma + filtro semantico LaBSE)...")
    resultado = refinar(df, corpus)
    resultado.insert(0, "id", [f"RT_{i:04d}" for i in range(len(resultado))])

    salida = args.salida or (AUMENTO_SALIDA / "retrotraduccion.csv")
    resultado.to_csv(salida, index=False, encoding="utf-8")
    print(f"\nTotal: {len(resultado)} filas -> {salida}")
    print(resultado["estado_filtro"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
