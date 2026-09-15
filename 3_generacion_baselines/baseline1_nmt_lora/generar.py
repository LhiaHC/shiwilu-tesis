"""
Generacion espanol -> shiwilu con el Baseline 1 (NLLB-200 + LoRA).

Reutiliza `load_checkpoint` del repositorio de referencia (F. Prado,
comunicacion personal, 14 de septiembre de 2026;
https://github.com/fapi19/Tesis_Spa-Jeb) para cargar el checkpoint
entrenado localmente (ver README.md de esta carpeta) y traduce oraciones
nuevas en espanol al shiwilu, produciendo un CSV con el mismo esquema que
`corpus/corpus_shiwilu_final.csv`.

Uso:
    python generar.py --checkpoint ruta/al/checkpoint \
        --entrada oraciones_nuevas.csv --salida salidas/baseline1_generado.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

import _ruta_raiz  # noqa: F401  (deja importable el paquete `shiwilu`)
from shiwilu.rutas import BASELINE1_REPO, BASELINE1_SALIDA

BASE_MODEL = "facebook/nllb-200-distilled-600M"
SRC_LANG_CODE = "spa_Latn"
TGT_LANG_CODE = "shw_Latn"

NUM_BEAMS = 5
LENGTH_PENALTY = 1.0
MAX_NEW_TOKENS = 128
NO_REPEAT_NGRAM_SIZE = 0
BATCH_SIZE = 8


def _importar_modulo_de_prado(repo_dir: Path):
    """Agrega el repo clonado de F. Prado al sys.path e importa su modulo
    de inferencia. El repo no se vendoriza aqui (ver README.md)."""
    if not repo_dir.exists():
        raise SystemExit(
            f"No se encontro el repositorio clonado en {repo_dir}.\n"
            "Sigue el paso 1 del README.md de esta carpeta: "
            "git clone https://github.com/fapi19/Tesis_Spa-Jeb.git "
            f"{repo_dir}"
        )
    sys.path.insert(0, str(repo_dir))
    from src.nmt.inference.generate import GenerationConfig, load_checkpoint  # type: ignore

    return GenerationConfig, load_checkpoint


def traducir(
    oraciones: list[str],
    checkpoint_dir: Path,
    repo_dir: Path = BASELINE1_REPO,
) -> list[str]:
    """Traduce una lista de oraciones en espanol al shiwilu."""
    _, load_checkpoint = _importar_modulo_de_prado(repo_dir)

    model, tokenizer, device = load_checkpoint(checkpoint_dir, base_model=BASE_MODEL)
    tokenizer.src_lang = SRC_LANG_CODE
    if TGT_LANG_CODE not in tokenizer.lang_code_to_id:
        raise RuntimeError(
            f"{TGT_LANG_CODE!r} no esta registrado en el tokenizer del checkpoint. "
            "Verifica que el entrenamiento haya extendido el vocabulario (ver "
            "config/nmt/training.yaml del repo de F. Prado, seccion 'tokenizer')."
        )
    forced_bos = tokenizer.lang_code_to_id[TGT_LANG_CODE]

    salidas: list[str] = []
    with torch.no_grad():
        for inicio in range(0, len(oraciones), BATCH_SIZE):
            lote = oraciones[inicio : inicio + BATCH_SIZE]
            enc = tokenizer(
                lote, return_tensors="pt", padding=True, truncation=True, max_length=128
            ).to(device)
            generados = model.generate(
                **enc,
                forced_bos_token_id=forced_bos,
                num_beams=NUM_BEAMS,
                length_penalty=LENGTH_PENALTY,
                max_new_tokens=MAX_NEW_TOKENS,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                early_stopping=True,
                num_return_sequences=1,
            )
            decodificados = tokenizer.batch_decode(generados, skip_special_tokens=True)
            salidas.extend(decodificados)
    return salidas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path,
                     help="Carpeta del checkpoint LoRA entrenado (ver README.md).")
    ap.add_argument("--entrada", required=True, type=Path,
                     help="CSV con columnas id, espanol, intencion (sin shiwilu).")
    ap.add_argument("--salida", type=Path, default=BASELINE1_SALIDA / "baseline1_generado.csv")
    ap.add_argument("--repo", type=Path, default=BASELINE1_REPO,
                     help="Carpeta del repo clonado de F. Prado (por defecto: tesis_spa_jeb/ junto a este script).")
    args = ap.parse_args()

    df = pd.read_csv(args.entrada)
    if "espanol" not in df.columns:
        raise SystemExit("El CSV de entrada debe tener una columna 'espanol'.")

    traducciones = traducir(df["espanol"].astype(str).tolist(), args.checkpoint, args.repo)
    df = df.copy()
    df["shiwilu"] = traducciones
    df["fuente"] = "baseline1_nllb_lora"

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.salida, index=False, encoding="utf-8")
    print(f"Generadas {len(df)} traducciones -> {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
