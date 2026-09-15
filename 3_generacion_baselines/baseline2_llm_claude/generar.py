"""
Generacion espanol -> shiwilu con el Baseline 2 (LLM few-shot, sin ajuste fino).

Usa la API de Claude (claude-sonnet-4-6) condicionada por:
  (a) ejemplos reales del corpus (corpus/corpus_shiwilu_final.csv), muestreados
      de la misma categoria de intencion que la oracion a traducir; y
  (b) los marcadores morfosintacticos documentados en el analisis intrinseco
      de la Fase 2 (2_analisis_corpus/resultados/tablas/analisis_marcadores_documentados.csv).

No hay ajuste fino del modelo: toda la adaptacion al shiwilu ocurre via
prompting. Este es el contraste metodologico frente al Baseline 1 (NMT
especializado y entrenado).

Entrada: CSV con columnas id, espanol, intencion (sin shiwilu).
Salida:  mismo esquema que corpus/corpus_shiwilu_final.csv.

Requiere ANTHROPIC_API_KEY (ver .env.example).

Uso:
    python generar.py --entrada oraciones_nuevas.csv \
        --salida salidas/baseline2_generado.csv
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import anthropic
import pandas as pd

import _ruta_raiz  # noqa: F401  (deja importable el paquete `shiwilu`)
from shiwilu.rutas import BASELINE2_SALIDA, CORPUS_CSV, MARCADORES_CSV

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
if not API_KEY:
    raise ValueError("[ERROR] No se encontro ANTHROPIC_API_KEY en el archivo .env")

MODELO = "claude-sonnet-4-6"
N_EJEMPLOS_FEW_SHOT = 5
SEMILLA = 42


def cargar_reglas_morfosintacticas(marcadores_csv: Path = MARCADORES_CSV) -> dict[str, str]:
    """Agrupa los marcadores documentados (R5) por categoria de intencion.

    Devuelve, por categoria, un bloque de texto listo para el prompt con la
    forma documentada, su funcion y la fuente linguistica.
    """
    df = pd.read_csv(marcadores_csv)
    df = df[df["documentacion"].isin(["documentada", "parcial"])]

    reglas_por_categoria: dict[str, list[str]] = {}
    for _, fila in df.iterrows():
        categoria = fila.get("intencion_esperada")
        if not isinstance(categoria, str) or categoria in ("(n/a)",):
            continue
        linea = (
            f"- \"{fila['forma_documentada']}\" ({fila['tipo']}): "
            f"{fila['funcion_documentada']} [{fila['fuente']}]"
        )
        reglas_por_categoria.setdefault(categoria, []).append(linea)

    return {cat: "\n".join(lineas) for cat, lineas in reglas_por_categoria.items()}


def cargar_ejemplos_por_categoria(corpus_csv: Path = CORPUS_CSV) -> dict[str, list[tuple[str, str]]]:
    """Agrupa pares (espanol, shiwilu) del corpus por categoria de intencion."""
    df = pd.read_csv(corpus_csv)
    ejemplos: dict[str, list[tuple[str, str]]] = {}
    for _, fila in df.iterrows():
        ejemplos.setdefault(fila["intencion"], []).append((fila["espanol"], fila["shiwilu"]))
    return ejemplos


def _construir_prompt(
    oracion: str,
    categoria: str,
    ejemplos: list[tuple[str, str]],
    reglas: str,
) -> tuple[str, str]:
    reglas_bloque = reglas or "(sin marcadores documentados para esta categoria)"
    ejemplos_bloque = "\n".join(f'  "{es}" -> "{shw}"' for es, shw in ejemplos)

    prompt_s = (
        "Eres un especialista en linguistica computacional y en la lengua "
        "shiwilu (jebero, ISO 639-3: jeb), lengua amazonica del Peru en "
        "peligro critico de extincion. Tu tarea es traducir oraciones breves "
        "del espanol al shiwilu, generando la forma mas natural y "
        "morfologicamente correcta posible a partir de la evidencia "
        "linguistica documentada. No tienes acceso a un traductor entrenado: "
        "debes razonar a partir de los ejemplos y las reglas morfosintacticas "
        "que se te entregan."
    )
    prompt_u = f"""Traduce la siguiente oracion en espanol al shiwilu.

CATEGORIA DE INTENCION: {categoria}

MARCADORES MORFOSINTACTICOS DOCUMENTADOS PARA ESTA CATEGORIA \
(analisis intrinseco del corpus, R5):
{reglas_bloque}

EJEMPLOS REALES DEL CORPUS PARA ESTA CATEGORIA:
{ejemplos_bloque}

ORACION A TRADUCIR:
"{oracion}"

Responde UNICAMENTE con la traduccion al shiwilu, sin comillas, sin \
explicaciones ni texto adicional."""
    return prompt_s, prompt_u


def traducir_oracion(
    oracion: str,
    categoria: str,
    ejemplos_por_categoria: dict[str, list[tuple[str, str]]],
    reglas_por_categoria: dict[str, str],
    client: anthropic.Anthropic,
    rng: random.Random,
) -> str:
    disponibles = ejemplos_por_categoria.get(categoria, [])
    muestra = rng.sample(disponibles, k=min(N_EJEMPLOS_FEW_SHOT, len(disponibles)))
    reglas = reglas_por_categoria.get(categoria, "")

    prompt_s, prompt_u = _construir_prompt(oracion, categoria, muestra, reglas)
    resp = client.messages.create(
        model=MODELO, max_tokens=128,
        system=prompt_s,
        messages=[{"role": "user", "content": prompt_u}],
    )
    return resp.content[0].text.strip().strip('"')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entrada", required=True, type=Path,
                     help="CSV con columnas id, espanol, intencion (sin shiwilu).")
    ap.add_argument("--salida", type=Path, default=BASELINE2_SALIDA / "baseline2_generado.csv")
    ap.add_argument("--semilla", type=int, default=SEMILLA)
    args = ap.parse_args()

    df = pd.read_csv(args.entrada)
    for col in ("espanol", "intencion"):
        if col not in df.columns:
            raise SystemExit(f"El CSV de entrada debe tener una columna '{col}'.")

    ejemplos_por_categoria = cargar_ejemplos_por_categoria()
    reglas_por_categoria = cargar_reglas_morfosintacticas()
    client = anthropic.Anthropic(api_key=API_KEY)
    rng = random.Random(args.semilla)

    traducciones = []
    for _, fila in df.iterrows():
        traduccion = traducir_oracion(
            fila["espanol"], fila["intencion"],
            ejemplos_por_categoria, reglas_por_categoria,
            client, rng,
        )
        traducciones.append(traduccion)
        print(f"  {fila['espanol']!r} -> {traduccion!r}")

    df = df.copy()
    df["shiwilu"] = traducciones
    df["fuente"] = "baseline2_llm_claude"

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.salida, index=False, encoding="utf-8")
    print(f"Generadas {len(df)} traducciones -> {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
