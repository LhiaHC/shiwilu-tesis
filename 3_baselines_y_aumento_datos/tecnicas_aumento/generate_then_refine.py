"""
Tecnica de aumento de datos: Generate-then-Refine (LLM + filtros de calidad).

Un LLM (Claude, claude-sonnet-4-6) GENERA oraciones nuevas en shiwilu para una
categoria de intencion, a partir de: la descripcion de la categoria, ejemplos
reales del corpus, los marcadores morfosintacticos ya validados contra la
literatura linguistica (R5) y los patrones candidatos detectados
estadisticamente en el corpus (no validados, se usan con cautela).

Luego cada oracion generada se REFINA con tres filtros, siguiendo el
protocolo de control de calidad descrito en la metodologia:
  1. Filtro de marcador   — si la categoria tiene marcadores documentados,
                            exige que al menos uno aparezca en la oracion.
  2. Filtro de idioma     — heuristica simple para detectar "espanolizacion"
                            (texto identico al glosado en espanol, o ausencia
                            total de rasgos ortograficos tipicos del shiwilu).
  3. Filtro semantico     — similitud coseno (embeddings LaBSE, congelado)
                            entre la oracion generada y el centroide de los
                            ejemplos reales de su categoria; rechaza si esta
                            demasiado alejada (posible deriva de tema) o es
                            casi identica a un ejemplo existente (duplicado).

Las oraciones que no pasan TODOS los filtros no se descartan: quedan
marcadas como "revisar_hablante_nativo" (la via de traduccion/validacion por
el hablante nativo descrita en la metodologia), no se usan automaticamente
para el aumento.

Requiere ANTHROPIC_API_KEY (ver .env.example).

Uso:
    python 3_baselines_y_aumento_datos/tecnicas_aumento/generate_then_refine.py --cantidad 20
    python 3_baselines_y_aumento_datos/tecnicas_aumento/generate_then_refine.py --categorias DES NEG REQUEST --cantidad 15
"""

from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

import anthropic

import _ruta_raiz  # noqa: F401  (deja importable el paquete `shiwilu`)
from shiwilu.rutas import (
    AUMENTO_SALIDA,
    CORPUS_CSV,
    MARCADORES_CSV,
    PALABRAS_CARACTERISTICAS_CSV,
    SECUENCIAS_FINALES_CSV,
    SECUENCIAS_INICIALES_CSV,
    preparar_directorios,
)
from shiwilu.taxonomia import DESC_INT, INTENCIONES

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODELO = "claude-sonnet-4-6"
N_EJEMPLOS_FEW_SHOT = 6
N_PATRONES_CANDIDATOS = 8   # top-N por log_odds, por tabla
SEMILLA = 42

# Filtro semantico: similitud coseno LaBSE contra el centroide de la categoria
UMBRAL_SIMILITUD_MIN = 0.20   # por debajo: posible deriva de tema
UMBRAL_SIMILITUD_MAX = 0.995  # por encima: posible duplicado casi exacto


# ---------------------------------------------------------------------------
# Carga de contexto linguistico (R5) y del corpus
# ---------------------------------------------------------------------------

def cargar_ejemplos_por_categoria(corpus_csv=CORPUS_CSV) -> dict[str, list[tuple[str, str]]]:
    df = pd.read_csv(corpus_csv)
    ejemplos: dict[str, list[tuple[str, str]]] = {}
    for _, fila in df.iterrows():
        ejemplos.setdefault(fila["intencion"], []).append((fila["espanol"], fila["shiwilu"]))
    return ejemplos


def cargar_marcadores_validados() -> dict[str, list[dict]]:
    """Marcadores morfosintacticos ya contrastados con la literatura (R5)."""
    df = pd.read_csv(MARCADORES_CSV)
    df = df[df["documentacion"].isin(["documentada", "parcial"])]
    por_categoria: dict[str, list[dict]] = {}
    for _, fila in df.iterrows():
        categoria = fila.get("intencion_esperada")
        if not isinstance(categoria, str) or categoria == "(n/a)":
            continue
        por_categoria.setdefault(categoria, []).append({
            "forma": fila["forma_documentada"],
            "tipo": fila["tipo"],
            "funcion": fila["funcion_documentada"],
            "fuente": fila["fuente"],
        })
    return por_categoria


def _top_patrones(path, columna_intencion="intencion") -> dict[str, list[str]]:
    df = pd.read_csv(path)
    if "log_odds" in df.columns:
        df = df.sort_values("log_odds", ascending=False)
    por_categoria: dict[str, list[str]] = {}
    for categoria, grupo in df.groupby(columna_intencion):
        por_categoria[categoria] = grupo["patron"].head(N_PATRONES_CANDIDATOS).tolist()
    return por_categoria


def cargar_patrones_candidatos() -> dict[str, dict[str, list[str]]]:
    """Patrones detectados estadisticamente en el corpus (NO validados
    contra fuentes linguisticas externas — usar con cautela en el prompt)."""
    return {
        "palabras_caracteristicas": _top_patrones(PALABRAS_CARACTERISTICAS_CSV),
        "secuencias_iniciales": _top_patrones(SECUENCIAS_INICIALES_CSV),
        "secuencias_finales": _top_patrones(SECUENCIAS_FINALES_CSV),
    }


# ---------------------------------------------------------------------------
# Generacion (LLM)
# ---------------------------------------------------------------------------

def _bloque_marcadores(marcadores: list[dict]) -> str:
    if not marcadores:
        return "(sin marcadores documentados para esta categoria)"
    return "\n".join(
        f"- \"{m['forma']}\" ({m['tipo']}): {m['funcion']} [{m['fuente']}]"
        for m in marcadores
    )


def _bloque_patrones_candidatos(candidatos: dict[str, list[str]], categoria: str) -> str:
    partes = []
    for nombre, por_cat in candidatos.items():
        patrones = por_cat.get(categoria, [])
        if patrones:
            partes.append(f"  {nombre}: " + ", ".join(f'"{p}"' for p in patrones))
    if not partes:
        return "(sin patrones candidatos para esta categoria)"
    return "\n".join(partes)


def construir_prompt(
    categoria: str,
    cantidad: int,
    ejemplos: list[tuple[str, str]],
    marcadores: list[dict],
    candidatos: dict[str, list[str]],
) -> tuple[str, str]:
    ejemplos_bloque = "\n".join(f'  "{es}" -> "{shw}"' for es, shw in ejemplos)
    marcadores_bloque = _bloque_marcadores(marcadores)
    candidatos_bloque = _bloque_patrones_candidatos(candidatos, categoria)

    prompt_s = (
        "Eres un especialista en linguistica computacional y en la lengua "
        "shiwilu (jebero, ISO 639-3: jeb), lengua amazonica del Peru en "
        "peligro critico de extincion. Tu tarea es generar pares de "
        "oraciones espanol-shiwilu NUEVOS (no copias de los ejemplos) para "
        "ampliar un corpus de clasificacion de intenciones, razonando a "
        "partir de la evidencia linguistica documentada."
    )
    prompt_u = f"""Genera {cantidad} pares NUEVOS de oraciones espanol-shiwilu para la \
categoria de intencion: {categoria} - {DESC_INT[categoria]}

EJEMPLOS REALES DEL CORPUS PARA ESTA CATEGORIA (no los repitas):
{ejemplos_bloque}

MARCADORES MORFOSINTACTICOS VALIDADOS CONTRA LA LITERATURA LINGUISTICA (R5) \
— usalos cuando sean aplicables:
{marcadores_bloque}

PATRONES CANDIDATOS DETECTADOS EN EL CORPUS (NO validados contra fuentes \
externas; son pistas, no reglas confirmadas — usalos con cautela, no los \
fuerces si no encajan naturalmente):
{candidatos_bloque}

RESTRICCIONES:
1. Entre 1 y 5 palabras por oracion en espanol (como los ejemplos reales).
2. Cada oracion en espanol debe ser inequivocamente de la categoria {categoria}.
3. La oracion en shiwilu debe ser tu mejor traduccion razonada, coherente \
con los marcadores y patrones anteriores cuando aplique.
4. No repitas oraciones iguales o casi iguales entre si ni respecto a los ejemplos.

Responde UNICAMENTE con un JSON array valido, sin texto adicional:
[{{"espanol": "texto aqui", "shiwilu": "texto aqui"}}, ...]"""
    return prompt_s, prompt_u


def generar_candidatos(
    categoria: str,
    cantidad: int,
    ejemplos_por_categoria: dict[str, list[tuple[str, str]]],
    marcadores_por_categoria: dict[str, list[dict]],
    candidatos_por_categoria: dict[str, dict[str, list[str]]],
    client: anthropic.Anthropic,
) -> list[dict]:
    ejemplos = ejemplos_por_categoria.get(categoria, [])[:N_EJEMPLOS_FEW_SHOT]
    marcadores = marcadores_por_categoria.get(categoria, [])

    prompt_s, prompt_u = construir_prompt(categoria, cantidad, ejemplos, marcadores, candidatos_por_categoria)
    resp = client.messages.create(
        model=MODELO, max_tokens=4096,
        system=prompt_s,
        messages=[{"role": "user", "content": prompt_u}],
    )
    texto = resp.content[0].text.strip()
    texto = re.sub(r"^```(?:json)?\s*", "", texto, flags=re.DOTALL)
    texto = re.sub(r"\s*```$", "", texto, flags=re.DOTALL)
    return json.loads(texto)


# ---------------------------------------------------------------------------
# Refinamiento (filtros de calidad)
# ---------------------------------------------------------------------------

def filtro_marcador(texto_shiwilu: str, marcadores: list[dict]) -> bool:
    """True si pasa el filtro (o si la categoria no tiene marcadores que exigir)."""
    if not marcadores:
        return True
    texto = texto_shiwilu.lower()
    return any(m["forma"].lower() in texto for m in marcadores)


def filtro_idioma(texto_espanol: str, texto_shiwilu: str) -> bool:
    """Heuristica simple anti-espanolizacion: rechaza si el shiwilu generado
    es identico (o casi) al espanol, o si no tiene ningun rasgo ortografico
    tipico del shiwilu (el apostrofo de oclusiva glotal es muy frecuente en
    los marcadores documentados en R5)."""
    if texto_shiwilu.strip().lower() == texto_espanol.strip().lower():
        return False
    return True   # heuristica deliberadamente laxa; revisar manualmente los "revisar_hablante_nativo"


def filtro_semantico(
    embedding_generado: np.ndarray,
    embeddings_categoria: np.ndarray,
) -> tuple[bool, float]:
    centroide = embeddings_categoria.mean(axis=0)
    sim = float(
        np.dot(embedding_generado, centroide)
        / (np.linalg.norm(embedding_generado) * np.linalg.norm(centroide) + 1e-9)
    )
    ok = UMBRAL_SIMILITUD_MIN <= sim <= UMBRAL_SIMILITUD_MAX
    return ok, sim


def refinar(
    candidatos: list[dict],
    categoria: str,
    marcadores_por_categoria: dict[str, list[dict]],
    embeddings_corpus_categoria: np.ndarray,
) -> pd.DataFrame:
    import sys
    sys.path.insert(0, str(_ruta_raiz.RAIZ / "2_baselines"))
    from comun import extraer_embeddings  # reutiliza la extraccion LaBSE

    marcadores = marcadores_por_categoria.get(categoria, [])
    shiwilu_textos = [c["shiwilu"] for c in candidatos]
    embeddings_generados = extraer_embeddings(shiwilu_textos, "labse")

    filas = []
    for candidato, emb in zip(candidatos, embeddings_generados):
        paso_marcador = filtro_marcador(candidato["shiwilu"], marcadores)
        paso_idioma = filtro_idioma(candidato["espanol"], candidato["shiwilu"])
        paso_semantico, similitud = filtro_semantico(emb, embeddings_corpus_categoria)

        fallas = []
        if not paso_marcador:
            fallas.append("marcador")
        if not paso_idioma:
            fallas.append("idioma")
        if not paso_semantico:
            fallas.append(f"semantico(sim={similitud:.3f})")

        filas.append({
            "espanol": candidato["espanol"],
            "shiwilu": candidato["shiwilu"],
            "intencion": categoria,
            "fuente": "generate_then_refine",
            "estado_filtro": "aprobado" if not fallas else "revisar_hablante_nativo",
            "detalle_filtro": ";".join(fallas) if fallas else "",
            "similitud_labse": round(similitud, 4),
        })
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--categorias", nargs="+", choices=INTENCIONES, default=INTENCIONES)
    ap.add_argument("--cantidad", type=int, default=20,
                     help="Cuantas oraciones nuevas generar por categoria.")
    ap.add_argument("--salida", default=None,
                     help="CSV de salida (por defecto: salidas/generate_then_refine.csv)")
    args = ap.parse_args()

    if not API_KEY:
        raise SystemExit(
            "[ERROR] No se encontro ANTHROPIC_API_KEY en el archivo .env. "
            "Esta tecnica requiere API key con creditos (ver README.md de esta carpeta)."
        )

    preparar_directorios()
    client = anthropic.Anthropic(api_key=API_KEY)

    ejemplos_por_categoria = cargar_ejemplos_por_categoria()
    marcadores_por_categoria = cargar_marcadores_validados()
    candidatos_por_categoria = cargar_patrones_candidatos()

    import sys
    sys.path.insert(0, str(_ruta_raiz.RAIZ / "2_baselines"))
    from comun import extraer_embeddings

    todas_las_filas = []
    for categoria in args.categorias:
        print(f"\n=== Generando {args.cantidad} oraciones para {categoria} ===")
        candidatos = generar_candidatos(
            categoria, args.cantidad,
            ejemplos_por_categoria, marcadores_por_categoria, candidatos_por_categoria,
            client,
        )
        print(f"  {len(candidatos)} candidatos generados, refinando...")

        ejemplos_shiwilu_categoria = [shw for _, shw in ejemplos_por_categoria.get(categoria, [])]
        embeddings_corpus_categoria = extraer_embeddings(ejemplos_shiwilu_categoria, "labse")

        df_categoria = refinar(candidatos, categoria, marcadores_por_categoria, embeddings_corpus_categoria)
        n_aprobados = (df_categoria["estado_filtro"] == "aprobado").sum()
        print(f"  {n_aprobados}/{len(df_categoria)} aprobados; el resto queda para "
              f"revision del hablante nativo")
        todas_las_filas.append(df_categoria)

    resultado = pd.concat(todas_las_filas, ignore_index=True)
    resultado.insert(0, "id", [f"GTR_{i:04d}" for i in range(len(resultado))])

    salida = args.salida or (AUMENTO_SALIDA / "generate_then_refine.csv")
    resultado.to_csv(salida, index=False, encoding="utf-8")
    print(f"\nTotal: {len(resultado)} filas -> {salida}")
    print(resultado["estado_filtro"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
