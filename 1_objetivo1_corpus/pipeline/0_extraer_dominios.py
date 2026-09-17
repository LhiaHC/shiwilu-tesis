"""
Etapa 0 — Extraccion de vocabulario por dominio semantico (OE1).

Clasifica el vocabulario en espanol de los textos narrativos bilingues segun los
ocho dominios semanticos del proyecto.

Entrada: 1_objetivo1_corpus/datos/II_TEXTOS_SHIWILU.pdf   (no versionado)
Salida:  1_objetivo1_corpus/intermedios/vocabulario_dominios.json
         1_objetivo1_corpus/intermedios/logs/vocabulario_dominios_log.json

Requiere ANTHROPIC_API_KEY.

Nota: esta etapa solo escribe el JSON de vocabulario. Ampliar las listas de
`semillas` de `shiwilu/dominios.py` con ese resultado es un paso manual y
deliberado, para que el codigo versionado no cambie por efecto secundario de
una ejecucion.
"""

import os
import json
import re
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import anthropic

import _ruta_raiz  # noqa: F401  (deja importable el paquete `shiwilu`)
from shiwilu.dominios import DOMINIOS as DOMINIOS_CONFIG
from shiwilu.rutas import LOGS, PDF, VOCABULARIO, preparar_directorios

preparar_directorios()

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
if not API_KEY:
    raise ValueError("[ERROR] No se encontro ANTHROPIC_API_KEY en el archivo .env")

MODELO = "claude-haiku-4-5-20251001"

DOMINIOS = {k: v["desc"] for k, v in DOMINIOS_CONFIG.items()}


def leer_pdf() -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(PDF))
    texto  = "".join(p.extract_text() or "" for p in reader.pages)
    print(f"  PDF: {len(reader.pages)} paginas, {len(texto):,} caracteres")
    return texto


def extraer_vocabulario(texto: str) -> dict:
    client    = anthropic.Anthropic(api_key=API_KEY)
    fragmento = texto[:12000]

    dominios_fmt = "\n".join(f'  "{k}": "{v}"' for k, v in DOMINIOS.items())

    prompt_s = (
        "Eres un linguista especialista en lenguas amazonicas del Peru, "
        "con experiencia en textos shiwilu (jebero, ISO 639-3: jeb). "
        "Tu tarea es analizar textos narrativos bilingues shiwilu-espanol "
        "e identificar vocabulario en espanol clasificado por dominio semantico."
    )
    prompt_u = f"""Analiza el fragmento de textos narrativos shiwilu y extrae vocabulario en espanol.

DOMINIOS SEMANTICOS:
{dominios_fmt}

INSTRUCCIONES:
1. Identifica palabras en espanol que aparezcan realmente en el texto (no inventes).
2. Clasifica entre 20 y 40 palabras por dominio.
3. Incluye solo palabras simples o frases cortas (max. 2 palabras).
4. Prioriza vocabulario culturalmente relevante para la comunidad kawapanana.
5. Si un dominio tiene pocas palabras en el texto, incluye las que encuentres (min. 5).

TEXTO A ANALIZAR:
---
{fragmento}
---

Responde UNICAMENTE con un JSON valido, sin texto adicional ni backticks:
{{
  "D1_Naturaleza": ["palabra1", "palabra2", ...],
  "D2_Cuerpo": ["palabra1", ...],
  "D3_Familia": ["palabra1", ...],
  "D4_Alimentos": ["palabra1", ...],
  "D5_Lugar": ["palabra1", ...],
  "D6_Tiempo": ["palabra1", ...],
  "D7_Actividades": ["palabra1", ...],
  "D8_Social": ["palabra1", ...]
}}"""

    resp = client.messages.create(
        model=MODELO, max_tokens=4096,
        system=prompt_s,
        messages=[{"role": "user", "content": prompt_u}],
    )
    texto_resp = resp.content[0].text.strip()
    texto_resp = re.sub(r"^```(?:json)?\s*", "", texto_resp, flags=re.DOTALL)
    texto_resp = re.sub(r"\s*```$",           "", texto_resp, flags=re.DOTALL)
    vocab = json.loads(texto_resp)

    log = {
        "modelo":    MODELO,
        "timestamp": datetime.now().isoformat(),
        "palabras_por_dominio": {k: len(v) for k, v in vocab.items()},
    }
    with open(LOGS / "vocabulario_dominios_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    return vocab


if __name__ == "__main__":
    print("[0] Extrayendo dominios del PDF con API de Claude...")
    texto = leer_pdf()
    print("  Llamando a la API...")
    vocab = extraer_vocabulario(texto)

    with open(VOCABULARIO, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print()
    print(f"  {'Dominio':<16} {'Palabras':>8}")
    print("  " + "-" * 26)
    for dom, palabras in vocab.items():
        print(f"  {dom:<16} {len(palabras):>8}")
    total = sum(len(v) for v in vocab.values())
    print("  " + "-" * 26)
    print(f"  {'TOTAL':<16} {total:>8}")
    print()
    print(f"  [OK] vocabulario guardado en {VOCABULARIO}")
