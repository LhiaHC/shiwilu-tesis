"""
Prepara una COPIA del corpus con las etiquetas propuestas para las oraciones DES que la auditoria marco como dudosas.

NO modifica `corpus/corpus_shiwilu_final.csv`. Escribe en analisis_intrinseco/resultados/auditoria_des/:
  corpus_shiwilu_propuesta_des.csv   misma estructura que el corpus (id, espanol, shiwilu, intencion, fuente) con las
                                     etiquetas propuestas aplicadas (solo las de confianza alta y media)
  cambios_propuestos_des.csv         cada propuesta: de DES a que categoria, confianza, regla, y si se aplico

Las propuestas son HIPOTESIS de revision, no decisiones: la etiqueta correcta la decide la autora (idealmente con un hablante).
Reglas usadas (se apoyan en como ya estan etiquetadas las otras categorias del corpus):
  REQUEST   imperativo explicito ("Lleva esto", "Anoten esto"); consejo o exhortacion; "Puedes + verbo" como "Puedes irte" y las
            15 "¿Puedes...?"; volitivo/necesidad como "Necesito descansar/trabajar/respuestas"; pedidos de auxilio como "Apoyanos"
  EMO       afecto en PRIMERA persona, como "Me gustan", "Me preocupa", "Me alegra el amanecer"; interjecciones como "¡Que horror!"
  SAL       aviso de partida/despedida, como "Nos vemos", "Hasta luego"
  AFI       capacidad en primera persona, como "Se correr", "Se leer", "Se nadar" (AFI en el corpus)
Se mantienen en DES las descripciones de estados o eventos de segunda y tercera persona ("Es feliz", "Le gusto", "Pareces triste").

Uso (desde la raiz del repositorio):
    python oe2_aumento_de_datos/analisis_intrinseco/proponer_correccion_des.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # raiz del repo: deja importable `shiwilu`

from shiwilu.clasificacion import cargar_corpus  # noqa: E402
from shiwilu.rutas import ANALISIS_INTRINSECO  # noqa: E402

SALIDA = ANALISIS_INTRINSECO / "resultados" / "auditoria_des"
APLICAR = {"alta", "media"}

# id -> (categoria propuesta, confianza, regla)
PROPUESTAS = {
    # --- REQUEST
    "FL2_0748": ("REQUEST", "alta", "imperativo explicito (como 'Lleva esto', 'Anoten esto')"),
    "FL2_0750": ("REQUEST", "alta", "imperativo explicito (como 'Lleva esto', 'Anoten esto')"),
    "FL2_2179": ("REQUEST", "media", "consejo o exhortacion al interlocutor"),
    "FL2_2180": ("REQUEST", "media", "consejo o exhortacion al interlocutor"),
    "FL2_2176": ("REQUEST", "media", "'Puedes + verbo' (como 'Puedes irte' y las 15 '¿Puedes...?', ya REQUEST)"),
    "FL2_2177": ("REQUEST", "media", "'Puedes + verbo' (como 'Puedes irte' y las 15 '¿Puedes...?', ya REQUEST)"),
    "FL2_0797": ("REQUEST", "media", "volitivo/necesidad (como 'Necesito descansar', ya REQUEST)"),
    "FL2_0799": ("REQUEST", "media", "volitivo/necesidad (como 'Necesito descansar', ya REQUEST)"),
    "FL2_0791": ("REQUEST", "media", "volitivo/necesidad (como 'Necesito respuestas', ya REQUEST)"),
    "FL2_0009": ("REQUEST", "media", "pedido de auxilio (como 'Apoyanos', ya REQUEST)"),
    "FL2_0011": ("REQUEST", "media", "pedido de auxilio (como 'Apoyanos', ya REQUEST)"),
    "FL2_0010": ("REQUEST", "media", "pedido de auxilio (como 'Apoyanos', ya REQUEST)"),
    "FL2_0021": ("REQUEST", "media", "exhortacion imperativa ('¡Al ataque!')"),
    # --- EMO
    "FL2_0778": ("EMO", "media", "afecto en primera persona (como 'Me preocupa')"),
    "FL2_0702": ("EMO", "media", "afecto en primera persona (como 'Me preocupa')"),
    "FL2_0817": ("EMO", "media", "afecto en primera persona (como 'Me preocupa')"),
    "FL2_0814": ("EMO", "media", "afecto en primera persona (como 'Me preocupa')"),
    "FL2_0687": ("EMO", "media", "afecto en primera persona (como 'Me gustan', ya EMO)"),
    "FL2_0786": ("EMO", "media", "afecto en primera persona ('La amaba')"),
    "FL2_0787": ("EMO", "media", "afecto en primera persona ('Lo extrañaba')"),
    "FL2_0017": ("EMO", "media", "interjeccion (como '¡Que horror!', ya EMO)"),
    "FL2_0018": ("EMO", "media", "interjeccion (como '¡Que horror!', ya EMO)"),
    # --- SAL
    "FL2_0820": ("SAL", "media", "aviso de partida/despedida (como 'Nos vemos', 'Hasta luego')"),
    # --- AFI
    "FL2_0770": ("AFI", "media", "capacidad en primera persona (como 'Se correr', 'Se leer', ya AFI)"),
    "FL2_0771": ("AFI", "media", "capacidad en primera persona (como 'Se correr', 'Se leer', ya AFI)"),
    # --- baja confianza: se listan pero NO se aplican
    "FL2_0690": ("REQUEST", "baja", "volitivo en plural ('Lo queremos'); es una declaracion, no un pedido claro"),
    "FL2_0689": ("REQUEST", "baja", "exhortacion ('Debemos irnos') o declaracion de obligacion"),
    "FL2_0008": ("EMO", "baja", "alerta exclamativa ('¡Incendio!'); tambien podria ser REQUEST o quedarse en DES"),
    "FL2_0016": ("EMO", "baja", "exclamacion de triunfo ('¡He ganado!')"),
    "FL2_0759": ("EMO", "baja", "'Le gusto': afecto en tercera persona; por la regla seria DES"),
    "FL2_0737": ("AFI", "baja", "ofrecimiento ('Cuenta conmigo'); no encaja del todo en ninguna categoria"),
    "FL2_0684": ("AFI", "baja", "posibilidad ('Podemos ganar'); es una declaracion de capacidad colectiva"),
    "FL2_0772": ("AFI", "baja", "admision ('Confese')"),
}


def main() -> int:
    SALIDA.mkdir(parents=True, exist_ok=True)
    corpus = cargar_corpus()
    por_id = corpus.set_index("id")
    filas = []
    for id_, (destino, confianza, regla) in PROPUESTAS.items():
        assert id_ in por_id.index, f"no existe {id_}"
        assert por_id.loc[id_, "intencion"] == "DES", f"{id_} no es DES hoy: {por_id.loc[id_, 'intencion']}"
        filas.append({"id": id_, "espanol": por_id.loc[id_, "espanol"], "shiwilu": por_id.loc[id_, "shiwilu"],
                      "intencion_original": "DES", "intencion_propuesta": destino, "confianza": confianza,
                      "regla": regla, "aplicado": confianza in APLICAR})
    cambios = pd.DataFrame(filas).sort_values(["aplicado", "intencion_propuesta", "confianza"], ascending=[False, True, True])
    cambios.to_csv(SALIDA / "cambios_propuestos_des.csv", index=False, encoding="utf-8")

    copia = corpus.copy()
    aplicar = cambios[cambios["aplicado"]].set_index("id")["intencion_propuesta"]
    copia["intencion"] = [aplicar.get(i, y) for i, y in zip(copia["id"], copia["intencion"])]
    copia.to_csv(SALIDA / "corpus_shiwilu_propuesta_des.csv", index=False, encoding="utf-8")

    print(f"Propuestas: {len(cambios)} ({int(cambios['aplicado'].sum())} aplicadas en la copia, {int((~cambios['aplicado']).sum())} de baja confianza sin aplicar)")
    print("\nDestinos de las aplicadas:\n", cambios[cambios["aplicado"]]["intencion_propuesta"].value_counts().to_string())
    comp = pd.DataFrame({"original": corpus["intencion"].value_counts(), "propuesta": copia["intencion"].value_counts()})
    print("\nOraciones por categoria:\n", comp.to_string())
    print("\nGuardado en", SALIDA)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
