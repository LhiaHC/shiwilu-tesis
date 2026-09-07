"""
Pruebas de las reglas de anotacion por intencion.

Estas reglas son el aporte metodologico de la Fase 1 y, ademas, el baseline
contra el que la Fase 2 contrasta los clasificadores basados en embeddings.
Un cambio silencioso aqui afecta a ambas fases, por eso conviene fijar el
comportamiento esperado.

Ejecutar desde la raiz del repositorio:  python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shiwilu.anotacion import anotar_dominio, anotar_intencion
from shiwilu.dominios import DOMINIOS
from shiwilu.taxonomia import DESC_INT, INTENCIONES, TIPO_SEARLE


@pytest.mark.parametrize("texto, esperada", [
    # Saludos y despedidas
    ("Hola.", "SAL"),
    ("Buenos días.", "SAL"),
    ("Hasta luego.", "SAL"),
    ("Nos vemos.", "SAL"),
    # Preguntas informativas
    ("¿Dónde está la chacra?", "PRG"),
    ("¿Cuántos hijos tienes?", "PRG"),
    # Solicitudes y mandatos
    ("¿Puedes traer agua?", "REQUEST"),
    ("Ayúdame.", "REQUEST"),
    ("Trae la olla.", "REQUEST"),
    ("Cantemos.", "REQUEST"),
    ("No fumes.", "REQUEST"),
    # Expresiones emocionales
    ("Gracias.", "EMO"),
    ("Lo siento.", "EMO"),
    ("¡Qué bonito!", "EMO"),
    # Afirmaciones
    ("De acuerdo.", "AFI"),
    ("Así es.", "AFI"),
    ("Sé nadar.", "AFI"),
    # Negaciones
    ("No puedo.", "NEG"),
    ("Nunca.", "NEG"),
    ("Me niego.", "NEG"),
])
def test_intencion_esperada(texto, esperada):
    intencion, _ = anotar_intencion(texto)
    assert intencion == esperada, f"{texto!r} -> {intencion}, se esperaba {esperada}"


@pytest.mark.parametrize("texto", ["", "   ", None, 123])
def test_entrada_invalida_cae_en_des(texto):
    """La regla por defecto nunca debe fallar ante entradas degeneradas."""
    intencion, confianza = anotar_intencion(texto)
    assert intencion == "DES"
    assert 0.0 <= confianza <= 1.0


def test_intencion_siempre_en_taxonomia():
    ejemplos = [
        "Hola.", "¿Qué comes?", "Dame agua.", "Gracias.", "Claro.",
        "No hay.", "El río crece.", "Cualquier texto suelto sin patron.",
    ]
    for texto in ejemplos:
        intencion, confianza = anotar_intencion(texto)
        assert intencion in INTENCIONES
        assert 0.0 <= confianza <= 1.0


def test_dominio_siempre_valido():
    ejemplos = ["Traigo yuca.", "Me duele la cabeza.", "Mi madre cocina.",
                "texto sin ninguna semilla reconocible"]
    for texto in ejemplos:
        assert anotar_dominio(texto) in DOMINIOS


def test_dominio_por_defecto_ante_entrada_invalida():
    assert anotar_dominio(None) == "D8_Social"


def test_taxonomia_completa_y_consistente():
    """Las tres tablas de la taxonomia deben cubrir las mismas intenciones."""
    assert set(DESC_INT) == set(INTENCIONES)
    assert set(TIPO_SEARLE) == set(INTENCIONES)
    assert set(TIPO_SEARLE.values()) <= {"expresivo", "directivo", "asertivo"}
