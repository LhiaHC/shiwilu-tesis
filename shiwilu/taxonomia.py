"""
Taxonomia de intenciones comunicativas del corpus shiwilu.

Definida a partir de la teoria de actos ilocucionarios de Searle (1975) y
validada contra el benchmark multilingue MASSIVE (FitzGerald et al., 2022).

Este modulo es la unica fuente de verdad de la taxonomia: lo importan tanto el
pipeline de construccion (Fase 1) como los notebooks de analisis (Fase 2).
"""

INTENCIONES = ["SAL", "EMO", "PRG", "REQUEST", "AFI", "NEG", "DES"]

DESC_INT = {
    "SAL":     "Saludos y despedidas",
    "EMO":     "Expresiones emocionales",
    "PRG":     "Preguntas informativas",
    "REQUEST": "Solicitudes y mandatos",
    "AFI":     "Afirmaciones y confirmaciones",
    "NEG":     "Negaciones y rechazos",
    "DES":     "Descripciones de estados o eventos",
}

TIPO_SEARLE = {
    "SAL":     "expresivo",
    "EMO":     "expresivo",
    "PRG":     "directivo",
    "REQUEST": "directivo",
    "AFI":     "asertivo",
    "NEG":     "asertivo",
    "DES":     "asertivo",
}

# Etiquetas anteriores unificadas en REQUEST durante la revision manual.
NORMALIZAR_INT = {"MAN": "REQUEST", "SOL": "REQUEST"}

# Color de relleno por intencion en los reportes Excel de la Fase 1.
COLOR_INT = {
    "SAL":     "C8E6C9",
    "EMO":     "FFF9C4",
    "PRG":     "B3E5FC",
    "REQUEST": "FFCCBC",
    "AFI":     "DCEDC8",
    "NEG":     "F8BBD0",
    "DES":     "CFD8DC",
}
