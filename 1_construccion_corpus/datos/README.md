# Materiales fuente

## `flashcards2.csv`

Tarjetas de vocabulario bilingüe español–shiwilu, usadas como fuente principal
para la extracción de pares del corpus (Etapa 1 del pipeline).

Columnas relevantes: `CODIGO`, `ESP`, `SHIWILU`, `Defectuosos?`

> **Pendiente de documentar:** procedencia exacta y condiciones de uso acordadas
> con la comunidad de Jeberos. Ver la sección de excepciones en
> [`LICENSE-DATOS`](../../LICENSE-DATOS).

## `II_TEXTOS_SHIWILU.pdf` (no versionado)

Textos narrativos bilingües shiwilu–español de la comunidad de Jeberos, Loreto.
Usado en la Etapa 0 para la extracción de vocabulario por dominio semántico.

Este archivo **no se incluye en el repositorio** por tratarse de material de
terceros. El vocabulario extraído de él sí está disponible en
`../intermedios/vocabulario_dominios.json`.

Para reejecutar la Etapa 0, colocar el PDF en esta carpeta con ese nombre exacto.
