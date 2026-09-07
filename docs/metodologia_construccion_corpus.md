# Metodología de construcción del corpus anotado shiwilu

Proyecto de Tesis 1 — PUCP 2026
Autora: Lhia Hurtado (lhia.hurtado@pucp.edu.pe)

---

## 1. Visión general del pipeline

La construcción del corpus se organizó en tres etapas automatizadas ejecutadas
secuencialmente mediante scripts de Python:

| Etapa | Script | Descripción | Uso de API |
|-------|--------|-------------|------------|
| 0 | `0_extraer_dominios.py` | Extracción de vocabulario por dominio semántico desde el PDF de textos shiwilu | Sí |
| 1 | `1_generar_pares_anotados.py` | Anotación automática de pares español–shiwilu desde las flashcards | No |
| 2 | `2_generar_oraciones.py` | Generación de oraciones nuevas en español para cubrir el déficit por categoría | Sí |

---

## 2. Etapa 0 — Extracción de dominios semánticos

### Objetivo
Identificar vocabulario culturalmente relevante del corpus textual shiwilu
y clasificarlo en 8 dominios semánticos predefinidos, para luego usarlo como
restricción léxica en la generación de oraciones (Etapa 2).

### Fuente de datos
Archivo `II_TEXTOS_SHIWILU.pdf`: textos narrativos bilingües shiwilu–español
de la comunidad de Jeberos, Loreto, Perú. Se procesaron los primeros 12 000
caracteres del PDF.

### Modelo utilizado
- **Proveedor:** Anthropic API
- **Modelo:** `claude-haiku-4-5-20251001`
- **Parámetros:** `max_tokens=4096`

### Técnica de prompting: Role Prompting + Zero-shot + Salida estructurada

Se utilizó una combinación de tres estrategias:

**a) Role prompting (system prompt)**
Se instruyó al modelo para que adoptara el rol de un lingüista especialista
en lenguas amazónicas del Perú con experiencia en textos shiwilu:

> *"Eres un lingüista especialista en lenguas amazónicas del Perú, con experiencia
> en textos shiwilu (jebero, ISO 639-3: jeb). Tu tarea es analizar textos
> narrativos bilingües shiwilu-español e identificar vocabulario en español
> clasificado por dominio semántico."*

**b) Zero-shot prompting**
No se proporcionaron ejemplos previos de clasificación. El modelo recibió
únicamente la definición de los 8 dominios semánticos y el fragmento de texto
a analizar, infiriendo la clasificación sin ejemplos de referencia.

**c) Salida estructurada (JSON)**
Se instruyó al modelo a responder exclusivamente con un objeto JSON válido
con una clave por dominio y una lista de términos como valor. Esto permitió
parsear la respuesta programáticamente e integrarla directamente en `config.py`.

### Dominios semánticos definidos

| Código | Dominio | Descripción |
|--------|---------|-------------|
| D1 | Naturaleza | Entorno natural amazónico: flora, fauna, agua, clima, astros |
| D2 | Cuerpo | Cuerpo humano, salud, estados físicos y enfermedad |
| D3 | Familia | Vínculos familiares, generacionales y de parentesco |
| D4 | Alimentos | Alimentación, cocina, recolección y preparación de bebidas |
| D5 | Lugar | Espacio, ubicación, deixis, movimiento y poblados |
| D6 | Tiempo | Temporalidad, ciclos naturales y referencias temporales |
| D7 | Actividades | Actividades cotidianas, laborales, productivas y rituales |
| D8 | Social | Interacción social, identidad, valores, emociones colectivas |

---

## 3. Etapa 1 — Anotación automática de pares

### Objetivo
Clasificar cada par español–shiwilu de las flashcards según la categoría de
intención comunicativa del enunciado en español.

### Método: Anotación basada en reglas (sin API)
Se implementó la función `anotar_intencion()` en `config.py` mediante un
sistema de reglas de expresiones regulares en cascada. No se utilizó ninguna
llamada a la API en esta etapa.

El sistema evalúa el texto en orden de prioridad:
1. Patrones léxicos de inicio de enunciado (p. ej., saludos, interrogativos)
2. Patrones morfosintácticos (imperativo, subjuntivo negativo, futuro)
3. Patrones de forma verbal (1.ª persona, imperativo con clítico)
4. Clasificación residual como DES si ningún patrón coincide

### Taxonomía de intenciones (Searle, 1975)

| Código | Intención | Tipo Searle | Descripción |
|--------|-----------|-------------|-------------|
| SAL | Saludos y despedidas | Expresivo | Apertura o cierre de interacción social |
| EMO | Expresiones emocionales | Expresivo | Manifestación de estados afectivos |
| PRG | Preguntas informativas | Directivo | Solicitud de información verbal |
| REQUEST | Solicitudes y mandatos | Directivo | Solicitud de acción al interlocutor |
| AFI | Afirmaciones y confirmaciones | Asertivo | Aceptación de algo como verdadero |
| NEG | Negaciones y rechazos | Asertivo | Rechazo o negación de una proposición |
| DES | Descripciones de estados o eventos | Asertivo | Enunciados descriptivos de hechos o situaciones |

Las categorías MAN (mandato) y SOL (solicitud) fueron unificadas bajo REQUEST,
siguiendo el esquema de clasificación de intenciones del benchmark MASSIVE
(Fitzgerald et al., 2022) y el framework Rasa.

### Criterio de selección
De cada categoría se seleccionaron hasta 100 pares ordenados por confianza
descendente. El puntaje de confianza (0–1) fue asignado por el mismo sistema
de reglas según la especificidad del patrón coincidente.

---

## 4. Etapa 2 — Generación de oraciones nuevas

### Objetivo
Generar oraciones en español para las categorías con déficit (menos de 100
pares en la Etapa 1), de modo que el corpus final alcance 100 ejemplos por
cada categoría de intención.

### Modelo utilizado
- **Proveedor:** Anthropic API
- **Modelo:** `claude-sonnet-4-6`
- **Parámetros:** `max_tokens=8192`

### Técnica de prompting: Role Prompting + Few-shot + Restricciones explícitas

**a) Role prompting (system prompt)**
El modelo adoptó el rol de especialista en lingüística computacional y lenguas
amazónicas del Perú:

> *"Eres un especialista en lingüística computacional y lenguas amazónicas del
> Perú. Tu tarea es generar oraciones en español para construir un corpus de
> clasificación de intenciones para la lengua shiwilu (jebero, ISO 639-3: jeb),
> hablada en el distrito de Jeberos, Loreto, Perú."*

**b) Few-shot prompting**
Se incluyeron entre 8 y 10 ejemplos reales del corpus para cada categoría,
de modo que el modelo pudiera calibrar el registro oral, la longitud y el
estilo esperado. Por ejemplo, para SAL:

> *"Hola.", "Hasta luego.", "Nos vemos.", "Bienvenido.", "Buenas tardes."*

**c) Restricciones explícitas en el prompt**
Cada llamada incluyó restricciones específicas:
- Longitud de 1 a 5 palabras por oración (registro oral conciso)
- Vocabulario de los dominios semánticos del shiwilu (D1–D8)
- Exclusión de tecnicismos modernos (internet, celular, computadora, etc.)
- Distribución de patrones sintácticos por categoría (máximos porcentuales
  para EMO, AFI y NEG, para evitar monotonía estructural)

**d) Salida estructurada (JSON)**
La respuesta se solicitó como un JSON array con campos `oracion` y `dominio`,
permitiendo el parseo automático y la asignación de dominio semántico.

**e) Sobregeneración con filtrado posterior**
Se solicitaron `cantidad × 1.5` oraciones a la API para luego filtrar por:
- Longitud (1–5 palabras)
- Deduplicación normalizada contra el corpus anotado existente
  (normalización: minúsculas, eliminación de puntuación en ambos extremos)

### Bloque de exclusiones en el prompt
Se incluyeron hasta 40 oraciones del corpus existente directamente en el
prompt como ejemplos a evitar, reforzando la deduplicación a nivel semántico
además del filtro exacto por código.

---

## 5. Resumen técnico

| Parámetro | Etapa 0 | Etapa 2 |
|-----------|---------|---------|
| Modelo | claude-haiku-4-5-20251001 | claude-sonnet-4-6 |
| Técnica principal | Zero-shot + role prompting | Few-shot + role prompting |
| Formato de salida | JSON estructurado | JSON array estructurado |
| Tokens máximos | 4 096 | 8 192 |
| Entrada | Fragmento de PDF (12 000 chars) | Déficit + ejemplos reales + vocabulario |

---

## 6. Referencias

- Searle, J. R. (1975). A taxonomy of illocutionary acts. En K. Gunderson (Ed.),
  *Language, mind and knowledge* (pp. 344–369). University of Minnesota Press.
- Fitzgerald, J. et al. (2022). MASSIVE: A 1M-example multilingual natural
  language understanding dataset with 51 typologically diverse languages.
  *arXiv:2204.08582*.
- Anthropic. (2025). *Claude API documentation*. https://docs.anthropic.com
