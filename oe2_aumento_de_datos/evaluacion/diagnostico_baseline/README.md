# Por qué "mBERT sin aumento" es el mejor y el aumento no lo mejora

Diagnósticos con el protocolo congelado (5 folds, texto normalizado, embeddings congelados). Las pruebas controladas usan **C fijo** (mBERT 0.3, LaBSE y XLM-R 10) para que la elección de C no intervenga, y lo sintético
es el de la etapa "pool" de cada fold (GtR 120 por categoría). Scripts: `ngramas_caracteres.py` y `por_que_no_mejora.py`; resultados en `resultados/`. No usan la API.

## 1. Por qué mBERT queda mejor que LaBSE y XLM-R
- Un clasificador que **solo ve las letras** (TF-IDF de n-gramas de caracteres 2-5 + Regresión Logística, mismos folds y protocolo de C) da **F1 0.751**, mejor que cualquier embedding (mBERT 0.665): en este corpus la intención se
  reconoce sobre todo por la forma superficial (sufijos, partículas).
- Los tres modelos fragmentan el shiwilu en 3.3-3.7 subpalabras por palabra. La silueta intrínseca (OE3) es negativa en los tres, pero mBERT es la menos mala (-0.010 contra -0.030 LaBSE y -0.051 XLM-R), el mismo orden que el F1.
- Interpretación (no probada causalmente): mBERT conserva más la forma de las letras; LaBSE está pensado para significado y XLM-R es el que peor lo hace.

## 2. Por qué el aumento no mejora a mBERT (resultados con C fijo)

**A. No es un techo de datos: mBERT todavía aprende de datos reales.** F1 del baseline con una fracción de las ~560 oraciones reales del pool:

| Fracción de reales | mBERT | LaBSE | XLM-R |
|---|---|---|---|
| 25% | 0.508 | 0.455 | 0.435 |
| 50% | 0.575 | 0.538 | 0.518 |
| 75% | 0.630 | 0.574 | 0.554 |
| 100% | 0.667 | 0.584 | 0.570 |

La curva de mBERT sigue subiendo (+0.037 en el último tramo), así que habría espacio para mejorar con datos reales; la de LaBSE y XLM-R se aplana (+0.010 y +0.016), y con ellos casi no hay qué ganar.

**B. Lo sintético sirve mucho menos que lo real** (GtR / Retrotraducción / Mixup, mBERT):

| Prueba | GtR | Retrotraducción | Mixup |
|---|---|---|---|
| B1. F1 entrenando solo con lo sintético (con ~480 reales el F1 sería ~0.63) | 0.484 | 0.452 | 0.594 |
| B2. Acierto de la etiqueta de lo sintético por un clasificador entrenado con reales (en reales nuevos acierta ~0.67) | 0.854 | **0.520** | 0.984 |
| B3. Similitud máxima con las reales: sintético / real nuevo | 0.843 / 0.848 | 0.863 / 0.848 | **0.983** / 0.848 |
| B4. AUC de distinguir real de sintético (1 = totalmente distinto) | **0.817** | 0.654 | 0.306 |

- GtR: se distingue de lo real (AUC 0.82), y sus etiquetas coinciden con las de un clasificador real mucho más que las de las oraciones reales nuevas (0.85 contra ~0.67): son ejemplos "típicos" y fáciles, con poca información sobre la frontera.
- Retrotraducción: la mitad de sus etiquetas no coincide con lo que predice un clasificador real (ruido de etiqueta), coherente con que empeora a mBERT.
- Mixup: es casi redundante (se parece a las reales mucho más que una oración real nueva).

**C. Dosis y origen del daño (GtR en mBERT).** F1 macro: sin sintético 0.667; con 25% 0.660, 50% 0.660, 75% 0.656, 100% 0.655 (baja con la dosis); con un sintético **balanceado** (mismo número por categoría) 0.669.
Agregando solo el sintético de una categoría, el que más perjudica es DES (-0.016), luego PRG (-0.010) y SAL (-0.009); EMO, AFI y REQUEST quedan neutros (-0.001 a -0.006) o ayudan a su propia categoría. El sintético de DES, PRG, NEG y SAL baja el F1 de su propia categoría
(DES 0.547 a 0.504, PRG 0.609 a 0.581, NEG 0.844 a 0.806, SAL 0.764 a 0.747). Con LaBSE y XLM-R, GtR queda entre +0.00 y +0.02.

## Limitaciones
- Los efectos por prueba son pequeños (≤0.02) frente al ancho de los intervalos (~0.07): son consistentes en dirección, no significativos uno por uno.
- C fijo no es lo mismo que C elegido por modelo y técnica.
- La similitud de XLM-R (≈0.999 en todo) no es informativa: su espacio es casi isotrópico en coseno.
- Que la forma morfológica de lo sintético sea la causa no está probado: haría falta la evaluación de un hablante.
