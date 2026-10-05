# Auditoría de las etiquetas de DES y su solape con REQUEST

Generada con [`../../auditoria_etiquetas_des.py`](../../auditoria_etiquetas_des.py). **Solo lista casos para revisar: no cambia el corpus ni ningún resultado.**
Las decisiones sobre qué etiqueta es correcta son de la autora (idealmente con un hablante); las columnas de "propuesta" de abajo son hipótesis mías.

## Por qué DES es tan heterogénea

La metodología del corpus ([`docs/metodologia_construccion_corpus.md`](../../../../docs/metodologia_construccion_corpus.md)) indica que las flashcards se anotaron con
**reglas de expresiones regulares en cascada** (saludos, interrogativos, imperativo, subjuntivo negativo, futuro, primera persona...) y que **DES era la clase residual: "si ningún
patrón coincide"**. Las 100 oraciones DES son flashcards (ninguna generada por la API). Por eso DES mezcla descripciones genuinas con imperativos, deseos, ofrecimientos y expresiones
emocionales que las reglas no reconocieron.

## Qué se midió

Las 100 oraciones DES evaluadas con "sin aumento" en 6 corridas: 3 modelos (LaBSE, mBERT, XLM-R) × 2 condiciones de texto (sin signos y con `¿?`).
`errores_de_6` = en cuántas de las 6 corridas la oración se clasificó como otra categoría. Distribución: 16 oraciones sin ningún error, **30 con 5 o 6 errores**, 15 con los 6 errores.
Los signos `¿?` casi no cambian a DES (F1 0.53 con y 0.54 sin signos en mBERT).

Archivos: `des_oraciones_con_errores.csv` (las 100, con lo que predijo cada corrida), `des_vs_request_indicios.csv`, `des_vs_request_pares_similares.csv`, `request_predichas_como_des.csv`.

## Las 30 DES con 5-6 errores, agrupadas

**A. Probables problemas de etiqueta (parecen REQUEST u otra intención).** Con `*` las que tienen menos de 5 errores pero entran por la inconsistencia con REQUEST:

| Oración | Por qué se sospecha | Predicción más frecuente |
|---|---|---|
| Termine esto. / Terminen esto. | imperativo (las reglas no lo reconocieron) | REQUEST |
| Quiero más. / Quiero este.* / Los necesito.* | volitivo/necesidad, igual que "Necesito descansar" (REQUEST) | REQUEST / AFI |
| Puedes entrar. / Puedes decírnoslo.* | permiso/oferta; compárese "Puedes irte." (REQUEST) | REQUEST |
| Deberías descansar. | consejo/exhortación | AFI |
| ¡Ayuda! / ¡Auxilio! | pedido de ayuda; compárese "Apóyanos." (REQUEST) | REQUEST |
| ¡Incendio! | alerta | EMO |
| Cuenta conmigo. | ofrecimiento / compromiso | SAL |
| Me voy. | despedida o aviso de partida | SAL |

**B. Descripciones genuinas pero muy variadas (la etiqueta parece correcta; el modelo no encuentra un patrón):**
Él corrió · Tú dormías · Él envejeció · Escaparon todos · Está fingiendo · Leo labios · Estoy ayunando · Soy mujer · Somos pobres · Nos reímos · Me despidieron · Me traicionaste ·
Tom morirá · Fracasaremos · Podemos ganar · Puedo hacerlo · Salí · Es feliz · Le gustó · Pareces triste · Odia correr.
Algunas de este grupo rozan EMO ("Es feliz", "Le gustó", "Pareces triste", "Odia correr") y podrían revisarse también.

## Inconsistencias con REQUEST

1. **"Puedes + verbo":** 15 oraciones `¿Puede(s) + verbo?` están etiquetadas REQUEST ("¿Puedes ayudar?", "¿Puedes ver?"...), y "Puedes irte." también; pero "Puedes entrar.", "Puedes decírnoslo.", "Puedo hacerlo.",
   "Podemos ganar." y "Puedo compartir." están como DES. Es la misma forma con etiquetas distintas. El par más parecido de todo el corpus es *Puedes entrar.* (DES) ↔ *Puedes irte.* (REQUEST), similitud 0.83.
2. **"Necesito / quiero":** "Necesito trabajar.", "Necesito descansar." y "Necesito respuestas." son REQUEST, pero "Los necesito.", "Quiero más.", "Quiero este." y "Lo queremos." son DES.
   Los modelos mandan "Necesito trabajar." y "Necesito descansar." a DES en las 6 corridas, y "Necesito respuestas." en 4 de 6.
3. **Imperativos:** "Termine esto." y "Terminen esto." (DES) frente a "Lleva esto." y "Anoten esto." (REQUEST).
4. **Pedidos de ayuda:** "¡Ayuda!" y "¡Auxilio!" (DES) frente a "Apóyanos." (REQUEST).
5. **Preguntas etiquetadas REQUEST:** las 15 "¿Puede(s)...?" llevan `¿?`, así que con los signos conservados se parecen a PRG; por eso el 18% de REQUEST tiene `¿?`.
6. **REQUEST que los modelos mandan a DES** (≥3 de 6 corridas): Necesito trabajar · Necesito descansar · Apóyanos · Puedes irte · Necesito respuestas · Comamos aquí · No forcejees · ¡No llores!.

## Qué sugiere

- Parte del error de DES es **ruido de etiqueta por construcción** (clase residual de un anotador por reglas), sobre todo en el grupo A y en las inconsistencias 1 a 4. Eso pone un techo al F1 de DES sin importar el modelo ni el aumento.
- Lo demás es **heterogeneidad real**: DES agrupa descripciones sin una estructura común, y un clasificador sobre embeddings congelados no la aprende bien con ~560 oraciones de entrenamiento.
- Cambiar etiquetas (por ejemplo pasar el grupo A a REQUEST) mueve los ejemplos que ve Generate-then-Refine para DES y REQUEST, así que hay que decidirlo **antes** de generar con Claude en Colab.

## Efecto medido de la propuesta (copia `corpus_shiwilu_propuesta_des.csv`)

Se evaluó el protocolo B (CV interna K=5) con la copia del corpus (25 oraciones movidas desde DES: 13 a REQUEST, 9 a EMO, 2 a AFI, 1 a SAL; 75 DES, 113 REQUEST, 109 EMO,
102 AFI, 101 SAL), con y sin `¿?`, para sin aumento y Mixup. Resultados en `evaluacion/resultados/cv_interna[_con_interrogacion]_corpus_shiwilu_propuesta_des/`.

**La propuesta NO mejora el F1; lo baja.** F1 macro sin aumento, original -> propuesta:

| | mBERT | LaBSE | XLM-R |
|---|---|---|---|
| sin signos | 0.665 -> 0.638 | 0.587 -> 0.581 | 0.570 -> 0.561 |
| con `¿?` | 0.735 -> 0.717 | 0.684 -> 0.673 | 0.648 -> 0.646 |

- El F1 de DES baja (mBERT 0.54 -> 0.41; LaBSE 0.41 -> 0.33); REQUEST sube (+0.02 a +0.05).
- Las 25 oraciones movidas se clasifican bien con su etiqueta NUEVA solo el 24-36% de las veces: los modelos no las agrupan con las categorías propuestas.
- Las 75 DES que quedan empeoran su recall (mBERT 0.57 -> 0.43, LaBSE 0.47 -> 0.28): las oraciones movidas eran vecinas de las demás DES en el espacio de embeddings.
- Con la propuesta, Mixup vs sin aumento queda dentro del ruido en las dos condiciones.

Lectura: el error de DES no se debe principalmente a ejemplos mal etiquetados que se puedan corregir con reglas semanticas; las formas de estas oraciones se parecen a las de DES para
los embeddings. Cambiar etiquetas con criterio semantico puede ser correcto linguisticamente pero no mejora la clasificacion con estos embeddings congelados. La decision de etiquetas debe tomarse por criterio
(idealmente con un hablante), no por el F1.

## Evaluación sin la clase DES (6 categorías, 600 oraciones)

`validacion_cruzada_cv_interna.py --excluir-categorias DES` quita DES al cargar el corpus (los folds congelados siguen valiendo; el catálogo de Retrotraducción se filtra y re-indexa).
Resultados en `evaluacion/resultados/cv_interna[_con_interrogacion]_sin_DES/` para sin aumento, Mixup y Retrotraducción (Generate-then-Refine pendiente).

F1 macro sin aumento: 7 clases (original) -> 6 clases:

| | mBERT | LaBSE | XLM-R |
|---|---|---|---|
| sin signos | 0.665 -> 0.702 | 0.587 -> 0.659 | 0.570 -> 0.663 |
| con `¿?` | 0.735 -> 0.807 | 0.684 -> 0.753 | 0.648 -> 0.735 |

Parte de la subida es mecánica (se deja de promediar la categoría más baja): el macro de las 6 categorías con las MISMAS predicciones de 7 clases es 0.686 / 0.616 / 0.602 (sin signos). El resto
viene de que las 6 categorías se confunden menos al desaparecer DES (mBERT +0.016, LaBSE +0.043, XLM-R +0.061 sobre ese valor). Ninguna técnica supera a sin aumento con 6 clases; con signos todas quedan dentro del ruido.
No es una comparación estricta con el corpus de 7 clases (cambian el test y el azar de referencia, 1/6 vs 1/7).
