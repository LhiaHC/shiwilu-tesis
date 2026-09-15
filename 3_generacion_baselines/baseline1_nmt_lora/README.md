# Baseline 1 — NLLB-200 + LoRA (reentrenado desde el pipeline de F. Prado)

Sistema NMT NLLB-200 adaptado mediante LoRA al par español-shiwilu. La
configuración objetivo es la que Fabian Prado identificó como campeona en su
propia tesis: `v2.1b LoRA+` (NLLB-200-distilled-600M + LoRA r=32/α=64 con
optimizador LoRA+, `lr_B = 16·lr_A`, sobre la variante `xl` de su corpus con
backtranslation). Resultado reportado por su autor: chrF++ promedio
(reranked) = 44.99 (F. Prado, comunicación personal, 14 de septiembre de
2026).

No copiamos su código ni sus datos a este repositorio: se clonan como
dependencia externa y se reentrena con su propio pipeline, para no duplicar
autoría ni arrastrar material de terceros dentro de este repo.

## 1. Clonar el repositorio de referencia

```bash
git clone https://github.com/fapi19/Tesis_Spa-Jeb.git tesis_spa_jeb
cd tesis_spa_jeb
```

La carpeta `tesis_spa_jeb/` queda ignorada por git en este repositorio (ver
`.gitignore`) — es un clon local, no un submódulo versionado.

## 2. Instalar sus dependencias

Sigue las instrucciones de su propio `README.md` (usa Poetry). Requiere GPU
para un tiempo de entrenamiento razonable (recomendado: Google Colab con GPU,
igual que el resto de esta tesis).

## 3. Entrenar la configuración campeona

Desde dentro de `tesis_spa_jeb/`, con sus propios datos procesados ya
incluidos en el repositorio (`data/processed/06_nmt_filtered_xl/`):

```bash
python -m scripts.nmt.30_train_lora \
    --variant xl \
    --rank 32 \
    --alpha 64 \
    --loraplus-lr-ratio 16 \
    --output-dir models/nmt/nllb_bidi_lora_v2_1b_loraplus_xl
```

Esto reproduce el checkpoint campeón sin necesitar los pesos originales de
Fabian. Verifica al final del entrenamiento con su propio script de
evaluación:

```bash
python -m scripts.nmt.40_evaluate --checkpoint models/nmt/nllb_bidi_lora_v2_1b_loraplus_xl --split test
```

Un chrF++ promedio cercano a 44.99 confirma que la reproducción es
equivalente a la reportada por el autor.

## 4. Generar traducciones para el aumento de datos (R6)

Con el checkpoint ya entrenado, usa `generar.py` de esta carpeta (no el de su
repo — este es el nuestro, pensado para tomar oraciones nuevas en español y
producir shiwilu en el formato del corpus de esta tesis):

```bash
python 3_generacion_baselines/baseline1_nmt_lora/generar.py \
    --checkpoint 3_generacion_baselines/baseline1_nmt_lora/tesis_spa_jeb/models/nmt/nllb_bidi_lora_v2_1b_loraplus_xl \
    --entrada ruta/a/oraciones_nuevas.csv \
    --salida 3_generacion_baselines/baseline1_nmt_lora/salidas/baseline1_generado.csv
```

`oraciones_nuevas.csv` debe tener al menos las columnas `id`, `espanol` e
`intencion` (mismo esquema que `corpus/corpus_shiwilu_final.csv`, pero sin la
columna `shiwilu`, que es la que este script completa).

## Cita

> F. Prado (comunicación personal, 14 de septiembre de 2026). Código:
> Prado, F. (2026). *Tesis_Spa-Jeb* [Repositorio de código]. GitHub.
> <https://github.com/fapi19/Tesis_Spa-Jeb>
