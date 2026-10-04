# Histórico (OE2)

Material de etapas anteriores de OE2. **Ya no se corre ni se actualiza**; se conserva
como evidencia de cómo se llegó al protocolo vigente (validación cruzada de 5 folds
sobre texto normalizado, en [`../evaluacion/`](../evaluacion/)). No usar estos números
para reportar resultados.

```
historico/
├── split_unico/            Primera versión: un solo split 70/15/15 y texto crudo
│   ├── split_fijo.csv        split congelado
│   ├── baseline.py, correr_todos.py, evaluar_aumento_texto.py
│   ├── bootstrap_ic.py, resumen_experimentos.py, curva_aprendizaje.py
│   ├── resultados_baselines/ y resultados_tecnicas/    salidas por modelo / técnica
│   └── intervalos_confianza.csv, resumen_experimentos.csv, curva_aprendizaje.{csv,png}
│
├── en_linea_protocolo_dev/ Generación en línea con elección de C en un dev de ~90 oraciones
│   └── resultados/           Generate-then-Refine 20, 40, 80 y 120 por categoría, Retrotraducción ×1
│                              (predicciones, resumen, pareadas, ROC y volumen). Ver su README
│
├── colab_zips/             Zips descargados de Colab (contienen lo generado y los resultados de arriba)
│
└── atajos_texto_crudo/     Validación cruzada con el texto tal cual (con signos y mayúsculas)
    ├── validacion_cruzada_{predicciones,resumen}*.csv   condiciones: texto crudo, solo
    │                                                    minúsculas, sin puntuación con mayúsculas
    ├── comparacion_pareada.csv
    └── auditoria_optimismo.py   análisis de por qué el F1 con texto crudo estaba inflado
```

**Por qué se descartó.** El corpus tiene atajos superficiales que delatan la
categoría sin ser señal lingüística: los signos `¿?` aparecen en el 100% de las
preguntas (PRG) y las oraciones de DES, PRG y REQUEST están 100% en mayúsculas
(10-21% en las demás). Con texto crudo el F1 macro era ~0.75; al normalizar
(minúsculas, sin puntuación, sin tildes/ñ) baja a ~0.55-0.66, que es el nivel real.
Ver [`split_unico/README.md`](split_unico/README.md) y
[`../tecnicas_aumento/README.md`](../tecnicas_aumento/README.md) para el detalle.

Los scripts siguen siendo ejecutables (usan `shiwilu/rutas.py`), pero sus salidas
escriben aquí mismo. `validacion_cruzada.py --condicion original|minusculas|
sin_puntuacion_mayusculas` también escribe en `atajos_texto_crudo/`.
