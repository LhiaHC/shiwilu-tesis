# Versión anterior del protocolo B (9 de 12 experimentos, sin Generate-then-Refine)

Archivos unidos de la primera versión del protocolo B, cuando solo estaban calculados sin aumento, Mixup y Retrotraducción (el nivel de GtR faltaba). Se conservan sin modificar; **los resultados vigentes
con los 12 experimentos son los `*_c120.csv`, `*_c80.csv` y `*_c40.csv` de la carpeta superior.**

| Modelo | Sin aumento | Mixup | Retrotraducción |
|---|---|---|---|
| mBERT | 0.665 | 0.629 | 0.631 |
| LaBSE | 0.587 | 0.587 | 0.581 |
| XLM-R | 0.570 | 0.546 | 0.551 |

Las cifras de esas tres filas no cambiaron al agregar GtR (se reutilizaron las mismas predicciones).
