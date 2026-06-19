# Reporte grafico de diferenciacion EMG

Este reporte resume si las senales de los movimientos se diferencian entre si.

- Archivos CSV analizados: 21
- Ventanas creadas: 46135

## Ventanas por clase

- apertura: 2255
- cierre: 2245
- dedo_anular_apertura: 2365
- dedo_anular_cierre: 2377
- dedo_indice_apertura: 2366
- dedo_indice_cierre: 2459
- dedo_medio_apertura: 2364
- dedo_medio_cierre: 2362
- dedo_menique_apertura: 2248
- dedo_menique_cierre: 2360
- dedo_pulgar_apertura: 2492
- dedo_pulgar_cierre: 2484
- pinza: 3012
- reposo: 14746

## Figuras generadas

- analisis_01_balance_clases.png
- analisis_02_pca_separabilidad.png
- analisis_03_ejemplos_senales.png
- analisis_04_importancia_caracteristicas.png

## Figuras de entrenamiento existentes

- sklearn_baseline_confusion_matrix.png
- confusion_matrix.png
- tiny_mlp_confusion_matrix.png
- training_history.png

## Como interpretarlo

- La matriz de confusion muestra en la diagonal los aciertos por movimiento.
- El grafico PCA muestra si las ventanas forman grupos separables.
- La importancia de caracteristicas indica que canales y medidas aportan mas al modelo base.
- Si dos clases aparecen mezcladas, se necesitan mas sesiones o mejorar la ubicacion de electrodos.
