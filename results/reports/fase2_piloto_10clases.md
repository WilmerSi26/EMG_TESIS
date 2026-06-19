# Fase 2 - piloto de 10 clases funcionales

Fecha: 2026-06-18

## Objetivo

Validar de forma preliminar si la reduccion de 14 clases individuales a 10 clases funcionales mantiene separabilidad suficiente antes de adquirir un nuevo dataset final con la XIAO a 200 Hz.

## Dataset usado

Origen:

```text
D:\ESPOCH\TESIS\CODEX\CARACTERIZACIOIN\data\raw
```

Patron usado:

```text
20260609_*_S001_trial-1_rutina_4ch.csv
```

Cantidad: 16 CSV principales.

No se usaron pruebas exploratorias `CAMBIO`, `CAMBIO2`, `ACTUAL` ni archivos de reubicacion de sensores.

## Procesamiento realizado

1. Remapeo de 14 clases a 10 clases funcionales.
2. Submuestreo por factor 5 para simular el flujo objetivo de 200 Hz.
3. Segmentacion por ventanas de 40 muestras con stride de 10 muestras.
4. Extraccion de 28 caracteristicas por ventana: 7 caracteristicas por cada uno de los 4 canales.

En 200 Hz, 40 muestras equivalen a 200 ms y stride 10 equivale a 50 ms.

## Clases funcionales

```text
reposo
apertura
cierre
pulgar_apertura
pulgar_cierre
indice_medio_apertura
indice_medio_cierre
anular_menique_apertura
anular_menique_cierre
pinza
```

## Resultados RandomForest

Archivo de modelo:

```text
D:\ESPOCH\TESIS\CARACTERIZACION_FINAL\models\baseline\emg_sklearn_baseline.joblib
```

Exactitud de prueba:

```text
84.10 %
```

Resumen por clase:

```text
reposo: F1 = 0.93
apertura: F1 = 0.86
cierre: F1 = 0.89
pulgar_apertura: F1 = 0.73
pulgar_cierre: F1 = 0.78
indice_medio_apertura: F1 = 0.81
indice_medio_cierre: F1 = 0.77
anular_menique_apertura: F1 = 0.81
anular_menique_cierre: F1 = 0.80
pinza: F1 = 0.79
```

Interpretacion: la agrupacion funcional es viable como estructura de adquisicion. El resultado no es final porque el dataset fue remapeado desde movimientos individuales, no grabado directamente con instrucciones funcionales.

## Resultados Tiny MLP

Archivo TFLite:

```text
D:\ESPOCH\TESIS\CARACTERIZACION_FINAL\models\tiny_mlp\emg_tiny_mlp.tflite
```

Exactitud de prueba:

```text
64.69 %
```

Resumen por clase:

```text
reposo: F1 = 0.89
apertura: F1 = 0.67
cierre: F1 = 0.76
pulgar_apertura: F1 = 0.42
pulgar_cierre: F1 = 0.50
indice_medio_apertura: F1 = 0.50
indice_medio_cierre: F1 = 0.48
anular_menique_apertura: F1 = 0.54
anular_menique_cierre: F1 = 0.54
pinza: F1 = 0.52
```

Interpretacion: el modelo compacto actual no es suficiente como version final embebida. Sirve como primera referencia TFLite, pero se recomienda mejorar el modelo antes de cargarlo a la XIAO.

## Decision tecnica

Se mantiene la propuesta de 10 clases funcionales para la adquisicion final:

- conserva control direccional de apertura/cierre;
- reduce complejidad respecto a 14 clases;
- coincide mejor con la mecanica de servos por grupos;
- evita entrenar movimientos individuales que la protesis no ejecutara individualmente.

## Recomendaciones para la siguiente fase

1. Crear firmware RTOS a 200 Hz.
2. Programar rutina de adquisicion con las 10 clases finales, no remapeadas.
3. Grabar 2 o 3 sesiones nuevas con la misma colocacion del brazalete.
4. Entrenar nuevamente RandomForest, Tiny MLP y una CNN1D compacta.
5. Elegir candidato embebido solo si supera al menos 80 % de exactitud y mantiene F1 aceptable por clase.

## Archivos generados

```text
D:\ESPOCH\TESIS\CARACTERIZACION_FINAL\data\pilot_remap_14_to_10
D:\ESPOCH\TESIS\CARACTERIZACION_FINAL\data\pilot_remap_14_to_10_200hz
D:\ESPOCH\TESIS\CARACTERIZACION_FINAL\models\baseline
D:\ESPOCH\TESIS\CARACTERIZACION_FINAL\models\tiny_mlp
D:\ESPOCH\TESIS\CARACTERIZACION_FINAL\results\figures
D:\ESPOCH\TESIS\CARACTERIZACION_FINAL\results\metrics
```

