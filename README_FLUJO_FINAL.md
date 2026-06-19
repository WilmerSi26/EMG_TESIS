# Caracterizacion final EMG - 10 clases funcionales

Este flujo conserva el proyecto anterior como referencia y trabaja en una carpeta nueva:

```text
D:\ESPOCH\TESIS\CARACTERIZACION_FINAL
```

## Clases finales

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

En `reposo` la mano queda relajada. Para control embebido, reposo debe mantener la posicion actual o retornar a neutral solo si se activa un modo de retorno temporizado.

## Fase 2 - piloto con datos previos

El piloto remapea los CSV principales de 14 clases a 10 clases funcionales:

```powershell
cd D:\ESPOCH\TESIS\CARACTERIZACION_FINAL
D:\ESPOCH\TESIS\CODEX\CARACTERIZACIOIN\.venv\Scripts\python.exe src\preprocessing\remap_14_to_10.py --clear-output
D:\ESPOCH\TESIS\CODEX\CARACTERIZACIOIN\.venv\Scripts\python.exe src\training\train_baseline_rf.py --input-dir data\pilot_remap_14_to_10 --models-dir models\baseline --figures-dir results\figures --metrics-dir results\metrics --window-size 40 --stride 10
```

El entrenamiento con `window-size 40` y `stride 10` corresponde a una configuracion objetivo de 200 Hz, equivalente a ventanas de 200 ms y desplazamiento de 50 ms.

## Nota metodologica

Los resultados de este piloto no sustituyen el dataset final. Sirven para comprobar si la reduccion funcional de clases mejora la separabilidad antes de adquirir datos nuevos con la XIAO configurada a 200 Hz.

