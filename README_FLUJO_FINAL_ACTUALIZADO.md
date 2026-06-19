# Flujo final actualizado - 10 clases funcionales

## Fase 2 ejecutada

Comandos usados:

```powershell
cd D:\ESPOCH\TESIS\CARACTERIZACION_FINAL
D:\ESPOCH\TESIS\CODEX\CARACTERIZACIOIN\.venv\Scripts\python.exe src\preprocessing\remap_14_to_10.py --clear-output
D:\ESPOCH\TESIS\CODEX\CARACTERIZACIOIN\.venv\Scripts\python.exe src\preprocessing\downsample_csv_by_factor.py --clear-output
D:\ESPOCH\TESIS\CODEX\CARACTERIZACIOIN\.venv\Scripts\python.exe src\training\train_baseline_rf.py --input-dir data\pilot_remap_14_to_10_200hz --models-dir models\baseline --figures-dir results\figures --metrics-dir results\metrics --window-size 40 --stride 10
D:\ESPOCH\TESIS\CODEX\CARACTERIZACIOIN\.venv-training\Scripts\python.exe src\training\train_tiny_mlp_tflite.py --input-dir data\pilot_remap_14_to_10_200hz --models-dir models\tiny_mlp --figures-dir results\figures --metrics-dir results\metrics --window-size 40 --stride 10 --epochs 80
```

## Resultado corto

RandomForest alcanzo 84.10 % de exactitud y valida la estructura funcional de 10 clases como punto de partida.

Tiny MLP alcanzo 64.69 %, por lo que todavia no debe considerarse modelo final para la XIAO.

El siguiente paso practico es firmware/adquisicion real a 200 Hz con las 10 clases finales.

