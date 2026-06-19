# Paso a paso practico: adquisicion, entrenamiento y comparacion

## 1. Cargar firmware en XIAO

Abre en Arduino IDE:

`D:\ESPOCH\TESIS\CARACTERIZACION_FINAL\firmware\xiao_esp32s3_emg_rtos_200hz\xiao_esp32s3_emg_rtos_200hz.ino`

Configura la placa Seeed Studio XIAO ESP32S3, carga el programa y valida por Monitor Serial que aparezcan lineas CSV con `sampling_rate_hz=200`.

## 2. Ver puertos disponibles

Ejecuta desde:

```powershell
cd D:\ESPOCH\TESIS\CARACTERIZACION_FINAL
D:\ESPOCH\TESIS\CODEX\CARACTERIZACIOIN\.venv\Scripts\python.exe src\acquisition\acquire_funcional_10clases.py --list-ports
```

## 3. Ensayar la rutina sin abrir serial

```powershell
D:\ESPOCH\TESIS\CODEX\CARACTERIZACIOIN\.venv\Scripts\python.exe src\acquisition\acquire_funcional_10clases.py --dry-run --repetitions 1
```

La rutina usa estas clases: reposo, apertura, cierre, pulgar apertura/cierre, indice-medio apertura/cierre, anular-menique apertura/cierre y pinza.

## 4. Adquirir un CSV por consola

Cambia `COM5` por el puerto real:

```powershell
D:\ESPOCH\TESIS\CODEX\CARACTERIZACIOIN\.venv\Scripts\python.exe src\acquisition\acquire_funcional_10clases.py --port COM5 --subject-id S001 --repetitions 1
```

Los CSV nuevos quedan en:

`D:\ESPOCH\TESIS\CARACTERIZACION_FINAL\data\final_10clases`

## 5. Adquirir con GUI

```powershell
D:\ESPOCH\TESIS\CODEX\CARACTERIZACIOIN\.venv\Scripts\python.exe src\acquisition\emg_gui_funcional_10clases.py
```

Usa la GUI si quieres ver las senales en vivo y grabar una toma guiada. Repite la toma cambiando el numero de trial para generar varios CSV.

## 6. Entrenar despues de tener datasets reales

RandomForest de referencia:

```powershell
D:\ESPOCH\TESIS\CODEX\CARACTERIZACIOIN\.venv\Scripts\python.exe src\training\train_baseline_rf.py --input-dir data\final_10clases --models-dir models\baseline_final --figures-dir results\figures --metrics-dir results\metrics --window-size 40 --stride 10
```

Tiny MLP TFLite:

```powershell
D:\ESPOCH\TESIS\CODEX\CARACTERIZACIOIN\.venv-training\Scripts\python.exe src\training\train_tiny_mlp_tflite.py --input-dir data\final_10clases --models-dir models\tiny_mlp_final --figures-dir results\figures --metrics-dir results\metrics --window-size 40 --stride 10 --epochs 100
```

CNN1D candidato embebido:

```powershell
D:\ESPOCH\TESIS\CODEX\CARACTERIZACIOIN\.venv-training\Scripts\python.exe src\training\train_reference_deep_models.py --model iyeleswarapu_cnn1d --input-dir data\final_10clases --models-dir models --figures-dir results\figures --metrics-dir results\metrics --window-size 40 --stride 10 --epochs 60
```

CNN-LSTM e Inception-LSTM solo como comparadores de computador:

```powershell
D:\ESPOCH\TESIS\CODEX\CARACTERIZACIOIN\.venv-training\Scripts\python.exe src\training\train_reference_deep_models.py --model ocjorge_cnn_lstm --input-dir data\final_10clases --epochs 50 --window-size 40 --stride 10
```

```powershell
D:\ESPOCH\TESIS\CODEX\CARACTERIZACIOIN\.venv-training\Scripts\python.exe src\training\train_reference_deep_models.py --model laboratorio_inception_lstm --input-dir data\final_10clases --epochs 30 --window-size 40 --stride 10
```

## 7. Regenerar comparacion

```powershell
D:\ESPOCH\TESIS\CODEX\CARACTERIZACIOIN\.venv\Scripts\python.exe src\training\compare_model_results.py
```

## Recomendacion de captura inicial

Para validar el flujo, toma primero 3 CSV con `S001`. Si el pipeline entrena sin errores, sube a 10-16 CSV. Para una prueba con paciente o usuario nuevo, reduce fatiga: una repeticion por CSV, descansos claros y pausas entre tomas.
