# Orden actual de ejecucion

## 1. Auditar dataset

Ejecutar cuando agregues o borres CSV.

```powershell
cd D:\ESPOCH\TESIS\CARACTERIZACION_FINAL
D:\ESPOCH\TESIS\CODEX\CARACTERIZACIOIN\.venv\Scripts\python.exe src\preprocessing\audit_final_dataset.py --copy-valid --clear-valid-dir
```

Que hace:

- Revisa los CSV de `data\final_10clases`.
- Excluye archivos cortados.
- Copia los CSV buenos a `data\final_10clases_valid`.

## 2. Entrenar modelo rapido para inferencia PC

```powershell
D:\ESPOCH\TESIS\CODEX\CARACTERIZACIOIN\.venv\Scripts\python.exe src\training\train_baseline_rf.py --input-dir data\final_10clases_valid --models-dir models\baseline_realtime --figures-dir results\figures\realtime_rf --metrics-dir results\metrics\realtime_rf --window-size 40 --stride 10 --trees 80
```

Que hace:

- Divide la senal en ventanas de 40 muestras.
- Calcula 28 caracteristicas por ventana.
- Entrena RandomForest liviano de 80 arboles.
- Guarda el modelo para inferencia en PC.

Resultado actual: 78.37 %.

## 3. Validar por sujeto

```powershell
D:\ESPOCH\TESIS\CODEX\CARACTERIZACIOIN\.venv\Scripts\python.exe src\training\validate_rf_by_subject.py --input-dir data\final_10clases_valid --window-size 40 --stride 10 --save-final-model
```

Que hace:

- Entrena con tres sujetos.
- Prueba con el sujeto restante.
- Repite hasta evaluar S001, S002, S003 y S004.

Resultado actual: 31.11 % promedio.

## 4. Probar inferencia en PC en tiempo real

Primero carga en la XIAO el firmware de adquisicion cruda:

```text
firmware\xiao_esp32s3_emg_rtos_200hz\xiao_esp32s3_emg_rtos_200hz.ino
```

Luego ejecuta:

```powershell
D:\ESPOCH\TESIS\CODEX\CARACTERIZACIOIN\.venv\Scripts\python.exe src\inference\live_inference_dashboard_fast.py
```

Que hace:

- Lee la senal cruda de la XIAO.
- Grafica los 4 canales.
- Calcula caracteristicas cada 10 muestras.
- Predice con `models\baseline_realtime`.
- Actualiza clase y probabilidades en pantalla.

## 5. Entrenar Tiny MLP para XIAO

```powershell
D:\ESPOCH\TESIS\CODEX\CARACTERIZACIOIN\.venv-training\Scripts\python.exe src\training\train_tiny_mlp_tflite.py --input-dir data\final_10clases_valid --models-dir models\tiny_mlp_final --figures-dir results\figures --metrics-dir results\metrics --window-size 40 --stride 10 --epochs 100 --batch-size 128
```

Que hace:

- Entrena una red pequena.
- Exporta `emg_tiny_mlp.tflite`.
- Este modelo es el candidato para XIAO.

Resultado actual: 62.70 %.

## 6. Exportar Tiny MLP a Arduino

```powershell
D:\ESPOCH\TESIS\CODEX\CARACTERIZACIOIN\.venv\Scripts\python.exe src\embedded\export_tiny_mlp_for_xiao.py --model-dir models\tiny_mlp_final
```

Que hace:

- Convierte el `.tflite` a `model_data.h`.
- Exporta media, escala y etiquetas a `emg_metadata.h`.

## 7. Probar inferencia dentro de XIAO

Abre y carga en Arduino IDE:

```text
firmware\xiao_esp32s3_tinyml_inference\xiao_esp32s3_tinyml_inference.ino
```

Luego lee la salida:

```powershell
D:\ESPOCH\TESIS\CODEX\CARACTERIZACIOIN\.venv\Scripts\python.exe src\inference\read_xiao_inference.py --port COM4
```

Que hace:

- La XIAO calcula caracteristicas e infiere.
- El computador solo lee `INFER,<indice>,<etiqueta>,<confianza>`.

## Importante

Para inferencia en PC usa el firmware de adquisicion cruda. Para inferencia en XIAO usa el firmware TinyML. No son el mismo firmware.
