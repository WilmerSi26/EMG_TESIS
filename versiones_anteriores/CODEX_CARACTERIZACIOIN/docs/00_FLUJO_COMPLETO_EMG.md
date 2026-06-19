# Flujo completo EMG

Este proyecto queda ordenado en pasos numerados.

## 00. Instalar TensorFlow correctamente

Tu `.venv` actual usa Python 3.14. TensorFlow no instala ahi. Para crear un
entorno separado con Python 3.12 y TensorFlow:

```powershell
cd D:\ESPOCH\TESIS\CODEX\CARACTERIZACIOIN
powershell -ExecutionPolicy Bypass -File .\scripts\00_instalar_python312_tensorflow.ps1
```

Esto descarga Python 3.12.10 oficial desde `python.org` y crea:

```text
tools/Python312/
.venv-training/
```

## 01. Adquirir datos

En PyCharm ejecuta:

```text
01_ADQUIRIR_DATOS_GUI.py
```

Flujo:

1. Actualizar puertos.
2. Seleccionar COM de la XIAO ESP32S3.
3. Conectar.
4. Verificar que se mueven las 4 senales.
5. Poner sujeto y toma.
6. Grabar rutina completa.

CSV generados:

```text
data/raw/
```

Movimientos capturados:

```text
dedo_pulgar
dedo_indice
dedo_medio
dedo_anular
dedo_menique
cierre_mano
apertura_mano
pinza_fina
```

## 02. Entrenar modelo base sin TensorFlow

Este paso ya funciono con tus 10 sesiones y obtuvo 94.28% de exactitud.

En PyCharm ejecuta:

```text
02_ENTRENAR_BASELINE_SIN_TENSORFLOW.py
```

O en terminal:

```powershell
.\.venv\Scripts\python.exe 02_ENTRENAR_BASELINE_SIN_TENSORFLOW.py --input-dir data\raw
```

Salidas:

```text
models/emg_sklearn_baseline.joblib
models/sklearn_feature_scaler.joblib
models/sklearn_label_encoder.joblib
models/sklearn_baseline_metadata.json
results/metrics/sklearn_baseline_report.txt
results/figures/sklearn_baseline_confusion_matrix.png
```

## 03. Probar simulacion grafica

En PyCharm ejecuta:

```text
03_PROBAR_SIMULACION_BASELINE.py
```

La ventana permite:

1. Escoger un CSV y simular una sesion guardada.
2. Seleccionar puerto COM y probar en vivo con sensores.

Para tiempo real:

1. Carga el firmware de adquisicion en la XIAO.
2. Cierra el Monitor Serial de Arduino.
3. Abre `03_PROBAR_SIMULACION_BASELINE.py`.
4. Selecciona el puerto COM.
5. Pulsa `Simular puerto`.
6. Haz un movimiento y observa la prediccion.

## 04. Entrenar RNN/LSTM con TensorFlow

Cuando `00` termine correctamente:

```powershell
.\.venv-training\Scripts\python.exe 04_ENTRENAR_RNN_TENSORFLOW.py --input-dir data\raw --epochs 60
```

Salidas:

```text
models/emg_rnn.keras
models/emg_rnn.tflite
models/scaler.joblib
models/label_encoder.joblib
models/metadata.json
```

Resultado obtenido con las 10 sesiones:

```text
accuracy = 0.7023
```

Nota: la RNN se guarda como `.keras`, pero la conversion directa a TFLite puede
fallar por operaciones LSTM. En ese caso se genera:

```text
models/emg_rnn.conversion_warning.txt
```

## 06. Entrenar modelo pequeno para XIAO/TFLite

Para la XIAO ESP32S3 se recomienda un modelo mas compacto:

```powershell
.\.venv-training\Scripts\python.exe 06_ENTRENAR_MODELO_XIAO_TFLITE.py --input-dir data\raw --epochs 80
```

Salidas:

```text
models/emg_tiny_mlp.keras
models/emg_tiny_mlp.tflite
models/tiny_feature_scaler.joblib
models/tiny_label_encoder.joblib
models/tiny_mlp_metadata.json
```

Resultado obtenido:

```text
accuracy = 0.7420
```

Este es el candidato inicial para portar a TensorFlow Lite Micro en la XIAO.

## 05. XIAO ESP32S3

Primero usa la XIAO solo para adquirir datos y probar inferencia en la PC:

```text
firmware/xiao_esp32s3_emg_rtos/xiao_esp32s3_emg_rtos.ino
```

Despues de validar el modelo, el despliegue embebido sigue este orden:

1. Convertir `emg_rnn.tflite` a arreglo C.
2. Integrar TensorFlow Lite Micro.
3. Agregar tarea FreeRTOS de inferencia.
4. Probar memoria y latencia.
5. Activar control de protesis solo cuando la clasificacion sea estable.

Por ahora, la forma correcta de validar es:

```text
XIAO sensores -> Serial USB -> PC -> modelo entrenado -> grafica de prediccion
```
