# Estado actual y resultados

Fecha de ejecucion: 2026-06-06.

## Datos

Carpeta:

```text
D:\ESPOCH\TESIS\CODEX\CARACTERIZACIOIN\data\raw
```

Sesiones CSV detectadas:

```text
10
```

Clases:

```text
apertura
cierre
dedo_anular
dedo_indice
dedo_medio
dedo_menique
dedo_pulgar
pinza
```

## Modelos entrenados

### 02. Baseline sin TensorFlow

Modelo:

```text
models/emg_sklearn_baseline.joblib
```

Resultado:

```text
accuracy = 0.9428
```

Uso recomendado:

- Validacion rapida.
- Simulacion en PC.
- Comparar si nuevas sesiones mejoran o empeoran.

### 04. RNN/LSTM TensorFlow

Modelo:

```text
models/emg_rnn.keras
```

Resultado:

```text
accuracy = 0.7023
```

TFLite:

```text
models/emg_rnn.conversion_warning.txt
```

La RNN se guardo correctamente para PC, pero no se convirtio a TFLite por las
operaciones internas LSTM. Esto no invalida el entrenamiento; solo indica que
no es el mejor candidato para TensorFlow Lite Micro en la XIAO.

### 06. Tiny MLP TFLite para XIAO

Modelo:

```text
models/emg_tiny_mlp.keras
models/emg_tiny_mlp.tflite
```

Resultado:

```text
accuracy = 0.7420
```

Uso recomendado:

- Candidato inicial para correr en XIAO ESP32S3.
- Usa 28 caracteristicas por ventana.
- Es mas viable para TensorFlow Lite Micro que la LSTM.

## Orden recomendado desde ahora

1. Capturar mas sesiones con `01_ADQUIRIR_DATOS_GUI.py`.
2. Reentrenar `02_ENTRENAR_BASELINE_SIN_TENSORFLOW.py`.
3. Probar en vivo `03_PROBAR_SIMULACION_BASELINE.py`.
4. Reentrenar `04_ENTRENAR_RNN_TENSORFLOW.py` para resultados de tesis en PC.
5. Probar `05_PROBAR_SIMULACION_RNN_TENSORFLOW.py` con `.venv-training`.
6. Reentrenar `06_ENTRENAR_MODELO_XIAO_TFLITE.py`.
7. Llevar `models/emg_tiny_mlp.tflite` a firmware con TensorFlow Lite Micro.

## Interpretacion

El baseline da la mejor exactitud porque Random Forest trabaja bien con
caracteristicas EMG por ventana. La RNN todavia necesita mas datos o ajustes.
El Tiny MLP es el punto medio: menor exactitud que Random Forest, pero portable
a XIAO mediante TFLite.
