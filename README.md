# Caracterizacion EMG para protesis de mano

Repositorio tecnico de la tesis para adquisicion, procesamiento, entrenamiento e inferencia de senales sEMG usando un brazalete de 4 canales, XIAO ESP32S3, modelos de aprendizaje automatico y firmware para control de servomotores.

## Contenido principal

- `src/acquisition`: programas para adquirir senales EMG y generar datasets guiados.
- `src/preprocessing`: remapeo de clases, reduccion de frecuencia de muestreo y auditoria de datasets.
- `src/training`: entrenamiento y comparacion de modelos RF, CNN/LSTM y TinyML.
- `src/inference`: inferencia en vivo desde computador y lectura de inferencia desde la XIAO.
- `src/embedded`: exportacion del modelo TinyML hacia TensorFlow Lite compatible con ESP32.
- `firmware`: programas Arduino/ESP32S3 para adquisicion, RTOS e inferencia TinyML.
- `data`: datasets CSV generados durante las pruebas.
- `models`: modelos entrenados, metadatos, escaladores, codificadores y modelos TensorFlow Lite.
- `docs`: guias tecnicas de exposicion y defensa del flujo de entrenamiento.

## Instalacion rapida

Crear un entorno de Python y luego instalar:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Para scripts que requieren TensorFlow, usar un entorno compatible con TensorFlow:

```powershell
python -m venv .venv-tf
.\.venv-tf\Scripts\python.exe -m pip install --upgrade pip
.\.venv-tf\Scripts\python.exe -m pip install -r requirements-training.txt
```

## Ejecucion habitual

Adquisicion guiada:

```powershell
.\.venv\Scripts\python.exe src\acquisition\emg_gui_funcional_10clases.py
```

Entrenamiento y comparacion:

```powershell
.\.venv-tf\Scripts\python.exe src\training\train_reference_deep_models.py
.\.venv-tf\Scripts\python.exe src\training\train_tiny_mlp_tflite.py
.\.venv\Scripts\python.exe src\training\train_baseline_rf.py
```

Inferencia en vivo en computador:

```powershell
.\.venv-tf\Scripts\python.exe src\inference\live_inference_compare_models.py
```

Lectura de inferencia desde la XIAO:

```powershell
.\.venv\Scripts\python.exe src\inference\read_xiao_inference.py
```

## Nota sobre modelos grandes

GitHub no permite subir archivos mayores a 100 MB en Git normal. Por eso los modelos Random Forest `.joblib` de gran tamano quedan fuera del repositorio mediante `.gitignore`. Los escaladores, codificadores, metadatos, modelos Keras y modelos `.tflite` pequenos si se conservan.

Si se necesita recuperar esos modelos RF exactos, pueden copiarse manualmente desde la estacion de trabajo original o regenerarse con los scripts de `src/training`.
