# Caracterizacion de movimientos EMG

Carpeta de trabajo para adquirir senales de 4 sensores mioelectricos, guardar
sesiones CSV de dedos y movimientos de mano, y entrenar una red neuronal
recurrente:

- `reposo`
- `dedo_pulgar_cierre`
- `dedo_pulgar_apertura`
- `dedo_indice_cierre`
- `dedo_indice_apertura`
- `dedo_medio_cierre`
- `dedo_medio_apertura`
- `dedo_anular_cierre`
- `dedo_anular_apertura`
- `dedo_menique_cierre`
- `dedo_menique_apertura`
- `cierre`
- `apertura`
- `pinza`

El formato serial esperado es el mismo del proyecto funcional:

```text
timestamp_ms,sample_index,emg_ch1,emg_ch2,emg_ch3,emg_ch4,board_id,sampling_rate_hz
```

## Estructura

```text
caracterizacion_de_movimientos/
|-- data/
|   |-- raw/                 # CSV activos para entrenar
|   |-- legacy_ambiguous_fingers/
|   |-- legacy_3mov_initial/
|   `-- processed/
|-- firmware/
|   `-- xiao_esp32s3_emg_rtos/
|-- models/                  # modelos entrenados
|-- results/
|   |-- figures/
|   `-- metrics/
|-- src/
|   |-- acquisition/
|   |-- training/
|   `-- inference/
|-- docs/
|-- requirements.txt           # adquisicion, CSV, graficas y utilidades
`-- requirements-training.txt  # entrenamiento RNN con TensorFlow
```

## Instalacion

### Opcion A: adquisicion y simulacion sin entrenar

Esta opcion funciona para leer la XIAO ESP32S3, guardar CSV y manejar datos. En
PyCharm Community puedes instalar `requirements.txt` sin TensorFlow:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Opcion B: entrenamiento RNN

Para entrenar con TensorFlow no uses Python 3.14. Si PyCharm creo el entorno
con Python 3.14, aparecera un error como `No matching distribution found for
tensorflow`.

Usa Python 3.11 o 3.12 de 64 bits, crea de nuevo el entorno virtual y luego:

```powershell
python -m pip install --upgrade pip
pip install -r requirements-training.txt
```

En PyCharm Community:

1. `File > Settings > Project > Python Interpreter`.
2. Elimina o cambia el entorno `.venv` creado con Python 3.14.
3. Crea un entorno nuevo seleccionando Python 3.11 o 3.12.
4. Instala primero `requirements.txt`.
5. Solo cuando vayas a entrenar, instala `requirements-training.txt`.

## 1. Cargar firmware de adquisicion

Abre en Arduino IDE:

```text
firmware/xiao_esp32s3_emg_rtos/xiao_esp32s3_emg_rtos.ino
```

Configura la placa Seeed Studio XIAO ESP32S3, selecciona el puerto COM y carga
el sketch. Cierra el Monitor Serial antes de ejecutar Python.

Pines configurados:

```text
CH1 -> A1
CH2 -> A2
CH3 -> A3
CH4 -> A4
```

Puedes cambiarlos en `EMG_PINS`.

## 2. Adquirir sesiones CSV

### Interfaz grafica recomendada

En PyCharm Community ejecuta:

```text
src/acquisition/emg_gui_all_movements.py
```

La ventana permite:

- Actualizar y seleccionar puerto COM.
- Conectar la XIAO ESP32S3.
- Ver senales crudas y filtradas.
- Grabar una rutina completa con todos los dedos, cierre, apertura y pinza.

La rutina incluida es:

```text
preparacion, reposo,
dedo_pulgar_cierre, dedo_pulgar_apertura,
dedo_indice_cierre, dedo_indice_apertura,
dedo_medio_cierre, dedo_medio_apertura,
dedo_anular_cierre, dedo_anular_apertura,
dedo_menique_cierre, dedo_menique_apertura,
cierre_mano, apertura_mano, pinza_fina, reposo
```

La fase `reposo` se guarda y se entrena como clase real: significa mano
relajada, sin apertura ni cierre voluntario. La fase `preparacion` se guarda,
pero el entrenamiento la ignora por defecto.

Nota: las etiquetas antiguas `dedo_pulgar`, `dedo_indice`, `dedo_medio`,
`dedo_anular` y `dedo_menique` no indican si el dedo se abrio o se cerro. Para
el nuevo entrenamiento direccional, graba sesiones nuevas con las etiquetas
`*_cierre` y `*_apertura`.

### Captura por consola

Lista puertos:

```powershell
python src\acquisition\acquire_movements.py --list-ports
```

Graba una sesion con 5 repeticiones para cierre, apertura y pinza:

```powershell
python src\acquisition\acquire_movements.py --port COM5 --subject-id S001 --session-id SES001 --repetitions 5
```

El programa guiara por consola y guardara un CSV en:

```text
data/raw/
```

Recomendacion inicial:

- 3 a 5 segundos por movimiento.
- 2 segundos de reposo entre movimientos.
- 5 a 10 repeticiones por sesion.
- Al menos 3 sesiones por sujeto antes de entrenar.
- Mantener los electrodos en la misma posicion dentro de una sesion.

## 3. Entrenar la RNN

```powershell
python src\training\train_rnn.py --input-dir data\raw --epochs 60
```

El entrenamiento reconoce estas etiquetas:

```text
dedo_pulgar_cierre, dedo_pulgar_apertura,
dedo_indice_cierre, dedo_indice_apertura,
dedo_medio_cierre, dedo_medio_apertura,
dedo_anular_cierre, dedo_anular_apertura,
dedo_menique_cierre, dedo_menique_apertura,
cierre_mano, apertura_mano, pinza_fina
reposo
```

Internamente normaliza:

```text
cierre_mano -> cierre
apertura_mano -> apertura
pinza_fina -> pinza
```

Salidas principales:

```text
models/emg_rnn.keras
models/emg_rnn.tflite
models/scaler.joblib
models/label_encoder.joblib
models/metadata.json
results/figures/confusion_matrix.png
results/figures/training_history.png
results/metrics/classification_report.txt
```

## 4. Simulacion grafica

Reproducir un CSV ya guardado:

```powershell
python src\inference\realtime_simulation.py --csv-file data\raw\TU_ARCHIVO.csv
```

Leer en tiempo real desde la XIAO:

```powershell
python src\inference\realtime_simulation.py --port COM5
```

La ventana muestra las 4 senales recientes, probabilidades por clase y la
prediccion actual.

## 5. Flujo recomendado con accesos numerados

Ejecuta estos archivos desde PyCharm o desde PowerShell, en este orden:

```text
01_ADQUIRIR_DATOS_GUI.py
02_ENTRENAR_BASELINE_SIN_TENSORFLOW.py
03_PROBAR_SIMULACION_BASELINE.py
04_ENTRENAR_RNN_TENSORFLOW.py
05_PROBAR_SIMULACION_RNN_TENSORFLOW.py
06_ENTRENAR_MODELO_XIAO_TFLITE.py
07_GENERAR_REPORTE_GRAFICO.py
08_INFERENCIA_EN_VIVO_DASHBOARD.py
```

`07_GENERAR_REPORTE_GRAFICO.py` crea evidencias para explicar si los
movimientos se diferencian:

```text
results/figures/analisis_01_balance_clases.png
results/figures/analisis_02_pca_separabilidad.png
results/figures/analisis_03_ejemplos_senales.png
results/figures/analisis_04_importancia_caracteristicas.png
results/metrics/reporte_grafico_entrenamiento.md
```

`08_INFERENCIA_EN_VIVO_DASHBOARD.py` es la prueba en tiempo real mas directa:
seleccionas el puerto COM, conectas la XIAO, ves las 4 senales actuales y el
movimiento que predice el modelo base con su confianza.

`03_PROBAR_SIMULACION_BASELINE.py` abre el mismo dashboard del modelo base,
pero tambien permite escoger un CSV para replay. Usalo para revisar si una
sesion guardada se reproduce con la prediccion esperada antes de pasar a la
prueba en vivo.

## Nota sobre RTOS y la red en la XIAO ESP32S3

El firmware incluido ya separa adquisicion y transmision usando tareas FreeRTOS.
Para correr la red en la placa, el siguiente paso es convertir `emg_rnn.tflite`
a un arreglo C e integrarlo con TensorFlow Lite Micro. Esta carpeta deja una
guia en:

```text
docs/rtos_y_despliegue_esp32.md
```

Para microcontrolador, una RNN pequena puede funcionar, pero hay que validar
RAM, latencia y operadores soportados por TensorFlow Lite Micro. Si no cabe o
los operadores LSTM complican la compilacion, se recomienda entrenar una red
compacta basada en ventanas/caracteristicas como alternativa embebida.
