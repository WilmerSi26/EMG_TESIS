# RTOS y despliegue en Seeed Studio XIAO ESP32S3

## Arquitectura recomendada

La XIAO ESP32S3 ya ejecuta FreeRTOS internamente. Para este proyecto conviene
separar tareas:

```text
TaskAdquisicion  -> lee ADC a periodo fijo y llena una cola/ring buffer
TaskInferencia   -> toma ventanas, normaliza y ejecuta modelo TFLite Micro
TaskComunicacion -> envia CSV, diagnostico o predicciones por USB Serial/BLE
```

Durante la fase de dataset se recomienda activar solo adquisicion y
comunicacion. Despues del entrenamiento se agrega inferencia.

## Flujo para llevar la red a la placa

1. Entrenar con `src/training/train_rnn.py`.
2. Revisar `models/metadata.json` para confirmar:
   - `window_size`
   - `channels`
   - `labels`
   - medias y escalas del `scaler`
3. Convertir `models/emg_rnn.tflite` a arreglo C:

```powershell
xxd -i models\emg_rnn.tflite > firmware\xiao_esp32s3_emg_rtos\emg_rnn_model.h
```

En Windows, si no tienes `xxd`, puedes usar Python para generar el arreglo.

4. Agregar TensorFlow Lite Micro al proyecto Arduino.
5. Reservar tensor arena y probar memoria.
6. Ejecutar inferencia cada vez que exista una ventana completa.

## Advertencia practica

Una RNN/LSTM puede ser pesada para microcontrolador por memoria y operadores.
La ESP32S3 tiene capacidad razonable, pero hay que validar:

- Latencia de inferencia menor al tiempo entre decisiones.
- RAM disponible para `tensor_arena`.
- Operadores soportados por TensorFlow Lite Micro.
- Consumo si luego se usa bateria.

Si la RNN no entra bien, conservar el mismo pipeline de CSV y entrenar una red
compacta para ventana temporal, por ejemplo:

- MLP con caracteristicas EMG: MAV, RMS, WL, ZC, SSC.
- CNN 1D pequena.
- GRU/LSTM mas pequena con menos unidades y ventanas mas cortas.

## Frecuencia de decision sugerida

Con muestreo de 1000 Hz:

- Ventana: 200 muestras.
- Salto: 50 muestras.
- Decision nueva: cada 50 ms.

Para controlar una protesis, aplicar suavizado de decisiones:

- Votacion de las ultimas 5 a 7 predicciones.
- Umbral de confianza, por ejemplo 0.70.
- Estado seguro si la confianza baja.
