# Analisis del codigo existente

Proyecto revisado:

```text
D:\ESPOCH\TESIS\CODEX\tesis-emg-protesis-transradial-ia
```

## Hallazgos principales

El proyecto ya tiene una base funcional para adquirir sEMG multicanal:

- Firmware para ESP32S3/XIAO ESP32S3.
- Lectura de 4 canales analogicos.
- Envio serial en formato CSV.
- Script de lectura serial con guardado CSV.
- GUI Tkinter para monitoreo, filtrado visual y rutina completa.
- Un CSV real de ejemplo con columnas crudas, filtradas y etiquetas.

Formato serial usado por el firmware:

```text
timestamp_ms,sample_index,emg_ch1,emg_ch2,emg_ch3,emg_ch4,board_id,sampling_rate_hz
```

Ese formato se conserva en esta nueva carpeta para no romper compatibilidad.

## Codigo base relevante

```text
src/acquisition/serial_reader.py
```

Permite adquirir una etiqueta por ejecucion y guardar CSV. Es util para pruebas
rapidas, pero para entrenamiento de tres movimientos requiere ejecutar varias
veces o manejar manualmente las etiquetas.

```text
src/acquisition/emg_gui.py
```

Tiene una rutina completa con varias fases: dedos individuales, cierre,
apertura, pinza, pausas y reposos. Guarda crudo y filtrado. Es buena para
exploracion, pero para el dataset pedido conviene una rutina mas enfocada en
solo tres clases.

```text
firmware/arduino_adquisicion_emg/xiao_esp32s3_adquisicion_emg/
```

Lee 4 canales a 1000 Hz y envia las muestras por Serial a 921600 baudios. La
version nueva conserva esta salida, pero organiza la adquisicion con FreeRTOS.

## Ajustes realizados en la nueva carpeta

1. Se creo una rutina dedicada a `cierre`, `apertura` y `pinza`.
2. Se normalizaron etiquetas para entrenamiento:
   - `cierre_mano` -> `cierre`
   - `apertura_mano` -> `apertura`
   - `pinza_fina` -> `pinza`
3. Se agrego entrenamiento RNN/LSTM por ventanas temporales.
4. Se agrego simulacion grafica para reproducir CSV o leer serial en vivo.
5. Se agrego firmware FreeRTOS base para XIAO ESP32S3.

## Recomendacion tecnica

Para entrenar el primer modelo, usar solo las columnas crudas:

```text
emg_ch1,emg_ch2,emg_ch3,emg_ch4
```

Asi el mismo pipeline puede usarse luego en tiempo real y en la placa. Si se
entrena con columnas filtradas por `filtfilt`, hay que reproducir un filtrado
equivalente en tiempo real, lo cual es mas delicado porque `filtfilt` no es
causal.

La limpieza y filtrado avanzado pueden agregarse despues, cuando ya exista un
dataset estable y comparable.
