# 01. Firmware para XIAO ESP32S3

Archivo a cargar:

```text
firmware/xiao_esp32s3_emg_rtos/xiao_esp32s3_emg_rtos.ino
```

## Orden

1. Abrir Arduino IDE.
2. Seleccionar placa Seeed Studio XIAO ESP32S3.
3. Activar `USB CDC On Boot: Enabled` si aparece esa opcion.
4. Seleccionar puerto COM.
5. Cargar el firmware.
6. Cerrar Monitor Serial.
7. Ejecutar `01_ADQUIRIR_DATOS_GUI.py` o `03_PROBAR_SIMULACION_BASELINE.py`.

## Conexion de sensores

```text
CH1 -> A1
CH2 -> A2
CH3 -> A3
CH4 -> A4
VCC -> 3V3
GND -> GND
```

## Formato serial

```text
timestamp_ms,sample_index,emg_ch1,emg_ch2,emg_ch3,emg_ch4,board_id,sampling_rate_hz
```
