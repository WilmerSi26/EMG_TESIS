# Firmware XIAO ESP32S3 RTOS 200 Hz

Carga `xiao_esp32s3_emg_rtos_200hz.ino` en Arduino IDE sobre la Seeed Studio XIAO ESP32S3.

Parametros esperados:

- Baudios: 921600
- Canales analogicos: A1, A2, A3, A4
- Frecuencia de muestreo: 200 Hz
- Salida serial CSV: `timestamp_ms,sample_index,emg_ch1,emg_ch2,emg_ch3,emg_ch4,board_id,sampling_rate_hz`

Primero valida que el Monitor Serial muestre lineas CSV estables. Luego cierra el Monitor Serial antes de abrir la GUI o el script de adquisicion en Python.
