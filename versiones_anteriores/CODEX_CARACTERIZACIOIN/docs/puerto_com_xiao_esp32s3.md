# XIAO ESP32S3 no aparece como puerto COM

Si `acquire_movements.py` muestra `No se detectaron puertos seriales`, el
problema esta antes del script: Windows no esta exponiendo la XIAO ESP32S3 como
puerto serial.

## Verificacion rapida

1. Usa un cable USB de datos, no solo de carga.
2. Conecta la XIAO directamente al computador, no a un hub USB.
3. Cierra el Monitor Serial de Arduino IDE.
4. Abre el Administrador de dispositivos de Windows.
5. Revisa `Puertos (COM y LPT)`.
6. Desconecta y conecta la placa: debe aparecer o desaparecer un `COMx`.

## Si no aparece ningun COM

Prueba:

- Otro cable USB.
- Otro puerto USB del computador.
- Presionar `RESET`.
- Entrar a modo bootloader con `BOOT` + `RESET`, segun la placa.
- Abrir Arduino IDE y verificar si aparece algun puerto.

## Configuracion importante en Arduino IDE

Para sketches que usan `Serial` por USB en ESP32S3/XIAO ESP32S3, revisa en
`Tools`:

```text
USB CDC On Boot: Enabled
Upload Mode: UART0 / USB CDC segun disponibilidad
Board: Seeed XIAO ESP32S3
```

Despues vuelve a cargar el firmware:

```text
firmware/xiao_esp32s3_emg_rtos/xiao_esp32s3_emg_rtos.ino
```

Luego cierra el Monitor Serial y ejecuta Python.

## Comando para listar puertos

Desde la terminal de PyCharm:

```powershell
python src\acquisition\acquire_movements.py --list-ports
```

Si sigue sin listar nada, PySerial tampoco ve un puerto COM disponible.

## Cuando ya aparezca

Ejemplo:

```powershell
python src\acquisition\acquire_movements.py --port COM5 --subject-id S001 --session-id SES001 --repetitions 5
```

Cambia `COM5` por el puerto real.
