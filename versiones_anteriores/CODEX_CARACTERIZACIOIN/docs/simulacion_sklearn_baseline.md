# Simulacion grafica sin TensorFlow

Como el modelo base ya entreno correctamente, puedes probar prediccion grafica
sin instalar TensorFlow.

## Reproducir un CSV

En PyCharm Community puedes ejecutar directamente:

```text
src/inference/realtime_simulation_sklearn.py
```

Si lo abres sin parametros, aparecera una ventana para escoger un CSV o un
puerto COM.

Desde terminal tambien puedes indicar el CSV:

```powershell
cd D:\ESPOCH\TESIS\CODEX\CARACTERIZACIOIN
.\.venv\Scripts\python.exe src\inference\realtime_simulation_sklearn.py --csv-file data\raw\TU_ARCHIVO.csv
```

Para reproducir aproximadamente a velocidad real:

```powershell
.\.venv\Scripts\python.exe src\inference\realtime_simulation_sklearn.py --csv-file data\raw\TU_ARCHIVO.csv --csv-real-time
```

## Leer desde la placa en vivo

Primero lista puertos:

```powershell
.\.venv\Scripts\python.exe src\inference\realtime_simulation_sklearn.py --list-ports
```

Luego:

```powershell
.\.venv\Scripts\python.exe src\inference\realtime_simulation_sklearn.py --port COM5
```

Cambia `COM5` por el puerto real.

## Que muestra la ventana

- Las 4 senales EMG recientes.
- Las probabilidades de cada clase.
- La prediccion actual y su confianza.

Este modelo usa caracteristicas por ventana y Random Forest. No reemplaza la
RNN final, pero sirve para validar el flujo de prediccion mientras instalas
Python 3.11 o 3.12 para TensorFlow.
