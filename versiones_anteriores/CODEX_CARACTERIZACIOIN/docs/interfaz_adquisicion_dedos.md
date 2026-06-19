# Interfaz grafica para adquisicion de dedos

Archivo principal:

```text
src/acquisition/emg_gui_all_movements.py
```

## Uso en PyCharm Community

1. Abre el proyecto:

```text
D:\ESPOCH\TESIS\CODEX\CARACTERIZACIOIN
```

2. Abre:

```text
src/acquisition/emg_gui_all_movements.py
```

3. Presiona `Run`.
4. En la ventana, pulsa `Actualizar`.
5. Selecciona el puerto COM de la XIAO ESP32S3.
6. Pulsa `1. CONECTAR`.
7. Verifica que se muevan las senales.
8. Escribe el sujeto anonimo y numero de toma.
9. Pulsa `2. GRABAR RUTINA COMPLETA`.

## Movimientos registrados

La rutina guarda apertura y cierre por separado para cada dedo:

```text
reposo
dedo_pulgar_cierre
dedo_pulgar_apertura
dedo_indice_cierre
dedo_indice_apertura
dedo_medio_cierre
dedo_medio_apertura
dedo_anular_cierre
dedo_anular_apertura
dedo_menique_cierre
dedo_menique_apertura
cierre_mano
apertura_mano
pinza_fina
```

Esto evita mezclar senales distintas: cerrar un dedo y abrirlo no se entrenan
como la misma clase.

La etiqueta `reposo` tambien se guarda y se entrena como una clase propia. En
esa fase la mano debe estar relajada, sin abrir ni cerrar ningun dedo.

Tambien guarda fases de control:

```text
preparacion
reposo_inicial
pausa
reposo_final
```

El script de entrenamiento ignora esas fases de control y entrena con los
movimientos utiles.

## Salida CSV

Los archivos se guardan en:

```text
data/raw/
```

con nombre similar a:

```text
20260606_093000_S001_trial-1_rutina_4ch.csv
```

## Nota

La interfaz conserva el comportamiento de la plataforma original:

- Selector de puerto.
- Boton actualizar.
- Boton conectar.
- Grafica cruda.
- Grafica filtrada.
- Filtro pasa banda.
- Filtro notch 50/60 Hz.
- Guardado CSV con 4 canales.
