# Organizacion actual del proyecto

## Carpetas principales

```text
D:\ESPOCH\TESIS\CODEX\CARACTERIZACIOIN
|-- 01_ADQUIRIR_DATOS_GUI.py
|-- 02_ENTRENAR_BASELINE_SIN_TENSORFLOW.py
|-- 03_PROBAR_SIMULACION_BASELINE.py
|-- 04_ENTRENAR_RNN_TENSORFLOW.py
|-- 05_PROBAR_SIMULACION_RNN_TENSORFLOW.py
|-- 06_ENTRENAR_MODELO_XIAO_TFLITE.py
|-- 07_GENERAR_REPORTE_GRAFICO.py
|-- 08_INFERENCIA_EN_VIVO_DASHBOARD.py
|-- data/
|-- models/
|-- results/
|-- src/
|-- docs/
|-- firmware/
|-- scripts/
`-- tools/
```

## Datos

```text
data/raw/
```

CSV activos para entrenar el modelo actual. Aqui deben quedar las sesiones con:

- `reposo`
- apertura y cierre de cada dedo;
- cierre de mano;
- apertura de mano;
- pinza.

```text
data/legacy_ambiguous_fingers/
```

CSV historicos con etiquetas como `dedo_pulgar`, `dedo_indice`, etc. Esos datos
no indican si el dedo se estaba abriendo o cerrando, por eso no se usan en el
entrenamiento direccional actual.

```text
data/legacy_3mov_initial/
```

CSV iniciales de cierre, apertura y pinza. Sirven como referencia historica,
pero no son el dataset principal del modelo actual de dedos.

## Modelos

```text
models/
```

Modelos activos que usan los pasos 03 y 08.

```text
models/archive_*/
```

Copias de modelos anteriores antes de reentrenar. No se usan por defecto, pero
quedan guardadas para comparacion.

## Resultados

```text
results/figures/
```

Matrices de confusion, PCA, ejemplos de senal e historiales de entrenamiento.

```text
results/metrics/
```

Reportes de clasificacion y resumen grafico del dataset.

## Dataset minimo recomendado

Para una prueba en clase, 5 sesiones direccionales pueden mostrar el flujo, pero
no suelen bastar para inferencia en vivo estable. Para una inferencia aceptable:

- minimo practico: 10 sesiones limpias con todas las clases;
- recomendado: 20 a 30 sesiones limpias;
- por clase: al menos 1000 a 2000 ventanas utiles;
- condiciones: distintas fuerzas, pequenas variaciones naturales y varios dias.

Como ahora hay 14 clases contando `reposo`, se necesita mas informacion que
cuando solo habia 8 clases.

## Latencia esperada

Con `window_size=200` y una frecuencia aproximada de 1000 Hz:

- la ventana necesita unos 200 ms de senal;
- el modelo predice cada 50 muestras, unos 50 ms;
- el suavizado del dashboard agrega estabilidad, pero puede sumar 300 a 500 ms;
- en PC la inferencia del RandomForest tarda pocos milisegundos.

En la practica, la respuesta visible suele sentirse entre 0.4 y 0.8 segundos.
No es exactamente instantanea porque primero se necesita completar una ventana
de senal. Para menor latencia se puede bajar la ventana o reducir el suavizado,
pero normalmente baja la estabilidad.
