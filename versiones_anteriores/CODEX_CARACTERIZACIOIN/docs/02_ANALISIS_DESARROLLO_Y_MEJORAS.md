# Analisis de la carpeta DESARROLLO y mejoras integradas

La carpeta `D:\ESPOCH\TESIS\DESARROLLO` ya tenia una idea importante para la
tesis: no basta con entrenar un modelo, tambien hay que demostrar graficamente
que los movimientos se diferencian.

## Que se encontro en DESARROLLO

- `entrenamiento esp32s3.py`: entrena un RandomForest, calcula matriz de
  confusion, reporte de clasificacion, validacion cruzada e importancia de
  caracteristicas.
- `data analisys.py`: compara senales crudas y filtradas, grafica dispersiones
  entre canales y genera modelos tipo arbol para microcontrolador.
- `ENVIO ESP.py`: reproduce senales desde CSV, muestra graficas y envia estados
  por serial a una ESP.
- `prueba tensor flow.py`: prueba la conversion de un modelo TensorFlow a un
  arreglo C para firmware.

## Que se integro en CARACTERIZACIOIN

- `07_GENERAR_REPORTE_GRAFICO.py` genera graficas para explicar la
  diferenciacion de movimientos:
  - balance de ventanas por clase;
  - separacion PCA de las caracteristicas;
  - ejemplos de senal por movimiento;
  - importancia de caracteristicas del RandomForest.
- `08_INFERENCIA_EN_VIVO_DASHBOARD.py` abre una interfaz en vivo:
  - selecciona el puerto COM;
  - conecta la XIAO ESP32S3;
  - muestra las 4 senales actuales;
  - muestra el movimiento predicho y su confianza;
  - muestra barras de probabilidad por clase.

## Por que se hizo asi

El entrenamiento base con RandomForest ya dio una exactitud alta con tus 10
sesiones, por eso se usa como modelo principal para la prueba en vivo mientras
se estabiliza TensorFlow. La RNN sigue siendo util para la tesis porque aprende
secuencias temporales completas, pero para validar adquisicion, etiquetas y
separabilidad, el modelo base es mas rapido y confiable en esta etapa.

## Como se explica en clase

Primero se adquieren senales EMG de cuatro sensores. Cada CSV queda etiquetado
por movimiento. Despues se corta la senal en ventanas de 200 muestras. De cada
ventana se extraen caracteristicas por canal: media, desviacion, minimo, maximo,
MAV, RMS y longitud de onda. Con esas caracteristicas se entrena el clasificador
y se valida con matriz de confusion.

Si la diagonal de la matriz de confusion tiene valores altos, significa que el
modelo reconoce bien ese movimiento. Si dos movimientos se confunden, se revisa
su posicion en PCA, el balance de datos y la calidad de la senal de los
electrodos. Finalmente, el dashboard en vivo toma la misma ventana de senal y
predice el movimiento en tiempo real.
