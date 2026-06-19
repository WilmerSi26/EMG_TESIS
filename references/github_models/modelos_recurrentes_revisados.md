# Modelos recurrentes revisados para sEMG

## laboratorioAI/2023-HGR5-CNN_LSTM

Repositorio: https://github.com/laboratorioAI/2023-HGR5-CNN_LSTM

Contiene el codigo del estudio "Assessing the influence of LSTM and post-processing in CNN based Hand Gesture Recognition using EMG". Es una referencia metodologica solida porque compara CNN y CNN-LSTM sobre EMG, reportando mejora por memoria temporal, aunque tambien mayor cantidad de parametros. Esta implementado en MATLAB y usa espectrogramas del dataset EMG-EPN-612, por lo que no se copia directamente al firmware de la XIAO.

Uso recomendado en este proyecto: tomarlo como soporte conceptual para justificar una comparacion de modelos temporales, no como modelo embebido principal.

## ocjorge/CNN-LSTM

Repositorio: https://github.com/ocjorge/CNN-LSTM

Implementacion Python compacta de CNN-LSTM para reconocimiento de gestos EMG. Puede servir como referencia de arquitectura si se desea construir un comparador de escritorio.

Uso recomendado en este proyecto: revisar su estructura CNN-LSTM y adaptar una version pequena al formato propio de ventanas `muestras x canales`.

## iyeleswarapu/emg-gesture-recognition

Repositorio: https://github.com/iyeleswarapu/emg-gesture-recognition

Implementacion Python pequena de CNN-LSTM para clasificacion EMG. Por su tamano puede orientar una prueba rapida, pero no resuelve directamente el problema de despliegue en microcontrolador.

Uso recomendado en este proyecto: referencia secundaria para entrenamiento offline.

## Conclusion practica

Para la XIAO ESP32S3 se prioriza Tiny MLP o CNN1D compacta. LSTM/CNN-LSTM se mantiene como modelo de comparacion en computador, porque la conversion y ejecucion de operadores recurrentes en TensorFlow Lite Micro puede aumentar memoria, latencia y complejidad de integracion.

