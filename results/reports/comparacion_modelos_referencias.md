# Comparacion final de modelos EMG

Este reporte compara los modelos entrenados bajo las mismas condiciones experimentales: dataset funcional de 10 clases, cuatro sujetos, 34 CSV validos, muestreo de 200 Hz, ventana de 40 muestras y desplazamiento de 10 muestras.

| Modelo | Origen | Exactitud test | Epocas | TFLite | Tamano TFLite | Lectura practica |
|---|---|---:|---:|---|---:|---|
| RandomForest baseline | Modelo base local de validacion rapida | 78.61% | - | No aplica | - | No es candidato directo para XIAO; sirve como referencia de escritorio. |
| Tiny MLP TFLite | Modelo compacto local para despliegue embebido | 63.16% | 100 | OK | 6.8 KB | Candidato embebido minimo por tamano y simplicidad. |
| CNN1D iyeleswarapu adaptado | Adaptado desde iyeleswarapu/emg-gesture-recognition | 72.66% | 56 | OK | 35.6 KB | Mejor candidato profundo para probar en TFLite/TFLite Micro por no usar LSTM. |
| CNN-LSTM ocjorge adaptado | Adaptado desde ocjorge/CNN-LSTM | 78.26% | 42 | Fallo conversion builtin | - | Bueno para escritorio; la LSTM fallo al convertir a TFLite builtin. |
| Inception-LSTM laboratorioAI adaptado | Inspirado en laboratorioAI/2023-HGR5-CNN_LSTM | 74.38% | 35 | Fallo conversion builtin | - | Referencia temporal; no conviene como primer despliegue en XIAO por LSTM. |

## Lectura tecnica

- RandomForest queda como referencia de escritorio para comprobar separabilidad sin exigir despliegue embebido.
- CNN-LSTM de ocjorge conserva una lectura temporal util en computador, pero su conversion TFLite builtin falla por operaciones TensorList asociadas a LSTM.
- CNN1D inspirado en iyeleswarapu es el candidato profundo mas razonable para despliegue, porque logro exportarse a TFLite y no usa capas recurrentes.
- Tiny MLP es el mas compacto y se mantiene como alternativa embebida minima por tamano y simplicidad.
- Inception-LSTM inspirado en laboratorioAI queda como referencia metodologica temporal; tambien falla la conversion TFLite builtin por LSTM.

## Decision para la siguiente fase

Para el despliegue en XIAO ESP32S3 conviene priorizar los modelos que generan TFLite sin operaciones recurrentes no soportadas. Los modelos con LSTM pueden mantenerse como comparadores de computador, pero no deberian bloquear el avance del firmware embebido.
