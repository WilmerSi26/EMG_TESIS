# Firmware TinyML de inferencia en XIAO ESP32S3

Este firmware ejecuta en la XIAO el modelo Tiny MLP exportado a TensorFlow Lite.

## 1. Generar cabeceras del modelo

Desde `D:\ESPOCH\TESIS\CARACTERIZACION_FINAL`:

```powershell
D:\ESPOCH\TESIS\CODEX\CARACTERIZACIOIN\.venv\Scripts\python.exe src\embedded\export_tiny_mlp_for_xiao.py --model-dir models\tiny_mlp_final
```

Esto genera:

- `generated/model_data.h`
- `generated/emg_metadata.h`

## 2. Compilar en Arduino IDE

Abre:

```text
D:\ESPOCH\TESIS\CARACTERIZACION_FINAL\firmware\xiao_esp32s3_tinyml_inference\xiao_esp32s3_tinyml_inference.ino
```

Necesitas una libreria compatible con TensorFlow Lite Micro para Arduino/ESP32. Si el IDE no encuentra los encabezados `tensorflow/lite/micro/...`, instala la libreria correspondiente y vuelve a compilar.

## 3. Ver resultado

Abre el Monitor Serial a `921600` baudios. La salida esperada es:

```text
INFER,<indice>,<etiqueta>,<confianza>
```

Ejemplo:

```text
INFER,9,reposo,0.8123
```

La inferencia se actualiza cada 10 muestras. A 200 Hz equivale a una actualizacion aproximada cada 50 ms despues de completar la primera ventana de 40 muestras.

## Error: `tensorflow/lite/micro/all_ops_resolver.h: No such file or directory`

Este error significa que Arduino IDE no tiene instalada una libreria TensorFlow Lite Micro compatible con ESP32. Para este firmware se valido el ZIP:

```text
D:\ESPOCH\TESIS\CARACTERIZACION_FINAL\firmware\arduino_libraries_cache\TensorFlowLite_ESP32-0.3.0.zip
```

Instalacion manual en Arduino IDE:

1. Abrir Arduino IDE.
2. Ir a `Sketch > Include Library > Add .ZIP Library...`.
3. Seleccionar `TensorFlowLite_ESP32-0.3.0.zip`.
4. Cerrar y volver a abrir Arduino IDE si no detecta la libreria.
5. Compilar nuevamente `xiao_esp32s3_tinyml_inference.ino`.

La libreria debe exponer estos encabezados:

```cpp
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
```

## Nota para XIAO ESP32S3

Si aparece `#error "unsupported board"` desde `Arduino_TensorFlowLite`, no usar esa libreria para este sketch. Esa libreria esta pensada para placas Arduino especificas y rechaza la XIAO ESP32S3.

Este firmware fue ajustado para usar:

```cpp
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
```

La libreria requerida en Arduino IDE es `TensorFlowLite_ESP32`. Si Arduino intenta usar `Arduino_TensorFlowLite`, cerrar el IDE, verificar que el sketch tenga `#include <TensorFlowLite_ESP32.h>` y compilar nuevamente.

## Error: `Didn't find op for builtin opcode 'FULLY_CONNECTED' version '12'`

La libreria `TensorFlowLite_ESP32` es antigua y no soporta la version 12 de `FULLY_CONNECTED`, generada por la conversion cuantizada/dynamic range de TensorFlow actual. Para la XIAO se genero una variante float compatible:

```text
models\tiny_mlp_final\emg_tiny_mlp_esp32_compatible.tflite
```

Esta variante usa:

```text
FULLY_CONNECTED version 1
SOFTMAX version 1
```

Los headers actuales de `generated/model_data.h` fueron regenerados con:

```powershell
.\.venv-tf\Scripts\python.exe src\embedded\export_tiny_mlp_for_xiao.py --model-dir models\tiny_mlp_final --tflite-file models\tiny_mlp_final\emg_tiny_mlp_esp32_compatible.tflite
```

Si vuelve a aparecer el error de version 12, significa que `model_data.h` fue regenerado accidentalmente con `emg_tiny_mlp.tflite` en lugar de la variante `emg_tiny_mlp_esp32_compatible.tflite`.
