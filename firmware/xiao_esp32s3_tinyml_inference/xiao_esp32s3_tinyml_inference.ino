/*
  Inferencia TinyML sEMG en XIAO ESP32S3

  Flujo:
  1. Lee 4 canales analogicos a 200 Hz.
  2. Acumula una ventana de 40 muestras.
  3. Extrae 7 caracteristicas por canal: media, desviacion, minimo,
     maximo, media absoluta, RMS y longitud de forma de onda.
  4. Normaliza las 28 caracteristicas con la media/escala del entrenamiento.
  5. Ejecuta el modelo Tiny MLP exportado a TensorFlow Lite.
  6. Imprime por USB Serial: INFER,<indice>,<etiqueta>,<confianza>

  Antes de compilar, genera:
  - generated/model_data.h
  - generated/emg_metadata.h
*/

#include <Arduino.h>
#include <TensorFlowLite_ESP32.h>

#include "generated/emg_metadata.h"
#include "generated/model_data.h"

#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

static const int EMG_PINS[EMG_NUM_CHANNELS] = {A1, A2, A3, A4};
static const uint32_t SERIAL_BAUD = 921600;
static const uint16_t SAMPLING_RATE_HZ = 200;
static const uint32_t SAMPLE_PERIOD_MS = 5;
static const int STEP_SIZE = 10;
static const bool PRINT_TOP3 = true;
static const bool PRINT_RAW_MEAN = true;

static uint16_t emgWindow[EMG_WINDOW_SIZE][EMG_NUM_CHANNELS];
static int writeIndex = 0;
static int samplesSeen = 0;
static int samplesSinceInference = 0;

constexpr int kTensorArenaSize = 96 * 1024;
alignas(16) static uint8_t tensorArena[kTensorArenaSize];

static tflite::ErrorReporter* errorReporter = nullptr;
static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input = nullptr;
static TfLiteTensor* output = nullptr;

float featureBuffer[EMG_FEATURE_COUNT];

float dequantizeValue(int8_t value, float scale, int zeroPoint) {
  return (static_cast<int>(value) - zeroPoint) * scale;
}

void pushSample() {
  for (int channel = 0; channel < EMG_NUM_CHANNELS; channel++) {
    emgWindow[writeIndex][channel] = analogRead(EMG_PINS[channel]);
  }
  writeIndex = (writeIndex + 1) % EMG_WINDOW_SIZE;
  if (samplesSeen < EMG_WINDOW_SIZE) {
    samplesSeen++;
  }
  samplesSinceInference++;
}

uint16_t getWindowValue(int sampleIndex, int channel) {
  int oldestIndex = writeIndex;
  int physicalIndex = (oldestIndex + sampleIndex) % EMG_WINDOW_SIZE;
  return emgWindow[physicalIndex][channel];
}

void extractAndScaleFeatures() {
  int featureIndex = 0;
  for (int channel = 0; channel < EMG_NUM_CHANNELS; channel++) {
    float sum = 0.0f;
    float sumSquares = 0.0f;
    float sumAbs = 0.0f;
    float minValue = 4095.0f;
    float maxValue = 0.0f;
    float waveformLength = 0.0f;
    float previous = static_cast<float>(getWindowValue(0, channel));

    for (int sample = 0; sample < EMG_WINDOW_SIZE; sample++) {
      float value = static_cast<float>(getWindowValue(sample, channel));
      sum += value;
      sumSquares += value * value;
      sumAbs += fabs(value);
      if (value < minValue) minValue = value;
      if (value > maxValue) maxValue = value;
      if (sample > 0) {
        waveformLength += fabs(value - previous);
      }
      previous = value;
    }

    float mean = sum / EMG_WINDOW_SIZE;
    float variance = 0.0f;
    for (int sample = 0; sample < EMG_WINDOW_SIZE; sample++) {
      float value = static_cast<float>(getWindowValue(sample, channel));
      float centered = value - mean;
      variance += centered * centered;
    }
    variance /= EMG_WINDOW_SIZE;

    float rawFeatures[7] = {
      mean,
      sqrt(variance),
      minValue,
      maxValue,
      sumAbs / EMG_WINDOW_SIZE,
      sqrt(sumSquares / EMG_WINDOW_SIZE),
      waveformLength,
    };

    for (int local = 0; local < 7; local++) {
      float scaled = (rawFeatures[local] - EMG_FEATURE_MEAN[featureIndex]) / EMG_FEATURE_SCALE[featureIndex];
      featureBuffer[featureIndex] = scaled;
      featureIndex++;
    }
  }
}

bool fillInputTensor() {
  if (input->type == kTfLiteFloat32) {
    for (int i = 0; i < EMG_FEATURE_COUNT; i++) {
      input->data.f[i] = featureBuffer[i];
    }
    return true;
  }

  if (input->type == kTfLiteInt8) {
    const float scale = input->params.scale;
    const int zeroPoint = input->params.zero_point;
    for (int i = 0; i < EMG_FEATURE_COUNT; i++) {
      int quantized = round(featureBuffer[i] / scale) + zeroPoint;
      if (quantized < -128) quantized = -128;
      if (quantized > 127) quantized = 127;
      input->data.int8[i] = static_cast<int8_t>(quantized);
    }
    return true;
  }

  Serial.println("# error=unsupported_input_tensor");
  return false;
}

float getOutputScore(int index) {
  if (output->type == kTfLiteFloat32) {
    return output->data.f[index];
  }
  if (output->type == kTfLiteInt8) {
    return dequantizeValue(output->data.int8[index], output->params.scale, output->params.zero_point);
  }
  return -1000000.0f;
}

void getTop3(int topIndex[3], float topScore[3]) {
  for (int rank = 0; rank < 3; rank++) {
    topIndex[rank] = -1;
    topScore[rank] = -1000000.0f;
  }

  for (int i = 0; i < EMG_CLASS_COUNT; i++) {
    float score = getOutputScore(i);
    for (int rank = 0; rank < 3; rank++) {
      if (score > topScore[rank]) {
        for (int shift = 2; shift > rank; shift--) {
          topScore[shift] = topScore[shift - 1];
          topIndex[shift] = topIndex[shift - 1];
        }
        topScore[rank] = score;
        topIndex[rank] = i;
        break;
      }
    }
  }
}

void printRawMeans() {
  if (!PRINT_RAW_MEAN) {
    return;
  }

  Serial.print(",RAW_MEAN");
  for (int channel = 0; channel < EMG_NUM_CHANNELS; channel++) {
    float sum = 0.0f;
    for (int sample = 0; sample < EMG_WINDOW_SIZE; sample++) {
      sum += static_cast<float>(getWindowValue(sample, channel));
    }
    Serial.print(",");
    Serial.print(sum / EMG_WINDOW_SIZE, 1);
  }
}

void printOutput() {
  int topIndex[3];
  float topScore[3];
  getTop3(topIndex, topScore);

  if (!PRINT_TOP3) {
    Serial.print("INFER,");
    Serial.print(topIndex[0]);
    Serial.print(",");
    Serial.print(EMG_LABELS[topIndex[0]]);
    Serial.print(",");
    Serial.println(topScore[0], 4);
    return;
  }

  Serial.print("INFER_TOP3");
  for (int rank = 0; rank < 3; rank++) {
    Serial.print(",");
    Serial.print(topIndex[rank]);
    Serial.print(",");
    Serial.print(EMG_LABELS[topIndex[rank]]);
    Serial.print(",");
    Serial.print(topScore[rank], 4);
  }
  printRawMeans();
  Serial.println();
}

void runInference() {
  extractAndScaleFeatures();
  if (!fillInputTensor()) {
    return;
  }
  TfLiteStatus invokeStatus = interpreter->Invoke();
  if (invokeStatus != kTfLiteOk) {
    Serial.println("# error=invoke_failed");
    return;
  }
  printOutput();
}

void setupTinyMl() {
  static tflite::MicroErrorReporter microErrorReporter;
  errorReporter = &microErrorReporter;

  model = tflite::GetModel(g_emg_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("# error=model_schema_mismatch");
    while (true) delay(1000);
  }

  static tflite::ops::micro::AllOpsResolver resolver;
  static tflite::MicroInterpreter staticInterpreter(model, resolver, tensorArena, kTensorArenaSize, errorReporter);
  interpreter = &staticInterpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("# error=allocate_tensors_failed");
    while (true) delay(1000);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.print("# input_type=");
  Serial.print(input->type);
  Serial.print(", output_type=");
  Serial.println(output->type);
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(1500);

  analogReadResolution(12);
  for (int channel = 0; channel < EMG_NUM_CHANNELS; channel++) {
    pinMode(EMG_PINS[channel], INPUT);
  }

  setupTinyMl();
  Serial.println("# tinyml_emg_ready");
}

void loop() {
  static uint32_t lastSampleMs = 0;
  uint32_t now = millis();

  if (now - lastSampleMs >= SAMPLE_PERIOD_MS) {
    lastSampleMs += SAMPLE_PERIOD_MS;
    pushSample();

    if (samplesSeen >= EMG_WINDOW_SIZE && samplesSinceInference >= STEP_SIZE) {
      samplesSinceInference = 0;
      runInference();
    }
  }
}


