/*
  Adquisicion sEMG multicanal con FreeRTOS - Seeed Studio XIAO ESP32S3

  Version final de adquisicion piloto a 200 Hz para el protocolo funcional
  de 10 clases. Envia por USB Serial:
  timestamp_ms,sample_index,emg_ch1,emg_ch2,emg_ch3,emg_ch4,board_id,sampling_rate_hz

  La adquisicion ADC y la transmision serial se ejecutan en tareas FreeRTOS
  separadas. Esta base se usa para validar estabilidad antes de integrar
  inferencia TFLite Micro y control PWM de servomotores.
*/

#include <Arduino.h>

static const char* BOARD_ID = "xiao_esp32s3_rtos_200hz";

static const int NUM_CHANNELS = 4;
static const int EMG_PINS[NUM_CHANNELS] = {A1, A2, A3, A4};

static const uint32_t SERIAL_BAUD = 921600;
static const uint16_t SAMPLING_RATE_HZ = 200;
static const uint32_t SAMPLE_PERIOD_TICKS = pdMS_TO_TICKS(5);
static const size_t SAMPLE_QUEUE_LENGTH = 128;

struct EmgSample {
  uint32_t timestampMs;
  uint32_t sampleIndex;
  uint16_t values[NUM_CHANNELS];
};

static QueueHandle_t sampleQueue = nullptr;
static volatile uint32_t droppedSamples = 0;

void acquisitionTask(void* parameter) {
  (void)parameter;
  uint32_t sampleIndex = 0;
  TickType_t lastWakeTime = xTaskGetTickCount();

  while (true) {
    EmgSample sample;
    sample.timestampMs = millis();
    sample.sampleIndex = sampleIndex++;

    for (int channel = 0; channel < NUM_CHANNELS; channel++) {
      sample.values[channel] = analogRead(EMG_PINS[channel]);
    }

    if (xQueueSend(sampleQueue, &sample, 0) != pdPASS) {
      droppedSamples++;
    }

    vTaskDelayUntil(&lastWakeTime, SAMPLE_PERIOD_TICKS);
  }
}

void serialTask(void* parameter) {
  (void)parameter;
  EmgSample sample;

  Serial.println("timestamp_ms,sample_index,emg_ch1,emg_ch2,emg_ch3,emg_ch4,board_id,sampling_rate_hz");

  while (true) {
    if (xQueueReceive(sampleQueue, &sample, portMAX_DELAY) == pdPASS) {
      Serial.print(sample.timestampMs);
      Serial.print(',');
      Serial.print(sample.sampleIndex);
      for (int channel = 0; channel < NUM_CHANNELS; channel++) {
        Serial.print(',');
        Serial.print(sample.values[channel]);
      }
      Serial.print(',');
      Serial.print(BOARD_ID);
      Serial.print(',');
      Serial.println(SAMPLING_RATE_HZ);
    }
  }
}

void diagnosticsTask(void* parameter) {
  (void)parameter;
  while (true) {
    if (droppedSamples > 0) {
      Serial.print("# dropped_samples=");
      Serial.println(droppedSamples);
    }
    vTaskDelay(pdMS_TO_TICKS(1000));
  }
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(1500);

  analogReadResolution(12);
  for (int channel = 0; channel < NUM_CHANNELS; channel++) {
    pinMode(EMG_PINS[channel], INPUT);
  }

  sampleQueue = xQueueCreate(SAMPLE_QUEUE_LENGTH, sizeof(EmgSample));
  if (sampleQueue == nullptr) {
    Serial.println("# error=no_queue");
    while (true) {
      delay(1000);
    }
  }

  xTaskCreatePinnedToCore(acquisitionTask, "emg_adc", 4096, nullptr, 3, nullptr, 0);
  xTaskCreatePinnedToCore(serialTask, "emg_serial", 4096, nullptr, 2, nullptr, 1);
  xTaskCreatePinnedToCore(diagnosticsTask, "emg_diag", 2048, nullptr, 1, nullptr, 1);
}

void loop() {
  vTaskDelay(pdMS_TO_TICKS(1000));
}
