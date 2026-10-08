// Sensor node: sample the ADXL345 and INMP441 and stream both as frames over UART0.
#include <Arduino.h>
#include <esp_timer.h>
#include <stdarg.h>

#include <atomic>

#include "adxl345.h"
#include "frame.h"
#include "inmp441.h"
#include "pins.h"

namespace {

constexpr uint32_t kBaud = 921600;
constexpr uint8_t kAccelDrainAt = 16;  // half the FIFO, so a late poll still has 5 ms of slack
constexpr uint32_t kHealthPeriodMs = 10000;

// VSPI for the accelerometer leaves HSPI free for the LoRa module, so radio traffic never stalls it.
SPIClass accelSpi(VSPI);
Adxl345 accel(accelSpi, PIN_ADXL_CS);
Inmp441 mic;

QueueHandle_t txQueue;
std::atomic<uint32_t> queueDrops{0};
std::atomic<uint32_t> accelOverruns{0};
std::atomic<uint32_t> audioOverruns{0};
uint32_t logSeq = 0;

// Never blocks: if the queue is full the frame is dropped and the host sees a sequence gap.
void send(frame::Stream stream, uint32_t seq, int64_t timestampUs, const void* payload,
          size_t len) {
  frame::Frame f;
  frame::encode(f, stream, seq, static_cast<uint64_t>(timestampUs), payload, len);
  if (xQueueSend(txQueue, &f, 0) != pdTRUE) queueDrops++;
}

void logf(const char* fmt, ...) {
  char text[160];
  va_list args;
  va_start(args, fmt);
  const int n = vsnprintf(text, sizeof text, fmt, args);
  va_end(args);
  if (n > 0) {
    const size_t len = min(static_cast<size_t>(n), sizeof text - 1);
    send(frame::Stream::Log, logSeq++, esp_timer_get_time(), text, len);
  }
}

void txTask(void*) {
  frame::Frame f;
  for (;;) {
    if (xQueueReceive(txQueue, &f, portMAX_DELAY) == pdTRUE) Serial.write(f.bytes, f.len);
  }
}

void accelTask(void*) {
  int16_t samples[frame::kAccelSamples * 3];
  size_t n = 0;
  int64_t frameStartUs = 0;
  uint32_t seq = 0;
  for (;;) {
    const uint8_t entries = accel.fifoEntries();
    if (entries < kAccelDrainAt) {
      vTaskDelay(1);
      continue;
    }
    if (entries >= Adxl345::kFifoDepth) accelOverruns++;
    const int64_t now = esp_timer_get_time();
    for (uint8_t k = 0; k < entries; k++) {
      AccelSample s;
      accel.readSample(s);
      // Entry k of n was sampled (n - 1 - k) periods of 312.5 us before the drain.
      if (n == 0) frameStartUs = now - static_cast<int64_t>(entries - 1 - k) * 625 / 2;
      samples[3 * n] = s.x;
      samples[3 * n + 1] = s.y;
      samples[3 * n + 2] = s.z;
      if (++n == frame::kAccelSamples) {
        send(frame::Stream::Accel, seq++, frameStartUs, samples, sizeof samples);
        n = 0;
      }
    }
  }
}

void audioTask(void*) {
  int16_t samples[Inmp441::kBlock];
  uint32_t seq = 0;
  for (;;) {
    mic.read(samples);
    // The block completed just now, so its first sample is one block (62.5 us each) older.
    const int64_t startUs = esp_timer_get_time() - static_cast<int64_t>(Inmp441::kBlock) * 125 / 2;
    send(frame::Stream::Audio, seq++, startUs, samples, sizeof samples);
    audioOverruns += mic.takeOverruns();
  }
}

void startAccel() {
  const uint8_t devid = accel.begin(PIN_ADXL_SCK, PIN_ADXL_MISO, PIN_ADXL_MOSI);
  if (devid != Adxl345::kDevId) {
    logf("adxl345 devid 0x%02X, expected 0x%02X, accel disabled", devid, Adxl345::kDevId);
    return;
  }
  xTaskCreatePinnedToCore(accelTask, "accel", 6144, nullptr, 5, nullptr, 1);
  logf("adxl345 ok, 3200 Hz, +/-16 g full resolution");
}

void startAudio() {
  const esp_err_t err = mic.begin(PIN_MIC_SCK, PIN_MIC_WS, PIN_MIC_SD);
  if (err != ESP_OK) {
    logf("i2s setup failed (%s), audio disabled", esp_err_to_name(err));
    return;
  }
  xTaskCreatePinnedToCore(audioTask, "audio", 6144, nullptr, 4, nullptr, 1);
  logf("inmp441 ok, 16 kHz");
}

}  // namespace

void setup() {
  Serial.setTxBufferSize(4096);
  Serial.begin(kBaud);
  txQueue = xQueueCreate(16, sizeof(frame::Frame));
  xTaskCreatePinnedToCore(txTask, "tx", 4096, nullptr, 3, nullptr, 0);
  logf("mhm node, frame format v%u", frame::kVersion);
  startAccel();
  startAudio();
}

void loop() {
  vTaskDelay(pdMS_TO_TICKS(kHealthPeriodMs));
  logf("health: accel fifo overruns %u, audio dma overruns %u, tx queue drops %u",
       accelOverruns.load(), audioOverruns.load(), queueDrops.load());
}
