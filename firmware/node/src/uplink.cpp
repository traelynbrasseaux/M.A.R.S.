#include "uplink.h"

#include <Arduino.h>
#include <esp_timer.h>
#include <stdarg.h>

#include <atomic>

namespace {

QueueHandle_t txQueue;
std::atomic<uint32_t> drops{0};
std::atomic<uint32_t> logSeq{0};

void txTask(void*) {
  frame::Frame f;
  for (;;) {
    if (xQueueReceive(txQueue, &f, portMAX_DELAY) == pdTRUE) Serial.write(f.bytes, f.len);
  }
}

}  // namespace

void uplinkBegin(uint32_t baud) {
  Serial.setTxBufferSize(4096);
  Serial.begin(baud);
  txQueue = xQueueCreate(16, sizeof(frame::Frame));
  xTaskCreatePinnedToCore(txTask, "tx", 4096, nullptr, 3, nullptr, 0);
}

void send(frame::Stream stream, uint32_t seq, int64_t timestampUs, const void* payload, size_t len) {
  frame::Frame f;
  frame::encode(f, stream, seq, static_cast<uint64_t>(timestampUs), payload, len);
  if (xQueueSend(txQueue, &f, 0) != pdTRUE) drops++;
}

void logf(const char* fmt, ...) {
  char text[200];
  va_list args;
  va_start(args, fmt);
  const int n = vsnprintf(text, sizeof text, fmt, args);
  va_end(args);
  if (n > 0) {
    const size_t len = min(static_cast<size_t>(n), sizeof text - 1);
    send(frame::Stream::Log, logSeq++, esp_timer_get_time(), text, len);
  }
}

uint32_t uplinkDrops() { return drops.load(); }
