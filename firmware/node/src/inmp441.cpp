#include "inmp441.h"

namespace {
constexpr i2s_port_t kPort = I2S_NUM_0;
}

esp_err_t Inmp441::begin(int sck, int ws, int sd) {
  i2s_config_t config = {};
  config.mode = static_cast<i2s_mode_t>(I2S_MODE_MASTER | I2S_MODE_RX);
  config.sample_rate = kSampleRate;
  config.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;
  // This hardware setup produces audio in the right slot; verify L/R wiring if changing slots.
  config.channel_format = I2S_CHANNEL_FMT_ONLY_RIGHT;
  config.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  config.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  config.dma_buf_count = 8;
  config.dma_buf_len = kBlock;
  esp_err_t err = i2s_driver_install(kPort, &config, 8, &events_);
  if (err != ESP_OK) return err;

  i2s_pin_config_t pins = {};
  pins.mck_io_num = I2S_PIN_NO_CHANGE;
  pins.bck_io_num = sck;
  pins.ws_io_num = ws;
  pins.data_out_num = I2S_PIN_NO_CHANGE;
  pins.data_in_num = sd;
  return i2s_set_pin(kPort, &pins);
}

void Inmp441::read(int16_t* out) {
  size_t got = 0;
  i2s_read(kPort, raw_, sizeof raw_, &got, portMAX_DELAY);
  // The 24-bit sample sits in bits 31..8; keeping the top 16 bits is the documented downshift.
  for (size_t i = 0; i < kBlock; i++) out[i] = static_cast<int16_t>(raw_[i] >> 16);
}

uint32_t Inmp441::takeOverruns() {
  uint32_t overruns = 0;
  i2s_event_t event;
  while (xQueueReceive(events_, &event, 0) == pdTRUE) {
    if (event.type == I2S_EVENT_RX_Q_OVF) overruns++;
  }
  return overruns;
}
