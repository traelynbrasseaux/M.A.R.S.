// INMP441 over I2S at 16 kHz, downshifted to int16 as in docs/interface.md.
#pragma once

#include <driver/i2s.h>

#include "frame.h"

class Inmp441 {
 public:
  static constexpr uint32_t kSampleRate = 16000;
  static constexpr size_t kBlock = frame::kAudioSamples;

  esp_err_t begin(int sck, int ws, int sd);
  // Blocks until one DMA buffer of kBlock samples has arrived.
  void read(int16_t* out);
  // DMA overruns since the last call, i.e. audio lost because reads fell behind.
  uint32_t takeOverruns();

 private:
  QueueHandle_t events_ = nullptr;
  int32_t raw_[kBlock];
};
