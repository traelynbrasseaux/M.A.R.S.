// ADXL345 over 4-wire SPI at 3200 Hz with the FIFO in stream mode.
#pragma once

#include <Arduino.h>
#include <SPI.h>

struct AccelSample {
  int16_t x, y, z;
};

class Adxl345 {
 public:
  static constexpr uint8_t kDevId = 0xE5;
  static constexpr uint8_t kFifoDepth = 32;

  Adxl345(SPIClass& spi, int cs) : spi_(spi), cs_(cs) {}

  // Returns the DEVID it read; anything but kDevId means the part is missing or miswired.
  uint8_t begin(int sck, int miso, int mosi);
  uint8_t fifoEntries();
  void readSample(AccelSample& out);

 private:
  void write(uint8_t reg, uint8_t value);
  void read(uint8_t reg, uint8_t* buf, size_t len);

  SPIClass& spi_;
  int cs_;
};
