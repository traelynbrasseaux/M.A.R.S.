// Sensor wiring for the node. Keep in sync with the pin map in docs/interface.md.
// A sensor whose pins are still -1 is skipped at boot and reported in a log frame.
#pragma once

// TODO(Martin): ADXL345 on its own SPI bus, separate from the LoRa radio's.
constexpr int PIN_ADXL_SCK = -1;
constexpr int PIN_ADXL_MOSI = -1;  // ADXL345 SDA/SDI
constexpr int PIN_ADXL_MISO = -1;  // ADXL345 SDO
constexpr int PIN_ADXL_CS = -1;

// TODO(Martin): INMP441 over I2S, with its L/R pin tied to GND.
constexpr int PIN_MIC_SCK = -1;
constexpr int PIN_MIC_WS = -1;
constexpr int PIN_MIC_SD = -1;

// TODO(Martin): set to 36 if the sensors run from Vext; -1 leaves Vext untouched.
constexpr int PIN_VEXT_CTRL = -1;
