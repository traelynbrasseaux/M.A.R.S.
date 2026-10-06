#include "adxl345.h"

namespace {

constexpr uint8_t REG_DEVID = 0x00;
constexpr uint8_t REG_BW_RATE = 0x2C;
constexpr uint8_t REG_POWER_CTL = 0x2D;
constexpr uint8_t REG_DATA_FORMAT = 0x31;
constexpr uint8_t REG_DATAX0 = 0x32;
constexpr uint8_t REG_FIFO_CTL = 0x38;
constexpr uint8_t REG_FIFO_STATUS = 0x39;

constexpr uint8_t RATE_3200HZ = 0x0F;
constexpr uint8_t FULL_RES_16G = 0x0B;      // FULL_RES plus range bits 11, 4-wire SPI
constexpr uint8_t FIFO_STREAM = 0x80;       // FIFO_MODE bits 10
constexpr uint8_t MEASURE = 0x08;
constexpr uint8_t SPI_READ = 0x80;
constexpr uint8_t SPI_MULTIBYTE = 0x40;

// 5 MHz is the ADXL345's SPI maximum; mode 3 is CPOL = 1, CPHA = 1.
const SPISettings kSettings(5000000, MSBFIRST, SPI_MODE3);

}  // namespace

uint8_t Adxl345::begin(int sck, int miso, int mosi) {
  pinMode(cs_, OUTPUT);
  digitalWrite(cs_, HIGH);
  spi_.begin(sck, miso, mosi, -1);

  uint8_t devid = 0;
  read(REG_DEVID, &devid, 1);
  if (devid != kDevId) return devid;

  write(REG_POWER_CTL, 0);  // standby while configuring
  write(REG_BW_RATE, RATE_3200HZ);
  write(REG_DATA_FORMAT, FULL_RES_16G);
  write(REG_FIFO_CTL, FIFO_STREAM);
  write(REG_POWER_CTL, MEASURE);
  return devid;
}

uint8_t Adxl345::fifoEntries() {
  uint8_t status = 0;
  read(REG_FIFO_STATUS, &status, 1);
  return status & 0x3F;
}

void Adxl345::readSample(AccelSample& out) {
  uint8_t raw[6];
  read(REG_DATAX0, raw, sizeof raw);
  out.x = static_cast<int16_t>(raw[0] | (raw[1] << 8));
  out.y = static_cast<int16_t>(raw[2] | (raw[3] << 8));
  out.z = static_cast<int16_t>(raw[4] | (raw[5] << 8));
  // Datasheet: allow 5 us after a data read for the FIFO to pop before the next read.
  delayMicroseconds(5);
}

void Adxl345::write(uint8_t reg, uint8_t value) {
  spi_.beginTransaction(kSettings);
  digitalWrite(cs_, LOW);
  spi_.transfer(reg);
  spi_.transfer(value);
  digitalWrite(cs_, HIGH);
  spi_.endTransaction();
}

void Adxl345::read(uint8_t reg, uint8_t* buf, size_t len) {
  spi_.beginTransaction(kSettings);
  digitalWrite(cs_, LOW);
  spi_.transfer(SPI_READ | (len > 1 ? SPI_MULTIBYTE : 0) | reg);
  for (size_t i = 0; i < len; i++) buf[i] = spi_.transfer(0);
  digitalWrite(cs_, HIGH);
  spi_.endTransaction();
}
