#include "frame.h"

#include <string.h>

namespace frame {

uint16_t crc16(const uint8_t* data, size_t len) {
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < len; i++) {
    crc ^= static_cast<uint16_t>(data[i] << 8);
    for (int bit = 0; bit < 8; bit++) {
      crc = (crc & 0x8000) ? static_cast<uint16_t>((crc << 1) ^ 0x1021)
                           : static_cast<uint16_t>(crc << 1);
    }
  }
  return crc;
}

static size_t bytesPerSample(Stream stream) {
  switch (stream) {
    case Stream::Accel:
      return 3 * sizeof(int16_t);
    case Stream::Audio:
      return sizeof(int16_t);
    case Stream::Log:
      break;
  }
  return 1;
}

void encode(Frame& out, Stream stream, uint32_t seq, uint64_t timestamp_us, const void* payload,
            size_t payload_len) {
  // Fields are copied as-is because the ESP32 is little-endian, like the wire format.
  uint8_t* p = out.bytes;
  const uint16_t count = static_cast<uint16_t>(payload_len / bytesPerSample(stream));
  p[0] = 0xA5;
  p[1] = 0x5A;
  p[2] = kVersion;
  p[3] = static_cast<uint8_t>(stream);
  memcpy(p + 4, &seq, sizeof seq);
  memcpy(p + 8, &timestamp_us, sizeof timestamp_us);
  memcpy(p + 16, &count, sizeof count);
  memcpy(p + kHeaderSize, payload, payload_len);
  const uint16_t crc = crc16(p, kHeaderSize + payload_len);
  memcpy(p + kHeaderSize + payload_len, &crc, sizeof crc);
  out.len = static_cast<uint16_t>(kHeaderSize + payload_len + kTrailerSize);
}

}  // namespace frame
