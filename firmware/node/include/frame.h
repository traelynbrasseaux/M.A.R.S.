// UART frame format from docs/interface.md; must match host/mhm/capture/frame.py.
#pragma once

#include <stddef.h>
#include <stdint.h>

namespace frame {

constexpr uint8_t kVersion = 1;
constexpr size_t kHeaderSize = 18;
constexpr size_t kTrailerSize = 2;

enum class Stream : uint8_t { Accel = 1, Audio = 2, Log = 3 };

constexpr size_t kAccelSamples = 32;   // 10 ms at 3200 Hz
constexpr size_t kAudioSamples = 256;  // 16 ms at 16 kHz
constexpr size_t kMaxPayload = kAudioSamples * sizeof(int16_t);
constexpr size_t kMaxFrame = kHeaderSize + kMaxPayload + kTrailerSize;

struct Frame {
  uint16_t len;
  uint8_t bytes[kMaxFrame];
};

// CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection, no final XOR.
uint16_t crc16(const uint8_t* data, size_t len);

void encode(Frame& out, Stream stream, uint32_t seq, uint64_t timestamp_us, const void* payload,
            size_t payload_len);

}  // namespace frame
