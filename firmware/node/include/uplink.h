// Frames out to the host: any task can queue one, and a single task writes them to UART0.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "frame.h"

void uplinkBegin(uint32_t baud);
// Never blocks: if the queue is full the frame is dropped and the host sees a sequence gap.
void send(frame::Stream stream, uint32_t seq, int64_t timestampUs, const void* payload, size_t len);
void logf(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
uint32_t uplinkDrops();
