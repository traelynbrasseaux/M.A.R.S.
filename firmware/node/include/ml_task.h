// The on-device detectors from lib/ml, running in their own task and fed by the sensor tasks.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "detector.h"

bool mlBegin();  // false if the models do not fit in memory
// Called by the sensor tasks with each frame's samples; never blocks.
void mlFeed(ml::Kind kind, const int16_t* samples, size_t frames);
// One command line from the host or the BOOT button, such as "learn 60 30".
void mlCommand(const char* line);
bool mlLearning();  // a stream is learning or calibrating
bool mlAlert();     // the chosen alert model flagged the latest block of either stream
void mlHealth(char* out, size_t len);
