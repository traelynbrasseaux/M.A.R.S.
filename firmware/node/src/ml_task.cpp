#include "ml_task.h"

#include <Arduino.h>
#include <esp_timer.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <atomic>
#include <new>

#include "frame.h"
#include "uplink.h"

namespace {

constexpr float kLearnSeconds = 60.0f;
constexpr float kCalibrateSeconds = 30.0f;
constexpr const char* kStreamNames[2] = {"accel", "audio"};
constexpr const char* kPhaseNames[5] = {"idle", "ranges", "bundle", "calibrate", "detect"};
constexpr const char* kModelNames[ml::kModels] = {"mahalanobis", "binary", "bipolar"};

struct Block {
  ml::Kind kind;
  uint16_t frames;
  int16_t samples[frame::kAudioSamples];
};
static_assert(frame::kAudioSamples >= 3 * frame::kAccelSamples, "an accel frame must fit a Block");

struct Command {
  char line[48];
};

QueueHandle_t blocks;
QueueHandle_t commands;
ml::ItemMemory* memory;
ml::FeatureExtractor* features;
ml::StreamDetector* detectors[2];  // accel, audio
std::atomic<uint32_t> drops{0};
std::atomic<uint32_t> worstUs[2];
std::atomic<bool> learning{false};
std::atomic<bool> alert{false};
float margin = ml::kDefaultMargin;
size_t alertModel = 0;
uint8_t latestAlerts[2];
ml::Phase lastPhase[2] = {ml::Phase::Idle, ml::Phase::Idle};
uint32_t scoreSeq = 0;

size_t indexOf(ml::Kind kind) { return kind == ml::Kind::Accel ? 0 : 1; }

void apply(const char* line) {
  char verb[16] = "";
  char arg1[16] = "";
  char arg2[16] = "";
  const int n = sscanf(line, "%15s %15s %15s", verb, arg1, arg2);
  if (n < 1) return;
  if (strcmp(verb, "learn") == 0) {
    const float learnS = n >= 2 ? static_cast<float>(atof(arg1)) : kLearnSeconds;
    const float calibrateS = n >= 3 ? static_cast<float>(atof(arg2)) : kCalibrateSeconds;
    for (ml::StreamDetector* d : detectors) d->learn(learnS, calibrateS);
    logf("learning %.0f s, then calibrating %.0f s", learnS, calibrateS);
  } else if (strcmp(verb, "detect") == 0) {
    bool ok = true;
    for (ml::StreamDetector* d : detectors) ok = d->detect() && ok;
    logf("%s", ok ? "detecting" : "nothing learned yet; send learn first");
  } else if (strcmp(verb, "stop") == 0) {
    for (ml::StreamDetector* d : detectors) d->stop();
    logf("stopped");
  } else if (strcmp(verb, "margin") == 0 && atof(arg1) > 0) {
    margin = static_cast<float>(atof(arg1));
    for (ml::StreamDetector* d : detectors) d->setMargin(margin);
    logf("margin %.2f", margin);
  } else if (strcmp(verb, "alert") == 0) {
    for (size_t m = 0; m < ml::kModels; m++) {
      if (strcmp(arg1, kModelNames[m]) == 0) alertModel = m;
    }
    logf("alert model %s", kModelNames[alertModel]);
  } else if (strcmp(verb, "status") == 0) {
    logf("status: accel %s, audio %s, margin %.2f, alert model %s",
         kPhaseNames[static_cast<int>(detectors[0]->phase())],
         kPhaseNames[static_cast<int>(detectors[1]->phase())], margin, kModelNames[alertModel]);
  } else {
    logf("unknown command: %s (try learn, detect, stop, margin, alert, status)", line);
  }
}

void report(const ml::BlockResult& r) {
  uint8_t payload[frame::kScoreBytes] = {static_cast<uint8_t>(r.kind),
                                         static_cast<uint8_t>(r.phase), r.alerts, 0};
  memcpy(payload + 4, r.scores, sizeof r.scores);
  memcpy(payload + 4 + sizeof r.scores, r.thresholds, sizeof r.thresholds);
  send(frame::Stream::Score, scoreSeq++, esp_timer_get_time(), payload, sizeof payload);

  const size_t i = indexOf(r.kind);
  latestAlerts[i] = r.phase == ml::Phase::Detect ? r.alerts : 0;
  alert = ((latestAlerts[0] | latestAlerts[1]) >> alertModel) & 1u;
}

void noticePhase(size_t i) {
  const ml::Phase now = detectors[i]->phase();
  if (now == lastPhase[i]) return;
  lastPhase[i] = now;
  if (now == ml::Phase::Detect) {
    const float* t = detectors[i]->thresholds();
    logf("%s detecting, thresholds mahalanobis %.4g, binary %.4g, bipolar %.4g", kStreamNames[i],
         t[0], t[1], t[2]);
  } else {
    logf("%s %s", kStreamNames[i], kPhaseNames[static_cast<int>(now)]);
  }
  bool busy = false;
  for (ml::StreamDetector* d : detectors) {
    busy = busy || (d->phase() != ml::Phase::Idle && d->phase() != ml::Phase::Detect);
  }
  learning = busy;
}

void mlTask(void*) {
  Block block;
  Command command;
  for (;;) {
    while (xQueueReceive(commands, &command, 0) == pdTRUE) apply(command.line);
    for (size_t i = 0; i < 2; i++) noticePhase(i);
    if (xQueueReceive(blocks, &block, pdMS_TO_TICKS(50)) != pdTRUE) continue;

    const size_t i = indexOf(block.kind);
    const int64_t start = esp_timer_get_time();
    ml::BlockResult result;
    const bool ready = detectors[i]->push(block.samples, block.frames, result);
    const uint32_t took = static_cast<uint32_t>(esp_timer_get_time() - start);
    if (took > worstUs[i]) worstUs[i] = took;
    if (ready) report(result);
  }
}

}  // namespace

bool mlBegin() {
  memory = new (std::nothrow) ml::ItemMemory;
  features = new (std::nothrow) ml::FeatureExtractor;
  if (!memory || !features) return false;
  detectors[0] = new (std::nothrow) ml::StreamDetector(ml::Kind::Accel, *memory, *features);
  detectors[1] = new (std::nothrow) ml::StreamDetector(ml::Kind::Audio, *memory, *features);
  if (!detectors[0] || !detectors[1]) return false;
  memory->generate(ml::kSeed);
  blocks = xQueueCreate(24, sizeof(Block));
  commands = xQueueCreate(4, sizeof(Command));
  xTaskCreatePinnedToCore(mlTask, "ml", 8192, nullptr, 2, nullptr, 0);
  return true;
}

void mlFeed(ml::Kind kind, const int16_t* samples, size_t frames) {
  if (!blocks) return;
  Block b;
  b.kind = kind;
  b.frames = static_cast<uint16_t>(frames);
  memcpy(b.samples, samples, frames * (kind == ml::Kind::Accel ? 3 : 1) * sizeof(int16_t));
  if (xQueueSend(blocks, &b, 0) != pdTRUE) drops++;
}

void mlCommand(const char* line) {
  if (!commands) {
    logf("ml is not running, ignoring: %s", line);
    return;
  }
  Command c;
  strncpy(c.line, line, sizeof c.line - 1);
  c.line[sizeof c.line - 1] = '\0';
  xQueueSend(commands, &c, 0);
}

bool mlLearning() { return learning; }

bool mlAlert() { return alert; }

void mlHealth(char* out, size_t len) {
  snprintf(out, len, "ml worst push accel %u us, audio %u us, ml drops %u",
           static_cast<unsigned>(worstUs[0].exchange(0)),
           static_cast<unsigned>(worstUs[1].exchange(0)), static_cast<unsigned>(drops.load()));
}
