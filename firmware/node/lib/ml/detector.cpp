#include "detector.h"

#include <math.h>
#include <string.h>

namespace ml {

namespace {

constexpr uint32_t kMaxBundleWindows = 65535;  // HdcModel counts votes in 16 bits

uint32_t windowsFor(float seconds, float fs) {
  const float windows = seconds * fs / kHop;
  return windows < 1 ? 1 : static_cast<uint32_t>(windows + 0.5f);
}

}  // namespace

StreamDetector::StreamDetector(Kind kind, const ItemMemory& memory, FeatureExtractor& features)
    : kind_(kind),
      channels_(kind == Kind::Accel ? 3 : 1),
      nFeatures_(kind == Kind::Accel ? kVibrationFeatures : kAudioFeatures),
      fs_(kind == Kind::Accel ? kAccelFs : kAudioFs),
      memory_(memory),
      features_(features),
      blockWindows_(windowsFor(kBlockSeconds, kind == Kind::Accel ? kAccelFs : kAudioFs)) {
  memset(blockSum_, 0, sizeof blockSum_);
  memset(calibrationMax_, 0, sizeof calibrationMax_);
  memset(thresholds_, 0, sizeof thresholds_);
}

bool StreamDetector::push(const int16_t* samples, size_t frames, BlockResult& out) {
  bool ready = false;
  for (size_t i = 0; i < frames; i++) {
    memcpy(&ring_[ringPos_ * channels_], &samples[i * channels_], channels_ * sizeof(int16_t));
    ringPos_ = (ringPos_ + 1) % kWindow;
    if (filled_ < kWindow) filled_++;
    // Same windows as the Python framing: the first at kWindow samples, then every kHop.
    if (++sinceWindow_ < kHop || filled_ < kWindow) continue;
    sinceWindow_ = 0;
    if (phase_ == Phase::Idle) continue;
    const size_t head = ringPos_ * channels_;
    const size_t total = kWindow * channels_;
    memcpy(window_, ring_ + head, (total - head) * sizeof(int16_t));
    memcpy(window_ + (total - head), ring_, head * sizeof(int16_t));
    ready = onWindow(out) || ready;
  }
  return ready;
}

bool StreamDetector::onWindow(BlockResult& out) {
  if (kind_ == Kind::Accel) {
    features_.vibration(window_, fs_, x_);
  } else {
    features_.audio(window_, x_);
  }

  float s[kModels] = {0, 0, 0};
  const Phase phase = phase_;
  switch (phase) {
    case Phase::Ranges:
      hdc_.observe(x_);
      mahalanobis_.observe(x_);
      if (--left_ == 0) {
        hdc_.finishRanges();
        enter(Phase::Bundle);
      }
      break;
    case Phase::Bundle:
      hdc_.bundle(memory_, x_);
      mahalanobis_.observe(x_);
      if (--left_ == 0) {
        hdc_.finish(memory_);
        trained_ = mahalanobis_.finish();
        enter(trained_ ? Phase::Calibrate : Phase::Idle);
      }
      break;
    case Phase::Calibrate:
    case Phase::Detect:
      s[0] = mahalanobis_.score(x_);
      hdc_.score(memory_, x_, s[1], s[2]);
      break;
    case Phase::Idle:
      return false;
  }
  if (phase_ != phase) return false;  // a learning phase ended mid-block and enter() reset it

  for (size_t m = 0; m < kModels; m++) blockSum_[m] += s[m];
  if (++inBlock_ < blockWindows_) return false;

  out.kind = kind_;
  out.phase = phase;
  out.alerts = 0;
  for (size_t m = 0; m < kModels; m++) {
    out.scores[m] = static_cast<float>(blockSum_[m] / inBlock_);
    out.thresholds[m] = thresholds_[m];
    blockSum_[m] = 0;
  }
  inBlock_ = 0;

  if (phase == Phase::Calibrate) {
    for (size_t m = 0; m < kModels; m++) calibrationMax_[m] = fmaxf(calibrationMax_[m], out.scores[m]);
    if (--left_ == 0) {
      calibrated_ = true;
      applyMargin();
      enter(Phase::Detect);
    }
  } else if (phase == Phase::Detect) {
    for (size_t m = 0; m < kModels; m++) {
      if (out.scores[m] > thresholds_[m]) out.alerts |= 1u << m;
    }
  }
  return true;
}

void StreamDetector::enter(Phase phase) {
  phase_ = phase;
  inBlock_ = 0;
  memset(blockSum_, 0, sizeof blockSum_);
  if (phase == Phase::Bundle) left_ = bundleWindows_;
  if (phase == Phase::Calibrate) left_ = calibrateBlocks_;
}

void StreamDetector::learn(float learnSeconds, float calibrateSeconds) {
  const uint32_t half = windowsFor(learnSeconds / 2, fs_);
  bundleWindows_ = half > kMaxBundleWindows ? kMaxBundleWindows : half;
  const float blocks = calibrateSeconds / kBlockSeconds;
  calibrateBlocks_ = blocks < 1 ? 1 : static_cast<uint32_t>(blocks + 0.5f);
  hdc_.begin(nFeatures_);
  mahalanobis_.begin(nFeatures_);
  trained_ = false;
  calibrated_ = false;
  memset(calibrationMax_, 0, sizeof calibrationMax_);
  memset(thresholds_, 0, sizeof thresholds_);
  enter(Phase::Ranges);
  left_ = half < 2 ? 2 : half;  // Mahalanobis needs at least two windows
}

bool StreamDetector::detect() {
  if (!calibrated_) return false;
  enter(Phase::Detect);
  return true;
}

void StreamDetector::stop() { enter(Phase::Idle); }

void StreamDetector::setMargin(float margin) {
  margin_ = margin;
  if (calibrated_) applyMargin();
}

void StreamDetector::applyMargin() {
  for (size_t m = 0; m < kModels; m++) thresholds_[m] = calibrationMax_[m] * margin_;
}

}  // namespace ml
