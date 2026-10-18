#include "hdc.h"

#include <math.h>
#include <string.h>

#include "stats.h"

namespace ml {

namespace {

constexpr uint32_t kLfsrTaps = 0x80200003;  // x^32 + x^22 + x^2 + x + 1, maximal length
constexpr size_t kThresholdBits = 16;

class Lfsr {
 public:
  explicit Lfsr(uint32_t seed) : state_(seed) {}

  // Right-shift Galois LFSR that emits its LSB before each shift.
  uint32_t bit() {
    const uint32_t out = state_ & 1u;
    state_ >>= 1;
    if (out) state_ ^= kLfsrTaps;
    return out;
  }

 private:
  uint32_t state_;
};

void fill(Lfsr& lfsr, uint32_t* words) {
  memset(words, 0, kWords * sizeof(uint32_t));
  for (size_t i = 0; i < kDim; i++) words[i / 32] |= lfsr.bit() << (i % 32);
}

bool bitAt(const uint32_t* words, size_t i) { return (words[i / 32] >> (i % 32)) & 1u; }

}  // namespace

void ItemMemory::generate(uint32_t seed) {
  Lfsr lfsr(seed);
  fill(lfsr, tiebreak);
  uint32_t base[kWords];
  fill(lfsr, base);
  memset(levels, 0, sizeof levels);
  for (size_t i = 0; i < kDim; i++) {
    uint32_t threshold = 0;
    for (size_t b = 0; b < kThresholdBits; b++) threshold |= lfsr.bit() << b;
    // Level l flips this position when its threshold is below l / (L - 1) of half the range.
    for (size_t l = 0; l < kLevels; l++) {
      const uint32_t cut = l * (1u << (kThresholdBits - 1)) / (kLevels - 1);
      const bool on = bitAt(base, i) != (threshold < cut);
      levels[l][i / 32] |= static_cast<uint32_t>(on) << (i % 32);
    }
  }
  for (size_t f = 0; f < kMaxFeatures; f++) fill(lfsr, ids[f]);
}

void HdcModel::begin(size_t nFeatures) {
  n_ = nFeatures;
  observed_ = 0;
  bundled_ = 0;
  memset(sum_, 0, sizeof sum_);
  memset(sumSq_, 0, sizeof sumSq_);
  memset(votes_, 0, sizeof votes_);
  memset(prototypeSum_, 0, sizeof prototypeSum_);
}

void HdcModel::observe(const float* x) {
  for (size_t f = 0; f < n_; f++) {
    sum_[f] += x[f];
    sumSq_[f] += static_cast<double>(x[f]) * x[f];
  }
  observed_++;
}

void HdcModel::finishRanges() {
  for (size_t f = 0; f < n_; f++) {
    const double mean = sum_[f] / observed_;
    const double std = featureStd(mean, sumSq_[f] / observed_ - mean * mean);
    lo_[f] = mean - kRangeSigma * std;
    width_[f] = 2 * kRangeSigma * std;
  }
}

bool HdcModel::majority(uint32_t ones, uint32_t votes, const ItemMemory& memory, size_t i) const {
  if (2 * ones == votes) return bitAt(memory.tiebreak, i);
  return 2 * ones > votes;
}

void HdcModel::encode(const ItemMemory& memory, const float* x) {
  memset(ones_, 0, sizeof ones_);
  for (size_t f = 0; f < n_; f++) {
    const double level = floor((x[f] - lo_[f]) / width_[f] * kLevels);
    const size_t q = level < 0 ? 0 : level > kLevels - 1 ? kLevels - 1 : static_cast<size_t>(level);
    const uint32_t* id = memory.ids[f];
    const uint32_t* lv = memory.levels[q];
    for (size_t w = 0; w < kWords; w++) {
      const uint32_t bound = id[w] ^ lv[w];
      uint8_t* out = ones_ + 32 * w;
      for (size_t b = 0; b < 32; b++) out[b] += (bound >> b) & 1u;
    }
  }
}

void HdcModel::bundle(const ItemMemory& memory, const float* x) {
  encode(memory, x);
  for (size_t i = 0; i < kDim; i++) {
    votes_[i] += majority(ones_[i], n_, memory, i);
    prototypeSum_[i] += static_cast<int32_t>(n_) - 2 * ones_[i];
  }
  bundled_++;
}

void HdcModel::finish(const ItemMemory& memory) {
  memset(prototype_, 0, sizeof prototype_);
  double normSq = 0;
  for (size_t i = 0; i < kDim; i++) {
    prototype_[i / 32] |= static_cast<uint32_t>(majority(votes_[i], bundled_, memory, i)) << (i % 32);
    normSq += static_cast<double>(prototypeSum_[i]) * prototypeSum_[i];
  }
  prototypeNorm_ = sqrt(normSq);
}

void HdcModel::score(const ItemMemory& memory, const float* x, float& binary, float& bipolar) {
  encode(memory, x);
  uint32_t mismatches = 0;
  int64_t dot = 0;
  int64_t normSq = 0;
  for (size_t i = 0; i < kDim; i++) {
    mismatches += majority(ones_[i], n_, memory, i) != bitAt(prototype_, i);
    const int32_t e = static_cast<int32_t>(n_) - 2 * ones_[i];
    dot += static_cast<int64_t>(e) * prototypeSum_[i];
    normSq += e * e;
  }
  binary = static_cast<float>(mismatches) / kDim;
  bipolar = static_cast<float>(1.0 - dot / (sqrt(static_cast<double>(normSq)) * prototypeNorm_));
}

}  // namespace ml
