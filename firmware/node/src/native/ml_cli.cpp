// Command line wrapper around lib/ml for the PC, so host tests can check it against the Python.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

#include "features.h"
#include "hdc.h"
#include "mahalanobis.h"

namespace {

template <typename T>
std::vector<T> readAll(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "cannot open %s\n", path);
    exit(2);
  }
  fseek(f, 0, SEEK_END);
  const long bytes = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::vector<T> data(static_cast<size_t>(bytes) / sizeof(T));
  if (fread(data.data(), sizeof(T), data.size(), f) != data.size()) {
    fprintf(stderr, "short read on %s\n", path);
    exit(2);
  }
  fclose(f);
  return data;
}

template <typename T>
void writeAll(const char* path, const std::vector<T>& data) {
  FILE* f = fopen(path, "wb");
  if (!f || fwrite(data.data(), sizeof(T), data.size(), f) != data.size()) {
    fprintf(stderr, "cannot write %s\n", path);
    exit(2);
  }
  fclose(f);
}

// features <accel|audio> <fs> <in.f32> <out.f32>: windows of kWindow every kHop samples.
int features(char** argv) {
  static ml::FeatureExtractor fx;
  const std::string kind = argv[2];
  const float fs = static_cast<float>(atof(argv[3]));
  const std::vector<float> x = readAll<float>(argv[4]);
  const bool accel = kind == "accel";
  const size_t channels = accel ? 3 : 1;
  const size_t width = accel ? ml::kVibrationFeatures : ml::kAudioFeatures;
  const size_t n = x.size() / channels;
  const size_t windows = n < ml::kWindow ? 0 : 1 + (n - ml::kWindow) / ml::kHop;
  std::vector<float> out(windows * width);
  for (size_t w = 0; w < windows; w++) {
    const float* window = x.data() + w * ml::kHop * channels;
    if (accel) {
      fx.vibration(window, fs, &out[w * width]);
    } else {
      fx.audio(window, &out[w * width]);
    }
  }
  writeAll(argv[5], out);
  return 0;
}

// score <n> <stats.f32> <bundle.f32> <test.f32> <out.f32>: learn in the node's two phases, then
// write mahalanobis, hdc binary and hdc bipolar scores for each test window.
int score(char** argv) {
  static ml::ItemMemory memory;
  static ml::HdcModel hdc;
  static ml::MahalanobisModel mahalanobis;
  const size_t n = static_cast<size_t>(atoi(argv[2]));
  const std::vector<float> stats = readAll<float>(argv[3]);
  const std::vector<float> bundle = readAll<float>(argv[4]);
  const std::vector<float> test = readAll<float>(argv[5]);

  memory.generate(ml::kSeed);
  hdc.begin(n);
  mahalanobis.begin(n);
  for (size_t i = 0; i + n <= stats.size(); i += n) {
    hdc.observe(&stats[i]);
    mahalanobis.observe(&stats[i]);
  }
  hdc.finishRanges();
  for (size_t i = 0; i + n <= bundle.size(); i += n) {
    hdc.bundle(memory, &bundle[i]);
    mahalanobis.observe(&bundle[i]);
  }
  hdc.finish(memory);
  if (!mahalanobis.finish()) {
    fprintf(stderr, "covariance is not positive definite\n");
    return 3;
  }

  std::vector<float> out;
  for (size_t i = 0; i + n <= test.size(); i += n) {
    float binary = 0;
    float bipolar = 0;
    hdc.score(memory, &test[i], binary, bipolar);
    out.push_back(mahalanobis.score(&test[i]));
    out.push_back(binary);
    out.push_back(bipolar);
  }
  writeAll(argv[6], out);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 6 && strcmp(argv[1], "features") == 0) return features(argv);
  if (argc == 7 && strcmp(argv[1], "score") == 0) return score(argv);
  fprintf(stderr,
          "usage: ml_cli features <accel|audio> <fs> <in.f32> <out.f32>\n"
          "       ml_cli score <n> <stats.f32> <bundle.f32> <test.f32> <out.f32>\n");
  return 1;
}
