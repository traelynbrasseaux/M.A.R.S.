// Command line wrapper around lib/ml for the PC, so host tests can check it against the Python.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

#include "features.h"

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

}  // namespace

int main(int argc, char** argv) {
  if (argc == 6 && strcmp(argv[1], "features") == 0) return features(argv);
  fprintf(stderr, "usage: ml_cli features <accel|audio> <fs> <in.f32> <out.f32>\n");
  return 1;
}
