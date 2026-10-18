// Shared numeric rule, matching feature_std in host/mhm/models/__init__.py.
#pragma once

#include <math.h>

namespace ml {

// A numerically constant feature gets std 1, so it carries no weight instead of dividing by ~0.
inline double featureStd(double mean, double variance) {
  return variance <= 1e-12 * fmax(1.0, mean * mean) ? 1.0 : sqrt(variance);
}

}  // namespace ml
