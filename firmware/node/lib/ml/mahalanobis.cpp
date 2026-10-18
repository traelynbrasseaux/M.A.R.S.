#include "mahalanobis.h"

#include <math.h>
#include <string.h>

#include "stats.h"

namespace ml {

namespace {

// Index of (i, j) with i <= j in an n x n upper triangle stored row by row.
size_t tri(size_t n, size_t i, size_t j) { return i * (2 * n - i + 1) / 2 + (j - i); }

}  // namespace

void MahalanobisModel::begin(size_t nFeatures) {
  n_ = nFeatures;
  count_ = 0;
  memset(sum_, 0, sizeof sum_);
  memset(cross_, 0, sizeof cross_);
}

void MahalanobisModel::observe(const float* x) {
  for (size_t i = 0; i < n_; i++) {
    sum_[i] += x[i];
    for (size_t j = i; j < n_; j++) cross_[tri(n_, i, j)] += static_cast<double>(x[i]) * x[j];
  }
  count_++;
}

bool MahalanobisModel::finish() {
  const double count = count_;
  double std[kMaxFeatures];
  for (size_t i = 0; i < n_; i++) {
    mean_[i] = sum_[i] / count;
    std[i] = featureStd(mean_[i], cross_[tri(n_, i, i)] / count - mean_[i] * mean_[i]);
    invStd_[i] = 1.0 / std[i];
  }

  // Covariance of the z-scored features plus the ridge; the lower triangle becomes its Cholesky L.
  double* a = new double[n_ * n_];
  for (size_t i = 0; i < n_; i++) {
    for (size_t j = i; j < n_; j++) {
      const double centered = cross_[tri(n_, i, j)] - count * mean_[i] * mean_[j];
      const double c = centered / ((count - 1) * std[i] * std[j]) + (i == j ? kRidge : 0.0);
      a[i * n_ + j] = a[j * n_ + i] = c;
    }
  }
  for (size_t j = 0; j < n_; j++) {
    double d = a[j * n_ + j];
    for (size_t k = 0; k < j; k++) d -= a[j * n_ + k] * a[j * n_ + k];
    if (d <= 0) {
      delete[] a;
      return false;
    }
    const double l = sqrt(d);
    a[j * n_ + j] = l;
    for (size_t i = j + 1; i < n_; i++) {
      double s = a[i * n_ + j];
      for (size_t k = 0; k < j; k++) s -= a[i * n_ + k] * a[j * n_ + k];
      a[i * n_ + j] = s / l;
    }
  }

  // Invert L in place, row by row; each entry only needs entries of L not yet overwritten.
  for (size_t i = 0; i < n_; i++) {
    const double inv = 1.0 / a[i * n_ + i];
    for (size_t j = 0; j < i; j++) {
      double s = 0;
      for (size_t k = j; k < i; k++) s += a[i * n_ + k] * a[k * n_ + j];
      a[i * n_ + j] = -s * inv;
    }
    a[i * n_ + i] = inv;
  }

  // precision = L^-T L^-1
  for (size_t i = 0; i < n_; i++) {
    for (size_t j = i; j < n_; j++) {
      double s = 0;
      for (size_t k = j; k < n_; k++) s += a[k * n_ + i] * a[k * n_ + j];
      precision_[i * n_ + j] = precision_[j * n_ + i] = static_cast<float>(s);
    }
  }
  delete[] a;
  return true;
}

float MahalanobisModel::score(const float* x) const {
  float z[kMaxFeatures];
  for (size_t i = 0; i < n_; i++) z[i] = static_cast<float>((x[i] - mean_[i]) * invStd_[i]);
  float d = 0;
  for (size_t i = 0; i < n_; i++) {
    float row = 0;
    for (size_t j = 0; j < n_; j++) row += precision_[i * n_ + j] * z[j];
    d += z[i] * row;
  }
  return d;
}

}  // namespace ml
