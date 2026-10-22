// EdgeForge additions to the BSD-2-Clause Sophgo runtime. No device APIs here.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

inline float edgeforge_fp16(uint16_t bits) {
  const int sign = (bits & 0x8000) ? -1 : 1;
  const int exp = (bits >> 10) & 31;
  const int frac = bits & 1023;
  if (exp == 31) return frac ? NAN : sign * INFINITY;
  if (exp == 0) return sign * std::ldexp(static_cast<float>(frac), -24);
  return sign * std::ldexp(static_cast<float>(1024 + frac), exp - 25);
}
inline float edgeforge_bf16(uint16_t bits) {
  uint32_t expanded = static_cast<uint32_t>(bits) << 16;
  float result;
  std::memcpy(&result, &expanded, sizeof(result));
  return result;
}

inline int edgeforge_sample(std::vector<float> logits, const std::vector<int>& visited,
                           int token_length, bool do_sample, float temperature,
                           float top_p, int top_k, float repetition_penalty,
                           std::mt19937& rng) {
  if (logits.empty() || temperature <= 0 || top_p <= 0 || top_p > 1 ||
      top_k < 0 || repetition_penalty <= 0 || !std::isfinite(temperature) ||
      !std::isfinite(top_p) || !std::isfinite(repetition_penalty))
    throw std::invalid_argument("Invalid sampling parameters");
  for (float value : logits)
    if (!std::isfinite(value)) throw std::runtime_error("Nonfinite lm_head logits");
  if (repetition_penalty != 1.0f) {
    std::vector<bool> seen(logits.size(), false);
    for (int i = 0; i < token_length; ++i) {
      int token = visited.at(i);
      if (token >= 0 && token < static_cast<int>(logits.size()) && !seen[token]) {
        logits[token] = logits[token] < 0 ? logits[token] * repetition_penalty
                                        : logits[token] / repetition_penalty;
        seen[token] = true;
      }
    }
  }
  if (!do_sample)
    return std::max_element(logits.begin(), logits.end()) - logits.begin();
  std::vector<int> order(logits.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return logits[a] > logits[b]; });
  if (top_k > 0 && top_k < static_cast<int>(order.size())) {
    const float cutoff = logits[order[top_k - 1]];
    // Like HF TopKLogitsWarper, keep ties at the kth logit's threshold.
    auto last = std::find_if(order.begin() + top_k, order.end(),
                            [&](int token) { return logits[token] < cutoff; });
    order.erase(last, order.end());
  }
  const double maximum = logits[order[0]] / static_cast<double>(temperature);
  std::vector<double> probabilities;
  double sum = 0;
  for (int token : order) {
    double p = std::exp(logits[token] / static_cast<double>(temperature) - maximum);
    probabilities.push_back(p);
    sum += p;
  }
  if (top_p < 1) {
    double cumulative = 0;
    size_t keep = 0;
    do { cumulative += probabilities[keep++] / sum; }
    while (keep < probabilities.size() && cumulative < top_p);
    probabilities.resize(keep);
    order.resize(keep);
  }
  std::discrete_distribution<size_t> distribution(probabilities.begin(), probabilities.end());
  return order[distribution(rng)];
}
