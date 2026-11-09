#pragma once
#include <tmol/utility/tensor/TensorAccessor.h>
#include <array>
#include <cstdint>
#include <stdexcept>

namespace tmol::pack::compiled {
// The caller owns storage; constructing with nullptr only computes its size.
template <Device D>
struct AnnealerWorkspace {
  unsigned char* data;
  int64_t bytes = 0;
  explicit AnnealerWorkspace(void* storage)
      : data(static_cast<unsigned char*>(storage)) {}
  template <class T, size_t N>
  TView<T, N, D> take(std::array<int64_t, N> shape, int alignment = 16) {
    if (bytes > INT64_MAX - (alignment - 1))
      throw std::overflow_error("annealer workspace alignment overflow");
    bytes = (bytes + alignment - 1) / alignment * alignment;
    int64_t stride[N], count = 1;
    for (int i = N - 1; i >= 0; --i) {
      if (shape[i] < 0 || (shape[i] && count > INT64_MAX / shape[i]))
        throw std::overflow_error("annealer workspace shape overflow");
      stride[i] = count;
      count *= shape[i];
    }
    if (count > (INT64_MAX - bytes) / int64_t(sizeof(T)))
      throw std::overflow_error("annealer workspace byte overflow");
    T* ptr = data ? reinterpret_cast<T*>(data + bytes) : nullptr;
    bytes += count * sizeof(T);
    return TView<T, N, D>(ptr, shape.data(), stride);
  }

  void validate(int64_t storage_bytes) const {
    if (storage_bytes >= 0 && storage_bytes < bytes)
      throw std::invalid_argument("annealer workspace is too small");
    if (data && reinterpret_cast<uintptr_t>(data) % 16)
      throw std::invalid_argument("annealer workspace must be 16-byte aligned");
  }
};
}  // namespace tmol::pack::compiled
