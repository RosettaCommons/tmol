#pragma once

#include <tmol/utility/tensor/TensorAccessor.h>
#include <array>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>

namespace tmol {

// Owning buffers for topology preparation. No framework allocator or runtime
// is needed. Copies share storage, so views remain valid until the last owner
// is released. Runtime execution continues to use caller-owned device buffers.
template <typename T, size_t N, Device D>
struct OwnedHostTensor {
  static_assert(D == Device::CPU, "Host preparation requires CPU storage");
  static_assert(N > 0, "A native tensor must have at least one dimension");
  std::shared_ptr<T[]> storage;
  TView<T, N, D> view;

  static OwnedHostTensor from_cpu(OwnedHostTensor source) { return source; }

  static OwnedHostTensor full(std::initializer_list<int64_t> shape, T value) {
    if (shape.size() != N)
      throw std::invalid_argument("Host tensor rank mismatch");
    std::array<int64_t, N> sizes, strides;
    std::copy(shape.begin(), shape.end(), sizes.begin());
    int64_t extent = 1;
    for (size_t axis = N; axis-- > 0;) {
      if (sizes[axis] < 0)
        throw std::invalid_argument("Negative host tensor extent");
      strides[axis] = extent;
      // Empty axes retain the native contiguous stride convention.
      int64_t const width = std::max<int64_t>(sizes[axis], 1);
      if (extent > std::numeric_limits<int64_t>::max() / width)
        throw std::overflow_error("Host tensor size overflow");
      extent *= width;
    }
    int64_t count = extent;
    for (auto size : sizes)
      if (size == 0) count = 0;
    if (uint64_t(count) > std::numeric_limits<size_t>::max() / sizeof(T))
      throw std::overflow_error("Host tensor allocation overflow");
    auto storage = std::shared_ptr<T[]>(new T[count]);
    std::fill_n(storage.get(), count, value);
    return {
        storage, TView<T, N, D>(storage.get(), sizes.data(), strides.data())};
  }

  static OwnedHostTensor full(int64_t size, T value) {
    static_assert(N == 1, "A scalar extent requires rank one");
    return full({size}, value);
  }
  static OwnedHostTensor zeros(std::initializer_list<int64_t> shape) {
    return full(shape, T(0));
  }
  static OwnedHostTensor ones(std::initializer_list<int64_t> shape) {
    return full(shape, T(1));
  }
};

}  // namespace tmol
