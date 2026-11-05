// Host-only typed array transport for native topology preparation.
#pragma once
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/portable/matrix.hh>
#include <tmol/score/unresolved_atom_types.hh>
#include <Eigen/Core>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>

namespace tmol::jax {
struct HostArray {
  const void* data = nullptr;
  int64_t shape[5] = {};
  int32_t rank = 0;
  int32_t scalar_type = 0;  // 1: int32, 2: bool8, 3: float32, 4: float64
};

template <typename T>
struct HostElement;
template <>
struct HostElement<int32_t> {
  static constexpr int32_t scalar_type = 1, lanes = 1;
};
template <>
struct HostElement<bool> {
  static_assert(sizeof(bool) == 1);
  static constexpr int32_t scalar_type = 2, lanes = 1;
};
template <>
struct HostElement<float> {
  static constexpr int32_t scalar_type = 3, lanes = 1;
};
template <>
struct HostElement<double> {
  static constexpr int32_t scalar_type = 4, lanes = 1;
};
template <int N>
struct HostElement<portable::Vec<int32_t, N>> {
  static_assert(sizeof(portable::Vec<int32_t, N>) == N * sizeof(int32_t));
  static constexpr int32_t scalar_type = 1, lanes = N;
};
template <typename T, int N, int Options, int MaxRows>
struct HostElement<Eigen::Matrix<T, N, 1, Options, MaxRows, 1>> {
  static_assert(
      sizeof(Eigen::Matrix<T, N, 1, Options, MaxRows, 1>) == N * sizeof(T));
  static constexpr int32_t scalar_type = HostElement<T>::scalar_type, lanes = N;
};
template <>
struct HostElement<UnresolvedAtomID<int32_t>> {
  static_assert(sizeof(UnresolvedAtomID<int32_t>) == 3 * sizeof(int32_t));
  static constexpr int32_t scalar_type = 1, lanes = 3;
};

inline size_t host_array_size(const HostArray& array) {
  if (array.rank < 1 || array.rank > 5 || array.scalar_type < 1
      || array.scalar_type > 4)
    throw std::invalid_argument("invalid native host array rank or dtype");
  size_t count = 1;
  for (int i = 0; i < array.rank; ++i) {
    if (array.shape[i] < 0 || array.shape[i] > INT32_MAX
        || (array.shape[i] && count > INT32_MAX / size_t(array.shape[i])))
      throw std::invalid_argument(
          "native host metadata exceeds int32 indexing");
    count *= array.shape[i];
  }
  if (count && !array.data)
    throw std::invalid_argument("native host array has no data");
  return count;
}

struct HostArguments {
  const HostArray* arrays;
  int64_t array_count;
  const int64_t* scalars;
  int64_t scalar_count;
  int64_t next_array = 0, next_scalar = 0;

  HostArguments(const HostArray* a, int64_t na, const int64_t* s, int64_t ns)
      : arrays(a), array_count(na), scalars(s), scalar_count(ns) {
    if (na < 0 || ns < 0 || (na && !a) || (ns && !s))
      throw std::invalid_argument("invalid native host argument list");
  }
  const HostArray& array() {
    if (next_array == array_count)
      throw std::invalid_argument("missing native host array argument");
    const auto& value = arrays[next_array++];
    host_array_size(value);
    return value;
  }
  int64_t scalar() {
    if (next_scalar == scalar_count)
      throw std::invalid_argument("missing native host scalar argument");
    return scalars[next_scalar++];
  }
  void finish() const {
    if (next_array != array_count || next_scalar != scalar_count)
      throw std::invalid_argument("extra native host arguments");
  }
};

template <typename T>
struct HostDecode;
template <>
struct HostDecode<int64_t> {
  static int64_t get(HostArguments& args) { return args.scalar(); }
};
template <>
struct HostDecode<bool> {
  static bool get(HostArguments& args) {
    auto value = args.scalar();
    if (value != 0 && value != 1)
      throw std::invalid_argument("native host boolean must be zero or one");
    return value;
  }
};
template <typename T, size_t N, Device D, PtrTag P>
struct HostDecode<TView<T, N, D, P>> {
  static_assert(D == Device::CPU);
  static TView<T, N, D, P> get(HostArguments& args) {
    using Element = HostElement<T>;
    const auto& array = args.array();
    constexpr int rank = N + (Element::lanes > 1);
    if (array.rank != rank || array.scalar_type != Element::scalar_type
        || (Element::lanes > 1 && array.shape[N] != Element::lanes)
        || reinterpret_cast<uintptr_t>(array.data) % alignof(T))
      throw std::invalid_argument(
          "native host array does not match its typed operand");
    int64_t strides[N], stride = 1;
    for (size_t i = N; i-- > 0;) {
      strides[i] = stride;
      stride *= array.shape[i];
    }
    return {
        reinterpret_cast<T*>(const_cast<void*>(array.data)),
        array.shape,
        strides};
  }
};

struct OwnedHostArray {
  std::shared_ptr<void> owner;
  HostArray array;

  template <typename T, size_t N, Device D, PtrTag P>
  void describe(TView<T, N, D, P> view) {
    static_assert(D == Device::CPU);
    using Element = HostElement<T>;
    array.rank = N + (Element::lanes > 1);
    static_assert(N + (Element::lanes > 1) <= 5);
    array.scalar_type = Element::scalar_type;
    for (size_t i = 0; i < N; ++i) array.shape[i] = view.size(i);
    if constexpr (Element::lanes > 1) array.shape[N] = Element::lanes;
    array.data = view.data();
    if (host_array_size(array)) {
      int64_t stride = 1;
      for (size_t i = N; i-- > 0;) {
        if (view.size(i) > 1 && view.stride(i) != stride)
          throw std::invalid_argument("native host output is not contiguous");
        stride *= view.size(i);
      }
    }
  }
  template <typename Pack>
  explicit OwnedHostArray(const Pack& pack) {
    // The core Pack has shared ownership. Retain it until NumPy copies the
    // result, avoiding both a callback allocator and an intermediate copy.
    auto retained = std::make_shared<Pack>(pack);
    describe(retained->view);
    owner = std::move(retained);
  }
};

struct HostResult {
  std::vector<OwnedHostArray> arrays;
  template <typename Pack>
  void capture(const Pack& pack) {
    arrays.emplace_back(pack);
  }
  template <typename... Packs>
  void capture(const std::tuple<Packs...>& packs) {
    std::apply([&](const auto&... pack) { (capture(pack), ...); }, packs);
  }
};

// Braced initialization sequences argument decoding in signature order. The
// signatures and all topology operations remain defined by the shared core.
template <typename Result, typename Context, typename... Args>
void invoke_host(
    Result (*operation)(Context&, Args...),
    Context& context,
    HostArguments& inputs,
    HostResult& outputs) {
  std::tuple<Args...> arguments{HostDecode<Args>::get(inputs)...};
  inputs.finish();
  outputs.capture(
      std::apply(
          [&](auto... values) { return operation(context, values...); },
          arguments));
}
}  // namespace tmol::jax
