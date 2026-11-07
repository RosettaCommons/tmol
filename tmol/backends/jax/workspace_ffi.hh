// XLA-owned native workspace, independent of JAX's floating precision setting.
#pragma once
#include "genbonded_ffi.hh"
#include <optional>
#include <type_traits>

namespace tmol::jax {
template <typename Kernel, typename = void>
struct ZeroEnergy : std::true_type {};
template <typename Kernel>
struct ZeroEnergy<Kernel, std::void_t<decltype(Kernel::zero_energy)>>
    : std::bool_constant<Kernel::zero_energy> {};

template <typename Kernel, typename = void>
struct ZeroWorkspace : std::true_type {};
template <typename Kernel>
struct ZeroWorkspace<Kernel, std::void_t<decltype(Kernel::zero_workspace)>>
    : std::bool_constant<Kernel::zero_workspace> {};

template <typename Kernel, typename = void>
struct HasWorkspace : std::false_type {};
template <typename Kernel>
struct HasWorkspace<
    Kernel,
    std::void_t<decltype(Kernel::workspace_bytes(
        std::declval<const std::vector<ffi::AnyBuffer>&>(), false))>>
    : std::true_type {};

struct NativeOutputs {
  std::optional<ffi::AnyBuffer> gradient;
  std::optional<ffi::AnyBuffer> workspace;

  const ffi::AnyBuffer* derivative() const {
    return gradient ? &*gradient : nullptr;
  }
};

template <typename Kernel>
NativeOutputs native_outputs(
    const std::vector<ffi::AnyBuffer>& inputs,
    ffi::AnyBuffer energy,
    ffi::RemainingRets rets) {
  constexpr size_t extra = HasWorkspace<Kernel>::value ? 1 : 0;
  if (rets.size() < extra || rets.size() > extra + 1)
    throw std::invalid_argument("incorrect native FFI result count");
  auto buffer = [&](size_t index) {
    auto value = rets.get<ffi::AnyBuffer>(index);
    if (!value)
      throw std::invalid_argument("native FFI expected a result buffer");
    return **value;
  };
  NativeOutputs outputs;
  if (rets.size() > extra) outputs.gradient = buffer(0);
  Kernel::validate(inputs, energy, outputs.derivative());
  if constexpr (HasWorkspace<Kernel>::value) {
    outputs.workspace = buffer(rets.size() - 1);
    const auto& work = *outputs.workspace;
    auto bytes = Kernel::workspace_bytes(inputs, outputs.gradient.has_value());
    if (work.element_type() != ffi::DataType::U8
        || work.dimensions().size() != 1 || work.size_bytes() != bytes
        || reinterpret_cast<uintptr_t>(work.untyped_data()) % alignof(double))
      throw std::invalid_argument(
          "native workspace size, dtype or alignment mismatch");
  }
  return outputs;
}

class ByteWorkspace {
 public:
  explicit ByteWorkspace(void* data = nullptr)
      : data_(static_cast<unsigned char*>(data)) {}

  template <typename T>
  T* take(size_t count) {
    constexpr size_t maximum = std::numeric_limits<size_t>::max();
    if (offset_ > maximum - (alignof(T) - 1))
      throw std::length_error("native workspace alignment overflow");
    offset_ = (offset_ + alignof(T) - 1) & ~(alignof(T) - 1);
    if (count > (maximum - offset_) / sizeof(T))
      throw std::length_error("native workspace extent overflow");
    auto* result = data_ ? reinterpret_cast<T*>(data_ + offset_) : nullptr;
    offset_ += count * sizeof(T);
    return result;
  }

  size_t size_bytes() const { return offset_; }

 private:
  unsigned char* data_;
  size_t offset_ = 0;
};

template <
    typename Kernel,
    template <Device> class Ops,
    Device D,
    typename Real,
    typename... Attributes>
void execute_native(
    const std::vector<ffi::AnyBuffer>& inputs,
    ffi::AnyBuffer energy,
    const NativeOutputs& outputs,
    Attributes... attributes) {
  if constexpr (HasWorkspace<Kernel>::value)
    Kernel::template execute<Ops, D, Real>(
        inputs,
        energy,
        outputs.derivative(),
        *outputs.workspace,
        attributes...);
  else
    Kernel::template execute<Ops, D, Real>(
        inputs, energy, outputs.derivative(), attributes...);
}
}  // namespace tmol::jax
