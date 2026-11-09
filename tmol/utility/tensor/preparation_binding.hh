#pragma once
// Typed registration adapter for framework-neutral preparation functions.
#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/tensor/context_manager.hh>
#include <tuple>
#include <type_traits>
#include <vector>
#ifdef WITH_CUDA
#include <tmol/utility/tensor/native_cuda.hh>
#endif

namespace tmol {
namespace native {
template <typename T>
struct PreparationArgument {
  using FrameworkType = T;
  static T convert(T value) { return value; }
};
template <typename T, size_t N, tmol::Device D, PtrTag P>
struct PreparationArgument<TView<T, N, D, P>> {
  using FrameworkType = Tensor;
  static TView<T, N, D, P> convert(Tensor const& value) {
    return view_tensor<T, N, D, P>(value, "native preparation argument");
  }
};
template <typename T>
struct PreparationResult;
template <typename T, size_t N, tmol::Device D, PtrTag P>
struct PreparationResult<TPack<T, N, D, P>> {
  using FrameworkType = Tensor;
  static Tensor convert(TPack<T, N, D, P> const& value) { return value.tensor; }
};
template <typename... P>
struct PreparationResult<std::tuple<P...>> {
  using FrameworkType = std::vector<Tensor>;
  static FrameworkType convert(std::tuple<P...> const& value) {
    return std::apply(
        [](auto const&... p) { return FrameworkType{p.tensor...}; }, value);
  }
};

template <auto Function>
struct PreparationCall;
template <typename R, typename... A, R (*Function)(ContextManager&, A...)>
struct PreparationCall<Function> {
  static auto call(
      ContextManager& context,
      typename PreparationArgument<A>::FrameworkType const&... args) {
    return PreparationResult<R>::convert(
        Function(context, PreparationArgument<A>::convert(args)...));
  }
};

template <ContextManager* Context, auto CPU, auto CUDA>
struct PreparationOperator;
template <
    ContextManager* Context,
    typename R,
    typename... A,
    R (*CPU)(ContextManager&, A...),
    auto CUDA>
struct PreparationOperator<Context, CPU, CUDA> {
  static auto call(typename PreparationArgument<A>::FrameworkType... args) ->
      typename PreparationResult<R>::FrameworkType {
    Tensor const* first = nullptr;
    auto inspect = [&](auto const& value) {
      if constexpr (std::is_same_v<std::decay_t<decltype(value)>, Tensor>)
        if (!first) first = &value;
    };
    (inspect(args), ...);
    TORCH_CHECK(first, "Native preparation requires tensor metadata");
    if (first->device().is_cpu())
      return PreparationCall<CPU>::call(*Context, args...);
#ifdef WITH_CUDA
    if (first->device().is_cuda()) {
      CUDAGuard guard(first->device());
      return PreparationCall<CUDA>::call(*Context, args...);
    }
#endif
    TORCH_CHECK(false, "Unsupported native preparation device");
  }
};
}  // namespace native
}  // namespace tmol
