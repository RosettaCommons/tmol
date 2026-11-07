#pragma once

#include <tmol/utility/tensor/native_autograd.hh>

namespace tmol::native {

// The registered signature is the autograd forward signature without its
// context. Keep the operator's apply path, including capture and saved tensors.
template <class Operator, class Signature = decltype(&Operator::forward)>
struct AutogradOperator;

template <class Operator, class Result, class... Args>
struct AutogradOperator<
    Operator,
    Result (*)(autograd::AutogradContext*, Args...)> {
  static Result call(Args... args) { return Operator::apply(args...); }
};

}  // namespace tmol::native
