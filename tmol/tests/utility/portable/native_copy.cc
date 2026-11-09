// Both stable and local-runtime Torch bindings must accept gradient slices.
#include <tmol/utility/tensor/native.hh>

void copy_gradient(tmol::native::Tensor source) {
  auto result = tmol::native::realigned_copy(source);
  (void)result;
}
