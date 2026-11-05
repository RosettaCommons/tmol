#pragma once
#include "native_sampling.hh"
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/context_manager.hh>

namespace tmol {
namespace pack {
namespace rotamer {
namespace dunbrack {
template <
    template <tmol::Device> class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int>
using DunbrackChiSampler =
    NativeDunbrackChiSampler<Dispatch, D, Real, Int, TPack, ContextManager>;
}  // namespace dunbrack
}  // namespace rotamer
}  // namespace pack
}  // namespace tmol
