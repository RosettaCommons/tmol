#pragma once
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/io/details/compiled/histidine_kernels.hh>

namespace tmol {
namespace io {
namespace details {
namespace compiled {

using tmol::portable::Vec;

template <
    template <tmol::Device> class DeviceOps,
    tmol::Device Dev,
    typename Real,
    typename Int>
struct ResolveHisTaut {
  static auto f(
      ContextManager& mgr,
      TView<Vec<Real, 3>, 3, Dev> coords,
      TView<Int, 2, Dev> res_types,
      TView<Int, 2, Dev> res_type_variants,
      TView<int64_t, 1, Dev> his_pose_ind,
      TView<int64_t, 1, Dev> his_res_ind,
      TView<bool, 3, Dev> atom_is_present,
      TView<HisAtomIndsInCanonicalOrdering<Int>, 1, Dev> his_atom_inds,
      TView<int64_t, 3, Dev> his_remapping_dst_index

      ) -> TPack<int64_t, 2, Dev> {
    auto his_taut_t =
        TPack<int64_t, 2, Dev>::zeros({coords.size(0), coords.size(1)});
    HisTautContext<Real, Int, Dev> c{
        coords,
        res_types,
        res_type_variants,
        his_pose_ind,
        his_res_ind,
        atom_is_present,
        his_atom_inds,
        his_remapping_dst_index,
        his_taut_t.view};
    launch_resolve_his_taut<DeviceOps, Dev, Real, Int>(mgr, c);
    return his_taut_t;
  };
};

}  // namespace compiled
}  // namespace details
}  // namespace io
}  // namespace tmol
