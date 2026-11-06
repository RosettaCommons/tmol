#include <tmol/utility/portable/reduce_ops.hh>
#include <tmol/utility/portable/math.hh>
#include <tmol/utility/tensor/native.hh>
#pragma once
#include <tmol/score/common/load_balance.hh>

#ifdef __NVCC__
error_this_should_not_be_compiled();  // nvcc should not include this file
#endif

#include <algorithm>
#include <tmol/score/common/serial_cpu_operations.hh>

#include <tmol/score/common/device_operations.hh>
#include <tmol/score/common/workgroup_lane.hh>
#include <tmol/score/common/counting.hh>
#include <tmol/utility/tensor/context_manager.hh>

namespace tmol {
namespace score {
namespace common {

template <>
struct DeviceOperations<tmol::Device::CPU> : SerialCPUOperations {
  template <typename launch_t, typename Func>
  static void forall_independent(ContextManager&, int N, Func f) {
    constexpr int64_t min_parallel_work = 32768;
    if (N < min_parallel_work) {
      for (int i = 0; i < N; ++i) {
        f(i);
      }
      return;
    }

    tmol::native::parallel_for(0, N, 1, [=](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        f(int(i));
      }
    });
  }

  template <typename launch_t, typename Func>
  static void forall_grouped(
      ContextManager& mgr, int n_groups, int items_per_group, Func f) {
    int const n_items = checked_dispatch_product(
        n_groups, items_per_group, "grouped CPU dispatch");
    if (n_groups <= 1) {
      forall<launch_t>(mgr, n_items, f);
      return;
    }

    tmol::native::parallel_for(0, n_groups, 1, [=](int64_t begin, int64_t end) {
      for (int64_t group = begin; group < end; ++group) {
        int const first_item = group * items_per_group;
        for (int item = 0; item < items_per_group; ++item) {
          f(first_item + item);
        }
      }
    });
  }

  static TMOL_HOST_DEVICE_FUNC void store_idempotent(
      int32_t& target, int32_t value) {
    __atomic_store_n(&target, value, __ATOMIC_RELAXED);
  }

  static TMOL_HOST_DEVICE_FUNC void store_idempotent(
      int64_t& target, int64_t value) {
    __atomic_store_n(&target, value, __ATOMIC_RELAXED);
  }

  static TMOL_HOST_DEVICE_FUNC void bitwise_or(int32_t& target, int32_t value) {
    __atomic_fetch_or(&target, value, __ATOMIC_RELAXED);
  }

  template <typename Int, typename Func>
  static void foreach_combination_triple(
      ContextManager&, Int dim1, Int dim2, Int dim3, Func f) {
    for (Int i = 0; i < dim1; ++i) {
      for (Int j = 0; j < dim2; ++j) {
        for (Int k = 0; k < dim3; ++k) {
          f(i, j, k);
        }
      }
    }
  }

  template <typename launch_t, typename Func>
  static void foreach_workgroup(ContextManager&, int n_workgroups, Func f) {
    for (int i = 0; i < n_workgroups; ++i) {
      f(i);
    }
  }

  template <typename launch_t, typename Func>
  static void foreach_independent_workgroup(
      ContextManager&, int n_workgroups, Func f) {
    constexpr int64_t min_parallel_workgroups = 256;
    if (n_workgroups < min_parallel_workgroups) {
      for (int i = 0; i < n_workgroups; ++i) {
        f(i);
      }
      return;
    }

    tmol::native::parallel_for(
        0, n_workgroups, 1, [=](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            f(int(i));
          }
        });
  }

  template <typename launch_t, typename Func>
  static void foreach_grouped_workgroup(
      ContextManager& mgr, int n_groups, int workgroups_per_group, Func f) {
    int const n_workgroups = checked_dispatch_product(
        n_groups, workgroups_per_group, "grouped CPU workgroup dispatch");
    if (n_groups <= 1) {
      foreach_workgroup<launch_t>(mgr, n_workgroups, f);
      return;
    }

    tmol::native::parallel_for(0, n_groups, 1, [=](int64_t begin, int64_t end) {
      for (int64_t group = begin; group < end; ++group) {
        int const first_workgroup = group * workgroups_per_group;
        for (int i = 0; i < workgroups_per_group; ++i) {
          f(first_workgroup + i);
        }
      }
    });
  }

  template <typename launch_t, typename Func>
  static void foreach_pose_workgroup(
      ContextManager& mgr, int n_poses, int workgroups_per_pose, Func f) {
    foreach_grouped_workgroup<launch_t>(mgr, n_poses, workgroups_per_pose, f);
  }

  // Construct load-balanced-search mapping of work items to their generator
  // index; see https://moderngpu.github.io/loadbalance.html
  // Arguments:
  //   - n_work_units_total: the sum of the number of work units
  //
  //   - exc_scan_offsets: the result of running exclusive scan on the
  //     the number of work units that each generator produces
  //.  - n_generators: the number of generators / length of exc_scan_offset
  template <typename launch_t, typename Offset, typename Int = int32_t>
  static TPack<Int, 1, tmol::Device::CPU> load_balancing_search(
      ContextManager&,
      int n_work_units_total,  // The count of the total number of work units
      Offset* exc_scan_offsets,
      int n_generators) {
    auto gen_for_work_item_t =
        TPack<Int, 1, tmol::Device::CPU>::zeros({n_work_units_total});
    auto gen_for_work_item = gen_for_work_item_t.view;

    load_balancing_search_into_cpu(
        n_work_units_total,
        exc_scan_offsets,
        n_generators,
        gen_for_work_item.data());
    return gen_for_work_item_t;
  }

  // Segmented scan expects the indices for the beginning of each segment rather
  // than, e.g., a boolean tensor indicating the start of each segment.
  // The identity value (e.g. 0) must be given because pre-initialization is not
  // always possible. seg_starts_inds must be sorted in ascending order.
  template <
      tmol::portable::scan_type scan_type,
      typename T,
      typename Int,
      typename OP>
  static auto segmented_scan(
      ContextManager& mgr,
      T* src,
      Int* seg_start_inds,
      int n,
      int n_segs,
      OP op,
      T identity) -> TPack<T, 1, tmol::Device::CPU> {
    auto dst_t = TPack<T, 1, Device::CPU>::empty({n});
    auto dst = dst_t.view;
    SerialCPUOperations::segmented_scan_into<scan_type>(
        mgr, src, seg_start_inds, dst.data(), n, n_segs, op, identity);
    return dst_t;
  }

  template <
      int N_T,
      int WIDTH,
      typename T,
      tmol::Device LaneD = tmol::Device::CPU>
  static void copy_contiguous_data(
      T* __restrict__ dst,
      T* __restrict__ src,
      int n,
      WorkgroupLane<LaneD> = {}) {
    for (int i = 0; i < n; ++i) {
      dst[i] = src[i];
    }
  }

  template <
      int N_T,
      int WIDTH,
      typename TD,
      typename TS,
      tmol::Device LaneD = tmol::Device::CPU>
  static void copy_contiguous_data_and_cast(
      TD* __restrict__ dst,
      TS* __restrict__ src,
      int n,
      WorkgroupLane<LaneD> = {}) {
    for (int i = 0; i < n; ++i) {
      dst[i] = static_cast<TD>(src[i]);
    }
  }

  // The trailing lane handle is empty here and so may be omitted; it exists
  // because the Metal backend has no implicit way to learn a thread's index.
  // See workgroup_lane.hh.
  template <int N_T, typename Func, tmol::Device LaneD = tmol::Device::CPU>
  static void for_each_in_workgroup(Func f, WorkgroupLane<LaneD> = {}) {
    for (int i = 0; i < N_T; ++i) {
      f(i);
    }
  }

  // No op on 1-core CPU
  template <
      int N_T,
      typename T,
      typename S,
      typename OP,
      tmol::Device LaneD = tmol::Device::CPU>
  static T reduce_in_workgroup(T val, S, OP, WorkgroupLane<LaneD> = {}) {
    return val;
  }

  template <
      int N_T,
      typename T,
      typename OP,
      tmol::Device LaneD = tmol::Device::CPU>
  static T shuffle_reduce_in_workgroup(T val, OP, WorkgroupLane<LaneD> = {}) {
    return val;
  }

  // See comments for shuffle_reduce_in_workgroup above
  template <
      int N_T,
      typename T,
      typename OP,
      tmol::Device LaneD = tmol::Device::CPU>
  static T shuffle_reduce_and_broadcast_in_workgroup(
      T val, OP op, WorkgroupLane<LaneD> = {}) {
    return val;
  }

  // No op on 1-core CPU
  template <tmol::Device LaneD = tmol::Device::CPU>
  static void synchronize_workgroup(WorkgroupLane<LaneD> = {}) {}

  // No op on 1-core CPU
  static void synchronize_device() {}
};

}  // namespace common
}  // namespace score
}  // namespace tmol
