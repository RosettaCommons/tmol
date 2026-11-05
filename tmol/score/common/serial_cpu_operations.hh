#pragma once
// Serial traversal primitives shared by host preparation and framework CPU
// dispatch. Exclusive scans preserve the caller-initialized identity at dst[0].
#include <algorithm>
#include <cassert>
#include <tmol/utility/portable/reduce_ops.hh>

namespace tmol {
namespace score {
namespace common {
struct SerialCPUOperations {
  template <typename launch_t, typename Func, typename Context>
  static void forall(Context&, int N, Func f) {
    for (int i = 0; i < N; ++i) {
      f(i);
    }
  }

  template <
      tmol::portable::scan_type scan_type,
      typename T,
      typename OP,
      typename Context>
  static void scan(Context&, T* src, T* dst, int n, OP op) {
    if (n <= 0) {
      return;
    }
    T last_val = src[0];
    if (scan_type == tmol::portable::scan_type::inclusive) {
      dst[0] = last_val;
    }
    for (int i = 1; i < n; ++i) {
      T i_val = src[i];
      T next_val = op(last_val, i_val);
      dst[i] = (scan_type == tmol::portable::scan_type::exclusive) ? last_val
                                                                   : next_val;
      last_val = next_val;
    }
  }

  template <
      tmol::portable::scan_type scan_type,
      typename T,
      typename OP,
      typename Context>
  static T scan_and_return_total(Context&, T* src, T* dst, int n, OP op) {
    if (n == 0) {
      return T(0);
    }
    T last_val = src[0];
    if (scan_type == tmol::portable::scan_type::inclusive) {
      dst[0] = last_val;
    }
    for (int i = 1; i < n; ++i) {
      T i_val = src[i];
      T next_val = op(last_val, i_val);
      dst[i] = (scan_type == tmol::portable::scan_type::exclusive) ? last_val
                                                                   : next_val;
      last_val = next_val;
    }
    return last_val;
  }

  template <typename T, typename Int, typename Comp, typename Context>
  static void segmented_sort(
      Context&, T* keys, int n, Int* seg_start_inds, int n_segs, Comp comp) {
    if (n <= 1) {
      return;
    }
    for (int i = 0; i < n_segs; ++i) {
      int const start = seg_start_inds[i];
      int const end = i + 1 < n_segs ? seg_start_inds[i + 1] : n;
      std::sort(keys + start, keys + end, comp);
    }
  }
  template <typename T, typename OP, typename Context>
  static T reduce(Context&, T* src, int n, OP op) {
    assert(n > 0);
    T val = src[0];
    for (int i = 1; i < n; ++i) {
      val = op(val, src[i]);
    }
    return val;
  }

  template <
      tmol::portable::scan_type scan_type,
      typename T,
      typename Int,
      typename OP,
      typename Context>
  static void segmented_scan_into(
      Context&,
      T* src,
      Int* seg_start_inds,
      T* dst,
      int n,
      int n_segs,
      OP op,
      T identity) {
    T last_val = identity;  // position 0 is always the start of a segment
    int count_seg = 0;
    for (int i = 0; i < n; ++i) {
      T i_val = src[i];
      if (count_seg < n_segs && i == seg_start_inds[count_seg]) {
        last_val = identity;
        count_seg++;
      }
      T next_val = op(last_val, i_val);
      dst[i] = (scan_type == tmol::portable::scan_type::exclusive) ? last_val
                                                                   : next_val;
      last_val = next_val;
    }
  }
};
}  // namespace common
}  // namespace score
}  // namespace tmol
