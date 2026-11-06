// Native candidate expansion; callbacks and differentiable parameters stay in
// the runtime.
#pragma once
#include "graph_topology_ffi.hh"
#include <tmol/score/constraint/potentials/constraint_topology.hh>
namespace tmol::jax {
struct ConstraintTopologyStage {
  static int fill(int64_t, size_t) { return 0; }
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t,
      int64_t) {
    if ((op != 0 && op != 1) || a.size() != (op ? 9 : 7)
        || out.size() != (op ? 3 : 1))
      throw std::invalid_argument("invalid constraint candidate operands");
    auto i = ffi::DataType::S32;
    require_shape(a[0], i, {-1, 4, 3});
    auto n = a[0].dimensions()[0];
    require_shape(a[1], i, {n});
    require_shape(a[2], i, {n, 3});
    require_shape(a[3], i, {-1});
    require_shape(a[4], i, {-1, -1});
    require_shape(a[5], i, {a[4].dimensions()[0], a[4].dimensions()[1]});
    require_shape(a.back(), i, {a[4].dimensions()[0], a[4].dimensions()[1]});
    auto rows = score::common::checked_dispatch_product(
        n, 2, "constraint candidate rows");
    if (op) {
      graph_wide(a[6], rows);
      require_shape(a[7], i, {-1});
      auto total = a[7].dimensions()[0];
      require_shape(out[0], i, {total});
      require_shape(out[1], i, {total, 4});
      require_shape(out[2], i, {total, 3});
    } else
      graph_wide(out[0], rows);
  }
  template <template <Device> class Ops, Device D>
  static void execute(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t,
      int64_t) {
    using namespace score::constraint::potentials;
    using I = int32_t;
    using I3 = Vec<I, 3>;
    using I4 = Vec<I, 4>;
#define IN(T, R, K) buffer_view<T, R, D>(a[K])
#define OUT(T, R, K) buffer_view<T, R, D>(out[K])
    ConstraintCandidateTopology<I, D> ctx{
        IN(I3, 2, 0),
        IN(I, 1, 1),
        IN(I, 1, 3),
        IN(I3, 1, 2),
        IN(I, 2, 4),
        IN(I, 2, 5),
        buffer_view<I, 2, D>(a.back())};
    ContextManager manager;
    if (op)
      constraint_candidate_fill_native<Ops, D>(
          manager,
          ctx,
          true,
          IN(int64_t, 1, 6),
          IN(I, 1, 7),
          OUT(I, 1, 0),
          OUT(I4, 1, 1),
          OUT(I3, 1, 2));
    else
      constraint_candidate_count_native<Ops, D>(
          manager, ctx, OUT(int64_t, 1, 0));
#undef IN
#undef OUT
  }
};
}  // namespace tmol::jax
