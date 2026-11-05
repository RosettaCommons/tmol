#include <tmol/pack/rotamer/dunbrack/host_sampling.hh>
#include <cassert>
#include <array>
using namespace tmol;
using namespace tmol::pack::rotamer::dunbrack;
template <class T>
auto view(T* x, int64_t n) {
  int64_t stride = 1;
  return TView<T, 1, Device::CPU>(x, &n, &stride);
}
int main() {
  ContextManager ctx;
  using Sampler = HostDunbrackSampling<double>;
  int offsets[] = {0, 0, 2, 2, 4}, counts[] = {0, 2, 0, 2, 0}, brt[4],
      boundary[] = {99, 99, 99, 99};
  Sampler::map_from_rotamer_index_to_brt_into(
      ctx, view(offsets, 5), view(brt, 4), view(boundary, 4));
  assert(
      (std::array<int, 4>{brt[0], brt[1], brt[2], brt[3]}
       == std::array<int, 4>{1, 1, 3, 3}));
  double probability[] = {.6, .4, .2, .8}, limits[] = {.5, .5, .5, .5, .5},
         prefix[4];
  int keep[4], count_prefix[4], out[] = {99, 99, 99, 99, 99};
  Sampler::determine_n_base_rotamers_to_build_into(
      ctx,
      view(limits, 5),
      view(counts, 5),
      view(brt, 4),
      view(offsets, 5),
      view(probability, 4),
      view(out, 5),
      view(prefix, 4),
      view(keep, 4),
      view(count_prefix, 4),
      {},
      {});
  assert(
      (std::array<int, 5>{out[0], out[1], out[2], out[3], out[4]}
       == std::array<int, 5>{0, 1, 0, 2, 0}));
  // A completely empty table group also has no map or count reads.
  int empty_offsets[] = {0, 0}, empty_counts[] = {0, 0}, empty_out[] = {99, 99};
  double empty_limits[] = {.5, .5};
  Sampler::map_from_rotamer_index_to_brt_into(
      ctx,
      view(empty_offsets, 2),
      view<int>(nullptr, 0),
      view<int>(nullptr, 0));
  Sampler::determine_n_base_rotamers_to_build_into(
      ctx,
      view(empty_limits, 2),
      view(empty_counts, 2),
      view<int>(nullptr, 0),
      view(empty_offsets, 2),
      view<double>(nullptr, 0),
      view(empty_out, 2),
      view<double>(nullptr, 0),
      view<int>(nullptr, 0),
      view<int>(nullptr, 0),
      {},
      {});
  assert(empty_out[0] == 0 && empty_out[1] == 0);
  // Libraryless rows contribute one base state; empty libraries contribute
  // zero. Planning must retain both rather than silently dropping a type.
  int types[] = {0, -1, 0, 0, 0, 1};
  int64_t shape[] = {3, 2}, strides[] = {2, 1}, libraries[] = {0, 3};
  int chis[] = {4, 0, 2};
  auto layout = prepare_dunbrack_sampling_layout(
      TView<int, 2, Device::CPU>(types, shape, strides),
      view(libraries, 2),
      view(chis, 3));
  assert(layout.n_possible_rotamers == 4 && layout.max_n_chi == 4);
  assert(
      layout.possible_counts.view[0] == 1 && layout.possible_counts.view[1] == 0
      && layout.possible_counts.view[2] == 3);
  assert(
      layout.possible_offsets.view[0] == 0
      && layout.possible_offsets.view[1] == 1
      && layout.possible_offsets.view[2] == 1);
  int poisoned_offsets[] = {10007, 10007, 10007};
  int canonical_counts[] = {1, 0, 3};
  assert(
      Sampler::checked_count_offsets(
          ctx, view(canonical_counts, 3), view(poisoned_offsets, 3))
      == 4);
  assert(
      poisoned_offsets[0] == 0 && poisoned_offsets[1] == 1
      && poisoned_offsets[2] == 1);
  bool over_budget = false;
  try {
    Sampler::checked_sample_offsets(
        ctx, view(canonical_counts, 3), view(poisoned_offsets, 3), 2);
  } catch (std::runtime_error const&) {
    over_budget = true;
  }
  assert(over_budget);
  assert(
      Sampler::checked_sample_offsets(
          ctx, view(canonical_counts, 3), view(poisoned_offsets, 3), 3)
      == 4);
}
