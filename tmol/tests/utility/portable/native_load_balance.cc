#include <tmol/score/common/load_balance.hh>
#include <cassert>
#include <cstdint>
template <typename Offset, typename Int>
void check() {
  // Empty leading, interior and trailing generators do not own work items.
  Offset starts[] = {0, 0, 2, 2, 5, 6, 6};
  Int expected[] = {1, 1, 3, 3, 3, 4}, result[6] = {};
  tmol::score::common::load_balancing_search_into_cpu(6, starts, 7, result);
  for (int i = 0; i < 6; ++i) assert(result[i] == expected[i]);
  tmol::score::common::load_balancing_search_into_cpu(0, starts, 0, result);
  Offset empty[] = {0, 0, 0};
  tmol::score::common::load_balancing_search_into_cpu(0, empty, 3, result);
  for (int i = 0; i < 6; ++i) assert(result[i] == expected[i]);
}
int main() {
  check<int32_t, int32_t>();
  check<int64_t, int32_t>();
  check<int32_t, int64_t>();
  check<int64_t, int64_t>();
}
