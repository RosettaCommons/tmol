#include <tmol/utility/portable/scan_workspace.hh>
#include <cassert>
#include <cstdint>
#include <limits>

int main() {
  using tmol::portable::scan_workspace_sizes;
  // Each boundary describes one full tile. The final work item must fit,
  // including the first integer that float32 cannot represent exactly.
  for (int threads : {128, 256}) {
    for (int values : {2, 3}) {
      for (int64_t count :
           {int64_t(0),
            int64_t(1),
            int64_t(256),
            int64_t(16777217),
            int64_t(INT32_MAX)}) {
        auto sizes = scan_workspace_sizes<int32_t>(count, threads, values);
        int64_t tiles = sizes[2] - 1;
        assert(sizes[0] == count);
        assert(tiles * threads * values >= count);
        assert(count == 0 || (tiles - 1) * threads * values < count);
        assert(sizes[1] >= tiles);
      }
    }
  }
  bool rejected = false;
  try {
    scan_workspace_sizes<int32_t>(int64_t(INT32_MAX) + 1, 128, 2);
  } catch (std::overflow_error const&) {
    rejected = true;
  }
  assert(rejected);
  rejected = false;
  try {
    scan_workspace_sizes<int64_t>(1, INT64_MAX, 2);
  } catch (std::overflow_error const&) {
    rejected = true;
  }
  assert(rejected);
}
