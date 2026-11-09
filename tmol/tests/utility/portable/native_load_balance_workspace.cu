#include <tmol/score/common/load_balance.hh>
#include <cassert>
#include <climits>
using namespace tmol::score::common;
struct NoAllocator : mgpu::context_t {
  cudaDeviceProp property{};
  const cudaDeviceProp& props() const override { return property; }
  int ptx_version() const override { return 80; }
  cudaStream_t stream() override { return nullptr; }
  void* alloc(size_t, mgpu::memory_space_t) override {
    throw std::logic_error("parent allocation");
  }
  void free(void*, mgpu::memory_space_t) override {
    throw std::logic_error("parent free");
  }
  void synchronize() override { throw std::logic_error("parent sync"); }
  cudaEvent_t event() override { throw std::logic_error("parent event"); }
  void timer_begin() override { throw std::logic_error("parent timer"); }
  double timer_end() override { throw std::logic_error("parent timer"); }
};
template <class F>
void rejects(F f) {
  bool rejected = false;
  try {
    f();
  } catch (std::exception const&) {
    rejected = true;
  }
  assert(rejected);
}
template <class Offset, class Int>
void check() {
  assert(load_balancing_search_workspace_size<Offset>(0, 5, 80) == 0);
  assert(load_balancing_search_workspace_size<Offset>(1, 1, 80) == 2);
  assert(load_balancing_search_workspace_size<Offset>(880, 16, 80) == 2);
  assert(load_balancing_search_workspace_size<Offset>(881, 16, 80) == 3);
  assert(load_balancing_search_workspace_size<Offset>(880, 16, 35) == 2);
  assert(load_balancing_search_workspace_size<Offset>(881, 16, 35) == 3);
  assert(load_balancing_search_workspace_size<Offset>(1392, 16, 20) == 2);
  assert(load_balancing_search_workspace_size<Offset>(1392, 16, 52) == 2);
  rejects([] { load_balancing_search_workspace_size<Offset>(-1, 1, 80); });
  rejects([] { load_balancing_search_workspace_size<Offset>(1, 0, 80); });
  rejects([] { load_balancing_search_workspace_size<Offset>(1, -1, 80); });
  rejects([] { load_balancing_search_workspace_size<Offset>(INT_MAX, 1, 80); });
  rejects([] { load_balancing_search_workspace_size<Offset>(1, 1, 0); });
  assert(load_balancing_search_workspace_size<Offset>(INT_MAX - 1, 1, 80) > 0);
  NoAllocator parent;
  Offset storage[3] = {}, offsets[1] = {};
  Int output[1] = {};
  load_balancing_search_into_cuda(
      0, offsets, 1, output, static_cast<Offset*>(nullptr), 0, parent);
  rejects([&] {
    load_balancing_search_into_cuda(1, offsets, 1, output, storage, 1, parent);
  });
  detail::LoadBalanceWorkspaceContext wrapper(parent, storage, sizeof(storage));
  assert(
      wrapper.ptx_version() == 80 && wrapper.stream() == nullptr
      && &wrapper.props() == &parent.property);
  rejects(
      [&] { wrapper.alloc(sizeof(storage) + 1, mgpu::memory_space_device); });
  rejects([&] { wrapper.alloc(1, mgpu::memory_space_host); });
  assert(wrapper.alloc(sizeof(storage), mgpu::memory_space_device) == storage);
  rejects([&] { wrapper.alloc(1, mgpu::memory_space_device); });
  wrapper.free(storage, mgpu::memory_space_device);
  rejects([&] { wrapper.alloc(1, mgpu::memory_space_device); });
  rejects([&] { wrapper.synchronize(); });
  // Instantiate the real GPU path; runtime calls above intentionally stop
  // before launch so this test does not pretend to validate CUDA execution on a
  // CPU node.
  auto volatile launch = static_cast<void (*)(
      int, Offset const*, int, Int*, Offset*, std::size_t, mgpu::context_t&)>(
      &load_balancing_search_into_cuda<Offset, Int>);
  (void)launch;
}
int main() {
  check<int32_t, int32_t>();
  check<int64_t, int32_t>();
  check<int32_t, int64_t>();
  check<int64_t, int64_t>();
}
