// Boilerplate shared by the tmol-jax CUDA pair adapters.
#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

using F32R1 = ffi::BufferR1<ffi::DataType::F32>;
using F32R2 = ffi::BufferR2<ffi::DataType::F32>;
using F32R3 = ffi::BufferR3<ffi::DataType::F32>;
using F32R4 = ffi::BufferR4<ffi::DataType::F32>;
using S32R2 = ffi::BufferR2<ffi::DataType::S32>;
using S32R3 = ffi::BufferR3<ffi::DataType::S32>;
using U8R3 = ffi::BufferR3<ffi::DataType::U8>;
using U32R4 = ffi::BufferR4<ffi::DataType::U32>;
using PredR2 = ffi::BufferR2<ffi::DataType::PRED>;

constexpr int kTile = 16;

// Map a linear index over the upper triangle of an n_tiles x n_tiles grid.
__device__ inline void tile_pair_from_linear(
    int index, int n_tiles, int& tile_i, int& tile_j) {
  int lo = 0;
  int hi = n_tiles;
  while (lo + 1 < hi) {
    int mid = (lo + hi) / 2;
    int start = mid * n_tiles - mid * (mid - 1) / 2;
    if (start <= index) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  tile_i = lo;
  int start = lo * n_tiles - lo * (lo - 1) / 2;
  tile_j = lo + index - start;
}

// Sum each of N per-thread channel values across the block, then add the
// block totals to dest[0..N). Must be called by every thread in the block.
template <int N>
__device__ inline void block_reduce_add(
    float (&shared)[N][kTile * kTile], const float (&values)[N], float* dest) {
#pragma unroll
  for (int channel = 0; channel < N; ++channel) {
    shared[channel][threadIdx.x] = values[channel];
  }
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
#pragma unroll
      for (int channel = 0; channel < N; ++channel) {
        shared[channel][threadIdx.x] += shared[channel][threadIdx.x + stride];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
#pragma unroll
    for (int channel = 0; channel < N; ++channel) {
      atomicAdd(dest + channel, shared[channel][0]);
    }
  }
}

inline ffi::Error launch_status() {
  if (auto status = cudaPeekAtLastError(); status != cudaSuccess) {
    return ffi::Error::Internal(cudaGetErrorString(status));
  }
  return ffi::Error::Success();
}
