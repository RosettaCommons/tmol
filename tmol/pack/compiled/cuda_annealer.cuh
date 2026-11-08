#pragma once
#include <tmol/pack/compiled/cuda_annealer_workspace.hh>
#include <tmol/pack/compiled/simulated_annealing.hh>
#include <tmol/utility/portable/reduce_ops.hh>
#include <moderngpu/cta_reduce.hxx>
#include <moderngpu/kernel_segsort.hxx>
#include <moderngpu/transform.hxx>
#include <cooperative_groups.h>
#include <curand_kernel.h>
#include <type_traits>

namespace tmol::pack::compiled {
static_assert(sizeof(mgpu::range_t) == 8 && sizeof(mgpu::merge_range_t) == 16);
using AnnealerSortLaunch = mgpu::detail::
    segsort_t<mgpu::empty_t, float, int, mgpu::less_t<float>>::launch_t;
static_assert(
    AnnealerSortLaunch::sm_ptx::nt * AnnealerSortLaunch::sm_ptx::vt >= 1408);
// Borrow caller storage while preserving ModernGPU's exact sorting algorithm.
// Device allocation, events and stream synchronization are not permitted here.
class AnnealerSortArena final : public mgpu::context_t {
  mgpu::context_t& parent_;
  unsigned char* data_;
  size_t capacity_, used_ = 0;

 public:
  AnnealerSortArena(mgpu::context_t& parent, void* data, size_t capacity)
      : parent_(parent),
        data_(static_cast<unsigned char*>(data)),
        capacity_(capacity) {}
  void reset() { used_ = 0; }
  const cudaDeviceProp& props() const override { return parent_.props(); }
  int ptx_version() const override { return parent_.ptx_version(); }
  cudaStream_t stream() override { return parent_.stream(); }
  void* alloc(size_t size, mgpu::memory_space_t space) override {
    if (space != mgpu::memory_space_device)
      throw std::logic_error("Annealer sort requested host memory");
    if (!size) return nullptr;
    used_ = (used_ + 15) / 16 * 16;
    if (used_ > capacity_ || size > capacity_ - used_)
      throw std::invalid_argument("CUDA annealer sort workspace is too small");
    void* result = data_ + used_;
    used_ += size;
    return result;
  }
  void free(void*, mgpu::memory_space_t) override {}
  void synchronize() override {
    throw std::logic_error("Annealer sort cannot synchronize");
  }
  cudaEvent_t event() override {
    throw std::logic_error("Annealer sort cannot create events");
  }
  void timer_begin() override {
    throw std::logic_error("Annealer sort cannot time events");
  }
  double timer_end() override {
    throw std::logic_error("Annealer sort cannot time events");
  }
};

template <unsigned int n_threads, typename T, typename op_t>
MGPU_DEVICE __inline__ T reduce_shfl_and_broadcast(
    cooperative_groups::thread_block_tile<n_threads> g, T val, op_t op) {
  for (unsigned int i = n_threads / 2; i > 0; i /= 2) {
    T const shfl_val = g.shfl_down(val, i);
    if (g.thread_rank() < 32 - i) {
      val = op(val, shfl_val);
    }
  }

  // thread 0 shares its reduced value with everyone
  // so that there is no disagreement on the partition
  // function value as a result of numerical imprecision
  T val_bcast = g.shfl(val, 0);

  return val_bcast;
}

template <tmol::Device D, typename Int, typename Real>
struct CudaInteractionGraph {
 public:
  int max_n_rotamers_per_pose_;
  TView<Int, 1, D> pose_n_res_;
  TView<Int, 1, D> pose_n_rotamers_;
  TView<Int, 1, D> pose_rotamer_offset_;
  TView<Int, 2, D> n_rotamers_for_res_;
  TView<Int, 2, D> oneb_offsets_;
  TView<Int, 1, D> res_for_rot_;
  int32_t chunk_size_;
  TView<int64_t, 1, D> neighbor_row_offsets_;
  TView<int32_t, 1, D> neighbor_blocks_;
  TView<int64_t, 1, D> neighbor_chunk_offset_offsets_;
  TView<int64_t, 1, D> reverse_neighbor_chunk_offset_offsets_;
  TView<int64_t, 1, D> chunk_offsets_;
  TView<Real, 1, D> energy1b_;
  TView<Real, 1, D> energy2b_;

  int n_poses_cpu() const { return pose_n_res_.size(0); }
  int max_n_res_cpu() const { return n_rotamers_for_res_.size(1); }
  int max_n_rotamers_per_pose_cpu() const { return max_n_rotamers_per_pose_; }

  MGPU_DEVICE
  int n_res(int pose) const { return pose_n_res_[pose]; }

  MGPU_DEVICE
  int n_rotamers(int pose) const { return pose_n_rotamers_[pose]; }

  MGPU_DEVICE
  int max_n_res() const { return n_rotamers_for_res_.size(1); }

  MGPU_DEVICE
  TView<Int, 2, D> const& n_rotamers_for_res() const {
    return n_rotamers_for_res_;
  }

  MGPU_DEVICE
  TView<Int, 1, D> const& res_for_rot() const { return res_for_rot_; }

  MGPU_DEVICE int64_t
  find_neighbor_edge(int pose, int block, int neighbor) const {
    int const row = pose * max_n_res() + block;
    int64_t lower = neighbor_row_offsets_[row];
    int64_t upper = neighbor_row_offsets_[row + 1];
    while (lower < upper) {
      int64_t const middle = lower + (upper - lower) / 2;
      if (neighbor_blocks_[middle] < neighbor) {
        lower = middle + 1;
      } else {
        upper = middle;
      }
    }
    return lower;
  }

  template <int ChunkSize, unsigned int n_threads>
  MGPU_DEVICE Real total_energy_for_assignment_parallel(
      int pose,
      cooperative_groups::thread_block_tile<n_threads> g,
      TensorAccessor<Int, 1, D> rotamer_assignment) const {
    int32_t const chunk_size =
        ChunkSize == 0 ? chunk_size_ : static_cast<int32_t>(ChunkSize);
    Real totalE = 0;
    int const n_res = pose_n_res_[pose];
    for (int i = g.thread_rank(); i < n_res; i += n_threads) {
      int const irot_local = rotamer_assignment[i];
      int const irot_global = irot_local + oneb_offsets_[pose][i];

      totalE += energy1b_[irot_global];
    }

    for (int i = g.thread_rank(); i < n_res; i += n_threads) {
      int const irot_local = rotamer_assignment[i];
      int const irot_chunk = irot_local / chunk_size;
      int const irot_in_chunk = irot_local - chunk_size * irot_chunk;

      int const row = pose * max_n_res() + i;
      for (int64_t edge = neighbor_row_offsets_[row];
           edge < neighbor_row_offsets_[row + 1];
           ++edge) {
        int const j = neighbor_blocks_[edge];
        if (j <= i) {
          continue;
        }
        int const jrot_local = rotamer_assignment[j];
        int64_t const ij_chunk_offset_offset =
            neighbor_chunk_offset_offsets_[edge];
        int const jrot_chunk = jrot_local / chunk_size;
        int const jrot_in_chunk = jrot_local - chunk_size * jrot_chunk;

        int const jres_n_rots = n_rotamers_for_res_[pose][j];
        int const jres_n_chunks = (jres_n_rots - 1) / chunk_size + 1;
        int const jrot_chunk_size =
            min(chunk_size, jres_n_rots - chunk_size * jrot_chunk);
        int64_t const ij_chunk_offset = chunk_offsets_
            [ij_chunk_offset_offset + irot_chunk * jres_n_chunks + jrot_chunk];
        if (ij_chunk_offset == -1) {
          continue;
        }

        float ij_energy = energy2b_
            [ij_chunk_offset + jrot_chunk_size * irot_in_chunk + jrot_in_chunk];
        totalE += ij_energy;
      }
    }
    totalE =
        reduce_shfl_and_broadcast(g, totalE, tmol::portable::plus_t<float>());
    return totalE;
  }
};

// Preserve the existing mapping, including cuRAND's inclusive upper endpoint.
// Most products already lie in range. Map cuRAND's inclusive endpoint to zero.
MGPU_DEVICE __inline__ int curand_in_range(float uniform, int n) {
  int const index = int(uniform * n);
  return index < n ? index : 0;
}

MGPU_DEVICE __inline__ int curand_in_range(
    curandStatePhilox4_32_10_t* state, int n) {
  return curand_in_range(curand_uniform(state), n);
}

template <tmol::Device D>
MGPU_DEVICE void set_quench_order(
    TensorAccessor<int, 1, D> quench_order,
    int n_rotamers,
    int rotamer_offset,
    curandStatePhilox4_32_10_t* state) {
  // Create a random permutation of all the rotamers
  // and visit them in this order to ensure all of them
  // are seen during the quench step

  for (int i = 0; i < n_rotamers; ++i) {
    quench_order[i] = i + rotamer_offset;
  }
  for (int i = 0; i <= n_rotamers - 2; ++i) {
    int rand_offset = curand_in_range(state, n_rotamers - i);
    int j = i + rand_offset;
    // swap i and j;
    int jval = quench_order[j];
    quench_order[j] = quench_order[i];
    quench_order[i] = jval;
  }
}

template <tmol::Device D>
MGPU_DEVICE int set_quench_32_order(
    int n_residues,
    TensorAccessor<int, 1, D> n_rotamers_for_res,
    TensorAccessor<int, 1, D> oneb_offsets,
    TensorAccessor<int, 1, D> quench_order,
    curandStatePhilox4_32_10_t* state) {
  // Create a random permutation of all the rotamers
  // and visit them in this order to ensure all of them
  // are seen during the quench step
  // int const n_residues = n_rotamers_for_res.size(0);
  // int const n_rots = quench_order.size(0);
  int count_n_quench_rots = 0;
  for (int i = 0; i < n_residues; ++i) {
    int const i_n_rots = n_rotamers_for_res[i];
    int const i_offset = oneb_offsets[i];
    int const i_n_quench_rots = (i_n_rots - 1) / 31 + 1;
    for (int j = 0; j < i_n_quench_rots; j++) {
      quench_order[count_n_quench_rots + j] = i_offset + 31 * j;
    }
    count_n_quench_rots += i_n_quench_rots;
  }
  for (int i = 0; i <= count_n_quench_rots - 2; ++i) {
    int rand_offset = curand_in_range(state, count_n_quench_rots - i);
    int j = i + rand_offset;
    // swap i and j;
    int jval = quench_order[j];
    quench_order[j] = quench_order[i];
    quench_order[i] = jval;
  }
  return count_n_quench_rots;
}

// Metropolis SA: now one warp per trajectory.
//   inner step:
//     thread 0 picks a candidate rotamer
//     all 32 threads cooperatively sum _dE_ over neighbor list
//     warp-reduce sum gives total dE
//     standard Metropolis accept/reject
template <
    int ChunkSize,
    bool ReturnFinalScore,
    bool EntryAssignmentIsBest,
    bool TrackBestAssignment,
    bool CacheAssignments = false,
    tmol::Device D,
    uint n_threads,
    typename Int,
    typename Real>
MGPU_DEVICE float warp_wide_sim_annealing(
    int pose,
    curandStatePhilox4_32_10_t* state,
    cooperative_groups::thread_block_tile<n_threads> g,
    CudaInteractionGraph<D, Int, Real> ig,
    TensorAccessor<Int, 1, D> current_rotamer_assignment,
    TensorAccessor<Int, 1, D> best_rotamer_assignment,
    TensorAccessor<Int, 1, D> quench_order,
    float hi_temp,
    float lo_temp,
    int n_outer_iterations,
    int n_inner_iterations,
    bool quench_on_last_iteration,
    bool quench_lite) {
  float const cooling_factor = 0.35f;
  int32_t const chunk_size =
      ChunkSize == 0 ? ig.chunk_size_ : static_cast<int32_t>(ChunkSize);
  int const n_res = ig.n_res(pose);
  int const n_rotamers = ig.n_rotamers(pose);
  int const pose_rotamer_offset = ig.pose_rotamer_offset_[pose];

  auto original_current = current_rotamer_assignment;
  auto original_best = best_rotamer_assignment;
  int64_t const assignment_size = n_res;
  int64_t const assignment_stride = 1;
  // Each warp owns one trajectory; cache its frequently read assignments.
  // The host enables this only when all residues fit and the CTA has 4 warps.
  if constexpr (CacheAssignments) {
    __shared__ Int current_cache[4][128];
    __shared__ Int best_cache[4][128];
    int const warp = threadIdx.x / 32;
    for (int k = g.thread_rank(); k < n_res; k += 32) {
      current_cache[warp][k] = current_rotamer_assignment[k];
      best_cache[warp][k] = best_rotamer_assignment[k];
    }
    g.sync();
    current_rotamer_assignment = TensorAccessor<Int, 1, D>(
        current_cache[warp], &assignment_size, &assignment_stride);
    best_rotamer_assignment = TensorAccessor<Int, 1, D>(
        best_cache[warp], &assignment_size, &assignment_stride);
  }

  float temperature = hi_temp;
  float best_energy = 0.0f;
  float current_total_energy = 0.0f;
  if constexpr (TrackBestAssignment) {
    best_energy = ig.template total_energy_for_assignment_parallel<ChunkSize>(
        pose, g, current_rotamer_assignment);
    current_total_energy = best_energy;
  }
  int n_steps = 0;

  for (int i = 0; i < n_outer_iterations; ++i) {
    bool quench = false;
    int i_n_inner_iterations = n_inner_iterations;

    if (i == n_outer_iterations - 1 && quench_on_last_iteration) {
      i_n_inner_iterations = n_rotamers;
      quench = true;
      temperature = 0.0f;
      // Recover the lowest-energy assignment before quenching. Ranking phases
      // enter with this assignment already current, so their first exact score
      // above can be reused.
      if (i != 0 || !EntryAssignmentIsBest) {
        for (int j = g.thread_rank(); j < n_res; j += 32) {
          current_rotamer_assignment[j] = best_rotamer_assignment[j];
        }
        g.sync();
        current_total_energy =
            ig.template total_energy_for_assignment_parallel<ChunkSize>(
                pose, g, current_rotamer_assignment);
      }
    }

    for (int j = 0; j < i_n_inner_iterations; ++j) {
      // --- Pick one candidate rotamer (thread 0, broadcast) ---
      int ran_res(0), local_new_rot(0), global_new_rot(0);
      float accept_rand(0);

      if (quench) {
        if (g.thread_rank() == 0) {
          // A quench visits its shuffled order once.
          if (j == 0) {
            if (quench_lite) {
              i_n_inner_iterations = set_quench_32_order(
                  n_res,
                  ig.n_rotamers_for_res_[pose],
                  ig.oneb_offsets_[pose],
                  quench_order,
                  state);
            } else {
              set_quench_order(
                  quench_order, n_rotamers, pose_rotamer_offset, state);
            }
          }
          int global_ran_rot = quench_order[j];
          ran_res = ig.res_for_rot()[global_ran_rot];
          global_new_rot = global_ran_rot;
          local_new_rot = global_ran_rot - ig.oneb_offsets_[pose][ran_res];
        }
        if (j == 0 && quench_lite) {
          i_n_inner_iterations = g.shfl(i_n_inner_iterations, 0);
        }
      } else {
        if (g.thread_rank() == 0) {
          float4 rands = curand_uniform4(state);
          int global_ran_rot =
              curand_in_range(rands.x, n_rotamers) + pose_rotamer_offset;
          ran_res = ig.res_for_rot()[global_ran_rot];
          int const ran_res_n_rots = ig.n_rotamers_for_res_[pose][ran_res];
          int const ran_res_offset = ig.oneb_offsets_[pose][ran_res];
          local_new_rot = curand_in_range(rands.y, ran_res_n_rots);
          global_new_rot = local_new_rot + ran_res_offset;
          accept_rand = rands.z;
        }
      }

      // Only the residue and local rotamer are consumed by the whole tile;
      // global_new_rot and accept_rand remain lane-0-only below.
      ran_res = g.shfl(ran_res, 0);
      local_new_rot = g.shfl(local_new_rot, 0);

      int const local_prev_rot = current_rotamer_assignment[ran_res];
      int const global_prev_rot =
          local_prev_rot + ig.oneb_offsets_[pose][ran_res];

      // Skip no-op substitutions
      if (local_new_rot == local_prev_rot) {
        continue;
      }

      // --- Compute dE: all 32 threads split the neighbor list ---
      int const ran_res_n_rots = ig.n_rotamers_for_res_[pose][ran_res];
      int const ran_res_n_chunks = (ran_res_n_rots - 1) / chunk_size + 1;

      int const new_chunk = local_new_rot / chunk_size;
      int const new_in_chunk = local_new_rot - chunk_size * new_chunk;
      int const new_chunk_size =
          min(chunk_size, ran_res_n_rots - chunk_size * new_chunk);

      int const prev_chunk = local_prev_rot / chunk_size;
      int const prev_in_chunk = local_prev_rot - chunk_size * prev_chunk;
      int const prev_chunk_size =
          min(chunk_size, ran_res_n_rots - chunk_size * prev_chunk);

      // Thread 0 contributes the 1b delta; others start at 0
      float delta_e =
          (g.thread_rank() == 0)
              ? ig.energy1b_[global_new_rot] - ig.energy1b_[global_prev_rot]
              : 0.0f;

      int const row = pose * ig.max_n_res() + ran_res;
      int64_t const row_begin = ig.neighbor_row_offsets_[row];
      int64_t const row_end = ig.neighbor_row_offsets_[row + 1];

      for (int64_t edge = row_begin + g.thread_rank(); edge < row_end;
           edge += 32) {
        int const k = ig.neighbor_blocks_[edge];
        int const local_k_rot = current_rotamer_assignment[k];
        int const k_chunk = local_k_rot / chunk_size;
        int const k_in_chunk = local_k_rot - k_chunk * chunk_size;

        int64_t const k_offset_offset =
            ig.reverse_neighbor_chunk_offset_offsets_[edge];

        int64_t const k_new_chunk_offset =
            ig.chunk_offsets_
                [k_offset_offset + k_chunk * ran_res_n_chunks + new_chunk];
        float new_contrib = 0.0f;
        if (k_new_chunk_offset >= 0) {
          new_contrib = ig.energy2b_
                            [k_new_chunk_offset + new_chunk_size * k_in_chunk
                             + new_in_chunk];
        }

        // The proposed and current rotamers commonly occupy the same chunk.
        // In that case both table entries share one sparse chunk offset; avoid
        // repeating this irregular global-memory lookup for every neighbor.
        int64_t const k_prev_chunk_offset =
            prev_chunk == new_chunk
                ? k_new_chunk_offset
                : ig.chunk_offsets_
                      [k_offset_offset + k_chunk * ran_res_n_chunks
                       + prev_chunk];
        float prev_contrib = 0.0f;
        if (k_prev_chunk_offset >= 0) {
          prev_contrib = ig.energy2b_
                             [k_prev_chunk_offset + prev_chunk_size * k_in_chunk
                              + prev_in_chunk];
        }

        delta_e += new_contrib - prev_contrib;
      }

      // Warp-reduce to sum delta_e across all threads
      float const total_delta_e = reduce_shfl_and_broadcast(
          g, delta_e, tmol::portable::plus_t<float>());

      // --- Metropolis accept/reject (thread 0 decides, broadcasts) ---
      bool accept;
      if (g.thread_rank() == 0) {
        if (quench) {
          accept = total_delta_e < 0.0f;
        } else {
          accept = total_delta_e < 0.0f
                   || accept_rand < expf(-total_delta_e / temperature);
        }
      }
      accept = g.shfl(accept, 0);

      if (accept) {
        if constexpr (CacheAssignments) {
          // Shuffles do not order shared-memory accesses. Complete reads from
          // this proposal and the preceding best-assignment copy before
          // writing.
          g.sync();
        }
        if (g.thread_rank() == 0) {
          current_rotamer_assignment[ran_res] = local_new_rot;
        }
        if constexpr (CacheAssignments) {
          g.sync();
        }
        if constexpr (TrackBestAssignment) {
          if (g.thread_rank() == 0) {
            current_total_energy += total_delta_e;
          }
          current_total_energy = g.shfl(current_total_energy, 0);
          if (current_total_energy < best_energy) {
            best_energy = current_total_energy;
            for (int k = g.thread_rank(); k < n_res; k += 32) {
              best_rotamer_assignment[k] = current_rotamer_assignment[k];
            }
          }
        }
      }

      // Periodically recompute total energy to correct floating-point drift
      if constexpr (TrackBestAssignment) {
        if (++n_steps > 10000) {
          n_steps = 0;
          current_total_energy =
              ig.template total_energy_for_assignment_parallel<ChunkSize>(
                  pose, g, current_rotamer_assignment);
        }
      }

    }  // end inner loop

    // geometric cooling toward lo_temp
    temperature = cooling_factor * (temperature - lo_temp) + lo_temp;

  }  // end outer loop

  if constexpr (CacheAssignments) {
    g.sync();
    for (int k = g.thread_rank(); k < n_res; k += 32) {
      original_current[k] = current_rotamer_assignment[k];
      if constexpr (TrackBestAssignment) {
        original_best[k] = best_rotamer_assignment[k];
      }
    }
    g.sync();
  }

  // Transition phases consume only the assignments; avoid their otherwise
  // unused exact rescore while retaining it for ranking phases.
  if constexpr (ReturnFinalScore) {
    return ig.template total_energy_for_assignment_parallel<ChunkSize>(
        pose, g, current_rotamer_assignment);
  }
  return 0.0f;
}

template <int ChunkSize, class IG>
void run_cuda_annealer(
    mgpu::context_t& context,
    IG ig,
    uint64_t const* seeds,
    CudaAnnealerWorkspace workspace,
    TView<float, 2, Device::CUDA> scores_fullquench,
    TView<int32_t, 3, Device::CUDA> rotamer_assignments_final) {
  int const n_poses = ig.n_poses_cpu(), max_n_res = ig.max_n_res_cpu(),
            max_n_rotamers = ig.max_n_rotamers_per_pose_cpu();
  if (scores_fullquench.size(0) != n_poses
      || scores_fullquench.size(1) != workspace.counts.final
      || rotamer_assignments_final.size(0) != n_poses
      || rotamer_assignments_final.size(1) != workspace.counts.final
      || rotamer_assignments_final.size(2) != max_n_res)
    throw std::invalid_argument("CUDA annealer output axes differ");
  if (!n_poses) return;
  if (!workspace.data || !seeds)
    throw std::invalid_argument(
        "CUDA annealer requires workspace and phase seeds");
  ig.reverse_neighbor_chunk_offset_offsets_ = workspace.reverse_offsets;
  auto check = [](cudaError_t e) {
    if (e != cudaSuccess) throw mgpu::cuda_exception_t(e);
  };
  check(cudaMemsetAsync(
      workspace.data, 0, workspace.zero_bytes, context.stream()));
  check(cudaMemsetAsync(
      scores_fullquench.data(),
      0,
      int64_t(n_poses) * workspace.counts.final * sizeof(float),
      context.stream()));
  if (max_n_res)
    check(cudaMemsetAsync(
        rotamer_assignments_final.data(),
        0,
        int64_t(n_poses) * workspace.counts.final * max_n_res * sizeof(int32_t),
        context.stream()));
  int const n_hitemp_simA_traj = workspace.counts.high;
  constexpr int n_lotemp_expansions = CudaAnnealerPolicy::low_expansions;
  int const n_lotemp_simA_traj = workspace.counts.low;
  int const n_fullquench_traj = workspace.counts.final;
  constexpr int n_outer_iterations_hitemp = CudaAnnealerPolicy::high_iterations;
  constexpr int n_outer_iterations_lotemp = CudaAnnealerPolicy::low_iterations;
  constexpr float high_temp_initial = CudaAnnealerPolicy::high_initial,
                  low_temp_initial = CudaAnnealerPolicy::low_initial;
  constexpr float high_temp_later = CudaAnnealerPolicy::high_later,
                  low_temp_later = CudaAnnealerPolicy::low_later;
  int const n_hitemp_simA_threads = 32 * n_poses * n_hitemp_simA_traj;
  int const n_lotemp_simA_threads = 32 * n_poses * n_lotemp_simA_traj;
  int const n_fullquench_threads = 32 * n_poses * n_fullquench_traj;
  int const n_inner_iterations_hitemp = int(max_n_rotamers * .4);
  int const n_inner_iterations_lotemp = int(max_n_rotamers * .15);
  AnnealerSortArena sort_context(
      context, workspace.sort_data, workspace.sort_bytes);
  auto scores_hitemp = workspace.scores_hitemp;
  auto current_rotamer_assignments_hitemp =
      workspace.current_rotamer_assignments_hitemp;
  auto best_rotamer_assignments_hitemp =
      workspace.best_rotamer_assignments_hitemp;
  auto sorted_hitemp_traj = workspace.sorted_hitemp_traj;
  auto segment_heads_hitemp = workspace.segment_heads_hitemp;
  auto segment_heads_lotemp = workspace.segment_heads_lotemp;
  auto segment_heads_fullquench = workspace.segment_heads_fullquench;
  auto scores_lotemp = workspace.scores_lotemp;
  auto current_rotamer_assignments_lotemp =
      workspace.current_rotamer_assignments_lotemp;
  auto best_rotamer_assignments_lotemp =
      workspace.best_rotamer_assignments_lotemp;
  auto sorted_lotemp_traj = workspace.sorted_lotemp_traj;
  auto current_rotamer_assignments_fullquench =
      workspace.current_rotamer_assignments_fullquench;
  auto sorted_fullquench_traj = workspace.sorted_fullquench_traj;
  auto quench_lite_order = workspace.quench_lite_order;
  auto fullquench_order = workspace.fullquench_order;
  // The graph is fixed throughout annealing. Resolve transposed chunk-table
  // offsets once per edge instead of binary-searching each neighbor row for
  // every proposal in every trajectory. Scratch scales with edges, not pairs.
  auto cache_reverse_offsets = [=] MGPU_DEVICE(int row) {
    int const pose = row / max_n_res;
    int const block = row % max_n_res;
    for (int64_t edge = ig.neighbor_row_offsets_[row];
         edge < ig.neighbor_row_offsets_[row + 1];
         ++edge) {
      int64_t const reverse =
          ig.find_neighbor_edge(pose, ig.neighbor_blocks_[edge], block);
      ig.reverse_neighbor_chunk_offset_offsets_[edge] =
          ig.neighbor_chunk_offset_offsets_[reverse];
    }
  };
  mgpu::transform<128, 1>(cache_reverse_offsets, n_poses * max_n_res, context);

  // Phase 1: run full SA, then score via quench-lite.
  auto hitemp_simulated_annealing = [=] MGPU_DEVICE(
                                        int thread_id, auto cache_tag) {
    auto seed = seeds[0];
    curandStatePhilox4_32_10_t state;
    curand_init(seed, thread_id, 0, &state);

    cooperative_groups::thread_block_tile<32> g =
        cooperative_groups::tiled_partition<32>(
            cooperative_groups::this_thread_block());
    int const cta_id = thread_id / 32;
    int const pose = cta_id / n_hitemp_simA_traj;
    int const traj_id = cta_id % n_hitemp_simA_traj;
    int const n_res = ig.n_res(pose);
    int const n_rotamers = ig.n_rotamers(pose);

    if (g.thread_rank() == 0) {
      sorted_hitemp_traj[pose][traj_id] = traj_id;
    }
    if (g.thread_rank() == 0 && traj_id == 0) {
      segment_heads_hitemp[pose] = pose * n_hitemp_simA_traj;
      segment_heads_lotemp[pose] = pose * n_lotemp_simA_traj;
      segment_heads_fullquench[pose] = pose * n_fullquench_traj;
    }
    // A fully frozen pose has no valid random-rotamer domain.
    if (n_res == 0) {
      if (g.thread_rank() == 0) {
        scores_hitemp[pose][traj_id] = 0.0f;
      }
      return;
    }

    // Random initial assignment
    for (int i = g.thread_rank(); i < n_res; i += 32) {
      int const i_n_rots = ig.n_rotamers_for_res()[pose][i];
      int chosen = curand_in_range(&state, i_n_rots);
      current_rotamer_assignments_hitemp[pose][traj_id][i] = chosen;
      best_rotamer_assignments_hitemp[pose][traj_id][i] = chosen;
    }

    // Full SA run with geometric cooling
    // Match the CPU annealer's per-pose work: a large neighbor in a jagged
    // batch must not inflate this pose's trajectory length.
    int const pose_inner_iterations = int(n_rotamers * .4);
    warp_wide_sim_annealing<
        ChunkSize,
        false,
        false,
        true,
        decltype(cache_tag)::value>(
        pose,
        &state,
        g,
        ig,
        current_rotamer_assignments_hitemp[pose][traj_id],
        best_rotamer_assignments_hitemp[pose][traj_id],
        quench_lite_order[pose][traj_id],
        high_temp_initial,
        low_temp_initial,
        n_outer_iterations_hitemp,
        pose_inner_iterations,
        false,
        false);

    // Preserve the unquenched best state for the next phase; ranking can
    // quench the best buffer in place because it is no longer needed.
    for (int i = g.thread_rank(); i < n_res; i += 32) {
      int i_assignment = best_rotamer_assignments_hitemp[pose][traj_id][i];
      current_rotamer_assignments_hitemp[pose][traj_id][i] = i_assignment;
    }

    // Quench-lite to produce a score for ranking
    float after_first_quench_lite_totalE =
        warp_wide_sim_annealing<ChunkSize, true, true, false>(
            pose,
            &state,
            g,
            ig,
            best_rotamer_assignments_hitemp[pose][traj_id],
            best_rotamer_assignments_hitemp[pose][traj_id],
            quench_lite_order[pose][traj_id],
            high_temp_initial,
            low_temp_initial,
            1,  // quench on the (only) iteration
            n_inner_iterations_hitemp,
            true,
            true);
    if (g.thread_rank() == 0) {
      scores_hitemp[pose][traj_id] = after_first_quench_lite_totalE;
    }
  };

  // Phase 2: low-temp SA seeded from top hitemp trajectories (each seeds
  // n_lotemp_expansions independent lotemp runs).
  auto lotemp_simulated_annealing = [=] MGPU_DEVICE(
                                        int thread_id, auto cache_tag) {
    cooperative_groups::thread_block_tile<32> g =
        cooperative_groups::tiled_partition<32>(
            cooperative_groups::this_thread_block());
    curandStatePhilox4_32_10_t state;
    if (g.thread_rank() == 0) {
      auto seed = seeds[1];
      curand_init(seed, thread_id, 0, &state);
    }

    int const cta_id = thread_id / 32;
    int const pose = cta_id / n_lotemp_simA_traj;
    int const traj_id = cta_id % n_lotemp_simA_traj;
    int const source_traj =
        sorted_hitemp_traj[pose][traj_id / n_lotemp_expansions];

    int const n_res = ig.n_res(pose);
    int const n_rotamers = ig.n_rotamers(pose);

    if (g.thread_rank() == 0) {
      sorted_lotemp_traj[pose][traj_id] = traj_id;
    }
    if (n_res == 0) {
      if (g.thread_rank() == 0) {
        scores_lotemp[pose][traj_id] = 0.0f;
      }
      return;
    }

    // Seed from the selected hitemp trajectory's best state
    for (int i = g.thread_rank(); i < n_res; i += 32) {
      int i_rot = current_rotamer_assignments_hitemp[pose][source_traj][i];
      current_rotamer_assignments_lotemp[pose][traj_id][i] = i_rot;
      best_rotamer_assignments_lotemp[pose][traj_id][i] = i_rot;
    }

    // Low-temperature cooling trajectory
    int const pose_inner_iterations = int(n_rotamers * .15);
    warp_wide_sim_annealing<
        ChunkSize,
        false,
        false,
        true,
        decltype(cache_tag)::value>(
        pose,
        &state,
        g,
        ig,
        current_rotamer_assignments_lotemp[pose][traj_id],
        best_rotamer_assignments_lotemp[pose][traj_id],
        quench_lite_order[pose][traj_id],
        high_temp_later,
        low_temp_later,
        n_outer_iterations_lotemp,
        pose_inner_iterations,
        false,
        false);

    // Quench-lite to score for the next round of selection
    for (int i = g.thread_rank(); i < n_res; i += 32) {
      current_rotamer_assignments_lotemp[pose][traj_id][i] =
          best_rotamer_assignments_lotemp[pose][traj_id][i];
    }
    float after_lotemp_quench_lite_totalE =
        warp_wide_sim_annealing<ChunkSize, true, true, false>(
            pose,
            &state,
            g,
            ig,
            current_rotamer_assignments_lotemp[pose][traj_id],
            best_rotamer_assignments_lotemp[pose][traj_id],
            quench_lite_order[pose][traj_id],
            high_temp_later,
            low_temp_later,
            1,  // quench on the (only) iteration
            n_inner_iterations_lotemp,
            true,
            true);
    if (g.thread_rank() == 0) {
      scores_lotemp[pose][traj_id] = after_lotemp_quench_lite_totalE;
    }
  };

  // Phase 3: full greedy quench from top lotemp trajectories.
  auto fullquench = ([=] MGPU_DEVICE(int thread_id) {
    cooperative_groups::thread_block_tile<32> g =
        cooperative_groups::tiled_partition<32>(
            cooperative_groups::this_thread_block());
    curandStatePhilox4_32_10_t state;
    if (g.thread_rank() == 0) {
      auto seed = seeds[2];
      curand_init(seed, thread_id, 0, &state);
    }
    int const cta_id = thread_id / 32;
    int const pose = cta_id / n_fullquench_traj;
    int const traj_id = cta_id % n_fullquench_traj;
    int const source_traj = sorted_lotemp_traj[pose][traj_id];

    int const n_res = ig.n_res(pose);

    if (g.thread_rank() == 0) {
      sorted_fullquench_traj[pose][traj_id] = traj_id;
    }
    if (n_res == 0) {
      if (g.thread_rank() == 0) {
        scores_fullquench[pose][traj_id] = 0.0f;
      }
      return;
    }

    for (int i = g.thread_rank(); i < n_res; i += 32) {
      int i_rot = current_rotamer_assignments_lotemp[pose][source_traj][i];
      current_rotamer_assignments_fullquench[pose][traj_id][i] = i_rot;
    }

    warp_wide_sim_annealing<ChunkSize, false, true, false>(
        pose,
        &state,
        g,
        ig,
        current_rotamer_assignments_fullquench[pose][traj_id],
        current_rotamer_assignments_fullquench[pose][traj_id],
        fullquench_order[pose][traj_id],
        high_temp_later,
        low_temp_later,
        1,  // quench on the (only) iteration
        n_inner_iterations_lotemp,
        true,
        false);
    // A greedy quench only accepts improvements, so current is also best.
    float best_totalE =
        ig.template total_energy_for_assignment_parallel<ChunkSize>(
            pose, g, current_rotamer_assignments_fullquench[pose][traj_id]);
    if (g.thread_rank() == 0) {
      scores_fullquench[pose][traj_id] = best_totalE;
    }
  });

  auto final_reindexing = ([=] MGPU_DEVICE(int thread_id) {
    cooperative_groups::thread_block_tile<32> g =
        cooperative_groups::tiled_partition<32>(
            cooperative_groups::this_thread_block());
    int const cta_id = thread_id / 32;
    int const pose = cta_id / n_fullquench_traj;
    int const traj_id = cta_id % n_fullquench_traj;
    int const source_traj = sorted_fullquench_traj[pose][traj_id];
    int const n_res = ig.n_res(pose);
    for (int i = g.thread_rank(); i < n_res; i += 32) {
      rotamer_assignments_final[pose][traj_id][i] =
          current_rotamer_assignments_fullquench[pose][source_traj][i];
    }
  });

  // Now launch the kernels we have created

  // Pack four independent trajectory warps per CTA. Their global thread IDs
  // and Philox streams stay unchanged while CTA scheduling overhead falls.
  constexpr int annealer_cta_threads = 128;

  // On Ampere and newer, longer trajectories amortize a small amount of spill
  // traffic and benefit from extra latency hiding. Short and older-GPU
  // workloads retain the unconstrained kernel.
  using HitempLaunch = mgpu::launch_params_t<annealer_cta_threads, 1, 1, 6>;
  auto launch_hitemp = [&](auto cache_tag) {
    if (max_n_rotamers >= 128 && context.ptx_version() >= 80) {
      mgpu::transform<HitempLaunch>(
          hitemp_simulated_annealing,
          n_hitemp_simA_threads,
          context,
          cache_tag);
    } else {
      mgpu::transform<annealer_cta_threads, 1>(
          hitemp_simulated_annealing,
          n_hitemp_simA_threads,
          context,
          cache_tag);
    }
  };
  bool const cache_assignments =
      max_n_res <= 128 && context.ptx_version() == 90;
  if (cache_assignments) {
    launch_hitemp(std::true_type{});
  } else {
    launch_hitemp(std::false_type{});
  }

  sort_context.reset();
  mgpu::segmented_sort(
      scores_hitemp.data(),
      sorted_hitemp_traj.data(),
      n_poses * n_hitemp_simA_traj,
      segment_heads_hitemp.data(),
      n_poses,
      mgpu::less_t<float>(),
      sort_context);

  if (cache_assignments) {
    mgpu::transform<annealer_cta_threads, 1>(
        lotemp_simulated_annealing,
        n_lotemp_simA_threads,
        context,
        std::true_type{});
  } else {
    mgpu::transform<annealer_cta_threads, 1>(
        lotemp_simulated_annealing,
        n_lotemp_simA_threads,
        context,
        std::false_type{});
  }

  sort_context.reset();
  mgpu::segmented_sort(
      scores_lotemp.data(),
      sorted_lotemp_traj.data(),
      n_poses * n_lotemp_simA_traj,
      segment_heads_lotemp.data(),
      n_poses,
      mgpu::less_t<float>(),
      sort_context);

  mgpu::transform<annealer_cta_threads, 1>(
      fullquench, n_fullquench_threads, context);

  sort_context.reset();
  mgpu::segmented_sort(
      scores_fullquench.data(),
      sorted_fullquench_traj.data(),
      n_poses * n_fullquench_traj,
      segment_heads_fullquench.data(),
      n_poses,
      mgpu::less_t<float>(),
      sort_context);

  mgpu::transform<annealer_cta_threads, 1>(
      final_reindexing, n_fullquench_threads, context);
}
}  // namespace tmol::pack::compiled
