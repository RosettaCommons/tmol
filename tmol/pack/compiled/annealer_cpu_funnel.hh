#pragma once
#include <tmol/pack/compiled/annealer_cpu.hh>
#include <tmol/pack/compiled/annealer_workspace.hh>
#include <tmol/pack/compiled/annealer_policy.hh>
#include <algorithm>
#include <cstring>
#include <numeric>
#include <utility>

namespace tmol::pack::compiled {
// Caller-owned, zero-initialized phase storage; outputs belong to Context.
struct CpuFunnelWorkspace : AnnealerWorkspace<Device::CPU> {
  static constexpr Device D = Device::CPU;
  AnnealerTrajectoryCounts counts;
  TView<int32_t, 3, D> hi_cur, hi_best, lo_cur, lo_best, fq;
  TView<float, 2, D> hi_score, lo_score, fq_score;
  CpuFunnelWorkspace(
      int64_t poses,
      int64_t residues,
      int high = AnnealerTrajectoryCounts::default_high,
      void* storage = nullptr,
      int64_t storage_bytes = -1)
      : AnnealerWorkspace(storage), counts(high, true) {
    if (poses < 0 || residues < 0 || poses > INT32_MAX || residues > INT32_MAX
        || (residues && poses > INT32_MAX / residues))
      throw std::invalid_argument(
          "CPU annealer dimensions exceed dispatch capacity");
    hi_cur = take<int32_t, 3>({poses, counts.high, residues});
    hi_best = take<int32_t, 3>({poses, counts.high, residues});
    hi_score = take<float, 2>({poses, counts.high});
    lo_cur = take<int32_t, 3>({poses, counts.low, residues});
    lo_best = take<int32_t, 3>({poses, counts.low, residues});
    lo_score = take<float, 2>({poses, counts.low});
    fq = take<int32_t, 3>({poses, counts.final, residues});
    fq_score = take<float, 2>({poses, counts.final});
    validate(storage_bytes);
  }
};

namespace detail {
struct Neighbor {
  int residue;
  int64_t chunk_offset_offset;
};

// One pose's CSR neighbor rows, each paired with the reverse edge's
// chunk_offset_offset, in the original neighbor order.
template <tmol::Device D>
std::vector<std::vector<Neighbor>> pose_neighbors(
    int pose,
    int n_res,
    int max_n_res,
    TView<int64_t, 1, D> neighbor_row_offsets,
    TView<int32_t, 1, D> neighbor_blocks,
    TView<int64_t, 1, D> neighbor_chunk_offset_offsets) {
  std::vector<std::vector<Neighbor>> neighbors(n_res);
  for (int b = 0; b < n_res; ++b) {
    int const row = pose * max_n_res + b;
    for (int64_t edge = neighbor_row_offsets[row];
         edge < neighbor_row_offsets[row + 1];
         ++edge) {
      int const neighbor = neighbor_blocks[edge];
      int const reverse_row = pose * max_n_res + neighbor;
      int64_t lower = neighbor_row_offsets[reverse_row];
      int64_t upper = neighbor_row_offsets[reverse_row + 1];
      while (lower < upper) {
        int64_t const middle = lower + (upper - lower) / 2;
        if (neighbor_blocks[middle] < b) {
          lower = middle + 1;
        } else {
          upper = middle;
        }
      }
      neighbors[b].push_back({neighbor, neighbor_chunk_offset_offsets[lower]});
    }
  }
  return neighbors;
}

// Multi-trajectory schedule, used when at least min_threads are available.
// Phases and rules follow the CUDA annealer; lengths follow the MPS one.
struct FunnelSchedule {
  static constexpr int min_threads = 8;
  static constexpr int lotemp_expansions = 10;
  static constexpr float round1_cut = 0.25f;
  static constexpr float round2_cut = 0.25f;
  static constexpr int outer_hitemp = 5;
  static constexpr int outer_lotemp = 5;
  static constexpr float inner_mult_hitemp = 0.4f;
  static constexpr float inner_mult_lotemp = 0.15f;
  static constexpr float high_temp_initial = 30.0f;
  static constexpr float low_temp_initial = 0.3f;
  static constexpr float high_temp_later = 0.2f;
  static constexpr float low_temp_later = 0.1f;
  static constexpr float cooling_factor = 0.35f;
  static constexpr int quench_lite_stride = 31;
};

inline std::mt19937_64 trajectory_rng(uint64_t seed, int pose, int traj) {
  std::seed_seq seq{
      uint32_t(seed), uint32_t(seed >> 32), uint32_t(pose), uint32_t(traj)};
  return std::mt19937_64(seq);
}

template <tmol::Device D>
struct FunnelAnnealer {
  using Assignment = TensorAccessor<int, 1, D>;

  TView<int, 1, D> pose_n_res;
  TView<int, 1, D> n_rotamers_for_pose;
  TView<int, 1, D> rotamer_offset_for_pose;
  TView<int, 2, D> n_rotamers_for_res;
  TView<int, 2, D> oneb_offsets;
  TView<int, 1, D> res_for_rot;
  int32_t chunk_size;
  int max_n_res;
  TView<int64_t, 1, D> neighbor_row_offsets;
  TView<int32_t, 1, D> neighbor_blocks;
  TView<int64_t, 1, D> neighbor_chunk_offset_offsets;
  TView<int64_t, 1, D> chunk_offsets;
  TView<float, 1, D> energy1b;
  TView<float, 1, D> energy2b;
  std::vector<std::vector<std::vector<Neighbor>>> neighbors;

  float total(int pose, Assignment a) const {
    return total_energy_for_assignment(
        n_rotamers_for_res[pose],
        oneb_offsets[pose],
        chunk_size,
        pose,
        max_n_res,
        neighbor_row_offsets,
        neighbor_blocks,
        neighbor_chunk_offset_offsets,
        chunk_offsets,
        energy1b,
        energy2b,
        a);
  }

  // Energy change from setting residue res to local rotamer new_rot.
  double delta(int pose, Assignment cur, int res, int new_rot) const {
    int const prev_rot = cur[res];
    int const n_rots = n_rotamers_for_res[pose][res];
    int const n_chunks = (n_rots - 1) / chunk_size + 1;
    int const offset = oneb_offsets[pose][res];
    int const new_chunk = new_rot / chunk_size;
    int const prev_chunk = prev_rot / chunk_size;
    int const new_in_chunk = new_rot - chunk_size * new_chunk;
    int const prev_in_chunk = prev_rot - chunk_size * prev_chunk;
    int const new_chunk_size =
        std::min(chunk_size, n_rots - chunk_size * new_chunk);
    int const prev_chunk_size =
        std::min(chunk_size, n_rots - chunk_size * prev_chunk);

    double d = energy1b[offset + new_rot] - energy1b[offset + prev_rot];
    for (Neighbor const& nb : neighbors[pose][res]) {
      int const k_rot = cur[nb.residue];
      if (k_rot < 0) {
        continue;
      }
      int const k_chunk = k_rot / chunk_size;
      int const k_in_chunk = k_rot - k_chunk * chunk_size;
      int64_t const base = nb.chunk_offset_offset + k_chunk * n_chunks;
      int64_t const new_offset = chunk_offsets[base + new_chunk];
      int64_t const prev_offset = chunk_offsets[base + prev_chunk];
      if (new_offset >= 0) {
        d += energy2b[new_offset + k_in_chunk * new_chunk_size + new_in_chunk];
      }
      if (prev_offset >= 0) {
        d -= energy2b
            [prev_offset + k_in_chunk * prev_chunk_size + prev_in_chunk];
      }
    }
    return d;
  }

  // Metropolis annealing with geometric cooling from hi toward lo; best holds
  // the lowest-energy assignment seen and must equal cur on entry.
  void anneal(
      int pose,
      std::mt19937_64& rng,
      Assignment cur,
      Assignment best,
      float hi,
      float lo,
      int n_outer,
      int n_inner) const {
    int const n_res = pose_n_res[pose];
    int const n_rots = n_rotamers_for_pose[pose];
    int const rot_offset = rotamer_offset_for_pose[pose];
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

    double current_e = total(pose, cur);
    double best_e = current_e;
    int n_steps = 0;
    float temperature = hi;
    for (int i = 0; i < n_outer; ++i) {
      for (int j = 0; j < n_inner; ++j) {
        int const res = res_for_rot[random_below(rng, n_rots) + rot_offset];
        int const new_rot = random_below(rng, n_rotamers_for_res[pose][res]);
        float const accept_rand = uniform(rng);
        if (new_rot == cur[res]) {
          continue;
        }
        double const d = delta(pose, cur, res, new_rot);
        if (d < 0 || accept_rand < std::exp(-d / temperature)) {
          cur[res] = new_rot;
          current_e += d;
          if (current_e < best_e) {
            best_e = current_e;
            for (int k = 0; k < n_res; ++k) {
              best[k] = cur[k];
            }
          }
        }
        if (++n_steps > 10000) {
          n_steps = 0;
          current_e = total(pose, cur);
        }
      }
      temperature = FunnelSchedule::cooling_factor * (temperature - lo) + lo;
    }
  }

  // One greedy pass over the pose's rotamers in random order; lite visits
  // every quench_lite_stride-th rotamer of each residue.
  void quench(
      int pose,
      std::mt19937_64& rng,
      Assignment cur,
      std::vector<std::pair<int, int>>& order,
      bool lite) const {
    int const stride = lite ? FunnelSchedule::quench_lite_stride : 1;
    order.clear();
    for (int res = 0; res < pose_n_res[pose]; ++res) {
      for (int rot = 0; rot < n_rotamers_for_res[pose][res]; rot += stride) {
        order.emplace_back(res, rot);
      }
    }
    std::shuffle(order.begin(), order.end(), rng);
    for (auto [res, rot] : order) {
      if (rot != cur[res] && delta(pose, cur, res, rot) < 0) {
        cur[res] = rot;
      }
    }
  }
};

// Each pose's trajectory indices, lowest score first.
template <tmol::Device D>
std::vector<std::vector<int>> rank_trajectories(
    TView<float, 2, D> scores, int n_poses, int n_traj) {
  std::vector<std::vector<int>> ranked(n_poses, std::vector<int>(n_traj));
  for (int pose = 0; pose < n_poses; ++pose) {
    std::iota(ranked[pose].begin(), ranked[pose].end(), 0);
    std::stable_sort(
        ranked[pose].begin(), ranked[pose].end(), [&](int a, int b) {
          return scores[pose][a] < scores[pose][b];
        });
  }
  return ranked;
}

// High-temperature trajectories, the best seeding low-temperature ones, the
// best of those fully quenched; trajectories run in parallel.
template <typename ParallelFor>
void run_funnel(
    FunnelAnnealer<Device::CPU> const& a,
    CpuFunnelWorkspace& w,
    TView<float, 2, Device::CPU> scores,
    TView<int32_t, 3, Device::CPU> assignments,
    uint64_t seed,
    ParallelFor parallel_for) {
  int const n_poses = scores.size(0), n_hi = w.counts.high;
  using S = FunnelSchedule;
  int const max_n_res = a.max_n_res;
  int const n_lo = w.counts.low;
  int const n_fq = w.counts.final;

  std::mt19937_64 master(seed);
  uint64_t const seeds[3] = {master(), master(), master()};

  auto for_each_trajectory = [&](int n_traj, auto fn) {
    parallel_for(
        0, int64_t(n_poses) * n_traj, 1, [&](int64_t begin, int64_t end) {
          std::vector<std::pair<int, int>> order;
          for (int64_t i = begin; i < end; ++i) {
            fn(int(i / n_traj), int(i % n_traj), order);
          }
        });
  };
  auto copy = [&](int pose, auto from, auto to) {
    for (int r = 0; r < a.pose_n_res[pose]; ++r) {
      to[r] = from[r];
    }
  };

  auto hi_cur = w.hi_cur;
  auto hi_best = w.hi_best;
  auto hi_score = w.hi_score;
  for_each_trajectory(n_hi, [&](int pose, int t, auto& order) {
    auto rng = trajectory_rng(seeds[0], pose, t);
    for (int r = 0; r < a.pose_n_res[pose]; ++r) {
      int const n = a.n_rotamers_for_res[pose][r];
      int const rot = n == 0 ? -1 : random_below(rng, n);
      hi_cur[pose][t][r] = rot;
      hi_best[pose][t][r] = rot;
    }
    int const n_rots = a.n_rotamers_for_pose[pose];
    if (n_rots == 0) {
      return;
    }
    a.anneal(
        pose,
        rng,
        hi_cur[pose][t],
        hi_best[pose][t],
        S::high_temp_initial,
        S::low_temp_initial,
        S::outer_hitemp,
        int(n_rots * S::inner_mult_hitemp));
    // The unquenched best seeds the next phase; its quench-lite ranks it.
    copy(pose, hi_best[pose][t], hi_cur[pose][t]);
    a.quench(pose, rng, hi_best[pose][t], order, true);
    hi_score[pose][t] = a.total(pose, hi_best[pose][t]);
  });
  auto hi_rank = rank_trajectories(hi_score, n_poses, n_hi);

  auto lo_cur = w.lo_cur;
  auto lo_best = w.lo_best;
  auto lo_score = w.lo_score;
  for_each_trajectory(n_lo, [&](int pose, int t, auto& order) {
    auto rng = trajectory_rng(seeds[1], pose, t);
    int const source = hi_rank[pose][t / S::lotemp_expansions];
    copy(pose, hi_cur[pose][source], lo_cur[pose][t]);
    copy(pose, hi_cur[pose][source], lo_best[pose][t]);
    int const n_rots = a.n_rotamers_for_pose[pose];
    if (n_rots == 0) {
      return;
    }
    a.anneal(
        pose,
        rng,
        lo_cur[pose][t],
        lo_best[pose][t],
        S::high_temp_later,
        S::low_temp_later,
        S::outer_lotemp,
        int(n_rots * S::inner_mult_lotemp));
    copy(pose, lo_best[pose][t], lo_cur[pose][t]);
    a.quench(pose, rng, lo_cur[pose][t], order, true);
    lo_score[pose][t] = a.total(pose, lo_cur[pose][t]);
  });
  auto lo_rank = rank_trajectories(lo_score, n_poses, n_lo);

  auto fq = w.fq;
  auto fq_score = w.fq_score;
  for_each_trajectory(n_fq, [&](int pose, int t, auto& order) {
    auto rng = trajectory_rng(seeds[2], pose, t);
    copy(pose, lo_cur[pose][lo_rank[pose][t]], fq[pose][t]);
    if (a.n_rotamers_for_pose[pose] == 0) {
      return;
    }
    a.quench(pose, rng, fq[pose][t], order, false);
    fq_score[pose][t] = a.total(pose, fq[pose][t]);
  });
  auto fq_rank = rank_trajectories(fq_score, n_poses, n_fq);

  for (int pose = 0; pose < n_poses; ++pose) {
    for (int t = 0; t < n_fq; ++t) {
      int const source = fq_rank[pose][t];
      scores[pose][t] = fq_score[pose][source];
      copy(pose, fq[pose][source], assignments[pose][t]);
    }
  }
}

}  // namespace detail
template <typename Context, typename ParallelFor>
void run_cpu_funnel(
    Context const& c,
    CpuFunnelWorkspace& w,
    uint64_t seed,
    ParallelFor parallel_for) {
  constexpr Device D = Device::CPU;
  int const poses = c.pose_n_res.size(0),
            residues = c.n_rotamers_for_res.size(1);
  if (c.scores.size(0) != poses || c.scores.size(1) != w.counts.final
      || c.best_rotamer_assignments.size(0) != poses
      || c.best_rotamer_assignments.size(1) != w.counts.final
      || c.best_rotamer_assignments.size(2) != residues
      || w.hi_cur.size(0) != poses || w.hi_cur.size(2) != residues)
    throw std::invalid_argument("CPU funnel output/workspace axes differ");
  if (!poses) return;
  if (!w.data) throw std::invalid_argument("CPU funnel requires workspace");
  std::memset(w.data, 0, w.bytes);
  std::memset(
      c.scores.data(), 0, int64_t(poses) * w.counts.final * sizeof(float));
  if (residues)
    std::memset(
        c.best_rotamer_assignments.data(),
        0,
        int64_t(poses) * w.counts.final * residues * sizeof(int32_t));
  detail::FunnelAnnealer<D> a{
      c.pose_n_res,
      c.n_rotamers_for_pose,
      c.rotamer_offset_for_pose,
      c.n_rotamers_for_res,
      c.oneb_offsets,
      c.res_for_rot,
      c.chunk_size,
      residues,
      c.neighbor_row_offsets,
      c.neighbor_blocks,
      c.neighbor_chunk_offset_offsets,
      c.chunk_offsets,
      c.energy1b,
      c.energy2b,
      {}};
  for (int pose = 0; pose < poses; ++pose)
    a.neighbors.push_back(
        detail::pose_neighbors(
            pose,
            c.pose_n_res[pose],
            residues,
            c.neighbor_row_offsets,
            c.neighbor_blocks,
            c.neighbor_chunk_offset_offsets));
  detail::run_funnel(
      a, w, c.scores, c.best_rotamer_assignments, seed, parallel_for);
}
}  // namespace tmol::pack::compiled
