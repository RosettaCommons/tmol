#include <tmol/utility/portable/math.hh>
#include <tmol/utility/tensor/native.hh>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/context_manager.hh>

#include <tmol/backends/cpu/score/common/device_operations.cpu.impl.hh>

#include <tmol/pack/compiled/simulated_annealing.hh>
#include <tmol/pack/compiled/compiled.impl.hh>
#include <tmol/pack/compiled/streaming_interaction_graph.impl.hh>

#include <algorithm>
#include <mutex>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

namespace tmol {
namespace pack {
namespace compiled {

namespace {
// Seeded from torch's default CPU generator, as the CUDA annealer draws from
// torch's CUDA generator, so torch.manual_seed makes either reproducible.
std::mt19937_64 annealer_rng() {
  auto seed = native::random_seed_tensor({}, native::Device(native::kCPU));
  return std::mt19937_64(static_cast<uint64_t>(seed.item<int64_t>()));
}

int random_below(std::mt19937_64& rng, int n) {
  return static_cast<int>(rng() % static_cast<uint64_t>(n));
}

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

std::mt19937_64 trajectory_rng(uint64_t seed, int pose, int traj) {
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
template <tmol::Device D>
auto run_funnel(FunnelAnnealer<D> const& a, int n_poses, int n_hi)
    -> std::tuple<TPack<float, 2, D>, TPack<int, 3, D>> {
  using S = FunnelSchedule;
  int const max_n_res = a.max_n_res;
  int const n_lo = int(n_hi * S::lotemp_expansions * S::round1_cut);
  int const n_fq = int(n_lo * S::round2_cut);

  auto master = annealer_rng();
  uint64_t const seeds[3] = {master(), master(), master()};

  auto for_each_trajectory = [&](int n_traj, auto fn) {
    native::parallel_for(
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

  auto hi_cur_t = TPack<int, 3, D>::zeros({n_poses, n_hi, max_n_res});
  auto hi_best_t = TPack<int, 3, D>::zeros({n_poses, n_hi, max_n_res});
  auto hi_score_t = TPack<float, 2, D>::zeros({n_poses, n_hi});
  auto hi_cur = hi_cur_t.view;
  auto hi_best = hi_best_t.view;
  auto hi_score = hi_score_t.view;
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

  auto lo_cur_t = TPack<int, 3, D>::zeros({n_poses, n_lo, max_n_res});
  auto lo_best_t = TPack<int, 3, D>::zeros({n_poses, n_lo, max_n_res});
  auto lo_score_t = TPack<float, 2, D>::zeros({n_poses, n_lo});
  auto lo_cur = lo_cur_t.view;
  auto lo_best = lo_best_t.view;
  auto lo_score = lo_score_t.view;
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

  auto fq_t = TPack<int, 3, D>::zeros({n_poses, n_fq, max_n_res});
  auto fq_score_t = TPack<float, 2, D>::zeros({n_poses, n_fq});
  auto fq = fq_t.view;
  auto fq_score = fq_score_t.view;
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

  auto scores_t = TPack<float, 2, D>::zeros({n_poses, n_fq});
  auto assignments_t = TPack<int, 3, D>::zeros({n_poses, n_fq, max_n_res});
  auto scores = scores_t.view;
  auto assignments = assignments_t.view;
  for (int pose = 0; pose < n_poses; ++pose) {
    for (int t = 0; t < n_fq; ++t) {
      int const source = fq_rank[pose][t];
      scores[pose][t] = fq_score[pose][source];
      copy(pose, fq[pose][source], assignments[pose][t]);
    }
  }
  return {scores_t, assignments_t};
}
}  // namespace

template <tmol::Device D>
void set_quench_order(
    TView<int, 1, D> quench_order,
    int const n_rots,
    int const pose_rotamer_offset,
    std::mt19937_64& rng) {
  // Create a random permutation of all the rotamers
  // and visit them in this order to ensure all of them
  // are seen during the quench step
  for (int i = 0; i < n_rots; ++i) {
    quench_order[i] = i + pose_rotamer_offset;
  }
  for (int i = 0; i <= n_rots - 2; ++i) {
    int j = i + random_below(rng, n_rots - i);
    // swap i and j;
    int jval = quench_order[j];
    quench_order[j] = quench_order[i];
    quench_order[i] = jval;
  }
}

template <tmol::Device D>
auto AnnealerDispatch<D>::forward(
    ContextManager&,
    int max_n_rotamers_per_pose,
    TView<int, 1, D> pose_n_res,               // n-poses
    TView<int, 1, D> n_rotamers_for_pose,      // n-poses
    TView<int, 1, D> rotamer_offset_for_pose,  // n-poses
    TView<int, 2, D> n_rotamers_for_res,       // n-poses x max-n-res
    TView<int, 2, D> oneb_offsets,             // n-poses x max-n-res
    TView<int, 1, D> res_for_rot,              // n-rots
    int32_t chunk_size,
    TView<int64_t, 1, D> neighbor_row_offsets,
    TView<int32_t, 1, D> neighbor_blocks,
    TView<int64_t, 1, D> neighbor_chunk_offset_offsets,
    TView<int64_t, 1, D> chunk_offsets,  // n-chunks-on-interacting-res
    TView<float, 1, D> energy1b,
    TView<float, 1, D> energy2b,
    int n_hitemp_traj) -> std::tuple<TPack<float, 2, D>, TPack<int, 3, D>> {
  int const n_poses = pose_n_res.size(0);
  int const max_n_res = n_rotamers_for_res.size(1);
  int const n_rotamers = res_for_rot.size(0);

  if (native::get_num_threads() >= FunnelSchedule::min_threads) {
    FunnelAnnealer<D> annealer{
        pose_n_res,
        n_rotamers_for_pose,
        rotamer_offset_for_pose,
        n_rotamers_for_res,
        oneb_offsets,
        res_for_rot,
        chunk_size,
        max_n_res,
        neighbor_row_offsets,
        neighbor_blocks,
        neighbor_chunk_offset_offsets,
        chunk_offsets,
        energy1b,
        energy2b,
        {}};
    for (int pose = 0; pose < n_poses; ++pose) {
      annealer.neighbors.push_back(pose_neighbors(
          pose,
          pose_n_res[pose],
          max_n_res,
          neighbor_row_offsets,
          neighbor_blocks,
          neighbor_chunk_offset_offsets));
    }
    return run_funnel(annealer, n_poses, n_hitemp_traj);
  }

  int n_traj = 1;
  int const n_outer_iterations = 20;
  // Rosetta uses 5x; our old value of 20x was 4x too many inner iterations
  int const n_inner_iterations_factor = 5;

  auto scores_t = TPack<float, 2, D>::zeros({n_poses, n_traj});
  auto current_rotamer_assignments_t =
      TPack<int, 3, D>::zeros({n_poses, n_traj, max_n_res});
  auto best_rotamer_assignments_t =
      TPack<int, 3, D>::zeros({n_poses, n_traj, max_n_res});
  auto quench_order_t = TPack<int, 1, D>::zeros({n_rotamers});

  auto scores = scores_t.view;
  auto current_rotamer_assignments = current_rotamer_assignments_t.view;
  auto best_rotamer_assignments = best_rotamer_assignments_t.view;
  auto quench_order = quench_order_t.view;

  float const high_temp = 100;
  float const low_temp = 0.3;  // matches Rosetta SimAnnealerBase::lowtemp

  auto rng = annealer_rng();
  std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

  for (int pose = 0; pose < n_poses; ++pose) {
    int const n_res = pose_n_res[pose];
    int const pose_n_rotamers = n_rotamers_for_pose[pose];
    int const pose_rotamer_offset = rotamer_offset_for_pose[pose];

    auto const neighbors = pose_neighbors(
        pose,
        n_res,
        max_n_res,
        neighbor_row_offsets,
        neighbor_blocks,
        neighbor_chunk_offset_offsets);
    int const n_inner_iterations = n_inner_iterations_factor * pose_n_rotamers;

    for (int traj = 0; traj < n_traj; ++traj) {
      // Initial assignment: assign a rotamer to every residue
      for (int i = 0; i < n_res; ++i) {
        int const i_n_rots = n_rotamers_for_res[pose][i];
        if (i_n_rots == 0) {
          current_rotamer_assignments[pose][traj][i] = -1;
          best_rotamer_assignments[pose][traj][i] = -1;
        } else {
          int rand_rot = random_below(rng, i_n_rots);
          current_rotamer_assignments[pose][traj][i] = rand_rot;
          best_rotamer_assignments[pose][traj][i] = rand_rot;
        }
      }

      float temperature = high_temp;
      double best_energy = total_energy_for_assignment(
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
          current_rotamer_assignments[pose][traj]);
      double current_total_energy = best_energy;
      int naccepts = 0;
      for (int i = 0; i < n_outer_iterations; ++i) {
        bool quench = false;
        if (i == n_outer_iterations - 1) {
          quench = true;
          temperature = 0;
          for (int j = 0; j < n_res; ++j) {
            current_rotamer_assignments[pose][traj][j] =
                best_rotamer_assignments[pose][traj][j];
          }
          current_total_energy = total_energy_for_assignment(
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
              current_rotamer_assignments[pose][traj]);
        }

        for (int j = 0; j < n_inner_iterations; ++j) {
          int global_ran_rot;
          if (quench) {
            if (j % pose_n_rotamers == 0) {
              // Reset the quench order every pose_n_rotamers iterations
              // as it will attempt all pose_n_rotamers in a particular
              // random order before starting over with a different
              // random order.
              set_quench_order(
                  quench_order, pose_n_rotamers, pose_rotamer_offset, rng);
            }
            global_ran_rot = quench_order[j % pose_n_rotamers];
          } else {
            global_ran_rot =
                random_below(rng, pose_n_rotamers) + pose_rotamer_offset;
          }

          int const ran_res = res_for_rot[global_ran_rot];
          int const local_prev_rot =
              current_rotamer_assignments[pose][traj][ran_res];
          int const ran_res_n_rots = n_rotamers_for_res[pose][ran_res];
          int const ran_res_n_chunks = (ran_res_n_rots - 1) / chunk_size + 1;
          int const ran_res_offset = oneb_offsets[pose][ran_res];
          int const local_ran_rot = global_ran_rot - ran_res_offset;
          int const ran_rot_chunk = local_ran_rot / chunk_size;
          int const prev_rot_chunk = local_prev_rot / chunk_size;
          int const ran_rot_in_chunk =
              local_ran_rot - chunk_size * ran_rot_chunk;
          int const prev_rot_in_chunk =
              local_prev_rot - chunk_size * prev_rot_chunk;
          int const ran_rot_chunk_size =
              std::min(chunk_size, ran_res_n_rots - chunk_size * ran_rot_chunk);
          int const prev_rot_chunk_size = std::min(
              chunk_size, ran_res_n_rots - chunk_size * prev_rot_chunk);
          int const global_prev_rot = local_prev_rot + ran_res_offset;

          double prev_e = energy1b[global_prev_rot];
          double deltaE = energy1b[global_ran_rot] - prev_e;

          for (Neighbor const& neighbor : neighbors[ran_res]) {
            int const k = neighbor.residue;
            int const local_k_rot = current_rotamer_assignments[pose][traj][k];
            int const krot_chunk = local_k_rot / chunk_size;
            int const krot_in_chunk = local_k_rot - krot_chunk * chunk_size;

            double k_new_e = 0;
            double k_prev_e = 0;

            // chunk_offset_offsets stores both orderings (k,ran_res) and
            // (ran_res,k), so always index as [k][ran_res] with k as the
            // outer/row dimension.
            int64_t const k_ran_chunk_offset_offset =
                neighbor.chunk_offset_offset;
            int64_t const krot_ranrot_chunk_offset = chunk_offsets
                [k_ran_chunk_offset_offset + krot_chunk * ran_res_n_chunks
                 + ran_rot_chunk];
            int64_t const krot_prevrot_chunk_offset = chunk_offsets
                [k_ran_chunk_offset_offset + krot_chunk * ran_res_n_chunks
                 + prev_rot_chunk];
            if (krot_ranrot_chunk_offset >= 0) {
              k_new_e = energy2b
                  [krot_ranrot_chunk_offset + krot_in_chunk * ran_rot_chunk_size
                   + ran_rot_in_chunk];
            }
            if (krot_prevrot_chunk_offset >= 0) {
              k_prev_e = energy2b
                  [krot_prevrot_chunk_offset
                   + krot_in_chunk * prev_rot_chunk_size + prev_rot_in_chunk];
            }

            deltaE += k_new_e - k_prev_e;
            prev_e += k_prev_e;
          }

          float const uniform_random = uniform(rng);

          if (pass_metropolis(
                  temperature, uniform_random, deltaE, prev_e, quench)) {
            current_rotamer_assignments[pose][traj][ran_res] = local_ran_rot;
            current_total_energy += deltaE;
            ++naccepts;
            if (naccepts > 1000) {
              naccepts = 0;
              float new_current_total_energy = total_energy_for_assignment(
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
                  current_rotamer_assignments[pose][traj]);
              current_total_energy = new_current_total_energy;
            }
            if (current_total_energy < best_energy) {
              for (int k = 0; k < n_res; ++k) {
                best_rotamer_assignments[pose][traj][k] =
                    current_rotamer_assignments[pose][traj][k];
              }
              best_energy = current_total_energy;
            }
          }

        }  // end inner loop

        // geometric cooling toward the low temperature
        // e.g. if towards 0.3, then:
        // temperature = 0.35 * (temperature - 0.3) + 0.3;
        temperature =
            (high_temp - low_temp) * tmol::portable::math::exp(-1 * (i + 1))
            + low_temp;

      }  // end outer loop

      scores[pose][traj] = total_energy_for_assignment(
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
          best_rotamer_assignments[pose][traj]);
    }  // end trajectory loop
  }  // end pose loop

  return {scores_t, best_rotamer_assignments_t};
}

template struct AnnealerDispatch<tmol::Device::CPU>;

template struct InteractionGraphBuilder<
    score::common::DeviceOperations,
    tmol::Device::CPU,
    float,
    int64_t>;
template struct InteractionGraphBuilder<
    score::common::DeviceOperations,
    tmol::Device::CPU,
    double,
    int64_t>;
template struct StreamingInteractionGraph<
    score::common::DeviceOperations,
    tmol::Device::CPU,
    float,
    int64_t>;
template struct StreamingInteractionGraph<
    score::common::DeviceOperations,
    tmol::Device::CPU,
    double,
    int64_t>;

}  // namespace compiled
}  // namespace pack
}  // namespace tmol
