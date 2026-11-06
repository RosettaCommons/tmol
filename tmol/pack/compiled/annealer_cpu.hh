#pragma once

#include <tmol/pack/compiled/simulated_annealing.hh>
#include <random>
#include <vector>

namespace tmol::pack::compiled {

// The mature CPU schedule uses one trajectory and float32 energy tables.
// Tensor buffers are caller-owned and zero-initialized, including padded rows.
// Host neighbor vectors and RNG state remain local native scratch.
template <template <typename, int> class View>
struct CpuAnnealerContext {
  View<int32_t, 1> pose_n_res, n_rotamers_for_pose, rotamer_offset_for_pose,
      res_for_rot, quench_order, neighbor_blocks;
  View<int32_t, 2> n_rotamers_for_res, oneb_offsets;
  View<int64_t, 1> neighbor_row_offsets, neighbor_chunk_offset_offsets,
      chunk_offsets;
  View<float, 1> energy1b, energy2b;
  View<float, 2> scores;
  View<int32_t, 3> current_rotamer_assignments, best_rotamer_assignments;
  int chunk_size;
};

namespace detail {
inline int random_below(std::mt19937_64& rng, int n) {
  return static_cast<int>(rng() % static_cast<uint64_t>(n));
}
template <typename IndexView>
void set_quench_order(
    IndexView quench_order,
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

}  // namespace detail

template <typename Context>
void run_cpu_annealer(Context c, uint64_t seed) {
  auto pose_n_res = c.pose_n_res;
  auto n_rotamers_for_pose = c.n_rotamers_for_pose;
  auto rotamer_offset_for_pose = c.rotamer_offset_for_pose;
  auto res_for_rot = c.res_for_rot;
  auto quench_order = c.quench_order;
  auto n_rotamers_for_res = c.n_rotamers_for_res;
  auto oneb_offsets = c.oneb_offsets;
  auto neighbor_row_offsets = c.neighbor_row_offsets;
  auto neighbor_chunk_offset_offsets = c.neighbor_chunk_offset_offsets;
  auto chunk_offsets = c.chunk_offsets;
  auto energy1b = c.energy1b;
  auto energy2b = c.energy2b;
  auto scores = c.scores;
  auto current_rotamer_assignments = c.current_rotamer_assignments;
  auto best_rotamer_assignments = c.best_rotamer_assignments;
  auto neighbor_blocks = c.neighbor_blocks;
  int const chunk_size = c.chunk_size;
  int const n_poses = pose_n_res.size(0);
  int const max_n_res = n_rotamers_for_res.size(1);
  int const n_traj = 1, n_outer_iterations = 20;
  int const n_inner_iterations_factor = 5;
  float const high_temp = 100;
  float const low_temp = 0.3;  // matches Rosetta SimAnnealerBase::lowtemp

  struct Neighbor {
    int residue;
    int64_t chunk_offset_offset;
  };

  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

  // The initialization below assigns -1 to zero-count residues. Energy
  // traversal can use the true pose length and these checked assignments.
  for (int pose = 0; pose < n_poses; ++pose) {
    int const n_res = pose_n_res[pose];
    int const pose_n_rotamers = n_rotamers_for_pose[pose];
    int const pose_rotamer_offset = rotamer_offset_for_pose[pose];

    // Materialize the CSR rows as host vectors without changing neighbor order.
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
        neighbors[b].push_back(
            {neighbor, neighbor_chunk_offset_offsets[lower]});
      }
    }
    int const n_inner_iterations = n_inner_iterations_factor * pose_n_rotamers;

    for (int traj = 0; traj < n_traj; ++traj) {
      // Initial assignment: assign a rotamer to every residue
      for (int i = 0; i < n_res; ++i) {
        int const i_n_rots = n_rotamers_for_res[pose][i];
        if (i_n_rots == 0) {
          current_rotamer_assignments[pose][traj][i] = -1;
          best_rotamer_assignments[pose][traj][i] = -1;
        } else {
          int rand_rot = detail::random_below(rng, i_n_rots);
          current_rotamer_assignments[pose][traj][i] = rand_rot;
          best_rotamer_assignments[pose][traj][i] = rand_rot;
        }
      }

      float temperature = high_temp;
      double best_energy = total_energy_for_assignment<true>(
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
          current_rotamer_assignments[pose][traj],
          n_res);
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
          current_total_energy = total_energy_for_assignment<true>(
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
              current_rotamer_assignments[pose][traj],
              n_res);
        }

        for (int j = 0; j < n_inner_iterations; ++j) {
          int global_ran_rot;
          if (quench) {
            if (j % pose_n_rotamers == 0) {
              // Reset the quench order every pose_n_rotamers iterations
              // as it will attempt all pose_n_rotamers in a particular
              // random order before starting over with a different
              // random order.
              detail::set_quench_order(
                  quench_order, pose_n_rotamers, pose_rotamer_offset, rng);
            }
            global_ran_rot = quench_order[j % pose_n_rotamers];
          } else {
            global_ran_rot = detail::random_below(rng, pose_n_rotamers)
                             + pose_rotamer_offset;
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
              float new_current_total_energy =
                  total_energy_for_assignment<true>(
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
                      current_rotamer_assignments[pose][traj],
                      n_res);
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

      scores[pose][traj] = total_energy_for_assignment<true>(
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
          best_rotamer_assignments[pose][traj],
          n_res);
    }  // end trajectory loop
  }  // end pose loop
}

}  // namespace tmol::pack::compiled
