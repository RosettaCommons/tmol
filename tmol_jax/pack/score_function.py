"""Compose shared native sparse physics using frozen score configuration."""

from functools import partial, lru_cache, cached_property

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.score.score_types import ScoreType
from tmol_jax.score.ref.ref_energy_term import RefEnergyTerm
from .scoring import (
    validate_rotamer_context,
    prepare_rotamer_nonbonded,
    nonbonded_candidate_groups,
    sparse_nonbonded_values,
)
from .bonded_scoring import (
    connected_candidates,
    prepare_rotamer_genbonded,
    prepare_rotamer_cartbonded,
    sparse_bonded_scores,
)
from .backbone_scoring import prepare_rotamer_backbone
from .connection_scoring import (
    prepare_rotamer_disulfide,
    prepare_rotamer_metal,
    sparse_connection_scores,
)
from .na_scoring import prepare_rotamer_na_torsion, sparse_na_torsion_scores
from .dunbrack_scoring import prepare_rotamer_dunbrack, sparse_dunbrack_scores
from .constraint_scoring import prepare_rotamer_constraints, sparse_constraint_scores
from .hbond_scoring import prepare_rotamer_hbond, sparse_hbond_values
from .lk_ball_scoring import prepare_rotamer_lk_ball, sparse_lk_ball_values

# The owning energy term supplies channel order; names select execution only.
_EXECUTION = {
    ScoreType.fa_ljatr: (prepare_rotamer_nonbonded, sparse_nonbonded_values, True),
    ScoreType.fa_elec: (
        partial(prepare_rotamer_nonbonded, electrostatic=True),
        sparse_nonbonded_values,
        True,
    ),
    ScoreType.hbond: (prepare_rotamer_hbond, sparse_hbond_values, True),
    ScoreType.lk_ball_iso: (prepare_rotamer_lk_ball, sparse_lk_ball_values, True),
    ScoreType.cart_lengths: (prepare_rotamer_cartbonded, sparse_bonded_scores, False),
    ScoreType.gen_torsions: (prepare_rotamer_genbonded, sparse_bonded_scores, False),
    ScoreType.rama: (prepare_rotamer_backbone, sparse_bonded_scores, False),
    ScoreType.disulfide: (prepare_rotamer_disulfide, sparse_connection_scores, False),
    ScoreType.metal_coordination: (
        prepare_rotamer_metal,
        sparse_connection_scores,
        False,
    ),
    ScoreType.dunbrack_rot: (prepare_rotamer_dunbrack, sparse_dunbrack_scores, False),
    ScoreType.na_torsion: (prepare_rotamer_na_torsion, sparse_na_torsion_scores, False),
    ScoreType.constraint: (
        prepare_rotamer_constraints,
        sparse_constraint_scores,
        False,
    ),
}


def _reference(coords, state):
    values, indices = state
    return values[None, :].astype(coords.dtype), indices


@partial(jax.jit, static_argnums=(0,))
def _packing_entry_indices(evaluate, coords, arrays):
    # Keep the existing term's identity construction. XLA can discard its
    # unconsumed energy computation during the two topology-only passes.
    return evaluate(coords, arrays, (None,))[1]


class RotamerScorer:
    """Eager candidate selection with a separately transformable evaluator.

    ``prepare`` fixes the neighbor identities at supplied coordinates. The
    resulting callable provides local coordinate gradients; arbitrary moves
    require another preparation to discover newly interacting conformers.
    """

    def __init__(self, score_function, pose, rotamers, sum_terms, apply_weights):
        validate_rotamer_context(pose, rotamers)
        self.rotamers = rotamers
        self.packed_block_types = pose.packed_block_types
        self.score_types = tuple(score_function.all_score_types())
        self.sum_terms = sum_terms
        self.apply_weights = apply_weights
        self.shape = rotamers.coords.shape
        entries = []
        bonded_candidates = None
        with jax.default_device(pose.device):
            for term in score_function.terms:
                provided = tuple(term.score_types_provided())
                weights = tuple(score_function.get_weight(st) for st in provided)
                if not any(weights):
                    continue
                selected = tuple(
                    i for i, st in enumerate(provided) if st in self.score_types
                )
                if provided == (ScoreType.ref,):
                    ref = RefEnergyTerm(score_function.param_db)
                    ref.set_options(score_function._options)
                    per_type = jax.device_put(
                        ref.block_type_values(pose.packed_block_types),
                        device=pose.device,
                    )
                    ids = jnp.arange(rotamers.n_rotamers_total, dtype=jnp.int32)
                    state = (
                        per_type[rotamers.block_type_ind_for_rot],
                        jnp.stack((rotamers.pose_for_rot, ids, ids)),
                    )
                    evaluate, spatial = _reference, False
                else:
                    prepare, evaluate, spatial = _EXECUTION[provided[0]]
                    options = (
                        {
                            "soft_repulsive": score_function._options.get(
                                "soft_rep", False
                            )
                        }
                        if provided[0] == ScoreType.fa_ljatr
                        else {}
                    )
                    if provided[0] in (
                        ScoreType.dunbrack_rot,
                        ScoreType.cart_lengths,
                        ScoreType.gen_torsions,
                        ScoreType.rama,
                    ):
                        options["term"] = term
                    if provided[0] in (ScoreType.cart_lengths, ScoreType.gen_torsions):
                        if bonded_candidates is None:
                            bonded_candidates = connected_candidates(pose, rotamers)
                        options["candidates"] = bonded_candidates
                    state = prepare(pose, rotamers, score_function.param_db, **options)
                entries.append((provided, selected, weights, evaluate, state, spatial))
        self.entries = tuple(entries)
        self._evaluate, self._arrays = self._compiled_evaluator(self.entries)

    def _validate_coordinates(self, coords):
        if isinstance(coords, jax.core.Tracer):
            raise ValueError("prepare sparse candidates outside JAX transformations")
        if coords.shape != self.shape or not np.isfinite(np.asarray(coords)).all():
            raise ValueError(f"expected finite coordinates with shape {self.shape}")
        if coords.device != self.rotamers.device:
            raise ValueError("coordinates and conformers must be on the same device")

    def prepare(self, coords):
        """Select discrete neighbors eagerly and return values/indices callable."""
        self._validate_coordinates(coords)
        reaches = tuple(
            dict.fromkeys(entry[4][3] for entry in self.entries if entry[5])
        )
        with jax.default_device(self.rotamers.device):
            candidates = (
                dict(
                    zip(
                        reaches,
                        nonbonded_candidate_groups(
                            self.rotamers,
                            self.packed_block_types,
                            reaches=reaches,
                            coords=coords,
                        ),
                    )
                )
                if reaches
                else {}
            )
            prepared_indices = tuple(
                candidates[state[3]] if spatial else None
                for *_, state, spatial in self.entries
            )
        return partial(
            self._evaluate, arrays=self._arrays, candidates=tuple(prepared_indices)
        )

    def _compiled_evaluator(self, entries, *, paged=False):
        # Floating tables remain explicit device operands, avoiding duplicate
        # embedded constants while compiling all runtime tensor operations once.
        leaves, tree = jax.tree.flatten(tuple(entry[4] for entry in entries))
        positions = tuple(
            i for i, leaf in enumerate(leaves) if isinstance(leaf, jax.Array)
        )
        arrays = tuple(leaves[i] for i in positions)
        template = tuple(
            None if i in positions else leaf for i, leaf in enumerate(leaves)
        )
        compile_program = (
            _compiled_page_execution if paged else _compiled_rotamer_execution
        )
        return (
            compile_program(
                tuple(entry[:4] for entry in entries),
                self.score_types,
                tree,
                template,
                positions,
                self.sum_terms,
                self.apply_weights,
                self.shape,
            ),
            arrays,
        )

    @cached_property
    def _packing_executions(self):
        return tuple(
            self._compiled_evaluator((entry,), paged=True) for entry in self.entries
        )

    def iter_packing_entries(
        self, coords, *, block_pair_window, score_window, topology_only=False
    ):
        """Yield explicit bounded spatial pages through the native evaluators.

        This eager packing adapter does not select a memory policy or construct
        an interaction graph. Spatial candidates are limited by block-pair
        windows and scored in smaller pages; nonspatial terms retain their
        existing native layouts. Consume and release each page before advancing.
        Correlated choices require the ordinary complete-table path.
        """
        self._validate_coordinates(coords)
        for size in (block_pair_window, score_window):
            if (
                isinstance(size, bool)
                or not isinstance(size, (int, np.integer))
                or size < 1
            ):
                raise ValueError("packing page sizes must be positive integers")
        if self.rotamers.correlated_groups:
            raise ValueError("correlated packing requires the complete score table")
        from tmol_core.pack.streaming import packing_windows

        poses, blocks = self.rotamers.n_rots_for_block.shape
        count = poses * blocks * (blocks + 1) // 2
        for entry, (evaluate, arrays) in zip(self.entries, self._packing_executions):
            if not entry[5]:
                if topology_only:
                    indices = _packing_entry_indices(evaluate, coords, arrays)
                    values = coords.reshape(-1)[:0]
                else:
                    values, indices = evaluate(coords, arrays, (None,))
                for start, size in packing_windows(indices.shape[1], score_window):
                    stop = start + size
                    yield (
                        values[..., :0] if topology_only else values[..., start:stop],
                        indices[:, start:stop],
                    )
                del values, indices
                continue
            for begin, size in packing_windows(count, block_pair_window):
                indices = nonbonded_candidate_groups(
                    self.rotamers,
                    self.packed_block_types,
                    reaches=(entry[4][3],),
                    coords=coords,
                    begin=begin,
                    count=size,
                )[0]
                for start, size in packing_windows(indices.shape[1], score_window):
                    page = indices[:, start : start + size]
                    if topology_only:
                        yield coords.reshape(-1)[:0], page
                    else:
                        yield evaluate(coords, arrays, (page,))
                    del page
                del indices

    def __call__(self, coords):
        return self.prepare(coords)(coords)


@lru_cache(maxsize=8)
def _compiled_rotamer_execution(
    entries,
    channels,
    tree,
    template,
    positions,
    sum_terms,
    apply_weights,
    shape,
):
    """Reuse programs across renders; all topology and parameter arrays are operands."""

    @jax.jit
    def score(x, arrays, candidates):
        if x.shape != shape or x.dtype not in (jnp.float32, jnp.float64):
            raise ValueError(f"expected float32/64 coordinates with shape {shape}")
        state_leaves = list(template)
        for i, value in zip(positions, arrays):
            state_leaves[i] = value
        states = tree.unflatten(state_leaves)
        all_values, all_indices = [], []
        for entry, state, indices in zip(entries, states, candidates):
            provided, selected, weights, evaluate = entry[:4]
            if indices is None:
                values, ids = evaluate(x, state)
            else:
                values, ids = evaluate(x, state, indices), indices
            values = values[jnp.asarray(selected)]
            if apply_weights:
                values *= jnp.asarray([weights[i] for i in selected], x.dtype)[:, None]
            if sum_terms:
                values = values.sum(axis=0)
            else:
                lanes = jnp.asarray([channels.index(provided[i]) for i in selected])
                values = (
                    jnp.zeros((len(channels), values.shape[1]), x.dtype)
                    .at[lanes]
                    .set(values)
                )
            all_values.append(values)
            all_indices.append(ids)
        if not all_values:
            return (
                jnp.zeros((0,) if sum_terms else (0, 0), x.dtype),
                jnp.zeros((3, 0), jnp.int32),
            )
        return jnp.concatenate(all_values, axis=-1), jnp.concatenate(
            all_indices, axis=1
        )

    return score


# A preset has more families than the complete-scorer cache has slots. Separate
# page programs so repeated packing does not evict/recompile every family.
_compiled_page_execution = lru_cache(maxsize=128)(
    _compiled_rotamer_execution.__wrapped__
)
