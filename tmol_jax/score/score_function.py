"""Weighted energy terms and explicitly rendered, reusable JAX scorers."""

from __future__ import annotations

import copy
import importlib
import math

import jax
import jax.numpy as jnp
import numpy as np

from .score_types import ScoreType
from ._execution import render_execution

from tmol_core.scoring.channels import TERM_CHANNELS, TERM_N_BODIES
from tmol_core.scoring.score_function import ScoreTermLifecycle, sorted_score_terms

_TERM_TYPES = {
    module: (name, TERM_CHANNELS[module])
    for module, name in {
        "ljlk": "LJLKEnergyTerm",
        "elec": "ElecEnergyTerm",
        "hbond": "HBondEnergyTerm",
        "cartbonded": "CartBondedEnergyTerm",
        "constraint": "ConstraintEnergyTerm",
        "disulfide": "DisulfideEnergyTerm",
        "backbone_torsion": "BackboneTorsionEnergyTerm",
        "dunbrack": "DunbrackEnergyTerm",
        "lk_ball": "LKBallEnergyTerm",
        "ref": "RefEnergyTerm",
        "genbonded": "GenBondedEnergyTerm",
        "metal": "MetalCoordinationEnergyTerm",
        "na_torsion": "NaTorsionEnergyTerm",
    }.items()
}
_TERM_FOR_TYPE = {
    ScoreType[name]: (module, class_name)
    for module, (class_name, names) in _TERM_TYPES.items()
    for name in names
}
# Stock executables retain array signatures and scalar options, never chemistry
# owners. Numeric state is supplied afresh by every rendered score.
_STOCK_EXECUTORS = {}

_FUSED_PAIR_TYPES = (
    ScoreType.fa_ljatr,
    ScoreType.fa_ljrep,
    ScoreType.fa_lk,
    ScoreType.fa_elec,
)


_NATIVE_CHANNEL_ORDER = [
    ScoreType[name] for names in TERM_CHANNELS.values() for name in names
]
_CHANNEL_BODIES = {
    ScoreType[name]: TERM_N_BODIES[group]
    for group, names in TERM_CHANNELS.items()
    for name in names
}


def _provided_channels(term):
    return sorted(term.score_types_provided(), key=_NATIVE_CHANNEL_ORDER.index)


def _term_bodies(term):
    if hasattr(term, "n_bodies"):
        return term.n_bodies()
    return _CHANNEL_BODIES[term.score_types_provided()[0]]


class ScoreFunction(ScoreTermLifecycle):
    """Mutable score configuration; render once for repeated coordinate scoring.

    Rendering snapshots weights and term options. Changing this configuration
    requires rendering again; an existing scorer keeps its original behavior.
    Topology and atom layout are fixed for each rendered scorer. Coordinates
    are its only input, so JAX can differentiate, compile and batch it normally.
    """

    def __init__(self, param_db, *, weight_dtype=np.float32):
        self.weight_dtype = np.dtype(weight_dtype)
        if self.weight_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("score weight storage requires float32 or float64")
        self._param_db = param_db
        self._weights = {}
        super().__init__(
            len(ScoreType), score_types=_provided_channels, n_bodies=_term_bodies
        )
        self._options = {}
        self._executors = {}

    @classmethod
    def from_sfxn_file(cls, path, param_db=None, *, weight_dtype=np.float32):
        """Load the shared versioned weights/options contract; poses select devices."""
        from tmol_core.scoring.score_function import load_score_function_record
        from tmol_jax.database import ParameterDatabase

        data = load_score_function_record(path)
        score = cls(
            ParameterDatabase.get_default() if param_db is None else param_db,
            weight_dtype=weight_dtype,
        )
        for name, weight in data["weights"].items():
            score.set_weight(getattr(ScoreType, name), weight)
        if "options" in data:
            score.set_options(data["options"])
        return score

    @property
    def param_db(self):
        return self._param_db

    @property
    def terms(self):
        return list(self.all_terms())

    @property
    def weights(self):
        return dict(self._weights)

    def clone(self):
        """Return an independent score configuration with identical weights."""
        cloned = type(self)(self._param_db, weight_dtype=self.weight_dtype)
        cloned.set_options(self._options)
        for score_type, weight in self._weights.items():
            cloned.set_weight(score_type, weight)
        return cloned

    def set_options(self, options):
        """Replace supported term options; unsupported options fail explicitly."""
        unsupported = set(options) - {"ref_weights", "soft_rep"}
        if unsupported:
            raise NotImplementedError(
                "JAX score options are not implemented: "
                + ", ".join(sorted(unsupported))
            )
        self._options = copy.deepcopy(dict(options))

    def set_option(self, key, value):
        self.set_options({**self._options, key: value})

    def set_weight(self, score_type, weight):
        if not isinstance(score_type, ScoreType):
            raise TypeError("score_type must be a ScoreType")
        with np.errstate(over="ignore"):
            weight = np.asarray(weight, dtype=self.weight_dtype).item()
        if not math.isfinite(weight):
            raise ValueError("score weights must be finite")
        if score_type not in _TERM_FOR_TYPE:
            raise NotImplementedError(f"JAX scoring does not support {score_type.name}")
        self.update_term_membership(score_type, weight)
        self._weights[score_type] = weight

    def _create_term_for_score_type(self, score_type):
        module, name = _TERM_FOR_TYPE[score_type]
        term_cls = getattr(
            importlib.import_module(f"tmol_jax.score.{module}.{module}_energy_term"),
            name,
        )
        return term_cls(self._param_db)

    def _term_membership_changed(self):
        self._executors.clear()

    def weights_tensor(self):
        """Weights in the same retained-term/native-lane order as rendered scores."""
        return jnp.asarray(
            [self.get_weight(st) for st in self.all_score_types()],
            dtype=self.weight_dtype,
        )

    def get_weight(self, score_type):
        return self._weights.get(score_type, 0.0)

    def active_score_types(self):
        return sorted(st for st in self.all_score_types() if self.get_weight(st) != 0)

    def pre_work_initialization(self, pose_stack):
        """Prepare the active molecular topology and parameters without execution."""
        self.render_whole_pose_scoring_module(pose_stack)

    @staticmethod
    def get_sorted_terms(term_list):
        return sorted_score_terms(term_list, len(ScoreType), _provided_channels)

    def _setup(self, pose_stack, constraint_set=None, omit=()):
        if not np.isfinite(np.asarray(pose_stack.coords)).all():
            raise ValueError("pose coordinates must be finite before scoring setup")
        states = []
        for term in self.all_terms():
            if term in omit or not any(
                self.get_weight(st) for st in term.score_types_provided()
            ):
                states.append(None)
                continue
            term.set_options(self._options)
            if ScoreType.constraint in term.score_types_provided():
                state = term.setup_for_pose_stack(
                    pose_stack,
                    pose_stack.packed_block_types,
                    constraint_set=constraint_set,
                )
            else:
                state = term.setup_for_pose_stack(
                    pose_stack, pose_stack.packed_block_types
                )
            states.append(state)
        return tuple(states)

    def render_whole_pose_scoring_module(
        self,
        pose_stack,
        *,
        sum_terms=True,
        apply_weights=True,
        constraint_set=None,
    ):
        """Return an unjitted coords -> scores callable with frozen configuration.

        Scores have shape [poses], or [retained term channels, poses] when
        sum_terms=False, aligned with all_score_types(). Zero-weight subchannels
        remain present while another channel retains their term. Disabling
        apply_weights exposes those same channels without their weights.
        """
        channels = tuple(self.all_score_types())
        weights = dict(self._weights)
        # Standalone LJ/LK and electrostatics retain independent parameter
        # prerequisites. When both are active, use their shared native traversal.
        fused_terms = ()
        if self.get_weight(ScoreType.fa_elec) and any(
            self.get_weight(st) for st in _FUSED_PAIR_TYPES[:3]
        ):
            fused_terms = tuple(
                dict.fromkeys(
                    self._term_for_st[st]
                    for st in (ScoreType.fa_ljatr, ScoreType.fa_elec)
                )
            )
        with jax.default_device(pose_stack.coords.device):
            states = self._setup(
                pose_stack, constraint_set=constraint_set, omit=fused_terms
            )
            if fused_terms:
                from tmol_jax.score.nonbonded import prepare_nonbonded, nonbonded_energy

                fused_state = prepare_nonbonded(
                    pose_stack,
                    self.param_db,
                    soft_repulsive=self._options.get("soft_rep", False),
                )
        entries = tuple(
            (copy.copy(term), state)
            for term, state in zip(self.all_terms(), states)
            if term not in fused_terms
            and any(weights.get(st, 0) for st in term.score_types_provided())
        )
        shape = pose_stack.coords.shape
        pair_weights = tuple(weights.get(st, 0) for st in _FUSED_PAIR_TYPES)

        terms = tuple(term for term, _ in entries)
        prepared = (
            tuple(state for _, state in entries),
            fused_state if fused_terms else None,
        )

        # Stock terms evaluate solely from their prepared state. Custom terms
        # may retain additional configuration, so preserve fresh render closures.
        stock = {
            (f"tmol_jax.score.{module}.{module}_energy_term", name)
            for module, (name, _) in _TERM_TYPES.items()
        }
        cache = (
            self._executors
            if all(
                (type(term).__module__, type(term).__name__) in stock for term in terms
            )
            else None
        )
        if cache is not None and ScoreType.constraint not in channels:
            cache = _STOCK_EXECUTORS
            evaluators = tuple(type(term).evaluate for term in terms)
            weighted_evaluators = tuple(
                getattr(type(term), "weighted_energy", None) for term in terms
            )
        else:
            # Custom terms may use their instance or pose, including callbacks.
            def bind(term):
                return lambda state, coords: term(pose_stack, state, coords)

            evaluators = tuple(bind(term) for term in terms)
            weighted_evaluators = tuple(
                getattr(term, "weighted_energy", None) for term in terms
            )
        provided_channels = tuple(tuple(term.score_types_provided()) for term in terms)
        has_fused = bool(fused_terms)

        def score(coords, prepared):
            term_states, pair_state = prepared
            energies = {}
            weighted_pair = None
            if has_fused:
                if sum_terms and apply_weights:
                    weighted_pair = nonbonded_energy(coords, pair_state, pair_weights)
                else:
                    energies.update(
                        zip(_FUSED_PAIR_TYPES, nonbonded_energy(coords, pair_state))
                    )
            weighted_terms = []
            combined_channels = set()
            for evaluate, weighted, provided, state in zip(
                evaluators, weighted_evaluators, provided_channels, term_states
            ):
                if sum_terms and weighted is not None:
                    term_weights = tuple(
                        weights.get(st, 0) if apply_weights else int(st in channels)
                        for st in provided
                    )
                    weighted_terms.append(weighted(state, coords, term_weights))
                    combined_channels.update(provided)
                else:
                    energies.update(evaluate(state, coords))
            if weighted_pair is not None:
                weighted_terms.append(weighted_pair)
                combined_channels.update(_FUSED_PAIR_TYPES)
            selected = tuple(st for st in channels if st not in combined_channels)
            if not selected:
                return sum(
                    weighted_terms,
                    jnp.zeros(
                        (coords.shape[0],) if sum_terms else (0, coords.shape[0]),
                        coords.dtype,
                    ),
                )
            values = jnp.stack(
                [
                    energies[st] * (weights.get(st, 0) if apply_weights else 1)
                    for st in selected
                ]
            )
            if sum_terms:
                result = values.sum(axis=0)
                return sum(weighted_terms, result)
            return values

        configuration = (
            tuple(type(term) for term in terms),
            tuple(sorted(weights.items())),
            sum_terms,
            apply_weights,
            bool(fused_terms),
        )
        return render_execution(
            cache,
            configuration,
            score,
            prepared,
            shape,
            native_only=(
                cache is not None
                and ScoreType.constraint not in channels
                and pose_stack.coords.device.platform == "gpu"
            ),
        )

    def jitted(self, pose_stack):
        """Compile execution; reuse its program across fresh prepared states."""
        return self.render_whole_pose_scoring_module(pose_stack).jitted()

    def render_block_pair_scoring_module(self, pose_stack, *, interaction_only=False):
        """Retain weighted pair energies; interaction-only output omits diagonals."""
        from .block_pair import BlockPairScorer

        return BlockPairScorer(self, pose_stack, interaction_only=interaction_only)

    def render_rotamer_scoring_module(
        self, pose_stack, rotamer_set, *, sum_terms=True, apply_weights=True
    ):
        """Render sparse values/indices with an explicit eager preparation step.

        Calling the scorer recomputes spatial candidates. For coordinate AD or
        jit/vmap, call ``scorer.prepare(coords)`` outside tracing and transform
        the returned fixed-candidate callable. Refresh candidates after moving
        conformers. Values are uncoalesced entries, with indices [3, entries];
        disabling sum_terms returns [active score types, entries].
        """
        from tmol_jax.pack.score_function import RotamerScorer

        return RotamerScorer(self, pose_stack, rotamer_set, sum_terms, apply_weights)

    def jitted_value_and_grad(self, pose_stack):
        """Return per-pose scores and the gradient of their sum."""
        return self.render_whole_pose_scoring_module(pose_stack).value_and_grad()
