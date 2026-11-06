"""Nucleic-acid conformers from shared chemistry and native sampling kernels."""

import copy
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from tmol_core.pack.na_sampling import (
    CHI_STEPS,
    MAX_SYN_WELL,
    annotate_na_sampler_residue,
    pack_na_sampler_params,
)
from tmol_core.pack.chi import DEFAULT_CHI_SAMPLE_LIMIT, sidechain_roots_for_chis
from tmol_core.pack.task import chi_sample_budget
from tmol_core.pack.kinematics import annotate_residue_rotamer_kinematics
from tmol_core.scoring.na_torsion import N_BASE_PER_POLYMER, SYN_MEAN
from tmol_core.scoring.na_torsion_params import prepare_na_torsion_parameters
from tmol_jax.kinematics._host import NativePosePlanner
from tmol_jax.score._native import register
from ._native import _sampling_put
from .dispatch import _wide_nonnegative, _narrow_prefix
from .graph import prefix_counts, load_balance
from .task import PackerTask, SetPackerTask


@partial(jax.jit, static_argnames=("operation", "outputs", "policy"))
def _stage(arguments, *, operation, outputs, policy):
    temperature, syn = policy
    register("tmol_na_sampling")
    return jax.ffi.ffi_call("tmol_na_sampling", outputs, vmap_method="sequential")(
        *arguments,
        operation=np.int64(operation),
        extent=np.int64(0),
        width=np.int64(N_BASE_PER_POLYMER),
        temperature=np.float64(temperature),
        sample_syn=np.int64(syn),
        max_syn_well=np.float64(MAX_SYN_WELL),
        syn_mean=np.float64(SYN_MEAN),
    )


@partial(jax.jit, static_argnames=("policy", "sampler_index"))
def _prepare_inputs(
    coords,
    types,
    offsets,
    choice_poses,
    choice_blocks,
    choice_types,
    choice_allowed,
    sampler_allowed,
    cache,
    parameters,
    steps,
    *,
    policy,
    sampler_index,
):
    # Measure each physical block once, then gather to considered types.
    # These are runtime indices; the shared chemistry provides ring order.
    rings = (
        cache["ring32"][jnp.maximum(types, 0)]
        if cache["ring32"].shape[0]
        else jnp.full((*types.shape, 5), -1, jnp.int32)
    )
    valid = (rings >= 0) & (types[..., None] >= 0)
    flat = (
        jnp.maximum(offsets[..., None] + rings, 0)
        + jnp.arange(types.shape[0], dtype=jnp.int32)[:, None, None] * coords.shape[1]
    )
    pucker = _stage(
        (
            coords.reshape(-1, 3),
            _wide_nonnegative(flat.reshape(-1, 5)),
            valid.reshape(-1, 5),
        ),
        operation=0,
        outputs=jax.ShapeDtypeStruct((types.size, 8), np.uint8),
        policy=policy,
    )
    rows = choice_poses * types.shape[1] + choice_blocks
    allowed = sampler_allowed[..., sampler_index]
    active = (
        allowed[choice_poses, choice_blocks]
        & choice_allowed
        & cache["builds_bt"][choice_types]
    )
    inputs = (
        _wide_nonnegative(choice_types),
        cache["base"],
        pucker[rows],
        cache["n_combos"],
        active,
        cache["chi_atom"],
        cache["proton"],
        *parameters,
        steps,
    )
    counts = _stage(
        inputs,
        operation=1,
        outputs=jax.ShapeDtypeStruct((rows.size, 8), np.uint8),
        policy=policy,
    )
    return inputs, counts, prefix_counts(counts)[:-1]


@partial(jax.jit, static_argnames=("total", "policy"))
def _sample_values(inputs, counts, offsets, *, total, policy):
    owners = load_balance(offsets, total=total)
    outputs = (
        jax.ShapeDtypeStruct((total, inputs[5].shape[1]), np.int32),
        jax.ShapeDtypeStruct((total, inputs[5].shape[1]), np.float32),
    )
    atoms, chi = _stage(
        (*inputs, offsets, owners), operation=2, outputs=outputs, policy=policy
    )
    return _narrow_prefix(counts), owners, atoms, chi


class PreparedNaSamples:
    """Fixed native sample identities with differentiable floating parameters.

    Prepare this object outside transformations. ``chi_values`` supports jit,
    vmap, JVP and higher parameter derivatives with fixed discrete choices.
    The mature NA sampler's public angle/table precision is float32.
    """

    def __init__(self, inputs, offsets, counts, owners, policy):
        self.inputs, self.offsets, self.counts = inputs, offsets, counts
        self.owners, self.policy = owners, policy
        self.outputs = (
            jax.ShapeDtypeStruct((owners.size, inputs[5].shape[1]), np.int32),
            jax.ShapeDtypeStruct((owners.size, inputs[5].shape[1]), np.float32),
        )

    def _fill(self, proton, means, sdev):
        args = (*self.inputs[:6], proton, self.inputs[7], means, sdev, self.inputs[10])
        return _stage(
            (*args, self.offsets, self.owners),
            operation=2,
            outputs=self.outputs,
            policy=self.policy,
        )

    def chi_values(self, proton=None, means=None, sdev=None):
        """Evaluate native angles; optional table overrides retain parameter AD."""
        supplied = (proton, means, sdev)
        originals = (self.inputs[6], self.inputs[8], self.inputs[9])
        parameters = tuple(
            old if new is None else new for old, new in zip(originals, supplied)
        )
        if any(
            x.shape != old.shape or x.dtype != jnp.float32
            for x, old in zip(parameters, originals)
        ):
            raise ValueError(
                "NA sampler parameter overrides must retain float32 table shapes"
            )
        types, base, pucker, combos = map(_narrow_prefix, self.inputs[:4])
        owners = self.owners
        rt = types[owners]
        local = (
            jnp.arange(owners.size, dtype=jnp.int32)
            - _narrow_prefix(self.offsets)[owners]
        )
        combo, slot = local % combos[rt], local // combos[rt]
        rb, rp = base[rt], pucker[owners]
        steps = self.inputs[10]

        @jax.custom_jvp
        def values(proton, means, sdev):
            return self._fill(proton, means, sdev)[1]

        @values.defjvp
        def derivative(primals, tangents):
            dp, dm, ds = tangents
            dchi = (
                jnp.where(slot < steps.size, dm[rb, rp], 0)
                + steps[slot % steps.size] * ds[rb // N_BASE_PER_POLYMER]
            ) * jnp.float32(np.pi / 180)
            tangent = jnp.concatenate((dchi[:, None], dp[rt, combo]), axis=1)
            return values(*primals), tangent

        return values(*parameters)

    def samples(self):
        atoms, chi = self._fill(self.inputs[6], self.inputs[8], self.inputs[9])
        return self.counts, self.owners, atoms, chi


class NaChiRotamerSampler:
    """Sample DNA/RNA glycosidic and proton chi with the scoring distributions."""

    samples_after_other_samplers = False

    def __init__(
        self,
        params,
        element_for_atom_type,
        *,
        chi_sample_level=0,
        sample_syn=True,
        expanded_limit=DEFAULT_CHI_SAMPLE_LIMIT,
        limit=DEFAULT_CHI_SAMPLE_LIMIT,
    ):
        if chi_sample_level not in CHI_STEPS:
            raise ValueError("NA chi sample level must be in 0..4")
        self.params = copy.deepcopy(params)
        self.element_for_atom_type = dict(element_for_atom_type)
        self.chi_sample_level, self.sample_syn = chi_sample_level, bool(sample_syn)
        self.expanded_limit, self.limit = chi_sample_budget(expanded_limit, limit)
        self._parameter_database = None
        self._annotations = {}
        self._storage = None
        self._steps = {}
        for value in self.params.values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)

    @classmethod
    def from_database(cls, parameter_database, **options):
        sampler = cls(
            prepare_na_torsion_parameters(parameter_database.scoring.na_torsion),
            {a.name: a.element for a in parameter_database.chemical.atom_types},
            **options,
        )
        sampler._parameter_database = parameter_database
        return sampler

    @classmethod
    def sampler_name(cls):
        return "NaChiRotamerSampler"

    def annotate_residue_type(self, restype, budget=None):
        budget = budget or (self.expanded_limit, self.limit)
        key = (id(restype), budget)
        if key not in self._annotations:
            inverse = NativePosePlanner().inverse_kinematics
            self._annotations[key] = (
                restype,
                annotate_na_sampler_residue(
                    restype,
                    self.element_for_atom_type,
                    *budget,
                    lambda rt: annotate_residue_rotamer_kinematics(rt, inverse),
                ),
            )
        return self._annotations[key][1]

    def _device_metadata(self, pbt, device, budget):
        if self._storage is None or self._storage[:3] != (id(pbt), device, budget):
            host = pack_na_sampler_params(
                [
                    self.annotate_residue_type(bt, budget)
                    for bt in pbt.active_block_types
                ]
            )
            cache = {
                name: _sampling_put(value, device)
                for name, value in host.items()
                if isinstance(value, np.ndarray) and name != "ring"
            }
            cache["ring32"] = jax.device_put(host["ring"].astype(np.int32), device)
            parameters = tuple(
                jax.device_put(self.params[name], device)
                for name in ("well_chi_syn", "chi_means", "sdev_chi")
            )
            self._storage = (id(pbt), device, budget, pbt, cache, parameters)
        return self._storage[4:]

    def defines_rotamers_for_rt(self, restype):
        return self.annotate_residue_type(restype)["base"] >= 0

    def first_sc_atoms_for_rt(self, restype):
        return sidechain_roots_for_chis(
            restype, ("chi1",) + tuple(c.chi_dihedral for c in restype.chi_samples)
        )

    def _prepare(self, pose, task):
        if isinstance(pose.coords, jax.core.Tracer):
            raise TypeError("dynamic NA sampling requires eager coordinates")
        if isinstance(task, PackerTask):
            task = SetPackerTask.from_packer_task(task)
        task._validate_pose(pose)
        if id(self) not in task.conformer_sampler_index:
            raise ValueError("sampler must be registered with the packing task")
        pbt, device = pose.packed_block_types, pose.device
        if (
            self._parameter_database is not None
            and pbt.chem_db is not self._parameter_database.chemical
        ):
            raise ValueError(
                "sampler database does not match the pose chemistry context"
            )
        budget = task.chi_sample_budget or (self.expanded_limit, self.limit)
        policy = (float(self.params["pucker_temperature"]), self.sample_syn)
        cache, parameters = self._device_metadata(pbt, device, budget)
        with jax.default_device(device):
            key = (device, self.chi_sample_level)
            if key not in self._steps:
                self._steps[key] = jnp.asarray(
                    CHI_STEPS[self.chi_sample_level], jnp.float32
                )
            steps = self._steps[key]
            inputs, counts, offsets = _prepare_inputs(
                pose.coords,
                pose.block_type_ind,
                pose.block_coord_offset,
                task.cons_bt_pose,
                task.cons_bt_block,
                task.cons_bt_block_type,
                task.is_cons_bt_allowed,
                task.per_block_conformer_sampler_allowed,
                cache,
                parameters,
                steps,
                policy=policy,
                sampler_index=task.conformer_sampler_index[id(self)],
            )
            host_counts = np.asarray(counts).copy().view(np.int64).reshape(-1)
            # Shared allocation/budget policy; no coordinates or floating tables
            # cross this output-cardinality boundary.
            from tmol_core.pack.chi import checked_sample_count

            total = checked_sample_count(
                host_counts.size,
                int(host_counts.sum()),
                int(host_counts.min(initial=0)),
                int(host_counts.max(initial=0)),
                *budget,
            )
            return inputs, counts, offsets, total, policy

    def prepare_chi_for_poses(self, pose, task):
        """Measure input sugars and allocate native sample identities eagerly."""
        inputs, counts, offsets, total, policy = self._prepare(pose, task)
        with jax.default_device(pose.device):
            owners = load_balance(offsets, total=total)
            return PreparedNaSamples(
                inputs, offsets, _narrow_prefix(counts), owners, policy
            )

    def sample_chi_for_poses(self, pose, task):
        inputs, counts, offsets, total, policy = self._prepare(pose, task)
        with jax.default_device(pose.device):
            return _sample_values(inputs, counts, offsets, total=total, policy=policy)

    def create_samples_for_poses(self, pose, task):
        counts, choices, atoms, chi = self.sample_chi_for_poses(pose, task)
        return (
            counts,
            choices,
            {
                "chi_defining_atom_for_rotamer": atoms,
                "chi_for_rotamers": chi.astype(pose.coords.dtype),
            },
        )
