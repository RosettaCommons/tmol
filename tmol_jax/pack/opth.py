"""OptH sample execution over shared chemistry, enumeration and source indices."""

from functools import partial
import attrs
import jax
import jax.numpy as jnp
import numpy as np

from tmol_core.pack.opth import (
    annotate_opth_residue,
    pack_opth_metadata,
    opth_builds_residue,
)
from tmol_core.pack.chi import checked_sample_count, sidechain_roots_for_chis
from tmol_core.pack.task import chi_sample_budget
from tmol_core.pack.kinematics import annotate_residue_rotamer_kinematics
from tmol_jax.kinematics._host import NativePosePlanner
from tmol_jax.score._native import register, coordinate_energy
from ._native import _sampling_put
from .current import IncludeCurrentSampler
from .imposition import _wide
from .dispatch import _narrow_prefix
from .graph import prefix_counts, load_balance


@partial(jax.jit, static_argnames=("operation", "outputs", "flip"))
def _stage(arguments, *, operation, outputs, flip):
    register("tmol_opth_sampling")
    return jax.ffi.ffi_call("tmol_opth_sampling", outputs, vmap_method="sequential")(
        *arguments,
        operation=np.int64(operation),
        extent=np.int64(0),
        width=np.int64(flip),
    )


@partial(jax.jit, static_argnames=("flip", "total"))
def _fill(inputs, counts, *, flip, total):
    offsets = prefix_counts(counts)[:-1]
    flip_counts = _stage(
        (*inputs, counts),
        operation=1,
        outputs=jax.ShapeDtypeStruct(counts.shape, np.uint8),
        flip=flip,
    )
    flip_prefix = prefix_counts(flip_counts)[:-1]
    owners = load_balance(offsets, total=total)
    width = inputs[13].shape[1]
    atoms, values, sources = _stage(
        (*inputs, offsets, flip_prefix, owners),
        operation=2,
        flip=flip,
        outputs=(
            jax.ShapeDtypeStruct((total, width), np.int32),
            jax.ShapeDtypeStruct((total, width), np.float32),
            jax.ShapeDtypeStruct((total, width, 8), np.uint8),
        ),
    )
    sources = _narrow_prefix(sources.reshape(-1, 8)).reshape(total, width)
    return owners, atoms, values, sources


@partial(jax.jit, static_argnames=("pose_shape",))
def _measure(coords, atoms, physical, *, pose_shape):
    register("tmol_precise_torsion")
    measured = coordinate_energy(
        "tmol_precise_torsion", coords.reshape(-1, 3)[atoms], ()
    )
    return (
        jnp.zeros((np.prod(pose_shape),), jnp.float32)
        .at[physical]
        .set(measured.astype(jnp.float32))
        .reshape(pose_shape)
    )


class PreparedOptHSamples:
    """Fixed discrete samples with native values and linear parameter AD.

    ``chi_values`` supports jit/vmap/JVP and higher derivatives in proton-table
    and measured-flip values. ``measure_flip_values`` recomputes each physical
    torsion once with double intermediates and supports first coordinate AD.
    Native torsion higher coordinate derivatives remain unavailable.
    """

    def __init__(self, inputs, counts, total, flip, measure_atoms, physical):
        self.inputs, self.counts, self.total, self.flip = inputs, counts, total, flip
        self.measure_atoms, self.physical = measure_atoms, physical
        self.initial = _fill(inputs, counts, flip=flip, total=total)

    def measure_flip_values(self, coords):
        return _measure(
            coords, self.measure_atoms, self.physical, pose_shape=self.inputs[14].shape
        )

    def chi_values(self, proton_samples=None, flip_values=None):
        proton = self.inputs[13] if proton_samples is None else proton_samples
        flips = self.inputs[14] if flip_values is None else flip_values
        if any(
            x.shape != old.shape or x.dtype != jnp.float32
            for x, old in zip((proton, flips), self.inputs[13:15])
        ):
            raise ValueError("OptH overrides must retain float32 parameter shapes")
        sources = self.initial[3]

        @jax.custom_jvp
        def values(p, f):
            return _fill(
                (*self.inputs[:13], p, f), self.counts, flip=self.flip, total=self.total
            )[2]

        @values.defjvp
        def derivative(primals, tangents):
            joined = jnp.concatenate(tuple(x.reshape(-1) for x in tangents))
            tangent = (
                jnp.where(sources >= 0, joined[jnp.maximum(sources, 0)], 0)
                if joined.size
                else jnp.zeros(sources.shape, jnp.float32)
            )
            return values(*primals), tangent

        return values(proton, flips)

    def samples(self):
        owners, atoms, values, _ = self.initial
        return _narrow_prefix(self.counts), owners, atoms, values


class OptHSampler(IncludeCurrentSampler):
    """Sample proton chi and NHQ flips using the common native sampler.

    Sampler preparation is eager; use ``prepare`` for a fixed differentiable
    parameter plan. OptH and Dunbrack must not be enabled on the same block.
    """

    def __init__(
        self, flip_NHQ=True, chi_sample_expanded_limit=100, chi_sample_limit=1000
    ):
        self.flip_NHQ = bool(flip_NHQ)
        self.budget = chi_sample_budget(chi_sample_expanded_limit, chi_sample_limit)
        self._annotations = {}
        self._storage = None

    def annotate_residue_type(self, restype, budget=None):
        budget = budget or self.budget
        key = (id(restype), budget)
        if key not in self._annotations:
            inverse = NativePosePlanner().inverse_kinematics
            cache = annotate_opth_residue(
                restype,
                *budget,
                lambda rt: annotate_residue_rotamer_kinematics(rt, inverse),
            )
            self._annotations[key] = (restype, cache)
        return self._annotations[key][1]

    def defines_rotamers_for_rt(self, restype):
        return opth_builds_residue(self.annotate_residue_type(restype), self.flip_NHQ)

    def first_sc_atoms_for_rt(self, restype):
        cache = self.annotate_residue_type(restype)
        chis = [cs.chi_dihedral for cs in restype.chi_samples if cs.is_proton]
        if self.flip_NHQ and cache.nhq_chi_col >= 0:
            chis.append(f"chi{cache.nhq_chi_col+1}")
        return sidechain_roots_for_chis(restype, chis)

    def _metadata(self, pbt, device, budget):
        key = (id(pbt), device, budget, self.flip_NHQ)
        if self._storage is None or self._storage[:4] != key:
            host = pack_opth_metadata(
                [
                    self.annotate_residue_type(bt, budget)
                    for bt in pbt.active_block_types
                ],
                pbt.max_n_atoms,
                self.flip_NHQ,
            )
            cache = {
                f.name: _sampling_put(getattr(host, f.name), device)
                for f in attrs.fields(type(host))
            }
            self._storage = (*key, pbt, cache, host)
        return self._storage[5:]

    def prepare(self, pose, task):
        task, _, sampler = self._selection_inputs(pose, task)
        allowed = task.per_block_conformer_sampler_allowed[:, :, sampler]
        for i, other in enumerate(task.conformer_samplers):
            if other.sampler_name() == "DunbrackChiSampler" and bool(
                jnp.any(allowed & task.per_block_conformer_sampler_allowed[:, :, i])
            ):
                raise RuntimeError(
                    "OptHSampler and DunbrackChiSampler cannot both be assigned to the same block"
                )
        budget = task.chi_sample_budget or self.budget
        cache, host = self._metadata(pose.packed_block_types, pose.device, budget)
        with jax.default_device(pose.device):
            inputs = tuple(
                _wide(x)
                for x in (
                    task.cons_bt_block_type,
                    task.cons_bt_pose,
                    task.cons_bt_block,
                    pose.block_type_ind,
                )
            )
            inputs += (task.is_cons_bt_allowed, allowed)
            inputs += tuple(
                cache[name]
                for name in (
                    "n_samples_for_bt_by_orig_bt",
                    "nhq_chi_col",
                    "nhq_chi_atom",
                    "is_his",
                    "has_proton_chi",
                    "chi_defining_atom",
                    "n_samples_per_chi",
                    "expanded_samples",
                )
            )
            inputs += (jnp.zeros(pose.block_type_ind.shape, jnp.float32),)
            counts, measure = _stage(
                inputs,
                operation=0,
                flip=self.flip_NHQ,
                outputs=(
                    jax.ShapeDtypeStruct((task.cons_bt_pose.size, 8), np.uint8),
                    jax.ShapeDtypeStruct(task.cons_bt_pose.shape, np.bool_),
                ),
            )
            n = np.asarray(counts).copy().view(np.int64).reshape(-1)
            total = checked_sample_count(
                len(n),
                int(n.sum()),
                int(n.min(initial=0)),
                int(n.max(initial=0)),
                *budget,
            )
            if inputs[13].size + inputs[14].size >= 1 << 31:
                raise OverflowError("OptH parameter source indices exceed int32")
            selected = np.asarray(measure)
            physical = np.unique(
                np.asarray(task.cons_bt_pose)[selected] * pose.max_n_blocks
                + np.asarray(task.cons_bt_block)[selected]
            )
            p, b = physical // pose.max_n_blocks, physical % pose.max_n_blocks
            types = np.asarray(pose.block_type_ind)[p, b]
            atoms = (
                p[:, None] * pose.max_n_atoms_per_pose
                + np.asarray(pose.block_coord_offset)[p, b, None]
                + host.nhq_chi_4atoms[types]
            )
            atoms = jax.device_put(atoms.astype(np.int32), pose.device)
            physical = jax.device_put(physical.astype(np.int32), pose.device)
            flips = _measure(
                pose.coords, atoms, physical, pose_shape=pose.block_type_ind.shape
            )
            inputs = (*inputs[:14], flips)
            return PreparedOptHSamples(
                inputs, counts, total, self.flip_NHQ, atoms, physical
            )

    def create_samples_for_poses(self, pose, task):
        counts, choices, atoms, values = self.prepare(pose, task).samples()
        return (
            counts,
            choices,
            dict(
                chi_defining_atom_for_rotamer=atoms,
                chi_for_rotamers=values.astype(pose.coords.dtype),
                opth_dof_initialization=True,
            ),
        )
