"""JAX storage for common Dunbrack sampling preparation."""

import attr
import copy
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np

from tmol_core.cache import AnnotationKey, PreparationCache
from tmol_core.database.content_hash import _content_hash
from tmol_core.pack.dunbrack import (
    DunbrackSamplingMetadata,
    pack_dunbrack_sampler_metadata,
    prepare_dunbrack_sampler_residue,
)
from tmol_core.pack.chi import DEFAULT_CHI_SAMPLE_LIMIT
from tmol_core.pack.task import chi_sample_budget
from tmol_core.pack.sampling_request import (
    prepare_dunbrack_sampling_request,
    apply_frozen_chi,
)
from tmol_core.pack.kinematics import annotate_residue_rotamer_kinematics
from tmol_jax.kinematics._host import NativePosePlanner
from tmol_jax.score.dunbrack.parameters import (
    prepare_dunbrack_parameters,
    _prepare_dunbrack_snapshot,
)
from ._sampler import SamplerSupport


@partial(jax.jit, static_argnames=("n_gbt", "max_n_chi"))
def _sampling_result(
    counts, identities, chi, selected, types, defining, *, n_gbt, max_n_chi
):
    """Map native samples to considered identities in one device execution."""
    return (
        jnp.zeros(n_gbt, dtype=jnp.int32).at[selected].set(counts),
        selected[identities],
        defining[types[identities]],
        jnp.pad(chi, ((0, 0), (0, max_n_chi - chi.shape[1]))),
    )


class DunbrackChiSampler(SamplerSupport):
    """Prepare and sample library or explicit chi on native CPU/CUDA.

    Sampling is eager because its output cardinality depends on the pose.
    It returns chi and atom identities; coordinate reconstruction and complete
    packing/design execution remain separate capabilities. Explicit task budgets
    override sampler defaults without mutating this reusable object.
    """

    samples_after_other_samplers = False

    def __init__(
        self,
        tables,
        *,
        expanded_limit=DEFAULT_CHI_SAMPLE_LIMIT,
        limit=DEFAULT_CHI_SAMPLE_LIMIT,
    ):
        # A sampler owns a parameter snapshot. Native host/device buffers may
        # be reused without observing mutable arrays supplied by its caller.
        self._initialize(expanded_limit=expanded_limit, limit=limit)
        self._tables = self._snapshot_tables(tables)
        self.metadata = DunbrackSamplingMetadata(
            self._tables.all_table_indices,
            self._tables.scoring_db_aux["nchi_for_table_set"],
            self._tables.sampling_db["n_rotamers_for_tableset"],
        )

    @staticmethod
    def _snapshot_tables(tables):
        tables = copy.deepcopy(tables)
        for mapping in (
            tables.scoring_db,
            tables.scoring_db_aux,
            tables.sampling_db,
        ):
            for value in mapping.values():
                if isinstance(value, np.ndarray):
                    value.setflags(write=False)
        return tables

    def _initialize(
        self, *, expanded_limit=DEFAULT_CHI_SAMPLE_LIMIT, limit=DEFAULT_CHI_SAMPLE_LIMIT
    ):
        self._tables = None
        self._pending_library = None
        self._snapshot_hash = None
        self._parameter_database = None
        self._native_storage = None
        self._metadata_cache = PreparationCache()
        self.expanded_limit, self.limit = chi_sample_budget(expanded_limit, limit)

    @classmethod
    def from_database(cls, parameter_database, **limits):
        """Snapshot the library now; fit coefficients when sampling needs them."""
        if cls is not DunbrackChiSampler:
            sampler = cls(prepare_dunbrack_parameters(parameter_database), **limits)
            sampler._parameter_database = parameter_database
            return sampler
        sampler = cls.__new__(cls)
        sampler._initialize(**limits)
        library = copy.deepcopy(parameter_database.scoring.dunbrack)
        sampler.metadata = DunbrackSamplingMetadata.from_database(library)
        sampler._snapshot_hash = _content_hash(library)
        sampler._pending_library = library
        sampler._parameter_database = parameter_database
        return sampler

    @property
    def tables(self):
        """Materialize the sampler's owned, read-only fitted parameter snapshot."""
        if self._tables is None:
            # The owned copy and its creation-time hash describe one snapshot.
            # A live source mutation cannot change it or the fitted cache key.
            tables = _prepare_dunbrack_snapshot(
                self._parameter_database.scoring.dunbrack,
                self._pending_library,
                self._snapshot_hash,
            )
            self._tables = self._snapshot_tables(tables)
            self._pending_library = None
        return self._tables

    @classmethod
    def sampler_name(cls):
        return "DunbrackChiSampler"

    def defines_rotamers_for_rt(self, restype):
        return self.metadata.defines_rotamers(restype)

    def first_sc_atoms_for_rt(self, restype):
        return self.metadata.sidechain_roots(restype)

    def annotate_residue_type(self, restype):
        inverse = NativePosePlanner().inverse_kinematics
        return prepare_dunbrack_sampler_residue(
            restype,
            self.metadata,
            expanded_limit=self.expanded_limit,
            limit=self.limit,
            prepare_kinematics=lambda residue: annotate_residue_rotamer_kinematics(
                residue, inverse
            ),
        )

    def _host_metadata(self, pbt):
        types = pbt.active_block_types
        key = AnnotationKey.from_sources(
            self.metadata,
            pbt.chem_db,
            *types,
            settings=(self.expanded_limit, self.limit),
        )
        return self._metadata_cache.get_or_create(
            "sampling",
            key,
            lambda: pack_dunbrack_sampler_metadata(
                types,
                [self.annotate_residue_type(rt) for rt in types],
                [self.defines_rotamers_for_rt(rt) for rt in types],
            ),
        )

    def prepare_packed_block_types(self, packed_block_types):
        pbt = packed_block_types
        host = self._host_metadata(pbt)
        return attr.evolve(
            host,
            **{
                field.name: jax.device_put(
                    getattr(host, field.name), pbt.n_atoms.device
                )
                for field in attr.fields(type(host))
                if isinstance(getattr(host, field.name), np.ndarray)
            },
        )

    def create_samples_for_poses(self, pose_stack, task):
        counts, choices, atoms, chi = self.sample_chi_for_poses(pose_stack, task)
        return (
            counts,
            choices,
            {
                "chi_defining_atom_for_rotamer": atoms,
                "chi_for_rotamers": chi,
            },
        )

    def sample_chi_for_poses(self, pose_stack, task):
        """Return per-choice counts, choice indices, defining atoms and chi.

        Choices refer to ``SetPackerTask.cons_bt_*`` rows. Residues outside this
        sampler's declared chemical domain receive zero samples, including
        ligands without an amino-acid chi sampling policy. No fallback identity
        or conformation is substituted. Returned arrays stay on the pose device.
        """
        from ._native import sample_chi, SamplingStorage
        from .task import PackerTask, SetPackerTask
        from tmol_jax.score._native import atom_downstream_of_conn
        from tmol_jax.score.constraint.functions import torsion

        if isinstance(pose_stack.coords, jax.core.Tracer):
            raise TypeError("dynamic chi sampling requires eager coordinates")
        if isinstance(task, PackerTask):
            task = SetPackerTask.from_packer_task(task)
        task._validate_pose(pose_stack)
        if id(self) not in task.conformer_sampler_index:
            raise ValueError("sampler must be registered with the packing task")
        pbt, device = pose_stack.packed_block_types, pose_stack.coords.device
        if (
            self._parameter_database is not None
            and pbt.chem_db is not self._parameter_database.chemical
        ):
            raise ValueError(
                "sampler database does not match the pose chemistry context"
            )
        configured = self
        if task.chi_sample_budget is not None:
            configured = copy.copy(self)
            configured.expanded_limit, configured.limit = task.chi_sample_budget
        metadata = configured._host_metadata(pbt)
        sampler_index = task.conformer_sampler_index[id(self)]
        request = prepare_dunbrack_sampling_request(
            pbt.active_block_types,
            metadata,
            np.asarray(pose_stack.block_type_ind),
            np.asarray(pose_stack.block_coord_offset),
            np.asarray(pose_stack.inter_residue_connections),
            atom_downstream_of_conn(pbt),
            pose_stack.coords.shape[1],
            np.asarray(task.cons_bt_pose),
            np.asarray(task.cons_bt_block),
            np.asarray(task.cons_bt_block_type),
            np.asarray(task.cons_bt_which_block_type),
            np.asarray(task.is_cons_bt_allowed),
            np.asarray(task.per_block_conformer_sampler_allowed)[..., sampler_index],
            np.asarray(task.per_block_chi_expansion),
            real_dtype=pose_stack.coords.dtype,
        )
        measured = np.empty(len(request.frozen_requests), pose_stack.coords.dtype)
        if request.frozen_requests:
            indices = request.frozen_atom_indices
            valid = np.all(indices >= 0, axis=1)
            measured.fill(np.nan)
            if np.any(valid):
                atoms = pose_stack.coords.reshape(-1, 3)[
                    jax.device_put(indices[valid], device)
                ]
                measured[valid] = np.asarray(torsion(atoms)).reshape(-1)
        apply_frozen_chi(request, measured)

        with jax.default_device(device):
            if not len(request.bbt_to_gbt):
                return (
                    jnp.zeros(request.n_gbt, dtype=jnp.int32),
                    jnp.zeros(0, dtype=jnp.int32),
                    jnp.zeros((0, request.max_n_chi), dtype=jnp.int32),
                    jnp.zeros((0, request.max_n_chi), dtype=pose_stack.coords.dtype),
                )
            if self._native_storage is None:
                self._native_storage = SamplingStorage(self.tables)
            counts, _, identities, chi = sample_chi(
                self._native_storage,
                pose_stack.coords.reshape(-1, 3),
                *(
                    getattr(request, name)
                    for name in (
                        "n_dihe_for_block",
                        "dihedral_offset_for_block",
                        "dihedral_atom_inds",
                        "bubl_and_rottable_set_for_bbt",
                        "chi_expansion_for_bbt",
                        "non_dunbrack_expansion_for_bbt",
                        "non_dunbrack_expansion_counts_for_bbt",
                        "prob_cumsum_limit_for_bbt",
                        "n_chi_for_bbt",
                    )
                ),
                max_samples_per_restype=max(task.chi_sample_budget or (0,)),
            )
            indices = jax.device_put(
                tuple(
                    np.asarray(value, dtype=np.int32)
                    for value in (
                        request.bbt_to_gbt,
                        request.block_type_ind_for_bbt,
                        metadata.chi_defining_atom,
                    )
                ),
                device,
            )
            return _sampling_result(
                counts,
                identities,
                chi,
                *indices,
                n_gbt=request.n_gbt,
                max_n_chi=request.max_n_chi,
            )
