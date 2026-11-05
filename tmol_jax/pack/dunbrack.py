"""JAX storage for common Dunbrack sampling preparation."""

import attr
import jax
import numpy as np

from tmol_core.pack.dunbrack import (
    DunbrackSamplingMetadata,
    pack_dunbrack_sampler_metadata,
    prepare_dunbrack_sampler_residue,
)
from tmol_core.pack.chi import DEFAULT_CHI_SAMPLE_LIMIT
from tmol_core.pack.kinematics import annotate_residue_rotamer_kinematics
from tmol_core.scoring.dunbrack import prepare_dunbrack_tables
from tmol_jax.kinematics._host import NativePosePlanner
from tmol_jax.score._native import fit_bspline_coefficients


class DunbrackChiSampler:
    """Prepare library, explicit-chi and retained-atom metadata.

    This currently exposes preparation only. Native sampling and complete
    packing/design execution are separate capabilities, not implemented here.
    Explicit budgets override the shared sampler defaults.
    """

    def __init__(
        self,
        tables,
        *,
        expanded_limit=DEFAULT_CHI_SAMPLE_LIMIT,
        limit=DEFAULT_CHI_SAMPLE_LIMIT,
    ):
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value <= 0
            for value in (expanded_limit, limit)
        ):
            raise ValueError("chi sample limits must be positive integers")
        self.tables = tables
        self.expanded_limit, self.limit = int(expanded_limit), int(limit)
        self.metadata = DunbrackSamplingMetadata(
            tables.all_table_indices,
            tables.scoring_db_aux["nchi_for_table_set"],
            tables.sampling_db["n_rotamers_for_tableset"],
        )

    @classmethod
    def from_database(cls, parameter_database, **limits):
        tables = prepare_dunbrack_tables(
            parameter_database.scoring.dunbrack,
            lambda values: [fit_bspline_coefficients(value) for value in values],
        )
        return cls(tables, **limits)

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

    def prepare_packed_block_types(self, packed_block_types):
        pbt = packed_block_types
        types = pbt.active_block_types
        host = pack_dunbrack_sampler_metadata(
            types,
            [self.annotate_residue_type(rt) for rt in types],
            [self.defines_rotamers_for_rt(rt) for rt in types],
        )
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
