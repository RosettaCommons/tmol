import attr
import cattr

import pandas

import torch


from tmol.types._torch import Tensor
from tmol.types._tensor import TensorGroup
from tmol.types._attrs import ValidateAttrs
from tmol.types._functional import _validate_args
from tmol.database.scoring import LJLKDatabase
from tmol.database._patched_chemdb import PatchedChemicalDatabase

from .._chemical_database import AtomTypeParamResolver
from tmol_core.scoring.ljlk import (
    LJLK_FIELDS,
    ljlk_type_parameters,
    validate_block_type,
)


@attr.s(auto_attribs=True, slots=True, frozen=True)
class LJLKGlobalParams(TensorGroup):
    max_dis: Tensor[torch.float32][...]
    lj_dlin_sigma_factor: Tensor[torch.float32][...]
    lj_dlin_sigma_factor_soft: Tensor[torch.float32][...]
    lj_hbond_OH_donor_dis: Tensor[torch.float32][...]
    lj_hbond_dis: Tensor[torch.float32][...]
    lj_hbond_hdis: Tensor[torch.float32][...]
    lk_min_dis2sigma: Tensor[torch.float32][...]

    # not clear if this belongs here or in a separate class
    lkb_water_dist: Tensor[torch.float32][...]
    lkb_water_angle_sp2: Tensor[torch.float32][...]
    lkb_water_angle_sp3: Tensor[torch.float32][...]
    lkb_water_angle_ring: Tensor[torch.float32][...]
    lkb_water_tors_sp2: Tensor[torch.float32][..., :]
    lkb_water_tors_sp3: Tensor[torch.float32][..., :]
    lkb_water_tors_ring: Tensor[torch.float32][..., :]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class LJLKTypeParams(TensorGroup):
    lj_radius: Tensor[torch.float32][...]
    lj_wdepth: Tensor[torch.float32][...]
    lk_dgfree: Tensor[torch.float32][...]
    lk_lambda: Tensor[torch.float32][...]
    lk_volume: Tensor[torch.float32][...]

    # Parameters imported from chemical database
    is_acceptor: Tensor[bool][...]
    acceptor_hybridization: Tensor[torch.int][...]

    is_donor: Tensor[bool][...]

    is_hydroxyl: Tensor[bool][...]
    is_polarh: Tensor[bool][...]

    is_hydrogen: Tensor[bool][...]

    # atom types Rosetta flattens LK desolvation for in C-C pairs
    is_carbon_lk: Tensor[bool][...]


@attr.s(frozen=True, slots=True, auto_attribs=True)
class LJLKParamResolver(ValidateAttrs):
    """Container for global/type/pair parameters, indexed by atom type name.

    Param resolver stores pair parameters for a collection of atom types, using
    a pandas Index to map from string atom type to a resolver-specific integer type
    index.
    """

    # str->int index from atom type name to index within parameter tensors
    atom_type_index: pandas.Index

    # shape [1] global parameters
    global_params: LJLKGlobalParams

    # shape [n] per-type parameters, source params used to calculate pair parameters
    type_params: LJLKTypeParams

    device: torch.device

    invalid_type_parameters: dict = attr.ib(factory=dict)

    def validate_block_type(self, block_type):
        """Require finite parameters only for atom types used by this block."""
        validate_block_type(block_type, self.invalid_type_parameters)

    @classmethod
    @_validate_args
    def from_database(
        cls,
        chemical_database: PatchedChemicalDatabase,
        ljlk_database: LJLKDatabase,
        device: torch.device,
    ):
        """Initialize param resolver for all atom types in database."""
        return cls.from_param_resolver(
            AtomTypeParamResolver.from_database(chemical_database, device=device),
            ljlk_database,
        )

    @classmethod
    @_validate_args
    def from_param_resolver(
        cls, atom_type_resolver: AtomTypeParamResolver, ljlk_database: LJLKDatabase
    ):
        # Reference existing atom type index from atom_type_resolver
        atom_type_index = atom_type_resolver.index
        device = atom_type_resolver.device

        # Convert float entries into 1-d tensors
        def at_least_1d(t):
            if t.dim() == 0:
                return t.expand((1,))
            return t

        global_params = LJLKGlobalParams(
            **{
                n: at_least_1d(torch.tensor(v, device=device))
                for n, v in cattr.unstructure(ljlk_database.global_parameters).items()
            }
        )

        host = ljlk_type_parameters(atom_type_index, ljlk_database)
        values = host.values
        fields = LJLK_FIELDS
        invalid_type_parameters = host.invalid_type_parameters
        is_carbon_lk = torch.from_numpy(host.is_carbon_lk).to(device)

        # Convert the param record dataframe into typed TensorGroup
        type_params = LJLKTypeParams(
            # Reference parameters from atom_type_resolver
            is_acceptor=atom_type_resolver.params.is_acceptor,
            acceptor_hybridization=atom_type_resolver.params.acceptor_hybridization,
            is_donor=atom_type_resolver.params.is_donor,
            is_hydroxyl=atom_type_resolver.params.is_hydroxyl,
            is_polarh=atom_type_resolver.params.is_polarh,
            **{
                name: torch.tensor(values[:, i], dtype=torch.float32, device=device)
                for i, name in enumerate(fields)
            },
            is_hydrogen=atom_type_resolver.params.is_hydrogen,
            is_carbon_lk=is_carbon_lk,
        )

        return cls(
            atom_type_index=atom_type_index,
            global_params=global_params,
            type_params=type_params,
            device=device,
            invalid_type_parameters=invalid_type_parameters,
        )
