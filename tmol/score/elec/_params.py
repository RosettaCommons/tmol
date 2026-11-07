import attr
import cattr

import torch

from tmol.types._torch import Tensor
from tmol.types._tensor import TensorGroup
from tmol.types._attrs import ValidateAttrs
from tmol.types._functional import _validate_args
from tmol.database.scoring import ElecDatabase

from tmol.chemical import RefinedResidueType
from tmol_core.scoring.elec import (
    lookup_order,
    partial_charge_table,
    count_pair_table,
    partial_charges_for_block,
    representative_mapping_for_block,
)


@attr.s(auto_attribs=True, slots=True, frozen=True)
class ElecGlobalParams(TensorGroup, ValidateAttrs):
    elec_min_dis: Tensor[torch.float32][...]
    elec_max_dis: Tensor[torch.float32][...]
    elec_sigmoidal_die_D: Tensor[torch.float32][...]
    elec_sigmoidal_die_D0: Tensor[torch.float32][...]
    elec_sigmoidal_die_S: Tensor[torch.float32][...]


@attr.s(frozen=True, slots=True, auto_attribs=True)
class ElecParamResolver(ValidateAttrs):
    """Shared host charge/count-pair lookup and Torch dielectric tensors."""

    global_params: ElecGlobalParams

    device: torch.device

    # map (AA,atom) to atom
    cp_reps: dict

    # map (AA,atom) to partial charge
    partial_charges: dict

    _lookup_order = staticmethod(lookup_order)

    def get_partial_charges_for_block(self, block_type: RefinedResidueType):
        return partial_charges_for_block(block_type, self.partial_charges)

    def get_bonded_path_length_mapping_for_block(self, block_type: RefinedResidueType):
        return representative_mapping_for_block(block_type, self.cp_reps)

    @classmethod
    @_validate_args
    def from_database(cls, elec_database: ElecDatabase, device: torch.device):
        """Initialize param resolver for all atoms defined in database."""
        # Load global params, coerce to 1D Tensors
        values = cattr.unstructure(elec_database.global_parameters)
        tensor = torch.tensor(list(values.values()), dtype=torch.float32, device=device)
        global_params = ElecGlobalParams(**dict(zip(values, tensor.unbind())))

        return cls(
            global_params=global_params,
            partial_charges=partial_charge_table(elec_database),
            cp_reps=count_pair_table(elec_database),
            device=device,
        )
