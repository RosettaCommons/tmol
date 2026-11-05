import attr
import numpy
from tmol_core.scoring.na_torsion_params import prepare_na_torsion_parameters
import torch

from tmol.database.scoring import NaTorsionDatabase
from tmol.utility import resolve_device
from tmol.utility.weak_identity_cache import WeakIdentityLRU
from tmol.types._torch import Tensor
from tmol.types._attrs import ValidateAttrs

from tmol_core.scoring.na_torsion import (  # noqa: F401
    BACKBONE_TORSIONS,
    SUGAR_TORSIONS,
    CHI_TORSION,
    TORSION_NAMES,
    TORSION_IND,
    POLYMERS,
    POLYMER_IND,
    UNKNOWN_BASE,
    BASES,
    N_BASE_PER_POLYMER,
    N_BASE,
    BASE_FOR_NAME3,
    UNKNOWN_BASE_REFERENCE,
    N_PUCKER,
    NORTH_PUCKERS,
    SYN_MEAN,
    SYN_RANGE,
    N_TORSION,
    DELTA,
    CHI,
    REQUIRED_TORSIONS,
    sugar_ring_atoms,
    _block_type_params,
    _scored_torsion_bonds,
)


def _polymer_index(base):
    """0 for DNA, 1 for RNA. Non-nucleotides (base < 0) fall to 0 and are masked."""
    return base.clamp_min(0) // N_BASE_PER_POLYMER


@attr.s(auto_attribs=True, frozen=True, slots=True)
class NaTorsionParams(ValidateAttrs):
    """Device-resident nucleic-acid torsion distributions and well energies."""

    _from_db_cache = WeakIdentityLRU()

    # means in degrees; 3 bins for alpha/gamma, 2 for beta/epsilon/zeta
    backbone_means: Tensor[torch.float32][2, 6, 3]
    backbone_n_bins: Tensor[torch.int32][2, 6]
    backbone_sdev: Tensor[torch.float32][2, 6]

    sugar_means: Tensor[torch.float32][2, N_PUCKER, 4]  # delta nu4 nu0 nu1
    chi_means: Tensor[torch.float32][N_BASE, N_PUCKER]

    # bin-population energies; each bin assignment is charged once
    well_pucker: Tensor[torch.float32][2, N_PUCKER]
    well_alpha_gamma: Tensor[torch.float32][2, 3, 3]
    well_bibii_pucker: Tensor[torch.float32][2, 2, 2]
    well_alphanext_bibii: Tensor[torch.float32][2, 3, 2]
    well_chi_syn: Tensor[torch.float32][2, N_PUCKER, N_BASE]
    is_north: Tensor[torch.bool][N_PUCKER]

    # per-polymer scalars, gathered like the tables above
    sdev_sugar: Tensor[torch.float32][2]
    sdev_chi: Tensor[torch.float32][2]
    weight_bb: Tensor[torch.float32][2]
    weight_chi: Tensor[torch.float32][2]
    weight_sugar: Tensor[torch.float32][2]

    # shared: these set the shape of the soft bin assignment, not its content
    pucker_temperature: float
    bin_blend_sdev: float

    @classmethod
    def from_database(cls, database: NaTorsionDatabase, device: torch.device):
        # score functions built from one database share its tables
        device = resolve_device(device)
        return cls._from_db_cache.get_or_create(
            database, (cls, device), lambda: cls._from_database(database, device)
        )

    @classmethod
    def _from_database(cls, database: NaTorsionDatabase, device: torch.device):
        return cls(
            **{
                name: (
                    torch.as_tensor(value, device=device)
                    if isinstance(value, numpy.ndarray)
                    else value
                )
                for name, value in prepare_na_torsion_parameters(database).items()
            }
        )
