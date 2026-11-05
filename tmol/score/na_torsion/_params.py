import attr
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


def _circular_mean(degrees):
    """Mean of angles, over the leading axis, in degrees.

    Averaging angles directly puts the mean of 179 and -179 at zero, which is
    the opposite side of the circle from either.
    """
    radians = degrees * (torch.pi / 180.0)
    mean = torch.atan2(radians.sin().mean(0), radians.cos().mean(0))
    # the database writes its means in [0, 360); wrap_degrees does not care,
    #    but a row in a different range invites a wrong comparison later
    return (mean * (180.0 / torch.pi)) % 360.0


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
    def _from_database(
        cls, database: NaTorsionDatabase, device: torch.device
    ):  # noqa: C901
        def t(v, dtype=torch.float32):
            return torch.tensor(v, dtype=dtype, device=device)

        def stack(build):
            return torch.stack([build(p) for p in POLYMERS])

        def zeros(*shape, dtype=torch.float32):
            return torch.zeros(shape, dtype=dtype, device=device)

        def backbone(poly):
            means = zeros(6, 3)
            for i, name in enumerate(BACKBONE_TORSIONS):
                vals = database.backbone_means[poly].get(name)
                if vals is None:  # delta lives in the sugar term
                    continue
                means[i, : len(vals)] = t(vals)
            return means

        def backbone_bins(poly):
            n = zeros(6, dtype=torch.int32)
            for i, name in enumerate(BACKBONE_TORSIONS):
                vals = database.backbone_means[poly].get(name)
                n[i] = 0 if vals is None else len(vals)
            return n

        def sugar(poly):
            out = zeros(N_PUCKER, 4)
            for j, name in enumerate(SUGAR_TORSIONS):
                out[:, j] = t(database.sugar_means[poly][name]["all"])
            return out

        chi = zeros(N_BASE, N_PUCKER)
        chi_syn = zeros(2, N_PUCKER, N_BASE)
        for pi, poly in enumerate(POLYMERS):
            w = database.well_energies[poly]
            named = [b for b in BASES[poly] if b != UNKNOWN_BASE]
            for b, base in enumerate(BASES[poly]):
                ind = pi * N_BASE_PER_POLYMER + b
                if base == UNKNOWN_BASE:
                    # chi means are angles, so they average on the circle; the
                    #    well energies are energies and average directly
                    chi[ind] = _circular_mean(
                        torch.stack(
                            [t(database.sugar_means[poly]["chi"][n]) for n in named]
                        )
                    )
                    for si, state in enumerate(("anti", "syn")):
                        chi_syn[si, :, ind] = torch.stack(
                            [t(w.chi_syn_given_pucker[state][n]) for n in named]
                        ).mean(0)
                    continue
                chi[ind] = t(database.sugar_means[poly]["chi"][base])
                for si, state in enumerate(("anti", "syn")):
                    chi_syn[si, :, ind] = t(w.chi_syn_given_pucker[state][base])

        is_north = zeros(N_PUCKER, dtype=torch.bool)
        is_north[list(NORTH_PUCKERS)] = True

        def well(field):
            return stack(lambda p: t(getattr(database.well_energies[p], field)))

        def scalar(field):
            return t([getattr(database.global_parameters[p], field) for p in POLYMERS])

        # shared across polymers by construction; the generator writes one value
        shared = database.global_parameters[POLYMERS[0]]

        return cls(
            backbone_means=stack(backbone),
            backbone_n_bins=stack(backbone_bins),
            backbone_sdev=stack(
                lambda p: t(database.global_parameters[p].sdev_backbone)
            ),
            sugar_means=stack(sugar),
            chi_means=chi,
            well_pucker=well("pucker"),
            well_alpha_gamma=well("alpha_gamma"),
            well_bibii_pucker=well("bibii_given_pucker"),
            well_alphanext_bibii=well("alphanext_given_bibii"),
            well_chi_syn=chi_syn,
            is_north=is_north,
            sdev_sugar=scalar("sdev_sugar"),
            sdev_chi=scalar("sdev_chi"),
            weight_bb=scalar("weight_bb"),
            weight_chi=scalar("weight_chi"),
            weight_sugar=scalar("weight_sugar"),
            pucker_temperature=shared.pucker_temperature,
            bin_blend_sdev=shared.bin_blend_sdev,
        )
