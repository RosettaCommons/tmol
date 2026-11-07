"""Common host preparation of nucleic-acid torsion parameters."""

import numpy

from tmol_core.scoring.na_torsion import (
    BACKBONE_TORSIONS,
    SUGAR_TORSIONS,
    POLYMERS,
    BASES,
    UNKNOWN_BASE,
    N_PUCKER,
    N_BASE,
    N_BASE_PER_POLYMER,
    NORTH_PUCKERS,
)


def _circular_mean(degrees):
    """Mean of angles, over the leading axis, in degrees.

    Averaging angles directly puts the mean of 179 and -179 at zero, which is
    the opposite side of the circle from either.
    """
    radians = degrees * (numpy.pi / 180.0)
    mean = numpy.arctan2(numpy.sin(radians).mean(0), numpy.cos(radians).mean(0))
    # the database writes its means in [0, 360); wrap_degrees does not care,
    #    but a row in a different range invites a wrong comparison later
    return (mean * (180.0 / numpy.pi)) % 360.0


def prepare_na_torsion_parameters(database):
    def t(v, dtype=numpy.float32):
        return numpy.asarray(v, dtype=dtype)

    def stack(build):
        return numpy.stack([build(p) for p in POLYMERS])

    def zeros(*shape, dtype=numpy.float32):
        return numpy.zeros(shape, dtype=dtype)

    def backbone(poly):
        means = zeros(6, 3)
        for i, name in enumerate(BACKBONE_TORSIONS):
            vals = database.backbone_means[poly].get(name)
            if vals is None:  # delta lives in the sugar term
                continue
            means[i, : len(vals)] = t(vals)
        return means

    def backbone_bins(poly):
        n = zeros(6, dtype=numpy.int32)
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
                    numpy.stack(
                        [t(database.sugar_means[poly]["chi"][n]) for n in named]
                    )
                )
                for si, state in enumerate(("anti", "syn")):
                    chi_syn[si, :, ind] = numpy.stack(
                        [t(w.chi_syn_given_pucker[state][n]) for n in named]
                    ).mean(0)
                continue
            chi[ind] = t(database.sugar_means[poly]["chi"][base])
            for si, state in enumerate(("anti", "syn")):
                chi_syn[si, :, ind] = t(w.chi_syn_given_pucker[state][base])

    is_north = zeros(N_PUCKER, dtype=numpy.bool_)
    is_north[list(NORTH_PUCKERS)] = True

    def well(field):
        return stack(lambda p: t(getattr(database.well_energies[p], field)))

    def scalar(field):
        return t([getattr(database.global_parameters[p], field) for p in POLYMERS])

    # shared across polymers by construction; the generator writes one value
    shared = database.global_parameters[POLYMERS[0]]

    return dict(
        backbone_means=stack(backbone),
        backbone_n_bins=stack(backbone_bins),
        backbone_sdev=stack(lambda p: t(database.global_parameters[p].sdev_backbone)),
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
