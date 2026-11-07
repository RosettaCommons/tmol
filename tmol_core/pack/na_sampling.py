"""Nucleic-acid conformer chemistry and padded sample tables, without a framework."""

import math
import numpy
from tmol_core.pack.chi import apply_chi_sample_budget, chi_depths
from tmol_core.scoring.na_torsion import _block_type_params

# k in chi = mean + k * sdev_chi, by extra-chi sample level, mirroring Rosetta's
# expansion of its fitted chi gaussians
CHI_STEPS = {
    0: (0.0,),
    1: (0.0, 1.0, -1.0),
    2: (0.0, 0.5, -0.5),
    3: (0.0, 1.0, 2.0, -1.0, -2.0),
    4: (0.0, 0.5, 1.0, -0.5, -1.0),
}

# a syn well deeper than this is too rare to be worth a rotamer
MAX_SYN_WELL = 5.0


def annotate_na_sampler_residue(
    rt, element_for_atom_type, expanded_limit, limit, prepare_kinforest
):
    """Prepare one chemical type using shared budgets and kinematic depths.

    ``prepare_kinforest`` attaches the common residue forest through the frontend
    adapter. Protein and ligand chis never enlarge the NA Cartesian product.
    """
    p = _block_type_params(rt, element_for_atom_type)
    chi1 = rt.torsion_to_uaids.get("chi1")
    # the third atom of a torsion is the one whose dof carries it
    p["chi1_atom"] = -1 if chi1 is None else chi1[2][0]
    # one residue at a time, so the budget is per residue: no grouping, but
    #    the same limits the rest of the packer applies
    # Protein/ligand chi must not enlarge this sampler's Cartesian product
    # or padded tensors. Other samplers own those residues.
    sampled = list(rt.chi_samples) if p["base"] >= 0 else []
    if sampled:
        prepare_kinforest(rt)
        depths = chi_depths(
            rt.residue_kinforest_data,
            [rt.torsion_to_uaids[cs.chi_dihedral][2][0] for cs in sampled],
        )
        sampled = list(
            apply_chi_sample_budget(
                sampled,
                depths,
                expanded_limit,
                limit,
            )
        )
    p["proton_chi"] = [
        (
            int(samp.chi_dihedral[3:]) - 1,
            rt.torsion_to_uaids[samp.chi_dihedral][2][0],
            expanded_samples(samp),
        )
        for samp in sampled
    ]
    return p


def pack_na_sampler_params(params):
    """Build the canonical type-indexed atom and proton-product arrays."""
    n_chi = max((1 + len(p["proton_chi"]) for p in params), default=1)

    ring = (
        numpy.stack([p["ring"] for p in params])
        if params
        else numpy.empty((0, 5), dtype=numpy.int64)
    )
    base = numpy.array([p["base"] for p in params])
    chi_atom = numpy.full((len(params), n_chi), -1, dtype=numpy.int32)
    combos = [proton_combinations(p["proton_chi"]) for p in params]
    n_combos = numpy.array([len(c) for c in combos], dtype=numpy.int64)
    proton = numpy.zeros(
        (len(params), (n_combos.max() if n_combos.size else 1), n_chi - 1),
        dtype=numpy.float32,
    )
    for i, p in enumerate(params):
        if p["base"] < 0:
            continue
        chi_atom[i, 0] = p["chi1_atom"]
        for slot, (_, atom, _) in enumerate(p["proton_chi"]):
            chi_atom[i, slot + 1] = atom
        for j, combo in enumerate(combos[i]):
            proton[i, j, : len(combo)] = combo

    return dict(
        base=base.astype(numpy.int64),
        ring=ring.astype(numpy.int64),
        chi_atom=chi_atom,
        builds_bt=base >= 0,
        n_combos=n_combos,
        proton=proton,
        n_chi=n_chi,
    )


def expanded_samples(samp):
    """A proton chi's samples in radians, each with its expansions either side."""
    values = []
    for sample in samp.samples:
        values.append(sample)
        for expansion in samp.expansions:
            values.extend((sample + expansion, sample - expansion))
    return tuple(math.radians(float(v)) for v in values)


def proton_combinations(proton_chi):
    """Cross product of the proton-chi samples, or a single empty set."""
    if not proton_chi:
        return [()]
    combos = [()]
    for _, _, samples in proton_chi:
        combos = [c + (s,) for c in combos for s in samples]
    return combos
