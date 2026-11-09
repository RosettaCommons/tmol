"""Shared chi sampling budgets, atom ownership and ring geometry preparation."""

import attr
import numpy
from tmol_core.geometry import numpy_coord_dihedrals

DEFAULT_CHI_SAMPLE_EXPANDED_LIMIT = 100
DEFAULT_CHI_SAMPLE_LIMIT = 1000


def checked_sample_count(n_counts, total, minimum, maximum, expanded_limit, limit):
    """Validate reduced sampling counts before narrowing indices or allocating.

    Frontends provide int64 reductions. Validate individual counts before
    trusting their sum: arbitrary invalid int64 values may overflow reduction.
    """
    if n_counts == 0:
        return 0
    capacity = numpy.iinfo(numpy.int32).max
    if n_counts > capacity:
        raise ValueError(f"Sampling count exceeds index capacity {capacity}")
    if minimum < 0 or maximum > capacity or total > capacity or total < 0:
        raise ValueError(
            f"Sampling count exceeds index capacity {capacity} "
            "or contains a negative count"
        )
    budget = max(expanded_limit, limit)
    if maximum > budget:
        raise ValueError(
            f"Sampling budget {budget} cannot fit the required library states"
        )
    return total


def n_conformers(samples, expanded: bool) -> int:
    """How many conformers a set of sampled chi enumerates."""
    total = 1
    for cs in samples:
        if cs is None:  # frozen
            continue
        total *= len(cs.samples) * (1 + 2 * len(cs.expansions) if expanded else 1)
    return total


def apply_chi_sample_budget(
    samples,
    depths,
    expanded_limit,
    limit,
    n_library_chi: int = 0,
    library_size: int = 1,
):
    """Trim sampled chi so the rotamers they enumerate stay bounded.

    ``depths`` gives each sample's depth in the kinforest that builds it, and
    ``n_library_chi`` how many chi a borrowed rotamer library already defines.
    The expansions are kept while the product stays under ``expanded_limit``
    and dropped at ``limit``; past it, chi freeze from the tip inward. A proton
    chi retains at least one mean, so its hydrogen is still placed. Library
    multiplicity is explicit; the number of chi cannot predict a library's size.
    """
    return tuple(
        cs
        for _, cs in budgeted_chi_samples(
            samples,
            depths,
            expanded_limit,
            limit,
            n_library_chi=n_library_chi,
            library_size=library_size,
        )
    )


def budgeted_chi_samples(
    samples, depths, expanded_limit, limit, n_library_chi=0, library_size=1
):
    """Return (input index, sample) pairs, preserving ownership across blocks.

    ``n_library_chi`` excludes chi already supplied by the same block's library.
    ``library_size`` accounts for an independent library in a group product
    without excluding identically named chi on its attached blocks.
    """
    kept = [
        (i, cs, d)
        for i, (cs, d) in enumerate(zip(samples, depths, strict=True))
        if cs.is_proton or int(cs.chi_dihedral[3:]) > n_library_chi
    ]
    indices = [i for i, _, _ in kept]
    samples = [cs for _, cs, _ in kept]
    depths = [d for _, _, d in kept]
    library = library_size
    if library * n_conformers(samples, True) <= expanded_limit:
        return tuple(zip(indices, samples))

    samples = [attr.evolve(cs, expansions=()) for cs in samples]
    order = sorted(
        (i for i, cs in enumerate(samples) if not cs.is_proton),
        key=lambda i: (depths[i], int(samples[i].chi_dihedral[3:])),
        reverse=True,
    )
    total = library * n_conformers(samples, False)
    for index in order:
        if total <= limit:
            break
        total //= len(samples[index].samples)
        samples[index] = None
    # A proton still needs a placement, not every alternative placement.
    for index in sorted(
        (i for i, cs in enumerate(samples) if cs is not None and cs.is_proton),
        key=lambda i: (depths[i], int(samples[i].chi_dihedral[3:])),
        reverse=True,
    ):
        if total <= limit:
            break
        total //= len(samples[index].samples)
        samples[index] = attr.evolve(samples[index], samples=samples[index].samples[:1])
    if total > limit:
        raise ValueError(
            f"Sampling budget {limit} cannot fit the required {library} library states"
        )
    return tuple((i, cs) for i, cs in zip(indices, samples) if cs is not None)


def chi_depths(rkd, chi_atoms):
    """Depth in a kinforest of the atom each chi turns about.

    The freeze order: a chi deep in the tree moves few atoms, one near the root
    swings everything past it. ``rkd`` is the ResidueKinforestData for the tree
    that will build these chi -- a single residue's for optH, a conjugated
    group's for full packing -- and ``chi_atoms`` gives each chi's defining
    atom in that tree's atom numbering.
    """
    depth = {}
    for atom in rkd.bfto_2_orig:
        parent = rkd.preds[atom]
        depth[int(atom)] = 0 if parent < 0 else depth[int(parent)] + 1
    return [depth.get(int(a), 0) for a in chi_atoms]


def chi_moving_roots(rt, chi_name: str) -> tuple[str, ...]:
    """The atoms a chi turns: everything bonded to its third atom but its second.

    Sidechain roots mark where a sampler stops copying degrees of freedom from
    the input structure and starts rebuilding them from ideal internal
    coordinates. The third atom of a torsion carries the degree of freedom but
    does not itself move, so it must not be a root.
    """
    uaids = rt.torsion_to_uaids.get(chi_name)
    if uaids is None:
        return ()
    held, turned = uaids[1][0], uaids[2][0]
    if held < 0 or turned < 0:
        return ()
    moved = {int(j) for i, j in rt.bond_indices if int(i) == turned and int(j) != held}
    return tuple(rt.atoms[at].name for at in sorted(moved))


def sidechain_roots_for_chis(rt, chi_names) -> tuple[str, ...]:
    """Sidechain roots for a sampler that turns the named chis."""
    roots = {}
    for chi_name in chi_names:
        for at in chi_moving_roots(rt, chi_name):
            roots[at] = None
    return tuple(roots)


def is_fixed_amino_acid(rt) -> bool:
    """Alpha amino acid whose only chi, if any, are proton chi: one conformer."""
    polymer = rt.properties.polymer
    if not polymer.is_polymer or polymer.polymer_type != "amino_acid":
        return False
    if polymer.backbone_type != "alpha_aa":
        return False
    proton_chis = {cs.chi_dihedral for cs in rt.chi_samples if cs.is_proton}
    return all(t.name in proton_chis for t in rt.torsions if t.name.startswith("chi"))


def fixed_amino_acid_roots(rt):
    """CA's sidechain branches: the substituents that carry further atoms.

    With none (glycine), every non-mainchain CA substituent is rebuilt:
    glycine's alpha hydrogens exchange under reflection, so rebuilding only
    one can place it on top of the retained one, and a glycine-to-amino-acid
    design must not inherit an arbitrary one.
    """
    ca, mainchain = rt.atom_to_idx["CA"], rt.properties.polymer.mainchain_atoms
    n_bonds = numpy.bincount(rt.bond_indices[:, 0], minlength=len(rt.atoms))
    substituents = [
        j for i, j in rt.bond_indices if i == ca and rt.atoms[j].name not in mainchain
    ]
    branches = [j for j in substituents if n_bonds[j] > 1]
    return tuple(rt.atoms[j].name for j in (branches or substituents))


def chi_moves_fourth_atom(rt, third, fourth):
    """Whether writing phi_c on a chi's third atom moves its fourth.

    A chi that closes a ring measures an atom upstream of the one carrying its
    degree of freedom, so nothing the packer writes can put that dihedral at a
    requested value; the write would only spin whatever hangs off the third
    atom instead.
    """
    kf = rt.rotamer_kinforest
    target = int(kf.kinforest_idx[third])
    node = int(kf.kinforest_idx[fourth])
    while True:
        parent = int(kf.parent[node])
        if parent == target:
            return True
        if parent < 0 or parent == node:
            return False
        node = parent


def chi4_by_defining_atom(block_types, max_n_atoms):
    """(n_types, max_n_atoms, 4) RTO atom indices of the chi each atom defines.

    Keyed by the chi-defining atom rather than by a chi's position, because a
    residue type's chi are not necessarily numbered without gaps and samplers
    do not agree on how to lay them out in a column. A chi whose fourth atom
    the third does not move is left out: it cannot be set.
    """
    table = numpy.full((len(block_types), max_n_atoms, 4), -1, dtype=numpy.int32)
    for ti, rt in enumerate(block_types):
        for name, uaids in rt.torsion_to_uaids.items():
            if not name.startswith("chi"):
                continue
            four = [int(u[0]) for u in uaids]
            if any(a < 0 for a in four):
                continue
            if not chi_moves_fourth_atom(rt, four[2], four[3]):
                continue
            table[ti, four[2]] = four

    return table


def ring_chi_phi_c_corrections(block_types, max_n_atoms):
    """(n_types, max_n_atoms) phi_c offset for chi that cannot be measured.

    A chi that closes a ring measures an atom upstream of the one carrying its
    degree of freedom, so no trial pass can tell what to add. Its offset is the
    difference between the chi and the phi_c of the ideal geometry, which turns
    what hangs off the third atom by however far the ring has moved from ideal.
    Zero everywhere else, where the measured correction is exact.
    """
    corrections = numpy.zeros((len(block_types), max_n_atoms), dtype=numpy.float32)
    for ti, rt in enumerate(block_types):
        ideal = rt.ideal_coords[rt.at_to_icoor_ind]
        for name, uaids in rt.torsion_to_uaids.items():
            if not name.startswith("chi"):
                continue
            four = [int(u[0]) for u in uaids]
            if any(a < 0 for a in four) or chi_moves_fourth_atom(rt, four[2], four[3]):
                continue
            xyz = ideal[four]
            chi_ideal = float(numpy_coord_dihedrals(xyz[0], xyz[1], xyz[2], xyz[3]))
            kfo = int(rt.rotamer_kinforest.kinforest_idx[four[2]])
            corrections[ti, four[2]] = chi_ideal - float(
                rt.rotamer_kinforest.dofs_ideal[kfo, 3]
            )

    return corrections
