"""Mirror-image rotamer libraries for the D-amino acids.

A D residue's rotamer statistics are its L counterpart's read at negated
backbone and sidechain torsions. Mirroring the libraries -- rather than the
packed tensors the scoring kernels read -- means the existing packing code
builds offsets and indices for D libraries using the same layout as L ones.
Explicit reflection metadata lets sampling choose ordering cells in the source
grid and lets spline fitting choose the reflected chi-mean branch.

Under phi,psi,chi -> -phi,-psi,-chi:

* the probability, mean and deviation tables are read at the reflected
  backbone bin,
* chi means negate while deviations do not, and
* a rotamer well swaps with its mirror, which for a chi with n wells is
  n + 1 - well.

Rows keep their positions. Every lookup that matters is built from the well
labels, and the one positional lookup -- the block of rows belonging to a
semirotameric residue's rotamer -- indexes those rows in order, so relabeling
in place leaves both correct.
"""

import attr
import numpy as np

# a d library is named for the l library it mirrors
D_PREFIX = "d"


def _d_table_name(table_name: str) -> str:
    return D_PREFIX + table_name


def _wells_per_chi(data) -> np.ndarray:
    """How many wells each chi is binned into.

    Taken from the alias table as well as the rotamer table: a library need not
    define every combination, and the aliases redirect the undefined ones, so
    they can name wells the rotamer table never shows. Proline is the case that
    matters -- its rotamer table uses two wells of chi1 and one each of chi2
    and chi3, while its aliases use three of each.
    """
    counts = data.rotamers.max(axis=0)
    alias = data.rotamer_alias
    if alias.size:
        n_chi = data.rotamers.shape[1]
        counts = np.maximum(
            counts,
            np.maximum(
                alias[:, :n_chi].max(axis=0),
                alias[:, n_chi:].max(axis=0),
            ),
        )
    return counts


def _mirror_wells(wells: np.ndarray, n_per_column: np.ndarray) -> np.ndarray:
    """Relabel rotamer wells with their mirror images."""
    if wells.size == 0:
        return wells
    return n_per_column[None, :] + 1 - wells


def _reflect(table: np.ndarray, dims, start, step) -> np.ndarray:
    """Permute reflected points onto the grid from ``_reflected_grid_start``.

    Point k moves to (c - k) modulo the number of bins, where c is the nearest
    integer to -2*start/step. Changing the target origin to -start-c*step
    preserves the exact reflected coordinates even for nonaligned grids.
    """
    out = table
    for dim, dim_start, dim_step in zip(dims, start, step):
        n = out.shape[dim]
        c = int(round(-2.0 * float(dim_start) / float(dim_step)))
        out = np.roll(np.flip(out, axis=dim), (c - n + 1) % n, axis=dim)
    return out


def _reflected_grid_start(start, step):
    """Choose the exact reflected origin nearest the original origin."""
    return -start - round(-2.0 * start / step) * step


def _backbone_dims(data):
    """(dims, starts, steps) of the backbone axes of the probability table."""
    n_bb = data.backbone_dihedral_start.shape[0]
    return (
        tuple(range(1, n_bb + 1)),
        [float(v) for v in data.backbone_dihedral_start],
        [float(v) for v in data.backbone_dihedral_step],
    )


def _mirror_rotameric_data(data):
    dims, start, step = _backbone_dims(data)
    target_start = [_reflected_grid_start(s, d) for s, d in zip(start, step)]
    # The default aligned grids retain the same tensor and physical origin.
    target_start = (
        data.backbone_dihedral_start
        if target_start == start
        else np.asarray(target_start, dtype=data.backbone_dihedral_start.dtype)
    )
    was_mirrored = getattr(data, "backbone_is_mirrored", False)
    # the rotamer table and its aliases share one count per chi, or an alias
    #    will redirect onto a well tuple that no row carries
    counts = _wells_per_chi(data)
    n_chi = data.rotamers.shape[1]
    alias = data.rotamer_alias
    return attr.evolve(
        data,
        backbone_is_mirrored=not was_mirrored,
        backbone_dihedral_start=target_start,
        backbone_source_start=None if was_mirrored else data.backbone_dihedral_start,
        rotamers=_mirror_wells(data.rotamers, counts),
        rotamer_probabilities=_reflect(data.rotamer_probabilities, dims, start, step),
        rotamer_means=-_reflect(data.rotamer_means, dims, start, step),
        rotamer_stdvs=_reflect(data.rotamer_stdvs, dims, start, step),
        # entries name rows, which keep their positions; only the backbone bin
        #    they are read from reflects
        prob_sorted_rot_inds=_reflect(
            data.prob_sorted_rot_inds, tuple(d - 1 for d in dims), start, step
        ),
        rotamer_alias=(
            np.concatenate(
                (
                    _mirror_wells(alias[:, :n_chi], counts),
                    _mirror_wells(alias[:, n_chi:], counts),
                ),
                axis=1,
            )
            if alias.size
            else alias
        ),
    )


def _mirror_rotameric_library(
    library,
):
    return attr.evolve(
        library,
        table_name=_d_table_name(library.table_name),
        rotameric_data=_mirror_rotameric_data(library.rotameric_data),
    )


def _mirror_semi_rotameric_library(
    library,
):
    data = library.rotameric_data
    dims, start, step = _backbone_dims(data)
    # the non-rotameric chi is the trailing axis and negates with the rest
    nonrot = _reflect(
        library.nonrotameric_chi_probabilities,
        (*dims, library.nonrotameric_chi_probabilities.ndim - 1),
        [*start, library.non_rot_chi_start],
        [*step, library.non_rot_chi_step],
    )
    chi_rotamers = library.rotameric_chi_rotamers
    return attr.evolve(
        library,
        table_name=_d_table_name(library.table_name),
        rotameric_data=_mirror_rotameric_data(data),
        non_rot_chi_start=_reflected_grid_start(
            library.non_rot_chi_start, library.non_rot_chi_step
        ),
        non_rot_chi_step=library.non_rot_chi_step,
        non_rot_chi_period=library.non_rot_chi_period,
        rotameric_chi_rotamers=_mirror_wells(chi_rotamers, chi_rotamers.max(axis=0)),
        nonrotameric_chi_probabilities=nonrot,
        # a well spanning [left, right] spans [-right, -left] mirrored
        rotamer_boundaries=-np.flip(library.rotamer_boundaries, axis=1),
    )


def _with_mirrored_libraries(
    library,
    d_residue_names,
    *,
    mirror_rotameric=_mirror_rotameric_library,
    mirror_semi=_mirror_semi_rotameric_library,
):
    """Append only required mirrored libraries and their missing lookup rows.

    ``d_residue_names`` maps an L residue name to the name of the D residue
    type that mirrors it; a residue whose L form has no rotamer library keeps
    none, as glycine and alanine do. Existing target mappings are authoritative.
    A generated table name must be free: an explicit lookup can reuse an
    existing table without guessing its provenance from a name prefix.
    """
    if not d_residue_names:
        return library

    tables = {}
    for lib in (*library.rotameric_libraries, *library.semi_rotameric_libraries):
        if lib.table_name in tables:
            raise ValueError(f"Duplicate Dunbrack table name: {lib.table_name}")
        tables[lib.table_name] = lib
    existing = {}
    for row in library.dun_lookup:
        if row.residue_name in existing:
            raise ValueError(f"Duplicate Dunbrack residue mapping: {row.residue_name}")
        existing[row.residue_name] = row.dun_table_name

    lookup = list(library.dun_lookup)
    required = set()
    added = {}
    for mapping in library.dun_lookup:
        d_name = d_residue_names.get(mapping.residue_name)
        if d_name is None or d_name in existing:
            continue
        source = mapping.dun_table_name
        target = _d_table_name(source)
        if d_name in added:
            if added[d_name] != target:
                raise ValueError(
                    f"Conflicting mirrored libraries requested for {d_name}"
                )
            continue
        if source not in tables:
            raise ValueError(f"Missing source Dunbrack table: {source}")
        if target in tables:
            raise ValueError(
                f"Mirrored Dunbrack table name collision: {target}; "
                f"provide an explicit mapping for {d_name} to reuse a table"
            )
        required.add(source)
        added[d_name] = target
        lookup.append(attr.evolve(mapping, dun_table_name=target, residue_name=d_name))

    if not added:
        return library

    return attr.evolve(
        library,
        dun_lookup=tuple(lookup),
        rotameric_libraries=(
            *library.rotameric_libraries,
            *(
                mirror_rotameric(lib)
                for lib in library.rotameric_libraries
                if lib.table_name in required
            ),
        ),
        semi_rotameric_libraries=(
            *library.semi_rotameric_libraries,
            *(
                mirror_semi(lib)
                for lib in library.semi_rotameric_libraries
                if lib.table_name in required
            ),
        ),
    )
