"""Common missing-atom chemistry, construction frames and hydrogen metadata."""

import attr
import numpy
from types import SimpleNamespace

_H_COMPLETION_MAX_NEIGH = 3


@attr.s(auto_attribs=True, slots=True, frozen=True)
class BlockTypeLeafAtomsAnnotation:
    is_leaf: numpy.ndarray


@attr.s(auto_attribs=True, slots=True, frozen=True)
class BlockTypeLeafAtomICoorAnnotation:
    geom: numpy.ndarray
    anc_uaids: numpy.ndarray
    geom_backup: numpy.ndarray
    anc_uaids_backup: numpy.ndarray


@attr.s(auto_attribs=True, slots=True, frozen=True)
class BlockTypeHCompletionAnnotation:
    # For each atom: is it a hydrogen whose position should be re-derived from
    # its parent's heavy-neighbor geometry (rather than a stored dihedral)?
    eligible: numpy.ndarray
    parent: numpy.ndarray
    neigh: numpy.ndarray
    n_neigh: numpy.ndarray
    dist: numpy.ndarray
    # connections borne by the parent: the atoms on their far side are
    # substituents too, and the open vertex is not right without them
    par_conn: numpy.ndarray
    n_par_conn: numpy.ndarray


def _annotate_block_type_atom_is_leaf_atom(block_type, is_hydrogen):
    if hasattr(block_type, "leaf_atom_ann"):
        return
    is_hydrogen = numpy.asarray(is_hydrogen)[: block_type.n_atoms]
    is_parent = numpy.zeros(block_type.n_atoms, dtype=bool)
    icoor_is_parent = numpy.zeros(block_type.n_icoors, dtype=bool)
    icoor_is_parent[block_type.icoors_ancestors[:, 0]] = True

    icoorind_to_atomind = numpy.full(block_type.n_icoors, -1, dtype=numpy.int32)
    icoorind_to_atomind[block_type.at_to_icoor_ind] = numpy.arange(
        block_type.n_atoms, dtype=numpy.int32
    )

    is_parent[icoorind_to_atomind[icoorind_to_atomind != -1]] = icoor_is_parent[
        icoorind_to_atomind != -1
    ]

    # We also need to turn off "is leaf" for any atom that is the
    # fourth one defining a named dihedral, unless, however, that atom
    # is a hydroxyl / free methyl (e.g. chi4 would be on methionine)
    # What atoms / dihedrals would this affect? OD1 / chi2 on ASP and
    # OE1 / chi3 on GLU are not parent atoms and so would be called
    # leaf atoms. Perhaps we could allow these atoms to be missing and
    # build them with a garbage chi2 / chi3 of 180 the way we build
    # missing hydroxyls? It seems like a poor decision.
    fourth_torsion_atoms = block_type.ordered_torsions[:, 3, 0]
    real_fourth_torsion_atoms = fourth_torsion_atoms[fourth_torsion_atoms != -1]

    is_fourth_chi_atom = numpy.zeros(block_type.n_atoms, dtype=bool)
    is_fourth_chi_atom[real_fourth_torsion_atoms] = True
    is_fourth_chi_atom[is_hydrogen == 1] = False
    is_parent = numpy.logical_or(is_parent, is_fourth_chi_atom)

    is_leaf = numpy.logical_not(is_parent)

    # special case: we cannot build missing atoms if there are not enough
    # atoms to define a coordinate frame
    if block_type.n_atoms == 3 and not block_type.connections:
        is_leaf[:] = False

    leaf_annotation = BlockTypeLeafAtomsAnnotation(is_leaf)

    setattr(block_type, "leaf_atom_ann", leaf_annotation)


def _uaid_for_at(bt, icoor_at_name):
    if icoor_at_name in bt.connection_to_cidx or icoor_at_name in ("up", "down"):
        return (-1, bt.connection_to_cidx.get(icoor_at_name, -1), 0)
    return (bt.atom_to_idx[icoor_at_name], -1, -1)


def _icoor_at_is_leaf(bt, icoor_at_name):
    if icoor_at_name not in bt.atom_to_idx:
        return 0
    return bt.leaf_atom_ann.is_leaf[bt.atom_to_idx[icoor_at_name]]


def _icoor_at_is_h(bt, atom_is_hydrogen, icoor_at_name):
    if icoor_at_name not in bt.atom_to_idx:
        return 0
    return atom_is_hydrogen[bt.atom_to_idx[icoor_at_name]]


def _icoor_at_is_inter_res(bt, icoor_at_name):
    return icoor_at_name not in bt.atom_to_idx


def _determine_leaf_atom_icoors_for_block_type(bt, atom_is_hydrogen):  # noqa: C901
    if hasattr(bt, "leaf_atom_icoor_ann"):
        return

    icoor_uaids = numpy.full((bt.n_atoms, 3, 3), -1, dtype=numpy.int32)
    icoor_geom = numpy.full((bt.n_atoms, 3), 0, dtype=numpy.float32)
    icoor_uaids_backup = numpy.full((bt.n_atoms, 3, 3), -1, dtype=numpy.int32)
    icoor_geom_backup = numpy.full((bt.n_atoms, 3), 0, dtype=numpy.float32)

    # We cannot build hydrogen atoms on a water molecule, for example
    # because there isn't a (meaningful) coordinate frame we can create
    # from only a single xyz coordinate. So, for now, we will skip
    # water.
    if bt.n_atoms <= 3 and not bt.connections:
        ann = BlockTypeLeafAtomICoorAnnotation(
            geom=icoor_geom,
            anc_uaids=icoor_uaids,
            geom_backup=icoor_geom_backup,
            anc_uaids_backup=icoor_uaids_backup,
        )
        setattr(bt, "leaf_atom_icoor_ann", ann)
        return
    cap_phi_offsets = {}
    torsion_defining_atoms = set(bt.ordered_torsions[:, 3, 0])
    for j, at in enumerate(bt.atoms):
        atname = at.name
        j_icoor_ind = bt.icoors_index[atname]
        j_icoor = bt.icoors[j_icoor_ind]

        # ok, let's turn p, gp, and ggp into uaids
        # if ggp is a leaf, then we need to recurse backwards through the ggps
        # and accumulate the phi offsets

        p_uaid = _uaid_for_at(bt, j_icoor.parent)
        gp_uaid = _uaid_for_at(bt, j_icoor.grand_parent)

        phi = j_icoor.phi
        theta = numpy.pi - j_icoor.theta
        d = j_icoor.d

        if _icoor_at_is_h(bt, atom_is_hydrogen, j_icoor.great_grand_parent):
            # use phi offsets from the non-leaf ggp* ancestor of the ggp
            # as the default strategy for building coords for hydrogen atoms
            # Note that this algorithm only works if we do not have two atoms
            # that are each other's ggps / aren't their own ggps, which can
            # and will happen if, e.g., the residue has only three atoms total.
            # that is why we handle that case
            seen = numpy.zeros(bt.n_icoors, dtype=bool)
            seen[j_icoor_ind] = True
            while _icoor_at_is_leaf(bt, j_icoor.great_grand_parent):
                ggp_ind = bt.icoors_index[j_icoor.great_grand_parent]
                if seen[ggp_ind]:
                    # cycle detected — can happen for small-molecule ligands
                    # where all atoms are leaves; break and use accumulated phi
                    break
                else:
                    seen[ggp_ind] = True
                j_icoor = bt.icoors[ggp_ind]
                phi += j_icoor.phi
        else:
            # if the ggp is not a hydrogen, even if it's a leaf atom, then try
            # and build the coordinate for this atom based on its position first
            # before falling back on the non-leaf ggp* ancestor. This "general"
            # code is specifically for building the OXT atom on a cterm residue
            # when the O atom is provided; the phi should be 180 off O and not
            # 260 off N (and some unknown offset of O).
            pass
        ggp_uaid = _uaid_for_at(bt, j_icoor.great_grand_parent)

        # A one-heavy-atom polymer cap (e.g. NH2) has no local third
        # reference for its hydrogen plane. Continue one bond into the
        # connected residue rather than reusing the cap's nitrogen. Its
        # first hydrogen is trans to that reference; retain the remaining
        # hydrogens' relative dihedrals from the generated chemical geometry.
        if gp_uaid[1] >= 0 and ggp_uaid == p_uaid:
            ggp_uaid = (-1, gp_uaid[1], 1)
            if atom_is_hydrogen[j]:
                phi += cap_phi_offsets.setdefault(p_uaid, numpy.pi - phi)

        ggp_ind_backup = None
        phi_backup = phi
        if _icoor_at_is_inter_res(bt, j_icoor.great_grand_parent):
            # Logic for when the great-grand parent atom is in another residue
            # and is absent. This "general" logic is specifically for building
            # the H atom on a residue where i-1 does not exist or is not
            # chemically bonded to residue i.
            while _icoor_at_is_inter_res(bt, j_icoor.great_grand_parent):
                ggp_ind_backup = bt.icoors_index[j_icoor.great_grand_parent]
                j_icoor = bt.icoors[ggp_ind_backup]
                phi_backup += j_icoor.phi
        elif not _icoor_at_is_h(bt, atom_is_hydrogen, j_icoor.great_grand_parent):
            # Logic for handling when the heavy-atom great-grand parent,
            # which itself is a leaf atom, is absent. This "general" logic
            # is specifically for building the OXT atom on a cterm residue
            # when the O atom is given but OXT is not.
            seen_backup = set()
            while True:
                reference = j_icoor.great_grand_parent
                candidate = bt.icoors[bt.icoors_index[reference]]
                # HA can reference an unresolved non-leaf CB. Its fixed CB
                # dihedral uses the same CA-N axis, so composing the two phi
                # offsets supplies an independent C reference. A sampled
                # torsion must not be replaced by its ideal dihedral.
                fixed_axis = (
                    candidate.parent == bt.icoors[j_icoor_ind].parent
                    and candidate.grand_parent == bt.icoors[j_icoor_ind].grand_parent
                    and bt.atom_to_idx.get(reference, -1) not in torsion_defining_atoms
                )
                if not (_icoor_at_is_leaf(bt, reference) or fixed_axis):
                    break
                ggp_ind_backup = bt.icoors_index[j_icoor.great_grand_parent]
                if ggp_ind_backup in seen_backup:
                    break
                seen_backup.add(ggp_ind_backup)
                j_icoor = bt.icoors[ggp_ind_backup]
                phi_backup += j_icoor.phi

        icoor_uaids[j, 0] = numpy.array(p_uaid, dtype=numpy.int32)
        icoor_uaids[j, 1] = numpy.array(gp_uaid, dtype=numpy.int32)
        icoor_uaids[j, 2] = numpy.array(ggp_uaid, dtype=numpy.int32)

        icoor_geom[j, 0] = phi
        icoor_geom[j, 1] = theta
        icoor_geom[j, 2] = d
        if ggp_ind_backup is not None:
            ggp_uaid_backup = _uaid_for_at(bt, j_icoor.great_grand_parent)
            icoor_uaids_backup[j, 0] = numpy.array(p_uaid, dtype=numpy.int32)
            icoor_uaids_backup[j, 1] = numpy.array(gp_uaid, dtype=numpy.int32)
            icoor_uaids_backup[j, 2] = numpy.array(ggp_uaid_backup, dtype=numpy.int32)

            icoor_geom_backup[j, 0] = phi_backup
            icoor_geom_backup[j, 1] = theta
            icoor_geom_backup[j, 2] = d

    ann = BlockTypeLeafAtomICoorAnnotation(
        geom=icoor_geom,
        anc_uaids=icoor_uaids,
        geom_backup=icoor_geom_backup,
        anc_uaids_backup=icoor_uaids_backup,
    )
    setattr(bt, "leaf_atom_icoor_ann", ann)


def _determine_h_completion_for_block_type(bt, atom_is_hydrogen):
    """Flag, per block type, which added H should be placed from geometry.

    An autogen ligand mixes two conformers: heavy atoms come from the input
    structure, H icoors from the OpenBabel one.
     - At an invertible center the stored dihedral can aim the H at a neighbor.
     - We flag such "inversion centers" here.
    """
    if hasattr(bt, "h_completion_ann"):
        return
    n = bt.n_atoms
    is_h = numpy.asarray(atom_is_hydrogen).astype(bool)[:n]
    eligible = numpy.zeros(n, dtype=bool)
    parent = numpy.full(n, -1, dtype=numpy.int32)
    neigh = numpy.full((n, _H_COMPLETION_MAX_NEIGH), -1, dtype=numpy.int32)
    n_neigh = numpy.zeros(n, dtype=numpy.int32)
    dist = numpy.zeros(n, dtype=numpy.float32)
    par_conn = numpy.full((n, _H_COMPLETION_MAX_NEIGH), -1, dtype=numpy.int32)
    n_par_conn = numpy.zeros(n, dtype=numpy.int32)

    # Two cases need a hydrogen placed from geometry rather than from a stored
    # dihedral: an autogen ligand, whose heavy atoms and H icoors come from
    # different conformers, and an atom a conjugation bonded to something,
    # whose icoors were measured before the bond existed and so describe the
    # wrong geometry once it does. Canonical residues are otherwise left with
    # their well-tested dihedral-built hydrogens, the backbone amide's
    # included: its connection is the polymer's, not a conjugation.
    structural = {bt.down_connection_ind, bt.up_connection_ind}
    conjugated_atoms = {
        int(at)
        for i, at in enumerate(bt.ordered_connection_atoms)
        if i not in structural
    }
    if getattr(bt, "hydrogens_regenerated", False) or conjugated_atoms:
        adj = [[] for _ in range(n)]
        for a0, a1 in bt.bond_indices:
            adj[int(a0)].append(int(a1))
        conns_on_atom = {}
        for ci, at in enumerate(bt.ordered_connection_atoms):
            conns_on_atom.setdefault(int(at), []).append(ci)
        for j in range(n):
            if not is_h[j]:
                continue
            j_neighbors = adj[j]
            if len(j_neighbors) != 1:
                continue  # only ordinary terminal H
            par = j_neighbors[0]
            if is_h[par]:
                continue
            par_heavy = [k for k in adj[par] if not is_h[k]]
            par_n_h = sum(1 for k in adj[par] if is_h[k])
            par_conns = conns_on_atom.get(par, [])
            if len(par_heavy) + len(par_conns) < 2 or par_n_h != 1:
                continue
            if not getattr(bt, "hydrogens_regenerated", False):
                # a canonical residue only needs this where a conjugation
                #    changed the site out from under its icoors
                if par not in conjugated_atoms:
                    continue
            par_heavy = par_heavy[:_H_COMPLETION_MAX_NEIGH]
            eligible[j] = True
            parent[j] = par
            n_neigh[j] = len(par_heavy)
            neigh[j, : len(par_heavy)] = par_heavy
            par_conns = par_conns[:_H_COMPLETION_MAX_NEIGH]
            n_par_conn[j] = len(par_conns)
            par_conn[j, : len(par_conns)] = par_conns
            dist[j] = float(bt.icoors[bt.icoors_index[bt.atoms[j].name]].d)

    setattr(
        bt,
        "h_completion_ann",
        BlockTypeHCompletionAnnotation(
            eligible=eligible,
            parent=parent,
            neigh=neigh,
            n_neigh=n_neigh,
            dist=dist,
            par_conn=par_conn,
            n_par_conn=n_par_conn,
        ),
    )


def prepare_leaf_atom_flags(block_types, atom_is_hydrogen, max_n_atoms):
    is_leaf_atom = numpy.zeros((len(block_types), max_n_atoms), dtype=bool)

    for i, block_type in enumerate(block_types):
        _annotate_block_type_atom_is_leaf_atom(block_type, atom_is_hydrogen[i, :])
        is_leaf_atom[i, : block_type.n_atoms] = block_type.leaf_atom_ann.is_leaf

    # A construction frame needs three other atoms, possibly across connections.
    conformer_types = [
        (
            not bt.properties.polymer.is_polymer
            or not bt.properties.is_canonical
            or bt.properties.polymer.polymer_type == "nucleic_acid"
        )
        and (bt.n_atoms > 3 or bool(bt.connections))
        for bt in block_types
    ]
    conformer_atoms = None
    has_conformer_frames = any(conformer_types)
    if has_conformer_frames:
        rebuildable = numpy.zeros((len(block_types), max_n_atoms), dtype=bool)
        for i, bt in enumerate(block_types):
            if conformer_types[i]:
                mainchain = set(bt.properties.polymer.mainchain_atoms or ())
                rebuildable[i, : bt.n_atoms] = numpy.asarray(
                    [atom.name not in mainchain for atom in bt.atoms]
                )
                if bt.properties.is_canonical:
                    rebuildable[i] &= is_leaf_atom[i]
                # A retained 5′ phosphate can be unresolved after a backbone
                # gap. Its O5′–C5′–C4′ frame is local to the resolved sugar.
                if bt.properties.polymer.polymer_type == "nucleic_acid":
                    # Fixed sugar branches can be reconstructed from observed
                    # ring atoms even though their hydrogens make them non-leaves.
                    for name in ("P", "O4'", "C1'", "C2'", "O2'"):
                        atom = bt.atom_to_idx.get(name)
                        if atom is not None:
                            rebuildable[i, atom] = True
        conformer_atoms = rebuildable
    return SimpleNamespace(
        is_leaf_atom=is_leaf_atom,
        has_conformer_frames=has_conformer_frames,
        conformer_atoms=conformer_atoms,
        conformer_types=numpy.asarray(conformer_types, dtype=bool),
    )


def prepare_leaf_icoor_metadata(block_types, atom_is_hydrogen, max_n_atoms):
    icoor_atom_ancestor_uaids = numpy.full(
        (len(block_types), max_n_atoms, 3, 3), -1, dtype=numpy.int32
    )
    icoor_geom = numpy.full((len(block_types), max_n_atoms, 3), -1, dtype=numpy.float32)
    icoor_atom_ancestor_uaids_backup = numpy.full(
        (len(block_types), max_n_atoms, 3, 3), -1, dtype=numpy.int32
    )
    icoor_geom_backup = numpy.full(
        (len(block_types), max_n_atoms, 3), -1, dtype=numpy.float32
    )
    for i, bt in enumerate(block_types):
        _annotate_block_type_atom_is_leaf_atom(bt, atom_is_hydrogen[i, :])
        _determine_leaf_atom_icoors_for_block_type(bt, atom_is_hydrogen[i, :])

        ann = bt.leaf_atom_icoor_ann
        icoor_geom[i, : bt.n_atoms] = ann.geom
        icoor_atom_ancestor_uaids[i, : bt.n_atoms] = ann.anc_uaids
        icoor_geom_backup[i, : bt.n_atoms] = ann.geom_backup
        icoor_atom_ancestor_uaids_backup[i, : bt.n_atoms] = ann.anc_uaids_backup

    return BlockTypeLeafAtomICoorAnnotation(
        icoor_geom,
        icoor_atom_ancestor_uaids,
        icoor_geom_backup,
        icoor_atom_ancestor_uaids_backup,
    )


def prepare_h_completion_metadata(block_types, atom_is_hydrogen, max_n_atoms):
    eligible = numpy.zeros((len(block_types), max_n_atoms), dtype=bool)
    parent = numpy.full((len(block_types), max_n_atoms), -1, dtype=numpy.int64)
    neigh = numpy.full(
        (len(block_types), max_n_atoms, _H_COMPLETION_MAX_NEIGH), -1, dtype=numpy.int64
    )
    n_neigh = numpy.zeros((len(block_types), max_n_atoms), dtype=numpy.int64)
    dist = numpy.zeros((len(block_types), max_n_atoms), dtype=numpy.float32)
    par_conn = numpy.full(
        (len(block_types), max_n_atoms, _H_COMPLETION_MAX_NEIGH), -1, dtype=numpy.int64
    )
    n_par_conn = numpy.zeros((len(block_types), max_n_atoms), dtype=numpy.int64)
    for i, bt in enumerate(block_types):
        _determine_h_completion_for_block_type(bt, atom_is_hydrogen[i, :])
        ann = bt.h_completion_ann
        eligible[i, : bt.n_atoms] = ann.eligible
        parent[i, : bt.n_atoms] = ann.parent
        neigh[i, : bt.n_atoms] = ann.neigh
        n_neigh[i, : bt.n_atoms] = ann.n_neigh
        dist[i, : bt.n_atoms] = ann.dist
        par_conn[i, : bt.n_atoms] = ann.par_conn
        n_par_conn[i, : bt.n_atoms] = ann.n_par_conn

    return BlockTypeHCompletionAnnotation(
        eligible, parent, neigh, n_neigh, dist, par_conn, n_par_conn
    )


def prepare_leaf_atom_metadata(block_types, atom_is_hydrogen, max_n_atoms):
    """Prepare all common flags/frames using the caller's chemical H identity."""
    flags = prepare_leaf_atom_flags(block_types, atom_is_hydrogen, max_n_atoms)
    return SimpleNamespace(
        **vars(flags),
        icoors=prepare_leaf_icoor_metadata(block_types, atom_is_hydrogen, max_n_atoms),
        h_completion=prepare_h_completion_metadata(
            block_types, atom_is_hydrogen, max_n_atoms
        ),
    )
