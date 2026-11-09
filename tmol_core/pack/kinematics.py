"""Shared residue and connected-group rotamer kinematics preparation.

Chemistry, reference frames, scan planning, coordinate order and padding are
common. ``inverse_kinematics`` executes the canonical native inverse on CPU;
it accepts float32 coordinates (including the zero virtual root), followed by
int32 parent/frame_x/frame_y/frame_z/doftype arrays, and returns float32 DOFs.
"""

import attr
import numpy
from types import SimpleNamespace
from typing import Generic, TypeVar
from tmol_core.array import invert_mapping
from tmol_core.kinematics import NodeType
from tmol_core.kinematics.chemical import (
    _annotate_block_type_with_residue_kinforest_data,
    _block_group_kinforest_data,
)
from tmol_core.kinematics.scan_ordering import prepare_scan_ordering

IdealArray = TypeVar("IdealArray")


@attr.s(auto_attribs=True, frozen=True, slots=True)
class RotamerKintree:
    """Residue-local kinematic tree and scan ordering in NumPy form."""

    kinforest_idx: numpy.ndarray  # mapping from restype order to kinforest order

    # the following arrays are all in kinforest order
    id: numpy.ndarray  # mapping from kinforest order to restype order
    doftype: numpy.ndarray
    parent: numpy.ndarray
    frame_x: numpy.ndarray
    frame_y: numpy.ndarray
    frame_z: numpy.ndarray
    nodes: numpy.ndarray
    scans: numpy.ndarray
    gens: numpy.ndarray
    n_scans_per_gen: numpy.ndarray
    dofs_ideal: numpy.ndarray


@attr.s(auto_attribs=True, frozen=True, slots=True)
class PackedRotamerKintree(Generic[IdealArray]):
    """Padded batch of residue-local kinematic trees."""

    kinforest_idx: numpy.ndarray  # mapping from restype order to kinforest order

    # the following arrays are all in kinforest order
    id: numpy.ndarray  # mapping from kinforest order to restype order
    doftype: numpy.ndarray
    parent: numpy.ndarray
    frame_x: numpy.ndarray
    frame_y: numpy.ndarray
    frame_z: numpy.ndarray
    n_nodes: numpy.ndarray
    nodes: numpy.ndarray
    scans: numpy.ndarray
    gens: numpy.ndarray
    n_scans_per_gen: numpy.ndarray
    dofs_ideal: IdealArray


def prepare_rotamer_kinematics(
    rkd, n_atoms, atom_names, mainchain_atoms, ideal, inverse_kinematics
):
    """Build a RotamerKintree from a spanning tree over a flat atom array.

    Shared by the single-residue and block-group paths: nothing here asks which
    block an atom belongs to. ``mainchain_atoms`` is the set whose atoms are
    preferred as reference-frame atoms (they hold still across rotamers); for a
    group it is the anchor block's.
    """
    kfo_2_to = rkd.bfto_2_orig.astype(numpy.int64)  # KFO → TO (0-indexed)
    preds = rkd.preds.astype(numpy.int64)  # BFS predecessors in TO; -9999 for root
    dof_type_to = rkd.dof_type  # NodeType per atom in TO

    # TO -> KFO mapping (used for kinforest_idx output field)
    to_2_kfo = invert_mapping(kfo_2_to, n_atoms).astype(numpy.int64)

    # KFO parent indices, 0-indexed (root points to itself)
    kfo_parents = numpy.zeros(n_atoms, dtype=numpy.int64)
    kfo_parents[1:] = to_2_kfo[preds[kfo_2_to[1:]]]

    # DOF types in KFO order
    dof_type_kfo = dof_type_to[kfo_2_to].astype(numpy.int32)

    # --- Vectorized frame defaults (correct for all bond atoms) ---
    # frame_x = self, frame_y = parent, frame_z = grandparent
    frame_x = numpy.arange(n_atoms, dtype=numpy.int64)
    frame_y = kfo_parents.copy()
    frame_z = kfo_parents[kfo_parents]

    # --- Fix jump root (KFO index 0) and its direct children ---
    # c1, c2 are the two frame atoms for root:
    #   if root has >= 2 children: c1, c2 = first two children of root
    #   if root has only 1 child:  c1 = that child, c2 = first child of c1
    root_children = numpy.where((kfo_parents == 0) & (numpy.arange(n_atoms) > 0))[0]

    if mainchain_atoms:
        # Polymer residues: prefer mainchain atoms as frame atoms because their
        # positions are fixed across rotamers (otherwise CB ends up as the ref
        # frame and is not idealized).
        mc_atoms = set(mainchain_atoms)

        def sort_key(i):
            return (atom_names[kfo_2_to[i]] not in mc_atoms, i)

        root_children = sorted(root_children, key=sort_key)
        c1 = int(root_children[0])
        if len(root_children) >= 2:
            c2 = int(root_children[1])
        else:
            c1_children = numpy.where(kfo_parents == c1)[0]
            c1_children = sorted(c1_children, key=sort_key)
            # a two-atom residue (1GJ2 O: O, HO) has no grandchild to frame on
            c2 = int(c1_children[0]) if len(c1_children) else 0
    else:
        # Ligand / non-polymer: no mainchain to prefer. Use the root's children
        #   (and grandchildren) in the order produced by the params-file icoor
        #   tree
        # need to handle case where natoms < 3
        if len(root_children) == 0:
            c1 = 0
            c2 = 0
        elif len(root_children) >= 2:
            c1 = int(root_children[0])
            c2 = int(root_children[1])
        else:
            c1 = int(root_children[0])
            c1_children = numpy.where(kfo_parents == c1)[0]
            c2 = int(c1_children[0]) if len(c1_children) > 0 else 0

    # Root (KFO 0): frame_x=c1, frame_y=self(=0), frame_z=c2
    frame_x[0] = c1
    frame_y[0] = 0
    frame_z[0] = c2

    # c1's own default frame_z is the root, which is no frame at all; it takes
    # c2 instead.
    frame_z[c1] = c2

    # Every other direct child of the root takes c1. Note that c1 is whichever
    # child the ordering above chose, NOT necessarily the first one: with a
    # mainchain to prefer it is often a later atom. Selecting these by index
    # rather than by "not c1" leaves whichever child sits at index 1 with the
    # root as its frame_z -- a frame of two distinct points, which gives no
    # reference direction, and every atom built on it comes out wrong.
    arange = numpy.arange(n_atoms)
    other_root_children = numpy.where(
        (kfo_parents == 0) & (arange > 0) & (arange != c1)
    )[0]
    frame_z[other_root_children] = c1

    # --- Build KinForest with global root at position 0 (required by inverse_kin) ---
    def _t(x):
        return numpy.asarray(x, dtype=numpy.int32)

    # kfo_parents[0] == 0 is a self-loop in KFO space (the jump root is its own parent).
    # After the +1 shift to kinforest positions this gives parent[1] = 1, which
    # get_scans treats as a disconnected second root (segfault).
    # Fix: use -1 for the jump root so that -1+1 = 0 (points to the global root).
    kfo_parents_kf = kfo_parents.copy()
    kfo_parents_kf[0] = -1

    kinforest = SimpleNamespace(
        id=_t(numpy.concatenate([[-1], kfo_2_to])),
        doftype=_t(numpy.concatenate([[NodeType.root], dof_type_kfo])),
        parent=_t(numpy.concatenate([[0], kfo_parents_kf + 1])),
        frame_x=_t(numpy.concatenate([[0], frame_x + 1])),
        frame_y=_t(numpy.concatenate([[0], frame_y + 1])),
        frame_z=_t(numpy.concatenate([[0], frame_z + 1])),
    )

    forward_scan_paths = prepare_scan_ordering(kinforest.parent).forward
    nodes = forward_scan_paths.nodes
    scans = forward_scan_paths.scans
    gens = forward_scan_paths.gens

    n_scans_per_gen = gens[1:, 1] - gens[:-1, 1]

    ideal_coords = numpy.zeros((n_atoms + 1, 3), dtype=numpy.float32)
    ideal_coords[1:] = ideal[kfo_2_to]

    dofs_ideal = inverse_kinematics(
        ideal_coords,
        kinforest.parent,
        kinforest.frame_x,
        kinforest.frame_y,
        kinforest.frame_z,
        kinforest.doftype,
    )
    dofs_ideal = numpy.asarray(dofs_ideal)
    if dofs_ideal.dtype != numpy.float32 or dofs_ideal.shape != (n_atoms + 1, 9):
        raise ValueError(
            "inverse_kinematics must return float32 DOFs shaped [n_atoms+1, 9]"
        )

    return RotamerKintree(
        kinforest_idx=to_2_kfo.astype(numpy.int32),
        id=kfo_2_to.astype(numpy.int32),
        doftype=dof_type_kfo,
        parent=kfo_parents_kf.astype(
            numpy.int32
        ),  # root is -1; load_rotamer_parents adds 1
        frame_x=frame_x.astype(numpy.int32),
        frame_y=frame_y.astype(numpy.int32),
        frame_z=frame_z.astype(numpy.int32),
        nodes=nodes,
        scans=scans,
        gens=gens,
        n_scans_per_gen=n_scans_per_gen,
        dofs_ideal=dofs_ideal[1:],
    )


def pack_rotamer_kinematics(block_types):
    """Coalesce prepared residue/group forests with native padding conventions."""
    n_types = len(block_types)

    max_n_nodes = max(
        (len(rt.rotamer_kinforest.nodes) for rt in block_types), default=0
    )
    max_n_scans = max(
        (rt.rotamer_kinforest.scans.shape[0] for rt in block_types), default=0
    )
    max_n_gens = max(
        (rt.rotamer_kinforest.gens.shape[0] for rt in block_types), default=0
    )
    max_n_atoms = max(
        (rt.rotamer_kinforest.id.shape[0] for rt in block_types), default=0
    )

    rt_n_nodes = numpy.zeros((n_types,), dtype=numpy.int32)
    rt_nodes = numpy.full((n_types, max_n_nodes), -1, dtype=numpy.int32)
    rt_scans = numpy.full((n_types, max_n_scans), -1, dtype=numpy.int32)
    rt_gens = numpy.full((n_types, max_n_gens, 2), 0, dtype=numpy.int32)
    rt_n_scans_per_gen = numpy.full(
        (n_types, max(max_n_gens - 1, 0)), -1, dtype=numpy.int32
    )

    rt_kinforest_idx = numpy.full((n_types, max_n_atoms), -1, dtype=numpy.int32)
    rt_id = numpy.full((n_types, max_n_atoms), -1, dtype=numpy.int32)
    rt_doftype = numpy.full((n_types, max_n_atoms), -1, dtype=numpy.int32)
    rt_parent = numpy.full((n_types, max_n_atoms), -1, dtype=numpy.int32)
    rt_frame_x = numpy.full((n_types, max_n_atoms), -1, dtype=numpy.int32)
    rt_frame_y = numpy.full((n_types, max_n_atoms), -1, dtype=numpy.int32)
    rt_frame_z = numpy.full((n_types, max_n_atoms), -1, dtype=numpy.int32)
    rt_dofs_ideal = numpy.zeros((n_types, max_n_atoms, 9), dtype=numpy.float32)

    for i, rt in enumerate(block_types):
        rt_n_nodes[i] = rt.rotamer_kinforest.nodes.shape[0]
        rt_nodes[i, : rt.rotamer_kinforest.nodes.shape[0]] = rt.rotamer_kinforest.nodes
        rt_scans[i, : rt.rotamer_kinforest.scans.shape[0]] = rt.rotamer_kinforest.scans
        rt_gens[i, : rt.rotamer_kinforest.gens.shape[0], :] = rt.rotamer_kinforest.gens
        # fill forward
        rt_gens[i, rt.rotamer_kinforest.gens.shape[0] :, :] = rt.rotamer_kinforest.gens[
            -1, :
        ]
        rt_n_scans_per_gen[i, : rt.rotamer_kinforest.n_scans_per_gen.shape[0]] = (
            rt.rotamer_kinforest.n_scans_per_gen
        )
        rt_kinforest_idx[i, : rt.n_atoms] = rt.rotamer_kinforest.kinforest_idx
        rt_id[i, : rt.n_atoms] = rt.rotamer_kinforest.id
        rt_doftype[i, : rt.n_atoms] = rt.rotamer_kinforest.doftype
        rt_parent[i, : rt.n_atoms] = rt.rotamer_kinforest.parent
        rt_frame_x[i, : rt.n_atoms] = rt.rotamer_kinforest.frame_x
        rt_frame_y[i, : rt.n_atoms] = rt.rotamer_kinforest.frame_y
        rt_frame_z[i, : rt.n_atoms] = rt.rotamer_kinforest.frame_z
        rt_dofs_ideal[i, : rt.n_atoms] = rt.rotamer_kinforest.dofs_ideal

    return PackedRotamerKintree(
        kinforest_idx=rt_kinforest_idx,
        id=rt_id,
        doftype=rt_doftype,
        parent=rt_parent,
        frame_x=rt_frame_x,
        frame_y=rt_frame_y,
        frame_z=rt_frame_z,
        n_nodes=rt_n_nodes,
        nodes=rt_nodes,
        scans=rt_scans,
        gens=rt_gens,
        n_scans_per_gen=rt_n_scans_per_gen,
        dofs_ideal=rt_dofs_ideal,
    )


def annotate_residue_rotamer_kinematics(restype, inverse_kinematics):
    """Create a kinforest for a single residue and its scan ordering data.

    The kinforest data structure on its own is incomplete and before it can be
    stored will need to be left-padded with 0s. In particular, the id, doftype,
    parent, frame_x, _y and _z data all require the 0th position to be occupied
    by the "root" atom.
    """
    if hasattr(restype, "rotamer_kinforest"):
        return

    _annotate_block_type_with_residue_kinforest_data(restype)
    polymer = getattr(getattr(restype, "properties", None), "polymer", None)
    setattr(
        restype,
        "rotamer_kinforest",
        prepare_rotamer_kinematics(
            restype.residue_kinforest_data,
            restype.n_atoms,
            [a.name for a in restype.atoms],
            getattr(polymer, "mainchain_atoms", None),
            restype.ideal_coords[restype.at_to_icoor_ind],
            inverse_kinematics,
        ),
    )


def prepare_block_group_rotamer_kinematics(
    block_types, links, *, inverse_kinematics, anchor: int = 0, kinforest_data=None
):
    """Create a kinforest spanning a group of blocks joined by their bonds.

    The single-block case of construct_single_residue_kinforest: a conjugated
    residue and everything hanging off it fold as one unit, so a torsion about
    a linkage bond moves what is beyond it. ``links`` are (parent, parent_conn,
    child, child_conn) tuples as taken by block_group_kinforest_data.

    The inter-block bond's own geometry is not in any block's ideal coords, so
    the dofs it reports are placeholders; they are overwritten by the values
    measured from the pose.
    """
    rkd, offsets = kinforest_data or _block_group_kinforest_data(
        block_types, links, anchor
    )
    names = [a.name for bt in block_types for a in bt.atoms]
    ideal = numpy.concatenate(
        [bt.ideal_coords[bt.at_to_icoor_ind] for bt in block_types]
    )
    polymer = getattr(getattr(block_types[anchor], "properties", None), "polymer", None)
    mainchain = getattr(polymer, "mainchain_atoms", None)
    return (
        prepare_rotamer_kinematics(
            rkd, int(offsets[-1]), names, mainchain, ideal, inverse_kinematics
        ),
        offsets,
    )
