"""Declared disulfides and the mature residue-ordered greedy SG matching rule."""

from collections import Counter

import attr
import numpy


@attr.s(auto_attribs=True, frozen=True)
class DisulfideSearchPlan:
    variants: numpy.ndarray
    declared: numpy.ndarray
    cys_pose: numpy.ndarray
    cys_residue: numpy.ndarray
    search: bool
    sg_atoms: numpy.ndarray | None = None


def prepare_disulfide_search(
    cys_classes, residue_types, declared=None, find_additional=True, sg_atoms=None
):
    """Prepare metadata without inspecting or transferring coordinates."""
    variants = numpy.zeros(numpy.shape(residue_types), dtype=numpy.int32)
    empty = numpy.empty((0, 3), dtype=numpy.int64)
    if not cys_classes:
        return DisulfideSearchPlan(
            variants,
            empty,
            numpy.empty(0, numpy.int64),
            numpy.empty(0, numpy.int64),
            False,
        )
    present = numpy.zeros(numpy.shape(residue_types), dtype=bool)
    for cys_class in cys_classes:
        present |= residue_types == cys_class
    rows = (
        empty
        if declared is None
        else numpy.asarray(declared, dtype=numpy.int64).reshape(-1, 3).copy()
    )
    if declared is not None:
        endpoints = Counter(
            (pose, residue)
            for pose, first, second in rows.tolist()
            for residue in (first, second)
        )
        repeated = [list(pair) for pair in sorted(endpoints) if endpoints[pair] > 1]
        if repeated:
            raise ValueError(
                "Each cysteine may occur in only one declared disulfide; "
                f"repeated (pose, residue) endpoints: {repeated}"
            )
        variants[rows[:, 0], rows[:, 1]] = 1
        variants[rows[:, 0], rows[:, 2]] = 1
    pose, residue = numpy.nonzero(present)
    search = bool(
        find_additional
        and pose.size
        and (declared is None or numpy.any(variants[pose, residue] != 1))
    )
    selected_sg = None
    if sg_atoms is not None:
        if len(sg_atoms) != len(cys_classes):
            raise ValueError("Provide one SG slot per cysteine class")
        if any(atom != sg_atoms[0] for atom in sg_atoms):
            selected_sg = numpy.empty(len(pose), dtype=numpy.int64)
            for cls, atom in zip(cys_classes, sg_atoms):
                selected_sg[residue_types[pose, residue] == cls] = atom
    return DisulfideSearchPlan(variants, rows, pose, residue, search, selected_sg)


def resolve_disulfides(plan, coords, sg_atom, cutoff=2.5):
    """Resolve a reusable metadata plan; coordinate and metadata inputs stay intact."""
    if not plan.search:
        return plan.declared, plan.variants
    found = numpy.zeros((len(plan.cys_pose), 3), dtype=numpy.int64)
    found[: len(plan.declared)] = plan.declared
    variants = plan.variants.copy()
    result = find_disulfide_pairs(
        coords,
        len(plan.declared),
        found,
        plan.cys_pose,
        plan.cys_residue,
        sg_atom if plan.sg_atoms is None else plan.sg_atoms,
        cutoff,
        variants,
    )
    return result, variants


def find_disulfide_pairs(
    coords,
    n_input_dslf,
    found_dslf,
    cys_pose_ind,
    cys_res_ind,
    sg_atom_for_co_cys,
    cutoff_dis,
    restype_variants,
):
    # Keep this small greedy matching pass on the CPU.

    # algorithm for CYD matching:
    # greedy
    # process the cys pairs in order from n->c
    # for cys i,
    #    take the closest as-of-yet unpaired SG to i's SG w/i 2.5A
    #    mark the two as now paired

    n_cys = cys_pose_ind.shape[0]
    scalar_sg = isinstance(sg_atom_for_co_cys, (int, numpy.integer))
    already_paired = numpy.zeros(n_cys, dtype=numpy.int32)
    for i in range(n_cys):
        already_paired[i] = restype_variants[cys_pose_ind[i], cys_res_ind[i]]
    n_found_dslf = n_input_dslf
    cutoff_dis2 = cutoff_dis * cutoff_dis

    for i in range(n_cys):
        if already_paired[i]:
            continue
        i_pose = cys_pose_ind[i]
        i_res = cys_res_ind[i]
        closest_match = -1
        closest_match_res = -1
        closest_dis2 = -1.0
        i_coord = coords[
            i_pose, i_res, sg_atom_for_co_cys if scalar_sg else sg_atom_for_co_cys[i]
        ]
        for j in range(i + 1, cys_pose_ind.shape[0]):
            j_pose = cys_pose_ind[j]
            if j_pose != i_pose:
                break
            j_res = cys_res_ind[j]
            if already_paired[j]:
                continue
            dis2 = numpy.sum(
                numpy.square(
                    i_coord
                    - coords[
                        j_pose,
                        j_res,
                        sg_atom_for_co_cys if scalar_sg else sg_atom_for_co_cys[j],
                    ]
                )
            )
            if dis2 < cutoff_dis2 and (closest_dis2 < 0 or dis2 < closest_dis2):
                closest_dis2 = dis2
                closest_match = j
                closest_match_res = j_res
        if closest_match >= 0:
            already_paired[i] = 1
            already_paired[closest_match] = 1
            found_dslf[n_found_dslf, 0] = i_pose
            found_dslf[n_found_dslf, 1] = i_res
            found_dslf[n_found_dslf, 2] = closest_match_res
            # mark these two as disulfides
            restype_variants[i_pose, i_res] = 1
            restype_variants[i_pose, closest_match_res] = 1
            n_found_dslf += 1
    return found_dslf[:n_found_dslf]
