"""Shared sampler routing for mature molecular input completion."""


def prepare_input_completion_task(
    pose,
    missing,
    *,
    task,
    dunbrack,
    fixed,
    opth,
    nucleic_acid,
    has_missing_atoms,
    supported_blocks,
    any_block,
):
    """Assign disjoint missing-heavy-atom and complete OptH packing positions.

    Masks and tasks belong to the execution frontend. Chemistry support comes
    from the existing samplers; no fallback conformation is introduced for an
    incomplete sidechain.
    """
    task.restrict_to_repacking()
    if has_missing_atoms:
        task.add_conformer_sampler_by_block_mask(dunbrack, missing)
        task.add_conformer_sampler_by_block_mask(fixed, missing)
    real = pose.block_type_ind >= 0
    if nucleic_acid is not None and has_missing_atoms:
        missing_na = missing & real & supported_blocks(nucleic_acid, pose)
        if any_block(missing_na):
            task.add_conformer_sampler_by_block_mask(nucleic_acid, missing_na)
    complete = ~missing
    if opth is None:
        task.disable_packing_by_block_mask(complete)
    else:
        complete = complete & real
        opth_blocks = complete & supported_blocks(opth, pose)
        task.add_conformer_sampler_by_block_mask(opth, opth_blocks)
        task.disable_packing_by_block_mask(complete & ~opth_blocks)
    return task


OPTH_INVARIANT_TERMS = (
    "disulfide",
    "omega",
    "rama",
    "ref",
    "na_torsion",
    "na_torsion_well",
)


def packer_atom_mask(residue, samplers, prepare_kinematics):
    """Atoms reserved for mature amino-acid sidechain packing during input."""
    import numpy as np
    from .sidechain import sidechain_region

    placed = np.zeros(residue.n_atoms, dtype=bool)
    for sampler in samplers:
        if sampler.defines_rotamers_for_rt(residue):
            prepare_kinematics(residue)
            roots = [
                residue.atom_to_idx[a] for a in sampler.first_sc_atoms_for_rt(residue)
            ]
            placed |= sidechain_region(residue, roots)[1].astype(bool)
    return placed
