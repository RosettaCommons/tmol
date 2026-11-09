"""PDB input with shared host parsing and JAX pose construction."""

from tmol_core.io.pdb_parsing import parse_pdb, _records_from_biotite  # noqa: F401


def pose_stack_from_pdb(
    pdb_input,
    *,
    canonical_ordering=None,
    packed_block_types=None,
    missing_density_distance_threshold=2.4,
    **kwargs,
):
    """Load a molecular PDB using its chemical ordering and residue palette."""
    from .canonical_ordering import (
        canonical_form_from_pdb,
        default_canonical_ordering,
        default_packed_block_types,
    )
    from .pose_stack_construction import pose_stack_from_canonical_form

    co = canonical_ordering or default_canonical_ordering()
    pbt = packed_block_types or default_packed_block_types()
    return pose_stack_from_canonical_form(
        dict(
            canonical_form_from_pdb(
                pdb_input,
                co,
                missing_density_distance_threshold=missing_density_distance_threshold,
            ),
            canonical_ordering=co,
        ),
        pbt,
        **kwargs,
    )
