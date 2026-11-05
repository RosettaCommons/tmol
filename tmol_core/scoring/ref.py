"""Reference energies resolved from one shared chemistry record and weight map."""

from tmol_core.chemical import _l_base_name


def reference_weight(block_type, weights):
    """Prefer an explicit residue weight, then its mirrored L residue, then zero."""
    value = weights.get(block_type.base_name)
    return weights.get(_l_base_name(block_type), 0.0) if value is None else value
