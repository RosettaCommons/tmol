"""Named nonkinematic donor ports for explicit metal coordination."""

from tmol_core.chemical import Connection, VariantScope, VariantType


def _metal_donor_patch(base_name: str, atom: str, n_metals: int = 1) -> VariantType:
    """A patch giving ``atom`` a connection per metal: metal_<atom>, metal2_<atom>, ...
    Named for its last connection, so a patched type's name is one of its connections.
    """
    names = [f"metal_{atom}"] + [f"metal{k}_{atom}" for k in range(2, n_metals + 1)]
    prefix = f"metal{n_metals}" if n_metals > 1 else "metal"
    return VariantType(
        name=f"{prefix}_{base_name}_{atom}",
        display_name=names[-1],
        pattern="",
        remove_atoms=(),
        add_atoms=(),
        add_atom_aliases=(),
        modify_atoms=(),
        add_connections=tuple(
            Connection(name=name, atom=f"<{atom}>", kinematic=False) for name in names
        ),
        add_bonds=(),
        icoors=(),
        applies_to=VariantScope(base_names=(base_name,)),
    )
