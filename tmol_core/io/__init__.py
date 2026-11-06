"""Framework-neutral molecular input preparation."""

__all__ = ["atom_array_from_file", "file_from_atom_array"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    from . import structure_io

    value = getattr(structure_io, name)
    globals()[name] = value
    return value
