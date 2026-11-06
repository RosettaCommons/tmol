"""Legacy imports for the shared metal coordination geometry."""

from tmol_core.io import metal_geometry as _common


def __getattr__(name):
    return getattr(_common, name)
