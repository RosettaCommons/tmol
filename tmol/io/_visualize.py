"""Notebook visualization with lazy Torch pose conversion and shared rendering."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Mapping

from tmol_core.io import visualize as _shared
from tmol_core.io.visualize import selection_gallery  # noqa: F401

if TYPE_CHECKING:
    from os import PathLike
    from biotite.structure import AtomArray, AtomArrayStack
    from tmol.pose._pose_stack import PoseStack

    StructureInput = PoseStack | AtomArray | AtomArrayStack | str | PathLike[str]
else:
    StructureInput = Any


def _pose_to_atom_array(model):
    from tmol.pose._pose_stack import PoseStack

    if not isinstance(model, PoseStack):
        return NotImplemented
    from tmol.io._pose_stack_from_biotite import atom_array_from_pose_stack

    return atom_array_from_pose_stack(model)


def _pose_stack_to_pdb_string(pose_stack):
    return _shared._biotite_to_pdb_string(_pose_to_atom_array(pose_stack))


def _pdb_text(model):
    return _shared._pdb_text(model, _pose_to_atom_array=_pose_to_atom_array)


def __getattr__(name):
    return getattr(_shared, name)


def view(  # noqa: C901
    model: StructureInput,
    *,
    width: int = 720,
    height: int = 420,
    style: Literal["cartoon", "stick"] = "cartoon",
    background_color: str = "white",
    cartoon_color: str = "spectrum",
    show_sidechains: bool = True,
    show_heteroatoms: bool = True,
    show_hover: bool = True,
    zoom_to: dict | None = None,
    highlighted: object | None = None,
    highlight_color: str = "#e83e8c",
):
    """Create a draggable py3Dmol viewer for a molecular structure.

    ``model`` may be a :class:`PoseStack`, a Biotite ``AtomArray`` or
    ``AtomArrayStack``, PDB text, or a PDB path. ``highlighted`` is an optional
    boolean mask over the atoms in the first model; highlighted atoms are shown
    as thicker sticks and spheres. The return value remains a real
    ``py3Dmol.view`` object for compatibility with existing notebooks.
    """
    return _shared.view(
        model,
        width=width,
        height=height,
        style=style,
        background_color=background_color,
        cartoon_color=cartoon_color,
        show_sidechains=show_sidechains,
        show_heteroatoms=show_heteroatoms,
        show_hover=show_hover,
        zoom_to=zoom_to,
        highlighted=highlighted,
        highlight_color=highlight_color,
        _pose_to_atom_array=_pose_to_atom_array,
    )


def switchable_view(
    structures: Mapping[str, StructureInput],
    *,
    notes: Mapping[str, str] | None = None,
    width: int = 720,
    height: int = 420,
):
    """Return HTML that switches one 3Dmol viewer among labeled structures.

    Args:
        structures: Ordered mapping of display labels to structures accepted by
            :func:`view`.
        notes: Optional mapping of structure labels to short explanatory text.
        width: Viewer width in pixels.
        height: Viewer height in pixels.
    """
    return _shared.switchable_view(
        structures,
        notes=notes,
        width=width,
        height=height,
        _pose_to_atom_array=_pose_to_atom_array,
    )


def overlay_view(
    structures: Mapping[str, StructureInput],
    *,
    notes: Mapping[str, str] | None = None,
    colors: Mapping[str, str] | None = None,
    default: str = "both",
    width: int = 720,
    height: int = 420,
):
    """Compare two or more structures, with the option to overlay them all.

    A selector shows any one structure alone (as :func:`view` would render
    it), or all of them at once in a single shared viewer, each colored
    distinctly.

    Args:
        structures: Two or more labeled structures, each accepted by :func:`view`.
        notes: Optional mapping of structure labels to short explanatory text,
            shown under each individual (non-overlay) panel.
        colors: Optional mapping of structure labels to a color used for that
            structure in the overlay panel. Unlisted labels draw from a
            default categorical palette, cycling if there are more structures
            than colors in it.
        default: Which panel is visible initially: ``"both"`` (the overlay,
            the default) or one of the structure labels.
        width: Viewer width in pixels.
        height: Viewer height in pixels.
    """
    return _shared.overlay_view(
        structures,
        notes=notes,
        colors=colors,
        default=default,
        width=width,
        height=height,
        _pose_to_atom_array=_pose_to_atom_array,
    )
