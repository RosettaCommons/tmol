"""Notebook display through common rendering and lazy JAX pose export."""

from tmol_core.io import visualize as _shared
from tmol_core.io.visualize import selection_gallery  # noqa: F401


def _pose_to_atom_array(model):
    from tmol_jax.pose import PoseStack

    if not isinstance(model, PoseStack):
        return NotImplemented
    from .structure_export import atom_array_from_pose_stack

    return atom_array_from_pose_stack(model)


def view(
    model,
    *,
    width=720,
    height=420,
    style="cartoon",
    background_color="white",
    cartoon_color="spectrum",
    show_sidechains=True,
    show_heteroatoms=True,
    show_hover=True,
    zoom_to=None,
    highlighted=None,
    highlight_color="#e83e8c",
):
    """Display a pose, Biotite structure, PDB text or path in a py3Dmol viewer."""
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


def switchable_view(structures, *, notes=None, width=720, height=420):
    """Return notebook HTML switching among labeled molecular structures."""
    return _shared.switchable_view(
        structures,
        notes=notes,
        width=width,
        height=height,
        _pose_to_atom_array=_pose_to_atom_array,
    )


def overlay_view(
    structures, *, notes=None, colors=None, default="both", width=720, height=420
):
    """Return notebook HTML comparing individual structures and their overlay."""
    return _shared.overlay_view(
        structures,
        notes=notes,
        colors=colors,
        default=default,
        width=width,
        height=height,
        _pose_to_atom_array=_pose_to_atom_array,
    )
