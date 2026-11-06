"""One tensor-only path from Atom37 coordinates to a PoseStack.

Canonical and noncanonical poses differ only in how residue identity and the
Atom37 slot layout are *resolved*, not in how a pose is built. Once a
:py:class:`~tmol.io.PoseBuildContext` has resolved the residue types, this
module takes tensors the whole way: it scatters Atom37 coordinates into
canonical atom slots and hands the result to the shared constructor.

Nothing here inspects an ``AtomArray``. Callers that have one should use it to
*derive* these tensors once (see
:py:func:`~tmol.io.prepare_atom37_pose_builder`) and then stay on this path.
"""

from typing import Mapping, Optional, Sequence

import torch

from tmol.io._build_context import PoseBuildContext
from tmol.io._canonical_form import CanonicalForm
from tmol.io._canonical_ordering import CanonicalOrdering
from tmol.pose._pose_stack import PoseStack


def atom37_slot_map_for_ordering(
    canonical_ordering: CanonicalOrdering,
    atom_names_by_slot: Mapping[str, Sequence[str]],
    device: torch.device,
) -> torch.Tensor:
    """Map each residue type's Atom37 slots to canonical atom indices.

    The mapping is a property of the residue-type table and the slot layout,
    not of any structure, so it is built once per (ordering, layout) pair.

    Atoms that a terminus patch adds are deliberately left unmapped. They exist
    only on terminal residue variants, and offering one on an interior residue
    would disqualify that residue's block types; tmol applies termini itself
    from the chain boundaries. Which atoms those are is read from the chemical
    database rather than named here.

    Args:
      canonical_ordering: Residue and atom ordering the ``res_types`` index
        into.
      atom_names_by_slot: Atom name occupying each slot, per residue name3. An
        empty name marks an unused slot.
      device: Device for the returned tensor.

    Returns:
      ``[n_residue_types, n_slots]`` canonical atom index per slot, ``-1``
      where the slot is unused or the atom is not part of that residue type.

    Examples:
      >>> slot_map = atom37_slot_map_for_ordering(co, layout, torch.device("cpu"))
      >>> slot_map.shape[1]
      37
    """
    from tmol_core.io.atom37 import atom37_slot_map_for_ordering as plan

    return torch.as_tensor(plan(canonical_ordering, atom_names_by_slot), device=device)


def _default_slot_map(
    canonical_ordering: CanonicalOrdering, device: torch.device
) -> torch.Tensor:
    """Build the slot map for the standard Atom37 layout."""
    # AtomWorks layout metadata has no framework dependency.
    from tmol_core.io.atomworks_encoding import ATOMWORKS_ATOM37_NAMES

    return atom37_slot_map_for_ordering(
        canonical_ordering, ATOMWORKS_ATOM37_NAMES, device
    )


def _canonical_form_from_atom37(
    atom37_coords: torch.Tensor,
    res_types: torch.Tensor,
    chain_id: torch.Tensor,
    canonical_ordering: CanonicalOrdering,
    *,
    slot_map: Optional[torch.Tensor] = None,
    disulfides: Optional[torch.Tensor] = None,
    cyclic_bonds: Optional[torch.Tensor] = None,
    covalent_bonds: Optional[torch.Tensor] = None,
    res_not_connected: Optional[torch.Tensor] = None,
) -> CanonicalForm:
    """Scatter Atom37 coordinates into a canonical form.

    Args:
      atom37_coords: ``[n_poses, n_tokens, n_slots, 3]``, autograd-tracked.
      res_types: ``[n_poses, n_tokens]`` index into ``canonical_ordering``;
        ``-1`` marks padding.
      chain_id: ``[n_poses, n_tokens]`` chain index per token.
      canonical_ordering: Ordering the ``res_types`` index into.
      slot_map: ``[n_residue_types, n_slots]`` from
        :py:func:`atom37_slot_map_for_ordering`. Defaults to the standard
        Atom37 layout, which covers the canonical amino acids; supply one built
        from an extended layout for other chemistry.
      disulfides: ``[n, 3]`` explicit ``(pose, res1, res2)`` rows.
      cyclic_bonds: ``[n, 3]`` explicit head-to-tail closures.
      covalent_bonds: ``[n, 5]`` explicit
        ``(pose, res1, atom1, res2, atom2)`` links in canonical ordering.
      res_not_connected: ``[n_poses, n_tokens, 2]`` suppressed polymer
        connections.

    Returns:
      A canonical form whose coordinates carry gradients back to
      ``atom37_coords``.
    """
    from tmol_core.io.atom37 import prepare_atom37_routing

    device = atom37_coords.device
    if atom37_coords.ndim != 4 or atom37_coords.shape[-1] != 3:
        raise ValueError("atom37_coords must have shape [pose, token, slot, 3]")
    if tuple(res_types.shape) != tuple(atom37_coords.shape[:2]):
        raise ValueError("res_types must match the coordinate pose and token axes")
    routing = prepare_atom37_routing(
        res_types.detach().cpu().numpy(),
        None if slot_map is None else slot_map.detach().cpu().numpy(),
        atom37_coords.shape[2],
        canonical_ordering,
    )
    source = tuple(torch.as_tensor(x, device=device) for x in routing.source_indices)
    target = tuple(torch.as_tensor(x, device=device) for x in routing.target_indices)
    coords = torch.full(
        (*routing.canonical_shape, 3),
        float("nan"),
        dtype=torch.float32,
        device=device,
    )
    coords[target] = atom37_coords[source]

    return CanonicalForm(
        chain_id=chain_id.to(torch.int32),
        res_types=res_types.to(torch.int32),
        coords=coords,
        res_labels=None,
        residue_insertion_codes=None,
        chain_labels=None,
        atom_occupancy=None,
        atom_b_factor=None,
        disulfides=disulfides,
        res_not_connected=res_not_connected,
        cyclic_bonds=cyclic_bonds,
        covalent_bonds=covalent_bonds,
    )


def pose_stack_from_atom37(
    atom37_coords: torch.Tensor,
    res_types: torch.Tensor,
    chain_id: torch.Tensor,
    context: PoseBuildContext,
    *,
    slot_map: Optional[torch.Tensor] = None,
    disulfides: Optional[torch.Tensor] = None,
    cyclic_bonds: Optional[torch.Tensor] = None,
    covalent_bonds: Optional[torch.Tensor] = None,
    res_not_connected: Optional[torch.Tensor] = None,
    no_optH: bool = True,
    **kwargs: object,
) -> PoseStack:
    """Build a differentiable PoseStack from Atom37 tensors alone.

    One path for canonical and noncanonical poses. Residue identity comes from
    ``res_types`` indexed into the context's canonical ordering, and any
    covalent chemistry the polymer connections do not already describe comes
    from ``covalent_bonds``; no ``AtomArray`` is consulted.

    Args:
      atom37_coords: ``[n_poses, n_tokens, n_slots, 3]``, autograd-tracked.
      res_types: ``[n_poses, n_tokens]`` index into
        ``context.canonical_ordering``; ``-1`` marks padding.
      chain_id: ``[n_poses, n_tokens]`` chain index per token.
      context: Chemistry resolved once, from any of the
        ``build_context_from_*`` helpers.
      slot_map: Atom37 slot layout; see
        :py:func:`atom37_slot_map_for_ordering`.
      disulfides: Explicit ``(pose, res1, res2)`` rows.
      cyclic_bonds: Explicit head-to-tail closures.
      covalent_bonds: Explicit ``(pose, res1, atom1, res2, atom2)`` links.
      res_not_connected: Suppressed polymer connections.
      no_optH: Preserve finite input hydrogens rather than optimizing them.

    Returns:
      A pose whose coordinates carry gradients back to ``atom37_coords``.

    Examples:
      >>> context = build_context_from_atom_array(array, device)
      >>> pose = pose_stack_from_atom37(coords, res_types, chain_id, context)
      >>> pose.coords.requires_grad
      True
    """
    # Imported here because the module defining the shared constructor is
    # loaded after this one.
    from tmol.io._pose_stack_from_biotite import (
        _pose_stack_from_canonical_form_and_context,
    )

    canonical_form = _canonical_form_from_atom37(
        atom37_coords,
        res_types,
        chain_id,
        context.canonical_ordering,
        slot_map=slot_map,
        disulfides=disulfides,
        cyclic_bonds=cyclic_bonds,
        covalent_bonds=covalent_bonds,
        res_not_connected=res_not_connected,
    )
    return _pose_stack_from_canonical_form_and_context(
        canonical_form,
        context,
        no_optH=no_optH,
        atom37_coords=atom37_coords,
        **kwargs,
    )
