"""Chemical compatibility for packing/design, independent of tensor execution.

The palette defines allowed residue replacements. Sampling conformers, applying
pose tasks and choosing a packing solution are separate execution operations.
"""

from dataclasses import dataclass

import numpy as np

from tmol_core.chemical import DEPROTONATED_STATE


@dataclass(frozen=True)
class PackerPaletteData:
    max_n_allowed: int
    n_allowed: np.ndarray
    allowed_types: np.ndarray
    is_original: np.ndarray
    repack_mask: np.ndarray


def _set_compare(x, y):
    """Treat the collections x and y as if they are sets. Return true if they
    contain the same elements and false otherwise
    """
    if len(x) != len(y):
        return False
    for val in x:
        if val not in y:
            return False
    return True


def _mainchain_elements(block_type, element_for_atom_type):
    """The element of each mainchain atom, in mainchain order.

    A mainchain fingerprint identifies an atom by its position along the
    mainchain, so two block types can only exchange dofs when their mainchains
    agree position by position. Comparing elements rather than atom names lets
    two backbones of the same shape exchange dofs whatever they call their
    atoms, and comparing elements rather than atom types keeps a difference
    that does not move an atom from blocking the exchange.
    """
    types = {atom.name: atom.atom_type for atom in block_type.atoms}
    return tuple(
        element_for_atom_type[types[name]]
        for name in (block_type.properties.polymer.mainchain_atoms or ())
    )


def _backbone_signatures(chemical_database, block_types):
    """Mainchain elements and applied terminal chemistry for each block type."""
    element_for_atom_type = {at.name: at.element for at in chemical_database.atom_types}
    terminal_patches = {
        patch.display_name
        for patch in chemical_database.variants
        if {"<{down}>", "<{up}>"}.intersection(patch.remove_atoms)
    }
    return [
        (
            _mainchain_elements(bt, element_for_atom_type),
            frozenset(bt.name.split(":")[1:]) & terminal_patches,
        )
        for bt in block_types
    ]


def _exchangeable(orig_bt, alt_bt, orig_backbone, alt_backbone) -> bool:
    """Whether block types can exchange, aside from side-chain protonation."""
    return (
        alt_bt.properties.polymer.is_polymer == orig_bt.properties.polymer.is_polymer
        and alt_bt.properties.polymer.polymer_type
        == orig_bt.properties.polymer.polymer_type
        and alt_bt.properties.polymer.backbone_type
        == orig_bt.properties.polymer.backbone_type
        and alt_bt.connections == orig_bt.connections
        and alt_bt.conjugation_context == orig_bt.conjugation_context
        and orig_backbone == alt_backbone
        and _set_compare(
            alt_bt.properties.chemical_modifications,
            orig_bt.properties.chemical_modifications,
        )
        and _set_compare(
            alt_bt.properties.connectivity, orig_bt.properties.connectivity
        )
    )


def prepare_packer_palette(chemical_database, block_types, noncanonical_design=False):
    """Prepare ordered choices for arbitrary residue types and explicit chemistry."""
    block_types = tuple(block_types)
    backbones = _backbone_signatures(chemical_database, block_types)
    allowed_block_types_for_block_type = [list() for _ in range(len(block_types))]
    allowed_block_is_orig = [list() for _ in range(len(block_types))]
    restrict_to_repacking_masks = [list() for _ in range(len(block_types))]
    # types that only metal detection selects never trade places with another
    selected_by_detection = [
        bool(bt.metal_sites)
        or bt.properties.protonation.protonation_state == DEPROTONATED_STATE
        for bt in block_types
    ]
    # types can exchange only within a group that agrees on these fields; each
    #    type's candidates keep the order of the full n^2 scan
    groups = {}
    for j, bt in enumerate(block_types):
        polymer = bt.properties.polymer
        key = (
            polymer.is_polymer,
            polymer.polymer_type,
            polymer.backbone_type,
            bt.properties.protonation.protonation_state,
            len(bt.connections),
            backbones[j],
        )
        groups.setdefault(key, []).append(j)
    group_of = {j: members for members in groups.values() for j in members}
    for i, orig_bt in enumerate(block_types):
        for j in group_of[i]:
            alt_bt = block_types[j]
            if i != j and (selected_by_detection[i] or selected_by_detection[j]):
                continue
            # a noncanonical type replaces only its own kind unless asked for
            if (
                not noncanonical_design
                and not alt_bt.properties.is_canonical
                and alt_bt.name3 != orig_bt.name3
            ):
                continue
            j_allowed_for_restrict_to_repack = alt_bt.name3 == orig_bt.name3
            if _exchangeable(orig_bt, alt_bt, backbones[i], backbones[j]):
                if (
                    alt_bt.properties.polymer.sidechain_chirality
                    == orig_bt.properties.polymer.sidechain_chirality
                ):
                    allowed_block_types_for_block_type[i].append(j)
                    allowed_block_is_orig[i].append(j == i)
                    restrict_to_repacking_masks[i].append(
                        j_allowed_for_restrict_to_repack
                    )
                elif orig_bt.properties.polymer.polymer_type == "amino_acid" and (
                    (
                        orig_bt.properties.polymer.sidechain_chirality == "l"
                        and alt_bt.properties.polymer.sidechain_chirality == "achiral"
                    )
                    or (
                        orig_bt.properties.polymer.sidechain_chirality == "achiral"
                        and alt_bt.properties.polymer.sidechain_chirality == "l"
                    )
                ):
                    # allow glycine <--> l-caa mutations
                    allowed_block_types_for_block_type[i].append(j)
                    allowed_block_is_orig[i].append(j == i)
                    restrict_to_repacking_masks[i].append(
                        j_allowed_for_restrict_to_repack
                    )
                elif (
                    orig_bt.properties.polymer.polymer_type == "amino_acid"
                    and orig_bt.properties.polymer.sidechain_chirality == "d"
                    and alt_bt.properties.polymer.sidechain_chirality == "achiral"
                ):
                    # allow d-caa --> glycine mutations;
                    # dangerous because this packer pallete will allow
                    # your d-caa to become glycine, and then later
                    # to an l-caa, but not the other way around
                    allowed_block_types_for_block_type[i].append(j)
                    allowed_block_is_orig[i].append(j == i)
                    restrict_to_repacking_masks[i].append(
                        j_allowed_for_restrict_to_repack
                    )
                elif (
                    noncanonical_design
                    and orig_bt.properties.polymer.polymer_type == "amino_acid"
                ):
                    # noncanonical design also crosses between l and d amino acids
                    allowed_block_types_for_block_type[i].append(j)
                    allowed_block_is_orig[i].append(j == i)
                    restrict_to_repacking_masks[i].append(
                        j_allowed_for_restrict_to_repack
                    )
            elif i == j:
                allowed_block_types_for_block_type[i].append(j)
                allowed_block_is_orig[i].append(j == i)
                restrict_to_repacking_masks[i].append(j_allowed_for_restrict_to_repack)

    n_types = len(block_types)
    max_n_allowed = max(map(len, allowed_block_types_for_block_type), default=0)
    n_allowed = np.asarray(
        [len(row) for row in allowed_block_types_for_block_type], dtype=np.int64
    )
    allowed_types = np.full((n_types, max_n_allowed), -1, dtype=np.int64)
    is_original = np.zeros((n_types, max_n_allowed), dtype=np.bool_)
    repack_mask = np.zeros_like(is_original)
    for i, row in enumerate(allowed_block_types_for_block_type):
        allowed_types[i, : len(row)] = row
        is_original[i, : len(row)] = allowed_block_is_orig[i]
        repack_mask[i, : len(row)] = restrict_to_repacking_masks[i]
    return PackerPaletteData(
        max_n_allowed, n_allowed, allowed_types, is_original, repack_mask
    )
