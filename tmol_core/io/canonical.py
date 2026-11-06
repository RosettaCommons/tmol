"""Canonical residue candidates and atom identity from common chemistry."""

from collections import defaultdict
from types import SimpleNamespace
import attr
import numpy
from tmol_core.chemical import N_VARIANT_INDICES, _special_case_variant_index

CONTEXT_PREFIX = "context_"


@attr.s(auto_attribs=True, frozen=True)
class CanonicalSelection:
    max_n_candidates_for_var_combo: int
    # n-co-equiv-class
    co_equiv_class_is_polymeric: numpy.ndarray
    co_polymer_connection_atoms: numpy.ndarray
    # n-co-equiv-class x n-term-opts x n-spcase-var x max-n-candidates
    var_combo_is_real_candidate: numpy.ndarray
    # n-co-equiv-class x n-term-opts x n-spcase-var x max-n-candidates
    var_combo_candidate_bt_index: numpy.ndarray
    # n-pbt-block-types x max-n-canonical-atoms
    bt_canonical_atom_is_absent: numpy.ndarray
    # n-pbt-block-types x max-n-canonical-atoms
    bt_non_term_patch_added_canonical_atom_is_present: numpy.ndarray
    # n-pbt-block-types
    bt_is_non_default_terminus: numpy.ndarray

    # n-pbt-block-types
    # needed for output?
    bt_ind_to_canonical_io_equiv_class_ind: numpy.ndarray

    # n-pbt-block-types x max-n-atoms
    bt_canonical_atom_ind_map: numpy.ndarray


def _map_term_to_int(is_down_term, is_up_term):
    if is_down_term and is_up_term:
        return 3
    if is_down_term:
        return 0
    if is_up_term:
        return 2
    return 1


def _term_and_spcase_var_candidate_lists(max_n_term, max_n_spcase):
    return [[[] for _ in range(max_n_spcase)] for _ in range(max_n_term)]


def _assign_var_inds_for_bt(co, bt):
    bt_vars = bt.name.split(":")
    # Caps are intrinsically terminal: their base type has only one polymer
    # connection and needs no terminal patch. Connection topology also agrees
    # with the terminal status of ordinary patched polymer types.
    is_polymer = bt.properties.polymer.is_polymer
    bt_is_down_term = is_polymer and bt.down_connection_ind < 0
    bt_is_up_term = is_polymer and bt.up_connection_ind < 0
    bt_is_non_default_term = False
    for var_name in bt_vars[1:]:
        if var_name in co.down_termini_patches:
            bt_is_down_term = True
            if var_name != co.restypes_default_termini_mapping[bt.io_equiv_class][0]:
                bt_is_non_default_term = True
        if var_name in co.up_termini_patches:
            bt_is_up_term = True
            if var_name != co.restypes_default_termini_mapping[bt.io_equiv_class][1]:
                bt_is_non_default_term = True
    term_ind = _map_term_to_int(bt_is_down_term, bt_is_up_term)
    spcase_var_ind = _special_case_variant_index(bt)
    return term_ind, spcase_var_ind, bt_is_non_default_term


def _get_co_equiv_class_is_polymeric(co, n_co_io_equiv_classes, bts_for_equiv_class):
    # We will consider the entire IO equivalence class to be
    # polymeric if a single block type within that equivalence class
    # is polymeric.
    co_equiv_class_is_polymeric = numpy.zeros((n_co_io_equiv_classes,), dtype=bool)
    for i, io_equiv_class in enumerate(co.restype_io_equiv_classes):
        for bt in bts_for_equiv_class[io_equiv_class]:
            if bt.properties.polymer.is_polymer:
                co_equiv_class_is_polymeric[i] = True
    return co_equiv_class_is_polymeric


def _collect_var_combo_candidates(
    co,
    max_n_termini_types,
    max_n_special_case_aa_variant_types,
    max_n_candidates_for_var_combo,
    pbt_io_equiv_class_candidates,
):
    n_co_io_equiv_classes = co.n_restype_io_equiv_classes
    var_combo_candidate_bt_index = numpy.full(
        (
            n_co_io_equiv_classes,
            max_n_termini_types,
            max_n_special_case_aa_variant_types,
            max_n_candidates_for_var_combo,
        ),
        -1,
        dtype=numpy.int64,
    )
    var_combo_is_real_candidate = numpy.zeros(
        (
            n_co_io_equiv_classes,
            max_n_termini_types,
            max_n_special_case_aa_variant_types,
            max_n_candidates_for_var_combo,
        ),
        dtype=bool,
    )

    for i, bt_name3 in enumerate(co.restype_io_equiv_classes):
        if bt_name3 not in pbt_io_equiv_class_candidates:
            continue
        for j in range(max_n_termini_types):
            for k in range(max_n_special_case_aa_variant_types):
                for candidate, (bt, bt_ind) in enumerate(
                    pbt_io_equiv_class_candidates[bt_name3][j][k]
                ):
                    var_combo_candidate_bt_index[i, j, k, candidate] = bt_ind
                    var_combo_is_real_candidate[i, j, k, candidate] = True
    return var_combo_candidate_bt_index, var_combo_is_real_candidate


def _note_atoms_present_and_absent_from_variants(co, pbt):
    # For bt i and canonical atom j, is canonical atom j absent from bt i?
    # needed so we can compute p & ~b[i]
    bt_canonical_atom_is_absent = numpy.ones(
        (
            pbt.n_types,
            co.max_n_canonical_atoms,
        ),
        dtype=bool,
    )
    # For bt i and  canonical atom j, is canonical atom j present in bt i
    # and not put there by a termini variant?
    bt_non_term_patch_added_canonical_atom_is_present = numpy.zeros(
        (
            pbt.n_types,
            co.max_n_canonical_atoms,
        ),
        dtype=bool,
    )
    for i, bt in enumerate(pbt.active_block_types):
        bt_at_names = set([at.name for at in bt.atoms])
        # first mark all atoms that are present as not absent
        # including the termini atoms
        for at_name in bt_at_names:
            can_ind = co.restypes_atom_index_mapping[bt.name3][at_name]
            bt_canonical_atom_is_absent[i, can_ind] = False

        # now we will go and strike out all the atoms that were not
        # added by termini patches so we can mark them as present
        for variant in bt.name.split(":")[1:]:
            bt_at_names.difference_update(
                co.termini_patch_added_atoms.get((bt.base_name, variant), ())
            )

        for at_name in bt_at_names:
            can_ind = co.restypes_atom_index_mapping[bt.io_equiv_class][at_name]
            bt_non_term_patch_added_canonical_atom_is_present[i, can_ind] = True
            bt_canonical_atom_is_absent[i, can_ind] = False

    return (
        bt_canonical_atom_is_absent,
        bt_non_term_patch_added_canonical_atom_is_present,
    )


def _create_bt_to_canonical_atom_mapping(co, pbt):
    bt_ind_to_canonical_ind = numpy.asarray(
        [
            co.restype_io_equiv_classes.index(bt.io_equiv_class)
            for bt in pbt.active_block_types
        ],
        dtype=numpy.int64,
    )
    bt_canonical_atom_ind = numpy.full(
        (pbt.n_types, pbt.max_n_atoms), -1, dtype=numpy.int64
    )
    for i, bt in enumerate(pbt.active_block_types):
        assert bt.io_equiv_class in co.restypes_ordered_atom_names
        i_canonical_ordering = co.restypes_ordered_atom_names[bt.io_equiv_class]
        for j, at in enumerate(bt.atoms):
            # probably this would be faster if we used a pandas indexer
            # but this is done only once, so, for now, use the slow form
            bt_canonical_atom_ind[i, j] = i_canonical_ordering.index(at.name.strip())
    bt_canonical_atom_ind = numpy.asarray(bt_canonical_atom_ind, dtype=numpy.int64)
    return bt_ind_to_canonical_ind, bt_canonical_atom_ind


def prepare_conjugations(block_types):
    """Annotate per block type its attachments (connections but up, down, disulfide and
    metal sites) as (is_metal, atom), its name without them, and the reverse lookup."""

    atoms_for_bt, base_for_bt = [], []
    by_base_and_atoms = {}
    contextual = {}
    for i, bt in enumerate(block_types):
        structural = {
            bt.down_connection_ind,
            bt.up_connection_ind,
            bt.connection_to_cidx.get("dslf", -1),
        }
        structural |= {
            bt.connection_to_cidx[name]
            for site in bt.metal_sites
            for name in site.site_connections
        }
        conjugations = [
            conn for ind, conn in enumerate(bt.connections) if ind not in structural
        ]
        # a multiset: an atom bridging two metals carries two connections;
        #   a donor's metal connections are named metal<k>_<atom>
        atoms = tuple(
            sorted((conn.name.startswith("metal"), conn.atom) for conn in conjugations)
        )
        atoms_for_bt.append(atoms)
        # the variant tags to drop are the connections' own names
        tags = {conn.name for conn in conjugations}
        combined = {"nterm"} if "conj_N" in tags else set()
        base = ":".join(
            part for part in bt.name.split(":") if part not in tags | combined
        )
        base_for_bt.append(base)
        if bt.conjugation_context:
            source = ":".join(
                p for p in bt.name.split(":") if not p.startswith(CONTEXT_PREFIX)
            )
            contextual.setdefault(source, {})[bt.conjugation_context] = i
            continue
        by_base_and_atoms[(base, atoms)] = i

    return SimpleNamespace(
        conjugation_atoms_for_bt=atoms_for_bt,
        conjugation_base_for_bt=base_for_bt,
        conjugated_bt_for_base_and_atoms=by_base_and_atoms,
        contextual_conjugates=contextual,
    )


def prepare_canonical_selection(co, active_block_types, max_n_atoms):
    """Build candidate tables without assuming proteins or default variants.

    Candidate order is the input palette order and remains the tie breaker.
    Conjugated forms are selected by declared bonds, except final ligand
    fragments, whose cut connections already describe their input chemistry.
    """
    block_types = tuple(active_block_types)
    pbt = SimpleNamespace(
        active_block_types=block_types,
        n_types=len(block_types),
        max_n_atoms=max_n_atoms,
    )
    conjugations = prepare_conjugations(block_types)
    candidates = {
        bt.io_equiv_class: _term_and_spcase_var_candidate_lists(4, N_VARIANT_INDICES)
        for bt in block_types
    }
    non_default = numpy.zeros(len(block_types), dtype=bool)
    by_class = defaultdict(list)
    for index, bt in enumerate(block_types):
        by_class[bt.io_equiv_class].append(bt)
        if conjugations.conjugation_atoms_for_bt[index] and not bt.is_ligand_fragment:
            continue
        terminus, variant, non_default[index] = _assign_var_inds_for_bt(co, bt)
        candidates[bt.io_equiv_class][terminus][variant].append((bt, index))
    maximum = max(
        (
            len(group)
            for types in candidates.values()
            for variants in types
            for group in variants
        ),
        default=0,
    )
    # A padded candidate axis retains well-defined no-candidate diagnostics.
    maximum = max(1, maximum)
    indices, valid = _collect_var_combo_candidates(
        co, 4, N_VARIANT_INDICES, maximum, candidates
    )
    absent, present = _note_atoms_present_and_absent_from_variants(co, pbt)
    classes, atoms = _create_bt_to_canonical_atom_mapping(co, pbt)
    return CanonicalSelection(
        max_n_candidates_for_var_combo=maximum,
        co_equiv_class_is_polymeric=_get_co_equiv_class_is_polymeric(
            co, co.n_restype_io_equiv_classes, by_class
        ),
        co_polymer_connection_atoms=numpy.asarray(
            list(
                zip(
                    co.polymer_conn_inds.down_atom_for_co_restype,
                    co.polymer_conn_inds.up_atom_for_co_restype,
                )
            ),
            dtype=numpy.int64,
        ),
        var_combo_is_real_candidate=valid,
        var_combo_candidate_bt_index=indices,
        bt_canonical_atom_is_absent=absent,
        bt_non_term_patch_added_canonical_atom_is_present=present,
        bt_is_non_default_terminus=non_default,
        bt_ind_to_canonical_io_equiv_class_ind=classes,
        bt_canonical_atom_ind_map=atoms,
    )


def candidate_score_thresholds(max_n_canonical_atoms):
    """Mature acceptance scores: one extraneous atom warns, two reject."""
    warning = 2 * (max_n_canonical_atoms + 1)
    return warning, 2 * warning
