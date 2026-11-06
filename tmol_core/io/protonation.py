"""Protonation evidence and donor-form metadata for the shared native selector."""

from collections import defaultdict
from dataclasses import dataclass
import attr
import numpy
from tmol_core.chemical import (
    DEPROTONATED_VAR_IND,
    NEUTRAL_TERMINUS,
    NEUTRAL_TERMINUS_VAR_BASE,
    _special_case_variant_index,
)


def variant_atoms(canonical_ordering, chemical_db, res_types):
    """Per present class and variant: hydrogen and metal-donor canonical atoms.

    Also each hydrogen's heavy parent and the connection atoms, per class.
    """
    classes = canonical_ordering.restype_io_equiv_classes
    present = {int(c) for c in numpy.unique(res_types[res_types >= 0]).tolist()}
    wanted = {classes[c]: c for c in present}
    atom_type = {at.name: at for at in chemical_db.atom_types}
    hydrogens = defaultdict(lambda: defaultdict(set))
    donates = defaultdict(lambda: defaultdict(set))
    parent = defaultdict(dict)
    linked = defaultdict(set)
    for res in chemical_db.residues:
        c = wanted.get(res.io_equiv_class)
        if c is None:
            continue
        v = _special_case_variant_index(res)
        index = canonical_ordering.restypes_atom_index_mapping[res.io_equiv_class]
        linked[c].update(index[x.atom] for x in res.connections if x.atom in index)
        element = {}
        for a in res.atoms:
            at = atom_type.get(a.atom_type)
            if a.name not in index or at is None:
                continue
            element[a.name] = at.element
            if at.element == "H" and v <= DEPROTONATED_VAR_IND:
                hydrogens[c][v].add(index[a.name])
            if at.is_metal_donor:
                donates[c][v].add(index[a.name])
        for a, b, *_ in res.bonds:
            for h, heavy in ((a, b), (b, a)):
                if element.get(h) == "H" and element.get(heavy, "H") != "H":
                    parent[c][index[h]] = index[heavy]
    return hydrogens, donates, parent, linked


def neutral_terminus_patches(canonical_ordering, chemical_db, res_types, variants):
    """The neutral amino terminus, for each base type of a class a residue asks it of.

    The database's patch applies to no type, so its forms exist only where needed.
    """
    asked = numpy.unique(
        res_types[(res_types >= 0) & (variants >= NEUTRAL_TERMINUS_VAR_BASE)]
    ).tolist()
    if not asked:
        return ()
    classes = {canonical_ordering.restype_io_equiv_classes[c] for c in asked}
    patch = next(v for v in chemical_db.variants if v.display_name == NEUTRAL_TERMINUS)
    return tuple(
        attr.evolve(
            patch,
            name=f"{patch.name}_{res.name}",
            applies_to=attr.evolve(patch.applies_to, base_names=(res.name,)),
        )
        for res in chemical_db.residues
        if res.name == res.base_name and res.io_equiv_class in classes
    )


@dataclass(eq=False)
class ProtonationPlan:
    hydrogen: numpy.ndarray
    titratable: numpy.ndarray
    parents: numpy.ndarray
    shared: numpy.ndarray
    expecting: numpy.ndarray
    donates: numpy.ndarray

    @property
    def arrays(self):
        return (
            self.hydrogen,
            self.titratable,
            self.parents,
            self.shared,
            self.expecting,
            self.donates,
        )


_LAST_PLAN = None


def prepare_protonation_plan(canonical_ordering, chemical_db, residue_types):
    """Pack exact hydrogen evidence/variant rules; key caches by actual databases."""
    global _LAST_PLAN
    types = numpy.asarray(residue_types)
    present = tuple(numpy.unique(types[types >= 0]).tolist())
    if _LAST_PLAN is not None:
        co, db, classes, plan = _LAST_PLAN
        if co is canonical_ordering and db is chemical_db and classes == present:
            return plan
    hydrogens, donates, parent, linked = variant_atoms(
        canonical_ordering, chemical_db, types
    )
    ntypes = len(canonical_ordering.restype_io_equiv_classes)
    natoms = canonical_ordering.max_n_canonical_atoms
    nvariants = (
        max(
            (v for forms in donates.values() for v in forms),
            default=DEPROTONATED_VAR_IND,
        )
        + 1
    )
    hydrogen, titratable, parents, shared = (
        numpy.zeros((ntypes, natoms), bool) for _ in range(4)
    )
    expecting = numpy.zeros((ntypes, nvariants), bool)
    donors = numpy.zeros((ntypes, nvariants, natoms), bool)
    for cls, forms in donates.items():
        for variant, atoms in forms.items():
            donors[cls, variant, list(atoms)] = True
    for cls, forms in hydrogens.items():
        all_h = set().union(*forms.values())
        hydrogen[cls, list(all_h)] = True
        if DEPROTONATED_VAR_IND not in forms:
            continue
        titr = (
            set().union(*(h for v, h in forms.items() if v != DEPROTONATED_VAR_IND))
            - forms[DEPROTONATED_VAR_IND]
        )
        expects = [
            v for v, h in forms.items() if v != DEPROTONATED_VAR_IND and h & titr
        ]
        if not titr or not expects:
            continue
        # A hydrogen-bearing form need not have any donor atoms.
        if max(expects) >= nvariants:
            size = max(expects) + 1
            expecting = numpy.pad(expecting, ((0, 0), (0, size - nvariants)))
            donors = numpy.pad(donors, ((0, 0), (0, size - nvariants), (0, 0)))
            nvariants = size
        expecting[cls, expects] = True
        titratable[cls, list(titr)] = True
        parents[cls, list({parent[cls][h] for h in titr if h in parent[cls]})] = True
        common = set.intersection(*map(set, forms.values()))
        shared[cls, [h for h in common if parent[cls].get(h) not in linked[cls]]] = True
    plan = ProtonationPlan(hydrogen, titratable, parents, shared, expecting, donors)
    _LAST_PLAN = canonical_ordering, chemical_db, present, plan
    return plan


def required_donor_rows(assignments):
    """Pose, residue and canonical atom for every assigned donor."""
    return numpy.asarray(
        [(p, r, a) for (p, _), got in assignments for r, a in got.donor_atoms],
        dtype=numpy.int64,
    ).reshape(-1, 3)
