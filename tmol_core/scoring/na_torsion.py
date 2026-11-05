"""Nucleic-acid torsion topology and ownership, independent of execution."""

import numpy

# torsions this term reads, by chemical-layer name; order fixes tensor layout
BACKBONE_TORSIONS = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta")
SUGAR_TORSIONS = ("delta", "nu4", "nu0", "nu1")
CHI_TORSION = "chi1"

TORSION_NAMES = BACKBONE_TORSIONS + ("nu4", "nu0", "nu1", CHI_TORSION)
TORSION_IND = {name: i for i, name in enumerate(TORSION_NAMES)}

# one index space for both polymers: DNA occupies the first block and RNA the
# second, so the polymer of a base is base // N_BASE_PER_POLYMER and the tables
# that vary only by polymer are gathered on that.
POLYMERS = ("dna", "rna")
POLYMER_IND = {name: i for i, name in enumerate(POLYMERS)}
# "x" is a nucleotide whose base is not one of the four: averaged over them, so
# a residue with an unrecognized or absent base is still placed on the polymer's
# distribution rather than left unscored
UNKNOWN_BASE = "x"
BASES = {"dna": ("a", "c", "g", "t", "x"), "rna": ("a", "c", "g", "u", "x")}
N_BASE_PER_POLYMER = 5
N_BASE = len(POLYMERS) * N_BASE_PER_POLYMER
BASE_FOR_NAME3 = {
    "DA": 0, "DC": 1, "DG": 2, "DT": 3, "DX": 4,
    "A": 5, "C": 6, "G": 7, "U": 8, "RX": 9,
}  # fmt: skip
# the reference a nucleotide falls back to when its base cannot be identified
UNKNOWN_BASE_REFERENCE = {"dna": "DX", "rna": "RX"}
N_PUCKER = 10

# pucker states on the C3'-endo side; the rest are south
NORTH_PUCKERS = (0, 1, 2, 3, 4)

# chi is scored against a fixed syn mean inside this window
SYN_MEAN = 50.0
SYN_RANGE = (20.0, 100.0)


N_TORSION = len(TORSION_NAMES)
DELTA = TORSION_IND["delta"]
CHI = TORSION_IND[CHI_TORSION]
# every torsion is optional and masked at scoring time: the backbone ones are
# absent at a terminus, chi wherever the base is not attached through a torsion
# we can name, and the sugar ones wherever the ring does not resolve
REQUIRED_TORSIONS: list = []


def sugar_ring_atoms(block_type, element_for_atom_type):
    """Ordered sugar ring, derived from the nu torsions rather than named.

    nu0 and nu1 each span four consecutive ring atoms offset by one, so
    together they give the whole cycle in order. The pucker slot arithmetic
    is defined relative to a cycle ending on the ring heteroatom, so rotate
    to put it last.
    """
    nu0 = block_type.torsion_to_uaids.get("nu0")
    nu1 = block_type.torsion_to_uaids.get("nu1")
    if nu0 is None or nu1 is None:
        return None
    cycle = [uaid[0] for uaid in nu0] + [nu1[-1][0]]
    if len(set(cycle)) != 5 or any(a < 0 for a in cycle):
        return None

    def element(atom_index):
        return element_for_atom_type[block_type.atoms[atom_index].atom_type]

    hetero = [i for i, a in enumerate(cycle) if element(a) != "C"]
    if len(hetero) != 1:
        return None
    k = hetero[0]
    return cycle[k + 1 :] + cycle[: k + 1]


def _block_type_params(block_type, element_for_atom_type):
    """Per-block-type indices this term needs, shared by scoring and packing.

    base is -1 for anything this term does not handle. A nucleotide whose sugar
    ring or glycosidic torsion cannot be resolved still scores: those subterms
    are masked, and what remains of its backbone is scored as usual.
    """
    # a modified nucleotide may borrow a canonical base table
    base = BASE_FOR_NAME3.get(block_type.na_base_reference or block_type.name3, -1)
    uaids = numpy.full((N_TORSION, 4, 3), -1, dtype=numpy.int32)
    ring = numpy.full((5,), -1, dtype=numpy.int32)
    if base >= 0:
        for i, name in enumerate(TORSION_NAMES):
            tor = block_type.torsion_to_uaids.get(name)
            if tor is not None:
                uaids[i] = numpy.array(tor, dtype=numpy.int32)
        ring_atoms = sugar_ring_atoms(block_type, element_for_atom_type)
        if ring_atoms is not None:
            ring[:] = ring_atoms
        # A residue whose sugar ring does not resolve has no pucker, and this
        # term's chi and sugar energies are both multiplied by it -- so it would
        # contribute nothing while still claiming the delta/nu/chi central bonds
        # and keeping genbonded off them. Nothing would restrain them at all.
        # Disown it instead. REQUIRED_TORSIONS is empty by design: the torsions
        # themselves are optional and masked at scoring time.
        if (ring < 0).any() or (uaids[REQUIRED_TORSIONS, :, 0] < 0).any():
            base = -1
    down = block_type.connection_to_cidx.get("down", -1)
    return dict(base=base, uaids=uaids, ring=ring, down=down)


def _scored_torsion_bonds(block_type, element_for_atom_type):
    """Central bonds, as index pairs, of the torsions this term scores.

    Empty for anything the term does not handle. Another term that would
    otherwise constrain the same rotation reads this to stay off it.
    """
    params = _block_type_params(block_type, element_for_atom_type)
    if params["base"] < 0:
        return frozenset()
    bonds = set()
    for torsion in params["uaids"]:
        # a uaid names an atom of this block only when its atom index is set
        second, third = int(torsion[1][0]), int(torsion[2][0])
        if second >= 0 and third >= 0:
            bonds.add(frozenset((second, third)))
    return frozenset(bonds)
