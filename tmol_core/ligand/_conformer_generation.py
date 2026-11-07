"""Distance-geometry 3D coordinate generation for ligands from a SMILES.

Replaces OpenBabel's ``make3D`` + rotor search. Pipeline: MMFF ideal geometry
(RDKit, read-only) -> distance-bounds matrix -> metric-matrix embedding -> a
execution-adapter stress refine (4D chiral annealing with random restarts, then a full-weight
3D pass) -> OpenBabel force-field cleanup. MMFF-untypeable but modelable ligands
(e.g. pentavalent phosphoranes) fall back to covalent-radius / hybridization
ideals with explicit trigonal-bipyramidal constraints.

The entry point :func:`_generate_conformer` returns a coordinate-carrying pybel
``Molecule``; partial charges and atom names are assigned by the caller.
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import Optional

import numpy as np
from rdkit import (
    Chem,
    RDLogger,
)
from rdkit.Chem import AllChem

from tmol_core.ligand._atom_typing import _assign_tmol_atom_types
from tmol_core.ligand._openbabel_compat import _OPENBABEL_FORCEFIELD_LOCK

RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger(__name__)


# Stress-refine weights: exact (bond/angle/ring) distances tethered hard; bracket
# and vdW bounds are softer hinges (there are O(N^2) of them).
W_EXACT = 50.0
W_BOUND = 2.0
W_PLANE = 10.0
W_CHIRAL = 30.0
CHIRAL_MARGIN = 4.0  # target |chiral volume|; drives centers firmly pyramidal so
# the force-field cleanup cannot flip a near-planar stereocenter.
REFINE_ITERS = 200
STAGE_A_ITERS = 15  # LBFGS iters per 4D anneal level
STAGE_A_ANNEAL = (0.0, 0.5, 5.0, 50.0)  # 4th-dimension penalty ramp
N_RESTART = 5  # max random stage-A restarts; stop on all chiral signs correct

_ELECTRONEGATIVITY = {
    1: 2.20,
    6: 2.55,
    7: 3.04,
    8: 3.44,
    9: 3.98,
    15: 2.19,
    16: 2.58,
    17: 3.16,
    35: 2.96,
    53: 2.66,
}
_BOND_ORDER_SHORTENING = {1.0: 0.0, 1.5: 0.13, 2.0: 0.18, 3.0: 0.32}
_HYB_ANGLE = {
    Chem.HybridizationType.SP: 180.0,
    Chem.HybridizationType.SP2: 120.0,
    Chem.HybridizationType.SP3: 109.5,
}


def _generate_conformer(
    smiles: str, *, minimize_steps: int = 50, seed: Optional[int] = None, execution
):
    """Conformer generation script SMILES -> pybel.

    Uses a custom generation scheme similar to RDKit's ETKDG, where an embedding
    + minimization in the embedding space is used to generate a very good guess
    at the initial conformer.  Uses the supplied minimization execution, followed
    by a very short MMFF minimization.

    Args:
        smiles: Ligand SMILES to embed and minimize.
        minimize_steps: OpenBabel conjugate-gradient steps for the final min.
        seed: Optional fixed RNG seed for reproducible coordinates.

    Raises:
        ValueError: Failures in parsing or final min.
    """
    if execution is None:
        raise NotImplementedError(
            "distance-geometry conformer refinement requires an execution adapter"
        )
    if seed is None:
        return _generate_conformer_impl(smiles, minimize_steps, seed, execution)
    ob, pybel = _import_openbabel()
    # Seeded results are reusable, but callers own their coordinates and charges.
    return pybel.Molecule(
        ob.OBMol(_seeded_conformer(smiles, minimize_steps, seed, execution))
    )


@lru_cache(maxsize=128)
def _seeded_conformer(smiles: str, minimize_steps: int, seed: int, execution):
    return _generate_conformer_impl(smiles, minimize_steps, seed, execution).OBMol


def _generate_conformer_impl(
    smiles: str, minimize_steps: int, seed: Optional[int], execution
):
    _, pybel = _import_openbabel()
    obmol = _smiles_to_obmol(smiles)
    rd = _obmol_to_rdkit(obmol)
    if rd is None:
        raise ValueError(f"could not parse ligand chemistry for SMILES {smiles!r}")
    targets = _geometry_targets(rd, _double_bond_stereo(smiles, rd))
    with execution.flush_denormals():
        _set_coords(obmol, _embed_coordinates(targets, seed=seed, execution=execution))
    _forcefield_minimize(obmol, steps=minimize_steps, frozen=targets["frozen_atoms"])
    return pybel.Molecule(obmol)


# OpenBabel / RDKit interop
def _import_openbabel():
    from tmol_core.ligand._openbabel_compat import _import_openbabel as _imp

    return _imp()


def _smiles_to_obmol(smiles: str):
    _, pybel = _import_openbabel()
    pymol = pybel.readstring("smi", smiles)
    pymol.addh()
    return pymol.OBMol


def _obmol_to_rdkit(obmol) -> Optional[Chem.Mol]:
    """OBMol -> sanitized RDKit mol carrying stereo (via OB atom parity)."""
    from tmol_core.ligand._openbabel_compat import _obmol_to_rdkit_mol

    mol = _obmol_to_rdkit_mol(obmol, sanitize=True)
    if mol is not None:
        Chem.AssignAtomChiralTagsFromMolParity(mol)
    return mol


def _set_coords(obmol, coords: np.ndarray) -> None:
    for i in range(len(coords)):
        obmol.GetAtom(i + 1).SetVector(
            float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2])
        )


def _forcefield_minimize(obmol, *, steps: int, frozen=()) -> None:
    """In-place OpenBabel minimize:
    mmff94 by default
    uff fallback (for chemistries mmff94 cannot type)
    'frozen' atoms are held fixed"""
    openbabel, _ = _import_openbabel()
    # OpenBabel shares one mutable force-field instance per model across threads.
    with _OPENBABEL_FORCEFIELD_LOCK:
        for name in ("mmff94", "uff"):
            ff = openbabel.OBForceField.FindForceField(name)
            if ff is None:
                continue
            constraints = openbabel.OBFFConstraints()
            for i in frozen:
                constraints.AddAtomConstraint(int(i) + 1)  # OB is 1-indexed
            # ff is an obabel global
            # if called consecutively on _isomers_, connectivity of 1st is used
            #    for both in minimization
            # setting up an empty molecule invalidates and forces recomputation
            ff.Setup(openbabel.OBMol())
            if not ff.IsSetupNeeded(obmol):
                raise ValueError(
                    f"{name} setup cache was not invalidated; the ligand would be "
                    "minimized against another molecule's connectivity"
                )
            if ff.Setup(obmol, constraints):
                ff.ConjugateGradients(steps)
                ff.GetCoordinates(obmol)
                return
        raise ValueError("no force field could minimize the ligand")


# Geometry targets (ideal distances / angles / chirality)
def _ideal_13_distance(r12: float, r23: float, theta0_deg: float) -> float:
    t = math.radians(theta0_deg)
    return math.sqrt(r12 * r12 + r23 * r23 - 2 * r12 * r23 * math.cos(t))


def _chiral_volume_targets(rd: Chem.Mol):
    """Signed chiral-volume targets (c0,c1,c2,c3,sign), read off rd."""
    stable3 = {"S"}
    tag4 = {
        Chem.ChiralType.CHI_TETRAHEDRAL_CW: 1.0,
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW: -1.0,
    }
    tag3 = {
        Chem.ChiralType.CHI_TETRAHEDRAL_CW: -1.0,
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW: 1.0,
    }
    out = []
    for a in rd.GetAtoms():
        tag = a.GetChiralTag()
        if tag not in tag4:
            continue
        nb = [x.GetIdx() for x in a.GetNeighbors()]
        if len(nb) == 4:
            out.append(tuple(nb) + (tag4[tag],))
        elif len(nb) == 3 and a.GetSymbol() in stable3:
            out.append((a.GetIdx(),) + tuple(nb) + (tag3[tag],))
    return out


def _mmff_bonds_angles(rd: Chem.Mol, props):
    bonds = []
    for b in rd.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        p = props.GetMMFFBondStretchParams(rd, i, j)
        if p:
            bonds.append((i, j, p[2]))
    angles = []
    for a in rd.GetAtoms():
        v = a.GetIdx()
        nb = [x.GetIdx() for x in a.GetNeighbors()]
        for m in range(len(nb)):
            for o in range(m + 1, len(nb)):
                p = props.GetMMFFAngleBendParams(rd, nb[m], v, nb[o])
                if p and p[2] > 0:
                    angles.append((nb[m], v, nb[o], p[2]))
    return bonds, angles, ()


def _fallback_bonds_angles(rd: Chem.Mol, elements):  # noqa: C901
    """MMFF-free ideals: covalent-radius bonds, hybridization angles. Degree-5
    centers get explicit trigonal-bipyramidal constraints (most electronegative
    substituents axial, by apicophilicity) and are frozen during the cleanup min.
    Returns (bonds, angles, frozen_atoms)."""
    pt = Chem.GetPeriodicTable()
    bonds = []
    for b in rd.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        r0 = (
            pt.GetRcovalent(elements[i])
            + pt.GetRcovalent(elements[j])
            - _BOND_ORDER_SHORTENING.get(b.GetBondTypeAsDouble(), 0.0)
        )
        bonds.append((i, j, r0))
    angles = []
    frozen = set()
    for a in rd.GetAtoms():
        v = a.GetIdx()
        nb = [x.GetIdx() for x in a.GetNeighbors()]
        if len(nb) == 5:  # trigonal bipyramidal
            ordered = sorted(
                nb, key=lambda k: -_ELECTRONEGATIVITY.get(elements[k], 2.5)
            )
            axial, equatorial = ordered[:2], ordered[2:]
            angles.append((axial[0], v, axial[1], 180.0))
            for x in axial:
                for y in equatorial:
                    angles.append((x, v, y, 90.0))
            for m in range(len(equatorial)):
                for o in range(m + 1, len(equatorial)):
                    angles.append((equatorial[m], v, equatorial[o], 120.0))
            frozen.update([v] + nb)
        elif len(nb) < 5:
            th = _HYB_ANGLE.get(a.GetHybridization(), 109.5)
            for m in range(len(nb)):
                for o in range(m + 1, len(nb)):
                    angles.append((nb[m], v, nb[o], th))
    return bonds, angles, tuple(sorted(frozen))


def _planar_systems(rd: Chem.Mol, bonds):  # noqa: C901
    """Fused planar atom sets: joined by an aromatic bond or a small (<=7) all-sp2
    ring. Returns (components, components extended with first-shell neighbors)."""
    n = rd.GetNumAtoms()
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    sp2 = Chem.HybridizationType.SP2
    planar = set()
    for b in rd.GetBonds():
        if b.GetIsAromatic():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            union(i, j)
            planar.update((i, j))
    for ring in rd.GetRingInfo().AtomRings():
        if len(ring) <= 7 and all(
            rd.GetAtomWithIdx(a).GetIsAromatic()
            or rd.GetAtomWithIdx(a).GetHybridization() == sp2
            for a in ring
        ):
            for a in ring:
                union(a, ring[0])
                planar.add(a)
    groups: dict[int, list] = {}
    for a in planar:
        groups.setdefault(find(a), []).append(a)
    components = [tuple(sorted(v)) for v in groups.values() if len(v) >= 3]
    nbr: dict[int, set] = {}
    for i, j, _ in bonds:
        nbr.setdefault(i, set()).add(j)
        nbr.setdefault(j, set()).add(i)
    sets = [
        tuple(sorted(set(c) | {b for a in c for b in nbr.get(a, ())}))
        for c in components
    ]
    return components, sets


def _double_bond_stereo(smiles: str, rd: Chem.Mol) -> dict:
    """Cis/trans the SMILES declares, which ``rd`` (same atom order) lost (4XXJ retinal):
    {(i, j): (a, d, is_cis)}, a on i and d on j."""
    source = Chem.MolFromSmiles(smiles, sanitize=False)
    if source is None:
        return {}
    Chem.SetBondStereoFromDirections(source)
    out = {}
    for bond in source.GetBonds():
        stereo = bond.GetStereo()
        if stereo not in (Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS):
            continue
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        a, d = bond.GetStereoAtoms()
        if max(i, j, a, d) >= rd.GetNumAtoms() or any(
            source.GetAtomWithIdx(x).GetAtomicNum()
            != rd.GetAtomWithIdx(x).GetAtomicNum()
            for x in (i, j, a, d)
        ):
            return {}
        out[(i, j)] = (a, d, stereo == Chem.BondStereo.STEREOCIS)
    return out


def _double_bond_torsions(rd: Chem.Mol, stereo: dict):
    """(a, i, j, d, is_cis) per substituent pair across a double bond outside rings of
    up to 7 atoms; unspecified bonds put each side's bulkiest substituent trans."""
    ring_info = rd.GetRingInfo()

    def bulk(x, center):
        atom = rd.GetAtomWithIdx(x)
        heavy = sum(
            1
            for n in atom.GetNeighbors()
            if n.GetIdx() != center and n.GetAtomicNum() > 1
        )
        return (atom.GetAtomicNum() > 1, heavy, atom.GetAtomicNum(), -x)

    out = []
    for bond in rd.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE or bond.GetIsAromatic():
            continue
        if any(len(r) <= 7 for r in ring_info.BondRings() if bond.GetIdx() in r):
            continue
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        left = [
            n.GetIdx() for n in rd.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() != j
        ]
        right = [
            n.GetIdx() for n in rd.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != i
        ]
        if not left or not right or len(left) > 2 or len(right) > 2:
            continue
        if (i, j) in stereo:
            ref_a, ref_d, ref_cis = stereo[(i, j)]
        elif (j, i) in stereo:
            ref_d, ref_a, ref_cis = stereo[(j, i)]
        else:
            ref_a = max(left, key=lambda x: bulk(x, i))
            ref_d = max(right, key=lambda x: bulk(x, j))
            ref_cis = False
        if ref_a not in left or ref_d not in right:
            continue
        for a in left:
            for d in right:
                out.append((a, i, j, d, ref_cis ^ (a != ref_a) ^ (d != ref_d)))
    return out


def _conjugated_single_bonds(rd: Chem.Mol):
    """(i, j) for the single bonds ligand typing promotes to double (conjugated, amide)."""
    mol = Chem.Mol(rd)
    try:
        _assign_tmol_atom_types(mol)
    except ValueError:
        return []
    return [
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        for b in rd.GetBonds()
        if b.GetBondType() == Chem.BondType.SINGLE
        and mol.GetBondWithIdx(b.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
    ]


def _planar_14_distance(r_ai, r_ij, r_jd, theta_aij, theta_ijd, is_cis):
    """Distance a-d across the bond i-j at dihedral 0 (cis) or 180 (trans), degrees."""
    t1, t2 = math.radians(theta_aij), math.radians(theta_ijd)
    ax, ay = r_ai * math.cos(t1), r_ai * math.sin(t1)
    dx = r_ij - r_jd * math.cos(t2)
    dy = r_jd * math.sin(t2) * (1.0 if is_cis else -1.0)
    return math.hypot(dx - ax, dy - ay)


def _geometry_targets(rd: Chem.Mol, double_bond_stereo: Optional[dict] = None) -> dict:
    elements = [a.GetAtomicNum() for a in rd.GetAtoms()]
    props = AllChem.MMFFGetMoleculeProperties(rd)
    if props is not None:
        bonds, angles, frozen = _mmff_bonds_angles(rd, props)
    else:
        bonds, angles, frozen = _fallback_bonds_angles(rd, elements)
    components, sets = _planar_systems(rd, bonds)
    double_bonds = _double_bond_torsions(rd, double_bond_stereo or {})
    # each double bond, and each single bond typing marks conjugated, is flat with
    # its substituents
    flat: dict[tuple, set] = {}
    for a, i, j, d, _ in double_bonds:
        flat.setdefault((i, j), {i, j}).update((a, d))
    for i, j in _conjugated_single_bonds(rd):
        flat.setdefault((i, j), {i, j}).update(
            n.GetIdx() for x in (i, j) for n in rd.GetAtomWithIdx(x).GetNeighbors()
        )
    sets = sets + [tuple(sorted(x)) for x in flat.values()]
    pt = Chem.GetPeriodicTable()
    return dict(
        n=rd.GetNumAtoms(),
        elements=elements,
        vdw=[pt.GetRvdw(z) for z in elements],
        bonds=bonds,
        angles=angles,
        planar_components=components,
        planar_sets=sets,
        chirals=_chiral_volume_targets(rd),
        double_bonds=double_bonds,
        frozen_atoms=frozen,
    )


# Distance-bounds matrix + smoothing + embedding
def _planar_component_distances(atoms, r0, ang, max_iters: int = 150, *, execution):
    """Given one planar ring system, compute the exact distance between
    _every_ pair of its atoms assuming it's perfectly flat.

    Do this using the same trick as the whole molecule (embed + min)
    but in 2D."""
    idx = {a: k for k, a in enumerate(atoms)}
    m = len(atoms)
    aset = set(atoms)
    tp = []
    for a in atoms:
        for b in atoms:
            if a < b and (a, b) in r0:
                tp.append((idx[a], idx[b], r0[(a, b)]))
    for (lo, v, hi), th0 in ang.items():
        if lo in aset and hi in aset and v in aset and (lo, v) in r0 and (v, hi) in r0:
            tp.append(
                (idx[lo], idx[hi], _ideal_13_distance(r0[(lo, v)], r0[(v, hi)], th0))
            )
    if len(tp) < m:
        return None
    Lc = np.zeros((m, m))
    Uc = np.full((m, m), 1e5)
    np.fill_diagonal(Uc, 0.0)
    for i, j, t in tp:
        Lc[i, j] = Lc[j, i] = Uc[i, j] = Uc[j, i] = t
    Lc, Uc = _smooth_bounds(Lc, Uc)
    Dm = 0.5 * (Lc + Uc)
    Jc = np.eye(m) - 1.0 / m
    w, V = np.linalg.eigh(-0.5 * (Jc @ (Dm**2) @ Jc))
    o = np.argsort(w)[::-1]
    Y0 = V[:, o[:2]] * np.sqrt(np.clip(w[o[:2]], 0.0, None))
    Yc = execution.planar_refine(Y0, tp, max_iters)
    return {
        (atoms[k], atoms[atom_j]): float(np.linalg.norm(Yc[k] - Yc[atom_j]))
        for k in range(m)
        for atom_j in range(k + 1, m)
    }


def _bounds_matrix(targets: dict, *, execution):
    """Build the distance bounds matrix:
    * 1-2 and 1-3 atom distances exact given ideal geometry
    * planar pairs in same ring system exact given 2D embedding
    * 1-4 pairs across an acyclic double bond exact, planar at its cis/trans
    * lower bounded given VDW radii for all others"""
    n = targets["n"]
    vdw = np.array(targets["vdw"], float)
    L = vdw[:, None] + vdw[None, :]
    U = np.full((n, n), 1e5)
    np.fill_diagonal(L, 0.0)
    np.fill_diagonal(U, 0.0)
    exact = np.zeros((n, n), bool)

    def fix(i, j, d):
        L[i, j] = L[j, i] = U[i, j] = U[j, i] = d
        exact[i, j] = exact[j, i] = True

    r0 = {}
    for i, j, d in targets["bonds"]:
        r0[(i, j)] = r0[(j, i)] = d
    ang = {}
    for i, v, j, th0 in targets["angles"]:
        ang[(min(i, j), v, max(i, j))] = th0
    for i, j, d in targets["bonds"]:
        fix(i, j, d)
    for i, v, j, th0 in targets["angles"]:
        if not exact[i, j]:
            fix(i, j, _ideal_13_distance(r0[(i, v)], r0[(v, j)], th0))
    for comp in targets["planar_components"]:
        dists = _planar_component_distances(comp, r0, ang, execution=execution)
        if dists:
            for (i, j), d in dists.items():
                fix(i, j, d)
    for a, i, j, d, is_cis in targets.get("double_bonds", ()):
        key_a, key_d = (min(a, j), i, max(a, j)), (min(i, d), j, max(i, d))
        if exact[a, d] or key_a not in ang or key_d not in ang:
            continue
        fix(
            a,
            d,
            _planar_14_distance(
                r0[(a, i)], r0[(i, j)], r0[(j, d)], ang[key_a], ang[key_d], is_cis
            ),
        )
    L = np.minimum(L, U)
    return L, U


def _smooth_bounds(L, U, lower_passes: int = 2):
    """Triangle-inequality smoothing:
    * upper bound via Floyd-Warshall shortest path,
    * lower bound via the inverse triangle inequality."""
    n = len(U)
    U = U.copy()
    L = L.copy()
    for k in range(n):
        U = np.minimum(U, U[:, k, None] + U[None, k, :])
    for _ in range(lower_passes):
        for k in range(n):
            cand = L[:, k, None] - U[k, None, :]
            L = np.maximum(L, cand)
            L = np.maximum(L, cand.T)
    np.fill_diagonal(L, 0.0)
    L = np.minimum(L, U)
    return L, U


def _metric_embed(L, U, seed: int) -> np.ndarray:
    """The main embedding:
    - randomly sample distances in [L, U],
    - double-center to Gram mtx
    - eigendecomposition to get coordinates."""
    n = len(L)
    rng = np.random.default_rng(seed)
    D = np.zeros((n, n))
    iu = np.triu_indices(n, 1)
    D[iu] = rng.uniform(L[iu], U[iu])
    D = D + D.T
    J = np.eye(n) - 1.0 / n
    G = -0.5 * (J @ (D**2) @ J)
    w, V = np.linalg.eigh(G)
    order = np.argsort(w)[::-1]
    coords = V[:, order[:3]] * np.sqrt(np.clip(w[order[:3]], 0.0, None))
    return np.pad(coords, ((0, 0), (0, 3 - n))) if n < 3 else coords


def _chiral_ok(X: np.ndarray, chirals):
    """
    Check stereochemical agreement to target.

    returns (n_matching, n_total) chiral centers."""
    ch = np.array(chirals, float).reshape(-1, 5)
    if len(ch) == 0:
        return (0, 0)
    c = ch[:, :4].astype(int)
    v = np.einsum(
        "ij,ij->i",
        X[c[:, 1]] - X[c[:, 0]],
        np.cross(X[c[:, 2]] - X[c[:, 0]], X[c[:, 3]] - X[c[:, 0]]),
    )
    return (int(np.sum(np.sign(v) == np.sign(ch[:, 4]))), len(ch))


def _embed_coordinates(
    targets: dict, *, seed: Optional[int] = None, execution
) -> np.ndarray:
    """Outermost embedding loop:
    - run stage 1 from random conditions until chirals are all correct
      (at most N_RESTART times)
    - run stage 2
    - return final embedding
    """
    L0, U0 = _bounds_matrix(targets, execution=execution)
    L, U = _smooth_bounds(L0, U0)
    components = targets["planar_sets"]
    chirals = targets["chirals"]
    seeds = np.random.SeedSequence(seed).generate_state(N_RESTART)
    if chirals:
        best = None
        for s in seeds:
            X = execution.chiral_anneal(
                _metric_embed(L, U, int(s)), L0, U0, L, U, components, chirals, int(s)
            )
            n_ok, n_total = _chiral_ok(X, chirals)
            if best is None or n_ok > best[0]:
                best = (n_ok, X)
            if n_ok == n_total:
                break
        X0 = best[1]
    else:
        X0 = _metric_embed(L, U, int(seeds[0]))
    return execution.stress_refine(X0, L0, U0, L, U, components, chirals)
