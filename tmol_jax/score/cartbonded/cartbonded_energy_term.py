"""Cartesian bonded lengths, angles, torsions, impropers and hydroxyl torsions.

Setup enumerates intra-residue and connection-spanning subgraphs, resolves
cartbonded.yaml parameters and maps atoms to pose indices. Masks follow
chemical topology, independent of coordinates. Torsions and impropers share
the canonical three-cosine potential with (k1, k2, k3, phi1, phi2, phi3)
parameters. Unparameterized subgraphs contribute zero, as in the native lookup."""

from __future__ import annotations

from itertools import permutations
import itertools
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.score.common.geom import (
    distance,
    dihedral_angle,
    pt_interior_angle,
)
from tmol_jax.score.energy_term import EnergyTerm
from tmol_jax.score.score_types import ScoreType
from tmol_jax.types import pytree_dataclass

from tmol_jax.score.cartbonded.potentials import (
    bond_angle_E,
    bond_length_E,
    hxltorsion_E,
)

# Precomputed per-pose-stack state (pytree)                                    #


@pytree_dataclass
class CartBondedTermState:
    """Conformation-independent CartBonded tables for a :class:`PoseStack`.

    Every field is padded across the pose / block axis. Real-vs-padding is
    tracked with a boolean mask per category so the hot-path reduction is
    a simple masked ``jnp.sum``.

    Shape legend: ``P`` = n_poses, ``B`` = max_n_blocks,
    ``L`` / ``A`` / ``T`` / ``I`` / ``H`` = per-category pad widths.
    """

    # Bond lengths — 2 atoms, params (d0, K).
    length_atoms: jnp.ndarray  # int32[P, B, L, 2] global atom indices
    length_params: jnp.ndarray  # float32[P, B, L, 2] (d0, K)
    length_mask: jnp.ndarray  # bool[P, B, L]

    # Bond angles — 3 atoms, params (theta0, K).
    angle_atoms: jnp.ndarray  # int32[P, B, A, 3]
    angle_params: jnp.ndarray  # float32[P, B, A, 2]
    angle_mask: jnp.ndarray  # bool[P, B, A]

    # Proper torsions (periodic, sum of three cosines).
    # We pass four atoms + six params (k1, k2, k3, phi1, phi2, phi3).
    torsion_atoms: jnp.ndarray  # int32[P, B, T, 4]
    torsion_params: jnp.ndarray  # float32[P, B, T, 6]
    torsion_mask: jnp.ndarray  # bool[P, B, T]

    # Improper torsions. Upstream tmol uses the same sum-of-three-cosines
    # ``cbtorsion_V_dV`` helper for impropers as for proper torsions —
    # only the score-type bucket differs — so we store the full 6-wide
    # ``(k1, k2, k3, phi1, phi2, phi3)`` vector.
    improper_atoms: jnp.ndarray  # int32[P, B, I, 4]
    improper_params: jnp.ndarray  # float32[P, B, I, 6]
    improper_mask: jnp.ndarray  # bool[P, B, I]

    # Hydroxyl torsions — same six-param layout as proper torsions.
    hxltorsion_atoms: jnp.ndarray  # int32[P, B, H, 4]
    hxltorsion_params: jnp.ndarray  # float32[P, B, H, 6]
    hxltorsion_mask: jnp.ndarray  # bool[P, B, H]

    # Inter-residue bond lengths — 2 atoms, params (d0, K). These are
    # keyed by ``(pose, entry)``; shape ``[P, IL, ...]`` where ``IL``
    # is the max number of inter-residue length subgraphs across the
    # stack. Emitted only for ``block_i < block_j`` to avoid double-
    # counting, matching upstream.
    inter_length_atoms: jnp.ndarray  # int32[P, IL, 2]
    inter_length_params: jnp.ndarray  # float32[P, IL, 2]
    inter_length_mask: jnp.ndarray  # bool[P, IL]

    # Inter-residue bond angles — 3 atoms, params (theta0, K).
    inter_angle_atoms: jnp.ndarray  # int32[P, IA, 3]
    inter_angle_params: jnp.ndarray  # float32[P, IA, 2]
    inter_angle_mask: jnp.ndarray  # bool[P, IA]

    # Inter-residue proper torsions — 4 atoms, 6 params.
    inter_torsion_atoms: jnp.ndarray  # int32[P, IT, 4]
    inter_torsion_params: jnp.ndarray  # float32[P, IT, 6]
    inter_torsion_mask: jnp.ndarray  # bool[P, IT]

    # Inter-residue impropers centred on a connection atom — 4 atoms, 6 params.
    inter_improper_atoms: jnp.ndarray  # int32[P, II, 4]
    inter_improper_params: jnp.ndarray  # float32[P, II, 6]
    inter_improper_mask: jnp.ndarray  # bool[P, II]


# Database lookup helpers                                                      #


def _bond_atom_map(bond_indices: np.ndarray, n_atoms: int) -> Dict[int, List[int]]:
    """Build a ``{atom -> [neighbour atoms]}`` adjacency map.

    Mirrors the ``bondmap`` built in the upstream ``find_subgraphs``.
    Accepts both directed (each edge listed once) and undirected
    (both directions present) bond index arrays.
    """
    adj: Dict[int, List[int]] = {i: [] for i in range(n_atoms)}
    for i in range(bond_indices.shape[0]):
        a, b = int(bond_indices[i, 0]), int(bond_indices[i, 1])
        if 0 <= a < n_atoms and 0 <= b < n_atoms:
            if b not in adj[a]:
                adj[a].append(b)
            if a not in adj[b]:
                adj[b].append(a)
    return adj


def _find_intra_subgraphs(
    adj: Dict[int, List[int]], improper_roots_local: Sequence[int]
) -> Tuple[
    List[Tuple[int, int]],
    List[Tuple[int, int, int]],
    List[Tuple[int, int, int, int]],
    List[Tuple[int, int, int, int]],
]:
    """Enumerate intra-residue subgraphs from a bond adjacency map.

    Returns ``(lengths, angles, torsions, impropers)`` — each a list of
    local-atom-index tuples. The enumeration logic mirrors upstream
    ``CartBondedEnergyTerm.find_subgraphs``:

    * Lengths ``(a, b)`` with ``a < b``.
    * Angles  ``(a, b, c)`` with ``a < c`` (canonicalized).
    * Torsions ``(a, b, c, d)`` with ``a != c``, ``b != d``, ``a < d``.
    * Impropers: for every local index ``a3`` in ``improper_roots_local``,
      every 3-permutation of its neighbours ``(a1, a2, a4)``.
    """
    lengths: List[Tuple[int, int]] = []
    angles: List[Tuple[int, int, int]] = []
    torsions: List[Tuple[int, int, int, int]] = []
    impropers: List[Tuple[int, int, int, int]] = []

    atoms = sorted(adj.keys())

    for a1 in atoms:
        for a2 in adj[a1]:
            if a1 < a2:
                lengths.append((a1, a2))

    for a1 in atoms:
        for a2 in adj[a1]:
            for a3 in adj[a2]:
                if a1 >= a3:
                    continue
                angles.append((a1, a2, a3))

    for a1 in atoms:
        for a2 in adj[a1]:
            for a3 in adj[a2]:
                if a1 == a3:
                    continue
                for a4 in adj[a3]:
                    if a2 == a4:
                        continue
                    if a1 >= a4:
                        continue
                    torsions.append((a1, a2, a3, a4))

    for a3 in improper_roots_local:
        neighbours = list(adj.get(a3, ()))
        for a1, a2, a4 in permutations(neighbours, 3):
            impropers.append((a1, a2, a3, a4))

    return lengths, angles, torsions, impropers


def _improper_root_atom_names(cart_db: Any) -> set:
    """Set of atom names that appear as the ``atm3`` of any improper in the DB.

    These are the "roots" around which the upstream code enumerates
    improper torsions. Mirrors ``find_improper_roots`` in the torch
    version.
    """
    roots: set = set()
    for _res, params in cart_db.residue_params.items():
        for imp in params.improper_parameters:
            roots.add(imp.atm3)
    return roots


def _parameter_name(rt, cart_db):
    """Use an exact residue variant when its own parameters are available."""
    if rt.name in cart_db.residue_params:
        return rt.name
    return rt.base_name


# Parameter keying                                                             #


def _length_key(p: Any) -> Tuple[str, ...]:
    return (p.atm1, p.atm2)


def _angle_key(p: Any) -> Tuple[str, ...]:
    return (p.atm1, p.atm2, p.atm3)


def _torsion_key(p: Any) -> Tuple[str, ...]:
    return (p.atm1, p.atm2, p.atm3, p.atm4)


def _reverse(key: Tuple[str, ...]) -> Tuple[str, ...]:
    """Canonical reversal used by upstream hash lookup.

    For length/angle/torsion subgraphs, reversing means swapping the
    first and last *real* atom; for lengths and torsions this is a full
    reversal, for angles it leaves the central atom fixed.
    """
    return tuple(reversed(key))


def _cos3_vals(p: Any) -> Tuple[float, ...]:
    """6-wide sum-of-three-cosines param vector for torsion-shaped entries."""
    return (p.k1, p.k2, p.k3, p.phi1, p.phi2, p.phi3)


def _fill_sym_db(
    db: Dict[Tuple[str, ...], Any],
    params_iter: Sequence[Any],
    key_fn: Any,
    val_fn: Any,
) -> None:
    """Insert ``(key, val)`` and ``(reversed key, val)`` for each param."""
    for p in params_iter:
        k = key_fn(p)
        v = val_fn(p)
        db[k] = v
        db[_reverse(k)] = v


# Per-block-type subgraph build                                                #


def _subgraphs_for_block_type(rt: Any, cart_db: Any, improper_roots: set) -> Tuple[
    List[Tuple[Tuple[int, int], Tuple[float, float]]],
    List[Tuple[Tuple[int, int, int], Tuple[float, float]]],
    List[
        Tuple[
            Tuple[int, int, int, int], Tuple[float, float, float, float, float, float]
        ]
    ],
    List[
        Tuple[
            Tuple[int, int, int, int], Tuple[float, float, float, float, float, float]
        ]
    ],
    List[
        Tuple[
            Tuple[int, int, int, int], Tuple[float, float, float, float, float, float]
        ]
    ],
]:
    """Enumerate + parameterize intra-residue cartbonded subgraphs.

    Returns five lists keyed by subgraph category, each entry a
    ``(local_atom_tuple, param_tuple)`` pair. Subgraphs with no database
    entry are dropped (matching the upstream "hash miss" behaviour).

    Impropers are treated identically to proper torsions in upstream
    tmol (they share ``cbtorsion_V_dV`` — a sum of three cosines). The
    only runtime distinction is which score-type bucket the per-entry
    energy accumulates into. We mirror that by storing improper params
    as the same 6-wide ``(k1, k2, k3, phi1, phi2, phi3)`` vector used
    by proper torsions / hxl torsions.
    """
    n_atoms = rt.n_atoms
    bond_indices = rt.bond_indices
    if bond_indices is None:
        bond_indices = np.zeros((0, 2), dtype=np.int32)

    adj = _bond_atom_map(bond_indices, n_atoms)

    # Improper roots for this residue are the local indices of the atoms
    # whose name is in the global improper-root name set.
    atom_to_idx = rt.atom_to_idx
    improper_roots_local: List[int] = [
        atom_to_idx[name] for name in improper_roots if name in atom_to_idx
    ]

    lengths_sg, angles_sg, torsions_sg, impropers_sg = _find_intra_subgraphs(
        adj, improper_roots_local
    )

    base = _parameter_name(rt, cart_db)
    res = cart_db.residue_params.get(base)
    # Build name-keyed parameter dicts. For each category we store both
    # the forward and reversed key so callers get symmetric lookup.
    length_db: Dict[Tuple[str, ...], Tuple[float, float]] = {}
    angle_db: Dict[Tuple[str, ...], Tuple[float, float]] = {}
    torsion_db: Dict[Tuple[str, ...], Tuple[float, ...]] = {}
    improper_db: Dict[Tuple[str, ...], Tuple[float, ...]] = {}
    hxl_db: Dict[Tuple[str, ...], Tuple[float, ...]] = {}

    wildcard = cart_db.residue_params.get("wildcard")
    sources = [source for source in (wildcard, res) if source is not None]

    # Impropers share the sum-of-three-cosines form with proper torsions
    # in upstream ``cbtorsion_V_dV`` (potentials.hh); only the score-type
    # bucket differs.
    def _xK(p):
        return (p.x0, p.K)

    for res in sources:
        _fill_sym_db(length_db, res.length_parameters, _length_key, _xK)
        _fill_sym_db(angle_db, res.angle_parameters, _angle_key, _xK)
        _fill_sym_db(torsion_db, res.torsion_parameters, _torsion_key, _cos3_vals)
        _fill_sym_db(improper_db, res.improper_parameters, _torsion_key, _cos3_vals)
        _fill_sym_db(hxl_db, res.hxltorsion_parameters, _torsion_key, _cos3_vals)

    atom_name = [a.name for a in rt.atoms]

    def _name_tuple(atoms: Sequence[int]) -> Tuple[str, ...]:
        return tuple(atom_name[a] for a in atoms)

    lengths_out: List[Tuple[Tuple[int, int], Tuple[float, float]]] = []
    for atoms in lengths_sg:
        key = _name_tuple(atoms)
        params = length_db.get(key) or length_db.get(_reverse(key))
        if params is not None:
            lengths_out.append((atoms, params))

    angles_out: List[Tuple[Tuple[int, int, int], Tuple[float, float]]] = []
    for atoms in angles_sg:
        key = _name_tuple(atoms)
        params = angle_db.get(key) or angle_db.get(_reverse(key))
        if params is not None:
            angles_out.append((atoms, params))

    torsions_out: List[
        Tuple[
            Tuple[int, int, int, int], Tuple[float, float, float, float, float, float]
        ]
    ] = []
    impropers_out: List[
        Tuple[
            Tuple[int, int, int, int], Tuple[float, float, float, float, float, float]
        ]
    ] = []
    hxl_out: List[
        Tuple[
            Tuple[int, int, int, int], Tuple[float, float, float, float, float, float]
        ]
    ] = []

    # Proper torsions: first try torsion_db, then hxl_db. hxl overrides
    # torsion if the exact same key is in both (matches upstream "score
    # type" priority).
    for atoms in torsions_sg:
        key = _name_tuple(atoms)
        rev = _reverse(key)
        if key in hxl_db or rev in hxl_db:
            vals = hxl_db.get(key) or hxl_db.get(rev)
            hxl_out.append((atoms, vals))
        elif key in torsion_db or rev in torsion_db:
            vals = torsion_db.get(key) or torsion_db.get(rev)
            torsions_out.append((atoms, vals))

    # Impropers are enumerated separately.
    for atoms in impropers_sg:
        key = _name_tuple(atoms)
        rev = _reverse(key)
        vals = improper_db.get(key) or improper_db.get(rev)
        if vals is not None:
            impropers_out.append((atoms, vals))

    return lengths_out, angles_out, torsions_out, impropers_out, hxl_out


# Inter-residue subgraph enumeration                                           #
#
# Mirrors the logic in ``tmol/score/common/connection.hh``:
#
# * For each connection on a residue, ``atom_paths_from_conn`` lists
#   partial paths of 1, 2, or 3 atoms starting at the connection atom.
#   Layout (matching ``get_paths_length_3`` in
#   ``tmol_jax/chemical/restypes.py``):
#
#     path 0       = (conn_atom,)                        -- length 1
#     paths 1-3    = (conn_atom, nb_k)                   -- length 2
#     paths 4-12   = (conn_atom, nb_k1, nb_k1's nb_k2)   -- length 3
#
# * The upstream C++ kernel iterates 34 combinations
#   ``(category, resA_path_ind, resB_path_ind)`` — see
#   ``CON_PATH_INDICES`` in ``connection.hh`` — yielding 1 length,
#   6 angles, and 27 torsions that span any given inter-residue
#   connection. For each combo it joins the (reversed) A-path tail with
#   the B-path head to build a 2-/3-/4-atom subgraph crossing the bond.
#
# We mirror that enumeration to get inter-residue (length, angle,
# torsion) subgraph candidates, then do the DB lookup in the order:
# residue A's per-residue DB -> residue B's per-residue DB (reversed) ->
# the ``wildcard`` DB. Missing-params entries are dropped, matching the
# upstream hash-miss behaviour.


_CON_PATH_INDICES: List[Tuple[int, int, int]] = [
    # category 0 = length
    (0, 0, 0),
    # category 1 = angle
    (1, 0, 1),
    (1, 0, 2),
    (1, 0, 3),
    (1, 1, 0),
    (1, 2, 0),
    (1, 3, 0),
    # category 2 = torsion
    (2, 0, 4),
    (2, 0, 5),
    (2, 0, 6),
    (2, 0, 7),
    (2, 0, 8),
    (2, 0, 9),
    (2, 0, 10),
    (2, 0, 11),
    (2, 0, 12),
    (2, 1, 1),
    (2, 1, 2),
    (2, 1, 3),
    (2, 2, 1),
    (2, 2, 2),
    (2, 2, 3),
    (2, 3, 1),
    (2, 3, 2),
    (2, 3, 3),
    (2, 4, 0),
    (2, 5, 0),
    (2, 6, 0),
    (2, 7, 0),
    (2, 8, 0),
    (2, 9, 0),
    (2, 10, 0),
    (2, 11, 0),
    (2, 12, 0),
]


def _path_atoms(path: np.ndarray) -> Tuple[int, ...]:
    """Extract the non-``-1`` atom indices from a length-3 path row.

    ``atom_paths_from_conn`` rows are right-padded with ``-1``; real
    atom indices fill the leading slots only.
    """
    return tuple(int(a) for a in path if int(a) != -1)


_ParamDBs = Tuple[
    Dict[Tuple[str, ...], Tuple[float, ...]],  # lengths (x0, K)
    Dict[Tuple[str, ...], Tuple[float, ...]],  # angles (x0, K)
    Dict[Tuple[str, ...], Tuple[float, ...]],  # torsions, six-wide
    Dict[Tuple[str, ...], Tuple[float, ...]],  # impropers, six-wide
]


def _build_cart_lat_dbs(res: Any) -> _ParamDBs:
    """Build name-keyed (length, angle, torsion, improper) DB dicts from a
    residue's cart-bonded params (or ``None``).

    Keys are atom-name tuples in their authored orientation; a ``+`` prefix
    marks an atom of the partner residue across a connection. Parameter
    tuples are ``(x0, K)`` for lengths and angles, and the full 6-wide
    ``(k1, k2, k3, phi1, phi2, phi3)`` vector for torsions and impropers.
    """
    if res is None:
        return {}, {}, {}, {}
    return (
        {_length_key(p): (p.x0, p.K) for p in res.length_parameters},
        {_angle_key(p): (p.x0, p.K) for p in res.angle_parameters},
        {_torsion_key(p): _cos3_vals(p) for p in res.torsion_parameters},
        {_torsion_key(p): _cos3_vals(p) for p in res.improper_parameters},
    )


def _build_cart_dbs_for_residue(rt: Any, cart_db: Any) -> _ParamDBs:
    """Per-residue name-keyed DB dicts.

    Uses current exact-variant parameter selection.
    """
    return _build_cart_lat_dbs(cart_db.residue_params.get(_parameter_name(rt, cart_db)))


def _build_cart_wildcard_dbs(cart_db: Any) -> _ParamDBs:
    """``wildcard`` DB dicts (same shape as :func:`_build_cart_dbs_for_residue`)."""
    return _build_cart_lat_dbs(cart_db.residue_params.get("wildcard"))


def _inter_residue_subgraphs_for_pair(
    rt_A,
    rt_B,
    conn_A,
    conn_B,
    cart_db,
    wild_dbs,
    res_dbs_A,
    res_dbs_B,
):
    """Match native connection-path traversal and explicit '+' atom ownership."""
    output = ([], [], [], [])
    # Native scoring visits each directed connection. Parameter keys retain
    # their authored orientation; a '+' marks the partner residue's atoms.
    for reverse in (False, True):
        first, second = (rt_B, rt_A) if reverse else (rt_A, rt_B)
        ca, cb = (conn_B, conn_A) if reverse else (conn_A, conn_B)
        databases = res_dbs_B if reverse else res_dbs_A
        for category, ia, ib in _CON_PATH_INDICES:
            path_a = _path_atoms(first.atom_paths_from_conn[ca, ia])[::-1]
            path_b = _path_atoms(second.atom_paths_from_conn[cb, ib])
            if not path_a or not path_b:
                continue
            names = tuple(first.atoms[i].name for i in path_a)
            names += tuple("+" + second.atoms[i].name for i in path_b)
            params = databases[category].get(names)
            if params is None:
                params = wild_dbs[category].get(names)
            if params is not None:
                atoms = path_a + path_b
                sides = (reverse,) * len(path_a) + (not reverse,) * len(path_b)
                output[category].append((atoms, sides, params))
        # A connection atom with exactly two other substituents is a
        # planarity centre (peptide N or C): its improper takes those two
        # atoms and the partner's connection atom in every order.
        centre = int(first.atom_paths_from_conn[ca, 0][0])
        partner = int(second.atom_paths_from_conn[cb, 0][0])
        neighbours = [
            int(first.atom_paths_from_conn[ca, p][1])
            for p in (1, 2, 3)
            if first.atom_paths_from_conn[ca, p][1] != -1
        ]
        if centre < 0 or partner < 0 or len(neighbours) != 2:
            continue
        items = ((neighbours[0], False), (neighbours[1], False), (partner, True))
        for order in itertools.permutations(items):
            slots = (order[0], order[1], (centre, False), order[2])
            names = tuple(
                ("+" if across else "") + (second if across else first).atoms[a].name
                for a, across in slots
            )
            params = databases[3].get(names)
            if params is None:
                params = wild_dbs[3].get(names)
            if params is not None:
                atoms = tuple(atom for atom, _ in slots)
                sides = tuple(across != reverse for _, across in slots)
                output[3].append((atoms, sides, params))
    return output


# Padded table construction                                                    #


def _pad_atoms(
    per_type: Sequence[Sequence[Sequence[int]]], k: int
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Turn ``per_type[bt][entry][k]`` into a padded [n_types, max, k] array.

    Returns the padded array, a boolean real-vs-pad mask, and the pad
    width. Empty slots get ``-1`` so accidental gathers fail loudly.
    """
    n_types = len(per_type)
    max_len = max((len(x) for x in per_type), default=0)
    max_len = max(max_len, 1)  # avoid zero-width arrays in jax
    atoms = np.full((n_types, max_len, k), -1, dtype=np.int32)
    mask = np.zeros((n_types, max_len), dtype=bool)
    for i, entries in enumerate(per_type):
        for j, atom_tuple in enumerate(entries):
            atoms[i, j, :] = atom_tuple
            mask[i, j] = True
    return atoms, mask, max_len


def _pad_params(per_type: Sequence[Sequence[Sequence[float]]], k: int) -> np.ndarray:
    """Pad per-block-type param lists to ``[n_types, max, k]`` float32."""
    n_types = len(per_type)
    max_len = max((len(x) for x in per_type), default=0)
    max_len = max(max_len, 1)
    params = np.zeros((n_types, max_len, k), dtype=np.float32)
    for i, entries in enumerate(per_type):
        for j, p in enumerate(entries):
            params[i, j, :] = p
    return params


# Energy term                                                                  #


class CartBondedEnergyTerm(EnergyTerm):
    """Cartesian-bonded harmonic + periodic penalties (pure JAX).

    Provides five score types:
    :attr:`ScoreType.cart_lengths`, :attr:`ScoreType.cart_angles`,
    :attr:`ScoreType.cart_torsions`, :attr:`ScoreType.cart_impropers`,
    :attr:`ScoreType.cart_hxltorsions`.

    Parameters
    ----------
    param_db
        A :class:`tmol_jax.database.ParameterDatabase`. Must expose
        ``scoring.cartbonded`` as a
        :class:`~tmol_jax.database.scoring.cartbonded.CartBondedDatabase`.
    """

    def __init__(self, param_db: Any):
        super().__init__(param_db=param_db)
        self.cart_database = param_db.scoring.cartbonded
        self.improper_roots = _improper_root_atom_names(self.cart_database)

        # One retained palette/parameter owner bounds memory and prevents id reuse.
        self._bt_cache = None

    # Metadata                                                         #

    def score_types_provided(self) -> List[ScoreType]:
        return [
            ScoreType.cart_lengths,
            ScoreType.cart_angles,
            ScoreType.cart_torsions,
            ScoreType.cart_impropers,
            ScoreType.cart_hxltorsions,
        ]

    # Setup                                                            #

    def _build_pbt_tables(self, pbt: Any) -> Dict[str, Any]:
        """Reuse tables only for the same live topology and parameter owners."""
        owners = (pbt.active_block_types, self.cart_database)
        if self._bt_cache is not None:
            cached_owners, tables = self._bt_cache
            if all(old is new for old, new in zip(cached_owners, owners)):
                return tables

        # (output prefix, atoms_arity, params_arity).
        _CATEGORIES = (
            ("length", 2, 2),
            ("angle", 3, 2),
            ("torsion", 4, 6),
            ("improper", 4, 6),
            ("hxltorsion", 4, 6),
        )
        per_type: Dict[str, Tuple[List, List]] = {
            name: ([], []) for name, _, _ in _CATEGORIES
        }
        for rt in pbt.active_block_types:
            sgs = _subgraphs_for_block_type(rt, self.cart_database, self.improper_roots)
            for (name, _, _), sg in zip(_CATEGORIES, sgs):
                atoms_l, params_l = per_type[name]
                atoms_l.append([a for a, _p in sg])
                params_l.append([p for _a, p in sg])

        intra_tables: Dict[str, np.ndarray] = {}
        for name, k_atoms, k_params in _CATEGORIES:
            atoms_l, params_l = per_type[name]
            a, m, _ = _pad_atoms(atoms_l, k_atoms)
            intra_tables[f"{name}_atoms"] = a
            intra_tables[f"{name}_params"] = _pad_params(params_l, k_params)
            intra_tables[f"{name}_mask"] = m

        # We enumerate every ordered (block_type_A, conn_A, block_type_B,
        # conn_B) pair in the palette. For each such pair we build a
        # list of (local_atoms, is_on_B, params) triples per category
        # (length/angle/torsion). Pairs with no matches still have an
        # empty list so the indexing at runtime is stable.
        wild_dbs = _build_cart_wildcard_dbs(self.cart_database)
        res_dbs = [
            _build_cart_dbs_for_residue(rt, self.cart_database)
            for rt in pbt.active_block_types
        ]

        # Per-pair (A, conn_A, B, conn_B) -> list of entries.
        # Each entry in ``length_*`` is (local_A_atom, local_B_atom, params).
        # Each entry in ``angle_*``  is (tuple of 3 local atoms, is_on_B mask, params).
        # Each entry in ``torsion_*`` is (tuple of 4 local atoms, is_on_B mask, params).
        inter_lengths_by_pair: Dict[
            Tuple[int, int, int, int],
            List[Tuple[Tuple[int, ...], Tuple[bool, ...], Tuple[float, ...]]],
        ] = {}
        inter_angles_by_pair: Dict[
            Tuple[int, int, int, int],
            List[Tuple[Tuple[int, ...], Tuple[bool, ...], Tuple[float, ...]]],
        ] = {}
        inter_torsions_by_pair: Dict[
            Tuple[int, int, int, int],
            List[Tuple[Tuple[int, ...], Tuple[bool, ...], Tuple[float, ...]]],
        ] = {}
        inter_impropers_by_pair: Dict[
            Tuple[int, int, int, int],
            List[Tuple[Tuple[int, ...], Tuple[bool, ...], Tuple[float, ...]]],
        ] = {}

        for bt_A, rt_A in enumerate(pbt.active_block_types):
            n_conn_A = len(rt_A.connections)
            for bt_B, rt_B in enumerate(pbt.active_block_types):
                n_conn_B = len(rt_B.connections)
                for conn_A in range(n_conn_A):
                    for conn_B in range(n_conn_B):
                        lengths, angles, torsions, impropers = (
                            _inter_residue_subgraphs_for_pair(
                                rt_A,
                                rt_B,
                                conn_A,
                                conn_B,
                                self.cart_database,
                                wild_dbs,
                                res_dbs[bt_A],
                                res_dbs[bt_B],
                            )
                        )
                        pair_key = (bt_A, conn_A, bt_B, conn_B)
                        if lengths:
                            inter_lengths_by_pair[pair_key] = lengths
                        if angles:
                            inter_angles_by_pair[pair_key] = angles
                        if torsions:
                            inter_torsions_by_pair[pair_key] = torsions
                        if impropers:
                            inter_impropers_by_pair[pair_key] = impropers

        tables = {
            **intra_tables,
            "inter_lengths_by_pair": inter_lengths_by_pair,
            "inter_angles_by_pair": inter_angles_by_pair,
            "inter_torsions_by_pair": inter_torsions_by_pair,
            "inter_impropers_by_pair": inter_impropers_by_pair,
        }
        self._bt_cache = (owners, tables)
        return tables

    def setup_for_pose_stack(
        self, pose_stack: Any, pbt: Any = None
    ) -> CartBondedTermState:
        """Map valid bonded subgraphs into each pose, excluding padding."""
        if pbt is None:
            pbt = pose_stack.packed_block_types
        tables = self._build_pbt_tables(pbt)

        block_type_ind = np.asarray(pose_stack.block_type_ind)  # [P, B]
        block_coord_offset = np.asarray(pose_stack.block_coord_offset)  # [P, B]

        P, B = block_type_ind.shape

        atom_present = np.asarray(pose_stack.atom_is_real)

        def _gather(
            atoms_bt: np.ndarray,  # [n_types, k_per_type, k]
            params_bt: np.ndarray,  # [n_types, k_per_type, kp]
            mask_bt: np.ndarray,  # [n_types, k_per_type]
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            # Take the rows of each (pose, block) -- safe-indexed by bt.
            safe_bt = np.where(block_type_ind >= 0, block_type_ind, 0)  # [P, B]
            real_block = block_type_ind >= 0  # [P, B]

            atoms_pb = atoms_bt[safe_bt]  # [P, B, k_per_type, k]
            params_pb = params_bt[safe_bt]  # [P, B, k_per_type, kp]
            mask_pb = mask_bt[safe_bt]  # [P, B, k_per_type]
            mask_pb = mask_pb & real_block[:, :, None]

            # Shift local atom indices to global atom indices within the
            # pose's coord slab. Padding slots keep -1.
            offsets = block_coord_offset[:, :, None, None]  # [P, B, 1, 1]
            valid = (atoms_pb >= 0) & mask_pb[..., None]
            atoms_global = np.where(valid, atoms_pb + offsets, -1).astype(np.int32)

            # Invalidate subgraphs that reference a missing atom. Use
            # ``atoms_global`` clamped to 0 for the per-pose gather into
            # ``atom_present`` and AND the boolean mask across the k axis.
            # Shape: [P, B, K, k]; reduce along k.
            safe_global = np.where(atoms_global >= 0, atoms_global, 0)
            # gather atom_present[p, safe_global[p, b, K, k]]
            pose_idx = np.arange(P)[:, None, None, None]  # [P, 1, 1, 1]
            atoms_present_per = atom_present[pose_idx, safe_global]  # [P, B, K, k]
            # For real atom slots, require present; for padding slots
            # (atoms_global == -1) they already won't produce spurious
            # contributions because the mask is ANDed below.
            all_atoms_present = np.all(valid & atoms_present_per, axis=-1)  # [P, B, K]
            mask_pb = mask_pb & all_atoms_present

            return atoms_global, params_pb.astype(np.float32), mask_pb.astype(bool)

        state_fields: Dict[str, jnp.ndarray] = {}
        for cat in ("length", "angle", "torsion", "improper", "hxltorsion"):
            atoms, params, mask = _gather(
                tables[f"{cat}_atoms"],
                tables[f"{cat}_params"],
                tables[f"{cat}_mask"],
            )
            state_fields[f"{cat}_atoms"] = jnp.asarray(atoms)
            state_fields[f"{cat}_params"] = jnp.asarray(params)
            state_fields[f"{cat}_mask"] = jnp.asarray(mask)

        irc = np.asarray(pose_stack.inter_residue_connections)  # [P, B, C, 2]
        inter_lengths_by_pair = tables["inter_lengths_by_pair"]
        inter_angles_by_pair = tables["inter_angles_by_pair"]
        inter_torsions_by_pair = tables["inter_torsions_by_pair"]
        inter_impropers_by_pair = tables["inter_impropers_by_pair"]

        # Walk each pose's connections, resolve partner blocks, emit
        # one packed list per (category) per pose. Only emit entries
        # where ``block_i < block_j`` to avoid double-counting.
        # (category-key, by-pair-dict, atoms-arity, params-arity).
        _INTER_CATS = (
            ("L", inter_lengths_by_pair, 2, 2),
            ("A", inter_angles_by_pair, 3, 2),
            ("T", inter_torsions_by_pair, 4, 6),
            ("I", inter_impropers_by_pair, 4, 6),
        )
        per_pose: Dict[str, Tuple[List, List]] = {
            cat: ([[] for _ in range(P)], [[] for _ in range(P)])
            for cat, _, _, _ in _INTER_CATS
        }

        n_conn = irc.shape[2]
        for pose in range(P):
            for block_i in range(B):
                bt_i = int(block_type_ind[pose, block_i])
                if bt_i < 0:
                    continue
                for conn_i in range(n_conn):
                    block_j = int(irc[pose, block_i, conn_i, 0])
                    conn_j = int(irc[pose, block_i, conn_i, 1])
                    if block_j < 0 or block_i >= block_j:
                        # block_i >= block_j ensures each edge is emitted once.
                        continue
                    bt_j = int(block_type_ind[pose, block_j])
                    if bt_j < 0:
                        continue

                    off_i = int(block_coord_offset[pose, block_i])
                    off_j = int(block_coord_offset[pose, block_j])
                    pair_key = (bt_i, conn_i, bt_j, conn_j)

                    # Skip pair keys with no subgraphs at all - the typical
                    # 1UBQ peptide bond hits one non-empty key and ~N-1
                    # empty ones, and the dict-get miss dominates loop cost.
                    hits = [d.get(pair_key) for _, d, _, _ in _INTER_CATS]
                    if not any(hits):
                        continue

                    for (cat, _, _, _), hit in zip(_INTER_CATS, hits):
                        if not hit:
                            continue
                        atoms_list, params_list = per_pose[cat]
                        for atoms, is_on_B, params in hit:
                            g = tuple(
                                (a + off_j) if b else (a + off_i)
                                for a, b in zip(atoms, is_on_B)
                            )
                            atoms_list[pose].append(g)
                            params_list[pose].append(params)

        # Inter-residue subgraphs also require real topology atoms.
        def _mask_missing(atoms: np.ndarray, mask: np.ndarray) -> np.ndarray:
            if atoms.size == 0:
                return mask
            safe = np.where(atoms >= 0, atoms, 0)
            pose_idx = np.arange(P)[:, None, None]
            present = atom_present[pose_idx, safe]  # [P, E, k]
            all_present = np.all(present, axis=-1)  # [P, E]
            return mask & all_present

        # (cat-key, state-field-prefix, atoms-arity, params-arity).
        _INTER_OUT = (
            ("L", "inter_length", 2, 2),
            ("A", "inter_angle", 3, 2),
            ("T", "inter_torsion", 4, 6),
            ("I", "inter_improper", 4, 6),
        )
        for cat, prefix, k_atoms, k_params in _INTER_OUT:
            atoms, mask, _ = _pad_atoms(per_pose[cat][0], k_atoms)
            params = _pad_params(per_pose[cat][1], k_params)
            mask = _mask_missing(atoms, mask)
            state_fields[f"{prefix}_atoms"] = jnp.asarray(atoms)
            state_fields[f"{prefix}_params"] = jnp.asarray(params)
            state_fields[f"{prefix}_mask"] = jnp.asarray(mask)

        return CartBondedTermState(**state_fields)

    # Evaluation                                                       #

    def __call__(
        self,
        pose_stack: Any,
        term_state: CartBondedTermState,
        coords_override: Optional[jax.Array] = None,
    ) -> Dict[ScoreType, jax.Array]:
        coords = pose_stack.coords if coords_override is None else coords_override
        # coords: [P, A, 3]

        def _gather_xyz(global_atoms: jax.Array) -> jax.Array:
            """Gather coords by (pose, global_atom) indices.

            ``global_atoms`` has any leading shape ``[P, ..., k]`` with
            ``-1`` sentinels in padding slots. We clamp the sentinel to
            0 for the gather (results are masked out downstream) and use
            ``jnp.take_along_axis`` along the per-pose atom axis.
            Returns ``[*orig_shape, 3]``.
            """
            safe = jnp.where(global_atoms >= 0, global_atoms, 0)
            orig_shape = safe.shape
            flat = safe.reshape(orig_shape[0], -1)  # [P, M]
            idx = flat[:, :, None].repeat(3, axis=-1)  # [P, M, 3]
            gathered = jnp.take_along_axis(coords, idx, axis=1)  # [P, M, 3]
            return gathered.reshape(*orig_shape, 3)

        def score_subgraphs(prefix, geometry, potential):
            atoms = getattr(term_state, f"{prefix}_atoms")
            params = getattr(term_state, f"{prefix}_params")
            mask = getattr(term_state, f"{prefix}_mask")
            xyz = _gather_xyz(atoms)
            measure = geometry(*(xyz[..., i, :] for i in range(atoms.shape[-1])))
            energy = potential(
                measure, *(params[..., i] for i in range(params.shape[-1]))
            )
            return jnp.where(mask, energy, 0.0).sum(axis=tuple(range(1, mask.ndim)))

        # Intra- and inter-residue graphs use identical potentials; only
        # their batch axes and score channel differ. Empty graphs sum to zero.
        categories = (
            ("length", ScoreType.cart_lengths, distance, bond_length_E),
            ("angle", ScoreType.cart_angles, pt_interior_angle, bond_angle_E),
            ("torsion", ScoreType.cart_torsions, dihedral_angle, hxltorsion_E),
            ("improper", ScoreType.cart_impropers, dihedral_angle, hxltorsion_E),
            ("hxltorsion", ScoreType.cart_hxltorsions, dihedral_angle, hxltorsion_E),
        )
        result = {
            channel: score_subgraphs(prefix, geometry, potential)
            for prefix, channel, geometry, potential in categories
        }
        for prefix, channel, geometry, potential in categories[:-1]:
            result[channel] = result[channel] + score_subgraphs(
                f"inter_{prefix}", geometry, potential
            )
        return result


__all__ = ["CartBondedEnergyTerm", "CartBondedTermState"]
