"""OptH chemical preparation and sample tables independent of framework arrays."""

import attr
import math
import numpy
from typing import Generic, TypeVar
from tmol_core.chemical import _l_base_name
from tmol_core.pack.chi import apply_chi_sample_budget, chi_depths

Array = TypeVar("Array")
_HIS_FLIP_BASES = frozenset(("HIS", "HIS_D"))


@attr.s(auto_attribs=True, slots=True, frozen=True)
class OptHSamplerRTCache:
    """Per-residue-type annotation for OptHSampler.

    Covers two orthogonal features:
    1. Proton chi sampling (SER/THR/TYR/CYS): samples the terminal (proton)
       chi angle using values from restype definition.
    2. NHQ flip (ASN/GLN/any HIS state): generates the input conformation plus a
       180-degree rotation about the terminal amide/ring chi angle, provided
       it does not move an atom connected to another block.
       HIS additionally generates both protonation states.
    """

    # proton chis (S/T/Y/C)
    has_proton_chi: bool
    n_chi_total: int
    chi_defining_atom: numpy.ndarray
    n_proton_samples: int
    expanded_samples: numpy.ndarray
    n_samples_per_chi: numpy.ndarray

    # N/H/Q flips
    nhq_chi_col: int  # chi index or -1
    nhq_chi_atom: int
    nhq_chi_4atoms: numpy.ndarray
    nhq_downstream_kfo: numpy.ndarray
    is_his: bool


@attr.s(auto_attribs=True, slots=True, frozen=True)
class OptHSamplerPackedBlockTypeCache(Generic[Array]):
    opth_sample_for_bt: Array
    has_proton_chi: Array
    n_chi_total: Array
    chi_defining_atom: Array
    n_proton_samples: Array
    expanded_samples: Array
    n_samples_per_chi: Array

    nhq_chi_col: Array
    nhq_chi_atom: Array
    nhq_chi_4atoms: Array
    nhq_downstream_kfo: Array
    nhq_downstream_count: Array
    is_his: Array

    # These two tensors use 0 or 1 as an index to dim=0 to represent
    # either flip_NHQ=False or flip_NHQ=True
    n_samples_for_bt_by_orig_bt: Array
    n_chi_needed_for_bt: Array


def _build_empty_proton_cache(
    nhq_chi_col, nhq_chi_atom, nhq_chi_4atoms, nhq_downstream_kfo, is_his
):
    return OptHSamplerRTCache(
        has_proton_chi=False,
        n_chi_total=0,
        chi_defining_atom=numpy.zeros(0, dtype=numpy.int32),
        n_proton_samples=0,
        expanded_samples=numpy.zeros((0, 0), dtype=numpy.float32),
        n_samples_per_chi=numpy.zeros(0, dtype=numpy.int32),
        nhq_chi_col=nhq_chi_col,
        nhq_chi_atom=nhq_chi_atom,
        nhq_chi_4atoms=nhq_chi_4atoms,
        nhq_downstream_kfo=nhq_downstream_kfo,
        is_his=is_his,
    )


def _compute_nhq_downstream_kfo(rt, nhq_rto: int) -> numpy.ndarray:
    """Return KFO indices of all atoms downstream of rt's NHQ chi-defining atom.

    Traverses the per-residue-type kinforest parent array starting from the
    children of nhq_chi_atom.  Computed once during _annotate_residue_type and
    cached on the OptHSamplerRTCache; returns an empty array for non-NHQ rts.
    """
    if nhq_rto < 0:
        return numpy.zeros(0, dtype=numpy.int32)
    kfidx = rt.rotamer_kinforest.kinforest_idx  # (n_atoms,) numpy, TO -> KFO
    parents = rt.rotamer_kinforest.parent  # (n_atoms,) numpy, KFO -> parent KFO
    kfo_nhq = int(kfidx[nhq_rto])
    n_at = len(parents)
    downstream = []
    queue = [k for k in range(n_at) if int(parents[k]) == kfo_nhq]
    while queue:
        k = queue.pop()
        downstream.append(k)
        queue.extend(ch for ch in range(n_at) if int(parents[ch]) == k)
    return numpy.array(downstream, dtype=numpy.int32)


def annotate_opth_residue(rt, expanded_limit, limit, prepare_kinforest):
    """Prepare proton/flip chemistry while preserving connected-atom ownership."""
    key = (expanded_limit, limit)
    if getattr(rt, "_opth_sampler_key", None) == key:
        return rt.opth_sampler_cache

    # A D residue flips on the same axis as the L form it mirrors, so read
    # the L base name; DASN and friends match nothing otherwise and emit no
    # rotamers at all. _fixed_aa_chi_sampler does the same.
    base = _l_base_name(rt)

    # NHQ flip annotation
    nhq_chi_col = -1
    nhq_chi_atom = -1
    nhq_chi_4atoms = numpy.zeros(4, dtype=numpy.int32)
    nhq_downstream_kfo = numpy.zeros(0, dtype=numpy.int32)
    is_his = base in _HIS_FLIP_BASES

    flip_chi = {"ASN": "chi2", "GLN": "chi3", "HIS": "chi2"}.get(base.split("_")[0])
    uaids = rt.torsion_to_uaids.get(flip_chi)
    if uaids is not None and all(u[0] >= 0 for u in uaids):
        # An attachment may append more chis. The terminal amide/ring flip
        # retains its own named axis, and cannot move a connection atom
        # independently of the block bonded to it.
        nhq_chi_atom = int(uaids[2][0])
        prepare_kinforest(rt)
        nhq_downstream_kfo = _compute_nhq_downstream_kfo(rt, nhq_chi_atom)
        connection_nodes = {
            int(rt.rotamer_kinforest.kinforest_idx[rt.atom_to_idx[c.atom]])
            for c in rt.connections
        }
        if connection_nodes.intersection(nhq_downstream_kfo):
            nhq_chi_atom = -1
            nhq_downstream_kfo = numpy.zeros(0, dtype=numpy.int32)
            is_his = False
        else:
            nhq_chi_col = int(flip_chi[3:]) - 1
            nhq_chi_4atoms = numpy.array([int(u[0]) for u in uaids], dtype=numpy.int32)
    else:
        is_his = False

    # proton chi annotation. A chi that turns heavy atoms is sampled by the
    #    packer alone, so optH never sees it.
    proton_chi = [cs for cs in rt.chi_samples if cs.is_proton]
    if proton_chi:
        # chi_samples are stored unbudgeted so that what optH enumerates and
        #    what the packer enumerates are each bounded on their own terms;
        #    a hydroxyl-rich sugar is otherwise thousands of rotamers here
        prepare_kinforest(rt)
        depths = chi_depths(
            rt.residue_kinforest_data,
            [rt.torsion_to_uaids[cs.chi_dihedral][2][0] for cs in proton_chi],
        )
        proton_chi = list(
            apply_chi_sample_budget(
                proton_chi,
                depths,
                expanded_limit,
                limit,
            )
        )
    if not proton_chi:
        setattr(
            rt,
            "opth_sampler_cache",
            _build_empty_proton_cache(
                nhq_chi_col,
                nhq_chi_atom,
                nhq_chi_4atoms,
                nhq_downstream_kfo,
                is_his,
            ),
        )
        setattr(rt, "_opth_sampler_key", key)
        return rt.opth_sampler_cache

    deg_to_rad = math.pi / 180

    chi_inds = [int(cs.chi_dihedral[3:]) - 1 for cs in proton_chi]
    n_chi_total = max(chi_inds) + 1

    chi_defining_atom = numpy.full(n_chi_total, -1, dtype=numpy.int32)
    n_samples_per_chi = numpy.zeros(n_chi_total, dtype=numpy.int32)

    max_n_expanded = max(
        len(cs.samples) * (1 + 2 * len(cs.expansions)) for cs in proton_chi
    )
    expanded_samples = numpy.zeros((n_chi_total, max_n_expanded), dtype=numpy.float32)

    for cs in proton_chi:
        ci = int(cs.chi_dihedral[3:]) - 1
        chi_defining_atom[ci] = rt.torsion_to_uaids[cs.chi_dihedral][2][0]

        n_samp = len(cs.samples)
        n_exp_per_samp = 1 + 2 * len(cs.expansions)
        n_samples_per_chi[ci] = n_samp * n_exp_per_samp

        for i in range(n_samp):
            for j in range(n_exp_per_samp):
                if j == 0:
                    expanded_samples[ci, n_exp_per_samp * i] = (
                        deg_to_rad * cs.samples[i]
                    )
                else:
                    exp_idx = (j - 1) // 2
                    factor = -1 if (j - 1) % 2 == 0 else 1
                    expanded_samples[ci, n_exp_per_samp * i + j] = deg_to_rad * (
                        cs.samples[i] + factor * cs.expansions[exp_idx]
                    )

    n_proton_samples_total = 1
    for n in n_samples_per_chi:
        if n > 0:
            n_proton_samples_total *= int(n)
    setattr(
        rt,
        "opth_sampler_cache",
        OptHSamplerRTCache(
            has_proton_chi=True,
            n_chi_total=n_chi_total,
            chi_defining_atom=chi_defining_atom,
            n_proton_samples=n_proton_samples_total,
            expanded_samples=expanded_samples,
            n_samples_per_chi=n_samples_per_chi,
            nhq_chi_col=nhq_chi_col,
            nhq_chi_atom=nhq_chi_atom,
            nhq_chi_4atoms=nhq_chi_4atoms,
            nhq_downstream_kfo=nhq_downstream_kfo,
            is_his=is_his,
        ),
    )

    setattr(rt, "_opth_sampler_key", key)

    return rt.opth_sampler_cache


def opth_builds_residue(cache, flip_NHQ):
    return cache.has_proton_chi or (flip_NHQ and cache.nhq_chi_col >= 0)


def pack_opth_metadata(caches, max_n_atoms, flip_NHQ):
    """Pack all chemical types into the canonical proton and flip lookup axes."""
    n_types = len(caches)
    max_n_chi = max(
        (max(c.n_chi_total, c.nhq_chi_col + 1 if flip_NHQ else 0) for c in caches),
        default=0,
    )
    max_n_expanded = max((c.expanded_samples.shape[1] for c in caches), default=0)

    # filled on the host and uploaded once: a write per element into a
    #    device tensor costs a kernel launch or a copy each
    opth_sample_for_bt = numpy.array(
        [opth_builds_residue(c, flip_NHQ) for c in caches], dtype=bool
    )
    has_proton_chi = numpy.zeros(n_types, dtype=bool)
    n_chi_total = numpy.zeros(n_types, dtype=numpy.int32)
    chi_defining_atom = numpy.full((n_types, max_n_chi), -1, dtype=numpy.int32)
    n_proton_samples = numpy.zeros(n_types, dtype=numpy.int32)
    expanded_samples = numpy.zeros(
        (n_types, max_n_chi, max_n_expanded), dtype=numpy.float32
    )
    n_samples_per_chi = numpy.zeros((n_types, max_n_chi), dtype=numpy.int32)
    nhq_chi_col = numpy.zeros(n_types, dtype=numpy.int32)
    nhq_chi_atom = numpy.zeros(n_types, dtype=numpy.int32)
    nhq_chi_4atoms = numpy.zeros((n_types, 4), dtype=numpy.int32)
    nhq_downstream_kfo = numpy.zeros((n_types, max_n_atoms), dtype=numpy.int32)
    nhq_downstream_count = numpy.zeros(n_types, dtype=numpy.int32)
    is_his = numpy.zeros(n_types, dtype=bool)
    n_samples_for_bt_by_orig_bt = numpy.zeros((2, n_types, n_types), dtype=numpy.int32)
    # the minimum chi tensor width needed for a GBT with non-zero rots
    n_chi_needed_for_bt = numpy.ones((2, n_types), dtype=numpy.int32)
    his_nhq = numpy.array([c.nhq_chi_col >= 0 and c.is_his for c in caches], dtype=bool)

    for i, c in enumerate(caches):
        if c.has_proton_chi:
            # use n_proton_samples wether or not we're in flip_NHQ mode
            n_samples_for_bt_by_orig_bt[:, i, i] = c.n_proton_samples
            n_chi_needed_for_bt[:, i] = c.n_chi_total
        elif c.nhq_chi_col >= 0:
            n_chi_needed_for_bt[1, i] = c.nhq_chi_col + 1
            if c.is_his:
                # HIS/HIS_D: 2 rotamers for EVERY HIS/HIS_D considered block type
                n_samples_for_bt_by_orig_bt[1, i, i] = 2
                n_samples_for_bt_by_orig_bt[1, i, his_nhq] = 2
            else:
                # ASN/GLN: 2 rotamers only for the original block type
                n_samples_for_bt_by_orig_bt[1, i, i] = 2
        has_proton_chi[i] = c.has_proton_chi
        n_chi_total[i] = c.n_chi_total
        chi_defining_atom[i, : c.n_chi_total] = c.chi_defining_atom
        n_proton_samples[i] = c.n_proton_samples
        expanded_samples[i, : c.n_chi_total, : c.expanded_samples.shape[1]] = (
            c.expanded_samples
        )
        n_samples_per_chi[i, : c.n_chi_total] = c.n_samples_per_chi
        nhq_chi_col[i] = c.nhq_chi_col
        nhq_chi_atom[i] = c.nhq_chi_atom
        nhq_chi_4atoms[i, :] = c.nhq_chi_4atoms
        nhq_downstream_kfo[i, : len(c.nhq_downstream_kfo)] = c.nhq_downstream_kfo
        nhq_downstream_count[i] = len(c.nhq_downstream_kfo)
        is_his[i] = c.is_his

    return OptHSamplerPackedBlockTypeCache(
        opth_sample_for_bt=opth_sample_for_bt,
        has_proton_chi=has_proton_chi,
        n_chi_total=n_chi_total,
        chi_defining_atom=chi_defining_atom,
        n_proton_samples=n_proton_samples,
        expanded_samples=expanded_samples,
        n_samples_per_chi=n_samples_per_chi,
        nhq_chi_col=nhq_chi_col,
        nhq_chi_atom=nhq_chi_atom,
        nhq_chi_4atoms=nhq_chi_4atoms,
        nhq_downstream_kfo=nhq_downstream_kfo,
        nhq_downstream_count=nhq_downstream_count,
        is_his=is_his,
        n_samples_for_bt_by_orig_bt=n_samples_for_bt_by_orig_bt,
        n_chi_needed_for_bt=n_chi_needed_for_bt,
    )
