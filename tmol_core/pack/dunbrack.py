"""Common chemical preparation for library and explicit-chi sampling."""

import attr
import numpy
from types import MappingProxyType
from typing import Generic, Mapping, TypeVar
from tmol_core.geometry import numpy_coord_dihedrals
from tmol_core.pack.chi import (
    apply_chi_sample_budget,
    chi_depths,
    sidechain_roots_for_chis,
)

Array = TypeVar("Array")


@attr.s(auto_attribs=True, slots=True, frozen=True)
class DunSamplerRTCache:
    """Data to store in RefinedResidueType that will be reused
    repeatedly in the creation of the DunSamplerPBTCache
    """

    bbdihe_uaids: numpy.ndarray
    chi_defining_atom: numpy.ndarray
    non_dunbrack_sample_counts: numpy.ndarray
    non_dunbrack_samples: numpy.ndarray
    rottable_set_for_bt: int
    frozen_chi: tuple = ()


@attr.s(auto_attribs=True, slots=True, frozen=True)
class DunSamplerPBTCache(Generic[Array]):
    """Data needed for chi sampling and for reporting how
    the chi are to be assigned to atoms
    """

    bbdihe_uaids: Array
    chi_defining_atom: Array
    non_dunbrack_sample_counts: Array
    non_dunbrack_samples: Array
    defines_rotamers_for_bts: Array
    rottable_set_for_bt: Array
    frozen_chi: tuple = ()

    @property
    def max_n_chi(self):
        return self.chi_defining_atom.shape[1]


@attr.s(auto_attribs=True, slots=True, frozen=True)
class DunbrackSamplingMetadata:
    """Immutable host library identities and enumeration sizes."""

    library_indices: Mapping[str, int] = attr.ib(
        converter=lambda values: MappingProxyType(dict(values))
    )
    nchi_for_table: tuple = attr.ib(converter=tuple)
    library_sizes: tuple = attr.ib(converter=tuple)

    @classmethod
    def from_database(cls, database):
        """Snapshot library identities and sizes without fitting coefficients.

        Accept the source Dunbrack library used by ``prepare_dunbrack_tables``.
        The same validation and shape metadata feed full table preparation.
        """
        from tmol_core.scoring.dunbrack import (
            prepare_dunbrack_lookup,
            _libraries,
            _library_nchi,
            _library_rotamer_counts,
        )

        lookup = prepare_dunbrack_lookup(database)
        libraries = _libraries(database)
        return cls(
            lookup[0], _library_nchi(libraries), _library_rotamer_counts(libraries)
        )

    def library_index(self, name):
        return int(self.library_indices.get(name, -1))

    def library_for_type(self, rt) -> int:
        """Index of the rotamer library this residue type reads, or -1.

        A residue with a library of its own uses it, under its own name: a
        d-amino acid has a mirrored library, and reading the l one would give
        it unmirrored rotamers. A noncanonical has none, and borrows whichever
        its ``dunbrack_reference`` names, if it names one.
        """
        index = self.library_index(rt.base_name)
        if index < 0 and rt.dunbrack_reference:
            index = self.library_index(rt.dunbrack_reference)
        return int(index)

    def defines_rotamers(self, rt):
        """Whether a rotamer library reaches this residue type.

        A library resolving answers it for most residues: glycine and alanine
        have nothing to rotate and no library, and a noncanonical has one only
        where a reference names it. A residue that samples heavy chi of its
        own is built from those alone.
        """
        if not rt.properties.polymer.is_polymer:
            return False
        if rt.properties.polymer.polymer_type != "amino_acid":
            return False
        if self.library_for_type(rt) >= 0:
            return True
        # no library, but chi of its own to turn. A proton chi alone is not
        #    enough: optH samples those, and rotamers varying only hydrogens
        #    would duplicate its work.
        return any(not cs.is_proton for cs in rt.chi_samples)

    def sidechain_roots(self, rt):
        """The atom chi1 rotates, which is where the sidechain starts.

        Read off the residue rather than named, so a noncanonical whose
        sidechain does not begin at an atom called CB still transfers its
        rotamers correctly. Empty for a residue with no chi to rotate.
        """
        assert self.defines_rotamers(rt)
        # Explicit chi can start after an unsampled axis. Copy upstream input
        # geometry instead of treating the entire chi1 branch as rebuilt.
        if self.library_for_type(rt) < 0:
            return sidechain_roots_for_chis(
                rt, (cs.chi_dihedral for cs in rt.chi_samples)
            )
        return tuple(t.c.atom for t in rt.torsions if t.name == "chi1")


def prepare_dunbrack_sampler_residue(  # noqa: C901
    restype, metadata, *, expanded_limit, limit, prepare_kinematics
):
    """Prepare angular samples and ownership using explicit host library metadata.

    The callback annotates the residue with its shared kinematic preparation;
    it is called only when sampled chi require the mature depth-based budget.
    """
    # ok; lets ask the appropriate library what chi it defines samples for
    # and also incorporate additional chi that the residue type itself
    # defines its own samples for, as long as this is actually a residue type
    # that the dunbrack library handles.
    # Strip away any patches and use the "base name" of the residue type to make
    # the "which library should I read from?" decision

    # fd  noncanonicals may borrow a canonical's rotamers (sampling only);
    #     a d-amino acid has a mirrored library of its own, so its own name
    #     is tried before any reference it carries
    dun_lib_ind = metadata.library_index(restype.base_name)
    borrowed = False
    if dun_lib_ind < 0 and restype.dunbrack_reference:
        dun_lib_ind = metadata.library_index(restype.dunbrack_reference)
        borrowed = dun_lib_ind >= 0

    # n-bb-dihedrals = 2; n-atoms in a dihedral = 4; n-entries in a uaid  = 3
    uaids = numpy.full((2, 4, 3), -1, dtype=numpy.int32)
    # fd  a borrowed library was fit against an alpha backbone; on any other
    #     backbone its phi/psi are not this residue's, so they are left
    #     unresolved and the library is read at neutral phi/psi
    if not (borrowed and restype.properties.polymer.backbone_type != "alpha_aa"):
        if "phi" in restype.torsion_to_uaids:
            uaids[0] = numpy.array(restype.torsion_to_uaids["phi"], dtype=numpy.int32)
        if "psi" in restype.torsion_to_uaids:
            uaids[1] = numpy.array(restype.torsion_to_uaids["psi"], dtype=numpy.int32)

    # a residue with no library still builds rotamers if it samples chi of
    #    its own; the library supplies the leading n_chi and its own
    #    samples supply the rest, so with no library it supplies them all
    n_chi = metadata.nchi_for_table[dun_lib_ind] if dun_lib_ind >= 0 else 0
    sampled = restype.chi_samples
    if sampled and metadata.defines_rotamers(restype):
        prepare_kinematics(restype)
        depths = chi_depths(
            restype.residue_kinforest_data,
            [restype.torsion_to_uaids[cs.chi_dihedral][2][0] for cs in sampled],
        )
        # Bound supplemental products before native enumeration. The full
        # library size is a conservative bound on phi/psi-specific coverage;
        # retain all selected library states even when a budget is smaller.
        library_size = (
            max(1, metadata.library_sizes[dun_lib_ind]) if dun_lib_ind >= 0 else 1
        )
        sampled = apply_chi_sample_budget(
            sampled,
            depths,
            max(1, expanded_limit // library_size),
            max(1, limit // library_size),
            n_library_chi=n_chi,
        )
    selected_names = {cs.chi_dihedral for cs in sampled}
    frozen_chi = tuple(
        cs.chi_dihedral
        for cs in restype.chi_samples
        if int(cs.chi_dihedral[3:]) > n_chi and cs.chi_dihedral not in selected_names
    )
    if frozen_chi:
        ideal = restype.ideal_coords[restype.at_to_icoor_ind]
        for cs in restype.chi_samples:
            if cs.chi_dihedral not in frozen_chi:
                continue
            atoms = [u[0] for u in restype.torsion_to_uaids[cs.chi_dihedral]]
            angle = (
                float(numpy_coord_dihedrals(*ideal[atoms]))
                if min(atoms) >= 0
                else numpy.nan
            )
            sampled += (
                attr.evolve(cs, samples=(numpy.degrees(angle),), expansions=()),
            )
    if dun_lib_ind >= 0 or restype.chi_samples:
        n_chi_total = n_chi
        for rt_chi in restype.chi_samples:
            chi_name = rt_chi.chi_dihedral
            assert chi_name[:3] == "chi"
            chi_ind = int(chi_name[3:])
            if chi_ind > n_chi_total:
                n_chi_total = chi_ind

        # Assumption: the Dunbrack library assigns chi for a contiguous
        # block of chi dihedrals starting with chi1.
        chi_names = ["chi%d" % (i + 1) for i in range(n_chi)]
        for chi_name in chi_names:
            assert chi_name in restype.torsion_to_uaids

        chi_defining_atom = numpy.full(n_chi_total, -1, dtype=numpy.int32)
        # The third atom (atom 2) in the chi defines the dihedral; this
        # atoms needs to be a member of the residue type, and not unresolved
        chi_defining_atom[:n_chi] = [
            restype.torsion_to_uaids[chi_name][2][0] for chi_name in chi_names
        ]

        if sampled:
            chi_inds = numpy.array(
                [int(samp.chi_dihedral[3:]) - 1 for samp in sampled],
                dtype=int,
            )
            sort_inds = numpy.argsort(chi_inds)
            chi_defining_atom[chi_inds[sort_inds]] = [
                restype.torsion_to_uaids[sampled[ind].chi_dihedral][2][0]
                for ind in sort_inds
            ]

        non_dunbrack_sample_counts = numpy.zeros((n_chi_total, 2), dtype=numpy.int32)
        max_chi_samples = 0
        for rt_chi in sampled:
            chi_name = rt_chi.chi_dihedral
            assert chi_name[:3] == "chi"
            chi_ind = int(chi_name[3:]) - 1
            n_expansions = 1 + 2 * len(rt_chi.expansions)
            n_samples = len(rt_chi.samples)
            n_expanded_samples = n_samples * n_expansions
            max_chi_samples = max(max_chi_samples, n_expanded_samples)
            non_dunbrack_sample_counts[chi_ind, 0] = n_samples
            non_dunbrack_sample_counts[chi_ind, 1] = n_expanded_samples

        non_dunbrack_samples = numpy.zeros(
            (n_chi_total, 2, max_chi_samples), dtype=numpy.float32
        )
        deg_to_rad = numpy.pi / 180
        for rt_chi in sampled:
            chi_name = rt_chi.chi_dihedral
            chi_ind = int(chi_name[3:]) - 1
            n_expansions = 1 + 2 * len(rt_chi.expansions)
            n_samples = len(rt_chi.samples)
            non_dunbrack_samples[chi_ind, 0, :n_samples] = deg_to_rad * numpy.array(
                rt_chi.samples
            )
            for i in range(n_samples):
                for j in range(n_expansions):
                    if j == 0:
                        non_dunbrack_samples[chi_ind, 1, n_expansions * i + j] = (
                            deg_to_rad * rt_chi.samples[i]
                        )
                    else:
                        expansion = (j - 1) // 2
                        factor = -1 if (j - 1) % 2 == 0 else 1
                        non_dunbrack_samples[chi_ind, 1, n_expansions * i + j] = (
                            deg_to_rad
                            * (
                                rt_chi.samples[i]
                                + factor * rt_chi.expansions[expansion]
                            )
                        )

    else:
        chi_defining_atom = numpy.zeros((0,), dtype=numpy.int32)
        non_dunbrack_sample_counts = numpy.zeros((0, 2), dtype=numpy.int32)
        non_dunbrack_samples = numpy.zeros((0, 2, 0), dtype=numpy.int32)

    cache = DunSamplerRTCache(
        bbdihe_uaids=uaids,
        chi_defining_atom=chi_defining_atom,
        non_dunbrack_sample_counts=non_dunbrack_sample_counts,
        non_dunbrack_samples=non_dunbrack_samples,
        rottable_set_for_bt=dun_lib_ind,
        frozen_chi=frozen_chi,
    )
    return cache


def pack_dunbrack_sampler_metadata(block_types, annotations, builds):
    """Pad residue sampling arrays once, before frontend device conversion."""
    n_types = len(block_types)
    if len(annotations) != n_types or len(builds) != n_types:
        raise ValueError("Sampling annotations must match the residue type axis")
    max_n_chi = max((a.chi_defining_atom.shape[0] for a in annotations), default=0)
    max_samples = max((a.non_dunbrack_samples.shape[2] for a in annotations), default=0)
    uaids = (
        numpy.stack([a.bbdihe_uaids for a in annotations])
        if n_types
        else numpy.empty((0, 2, 4, 3), dtype=numpy.int32)
    )
    chi_atoms = numpy.full((n_types, max_n_chi), -1, dtype=numpy.int32)
    counts = numpy.full((n_types, max_n_chi, 2), -1, dtype=numpy.int32)
    samples = numpy.full((n_types, max_n_chi, 2, max_samples), -1, dtype=numpy.float32)
    library = numpy.full(n_types, -1, dtype=numpy.int32)
    for i, data in enumerate(annotations):
        chi_atoms[i, : data.chi_defining_atom.shape[0]] = data.chi_defining_atom
        shape = data.non_dunbrack_sample_counts.shape[0]
        counts[i, :shape] = data.non_dunbrack_sample_counts
        library[i] = data.rottable_set_for_bt
        if shape and data.non_dunbrack_samples.shape[2]:
            samples[i, :shape, :, : data.non_dunbrack_samples.shape[2]] = (
                data.non_dunbrack_samples
            )
    return DunSamplerPBTCache(
        uaids,
        chi_atoms,
        counts,
        samples,
        numpy.asarray(builds, dtype=numpy.bool_),
        library,
        tuple(a.frozen_chi for a in annotations),
    )
