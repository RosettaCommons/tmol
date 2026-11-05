import numpy
import torch
import pandas

from tmol.types._torch import Tensor
from tmol.types._functional import _validate_args
from tmol.database import ParameterDatabase
from tmol.pose._packed_block_types import PackedBlockTypes
from tmol.chemical import ResidueTypeSet
from tmol.utility import resolve_device
from typing import List, Optional, Union
from tmol_core.io.canonical_ordering import (  # noqa: F401 (legacy exports)
    CanonicalOrdering as CoreCanonicalOrdering,
    PolymerConnectionIndices,  # legacy import compatibility
    CysSpecialCaseIndices,
    HisSpecialCaseIndices,
    ordered_set,
)
from ._canonical_form import CanonicalForm
from ._pdb_parsing import _parse_pdb
from ._alternates import _records_with_one_alternate
import toolz.functoolz


class CanonicalOrdering(CoreCanonicalOrdering):
    """Shared chemical ordering with Torch device-array conversion."""

    def create_src_2_tmol_mappings(
        self, src_aa_name3s, src_atom_names_for_name3s, device
    ):
        arrays = super().create_src_2_tmol_mappings(
            src_aa_name3s, src_atom_names_for_name3s
        )
        return tuple(torch.as_tensor(array, device=device) for array in arrays)


@_validate_args
@toolz.functoolz.memoize
def _default_canonical_ordering() -> CanonicalOrdering:
    """Create a CanonicalOrdering object from the default set of residue types"""

    chemdb = ParameterDatabase.get_default().chemical
    return CanonicalOrdering.from_chemdb(chemdb)


@_validate_args
def _default_packed_block_types(device: torch.device) -> PackedBlockTypes:
    """Create a PackedBlockTypes object from the default set of residue types"""
    # resolved before the memo lookup so 'cuda' and 'cuda:0' share one entry
    return _memoized_packed_block_types(resolve_device(device))


def _only_coordinates_a_metal(restype) -> bool:
    """Whether this type is an ion or cluster: a non-polymer whose connections are all
    non-kinematic (a capped cystine's disulfide is too, so polymers are kept)."""
    if restype.properties.polymer.is_polymer:
        return False
    connections = restype.connections
    return bool(connections) and not any(
        connection.kinematic for connection in connections
    )


@toolz.functoolz.memoize
def _memoized_packed_block_types(device: torch.device) -> PackedBlockTypes:
    restype_set = ResidueTypeSet.get_default()

    # a cluster's twelve connections would widen every pose packed with it; ions and
    #    clusters join the packed set only when a structure has a metal
    active = [
        rt for rt in restype_set.residue_types if not _only_coordinates_a_metal(rt)
    ]

    return PackedBlockTypes.from_restype_list(
        chem_db=restype_set.chem_db,
        restype_set=restype_set,
        active_block_types=active,
        device=device,
    )


@_validate_args
def _canonical_form_from_pdb(
    canonical_ordering: CanonicalOrdering,
    pdb_lines_or_fname: Union[str, List],
    device: torch.device,
    *,
    residue_start: Optional[int] = None,
    residue_end: Optional[int] = None,
    res_not_connected: Optional[Tensor[torch.bool][:, :, 2]] = None,
) -> CanonicalForm:
    """Create a canonical form from either the contents of a PDB file
    as one long string or a list of individual lines from the file or
    by providing the name/path of a PDB file

    pdb_lines_or_fname must either be a list of the lines in a PDB file or
    a string representing a file

    """
    atom_records = _records_with_one_alternate(_parse_pdb(pdb_lines_or_fname))
    if residue_start is not None or residue_end is not None:
        atom_records = _select_atom_records_res_subset(
            atom_records, residue_start, residue_end
        )
    return _canonical_form_from_atom_records(
        canonical_ordering, atom_records, device, res_not_connected
    )


def _select_atom_records_res_subset(
    atom_records: pandas.DataFrame,
    residue_start: Optional[int],
    residue_end: Optional[int],
):
    """Figure out the starting row index for each residue
    and take the slice of the atom_records dataframe containing
    every atom of every residue within the given inclusive range.
    If either residue_start or residue_end are omitted, then
    the are treated as being the first or last residue.
    """

    atom_records_begin_for_res = []
    last_res = None
    for i, row in atom_records.iterrows():
        this_res = (row["modeli"], row["chaini"], row["resi"])
        if last_res is not None and last_res == this_res:
            pass
        else:
            atom_records_begin_for_res.append(i)
            last_res = this_res
    atom_records_begin_for_res.append(i + 1)
    if residue_start is None:
        residue_start = 0
    if residue_end is None:
        residue_end = len(atom_records_begin_for_res) - 1
    begin = atom_records_begin_for_res[residue_start]
    end = atom_records_begin_for_res[residue_end]
    return atom_records.iloc[begin:end]


def _canonical_form_from_atom_records(  # noqa: C901
    canonical_ordering: CanonicalOrdering,
    atom_records: pandas.DataFrame,
    device: torch.device,
    res_not_connected: Optional[Tensor] = None,
) -> CanonicalForm:
    device = resolve_device(device)
    max_n_canonical_atoms = canonical_ordering.max_n_canonical_atoms

    models = {key: i for i, key in enumerate(atom_records.modeli.unique())}
    residues = [{} for _ in models]
    chains = [{} for _ in models]
    for row in atom_records.itertuples(index=False):
        model = models[row.modeli]
        key = (row.chain, row.resi, row.insert)
        residues[model].setdefault(key, len(residues[model]))
        chains[model].setdefault(row.chaini, len(chains[model]))
    n_models = len(models)
    n_res = max(map(len, residues), default=0)
    shape = (n_models, n_res)
    atom_shape = (*shape, max_n_canonical_atoms)
    chain_id = numpy.full(shape, -1, dtype=numpy.int32)
    res_types = numpy.full(shape, -1, dtype=numpy.int32)
    coords = numpy.full((*atom_shape, 3), numpy.nan, dtype=numpy.float32)
    res_labels = numpy.zeros(shape, dtype=int)
    res_ins_codes = numpy.full(shape, "", dtype=object)
    chain_labels = numpy.full(shape, "", dtype=object)
    atom_occupancy = numpy.ones(atom_shape, dtype=numpy.float32)
    atom_b_factor = numpy.zeros(atom_shape, dtype=numpy.float32)
    types = {
        name: i for i, name in enumerate(canonical_ordering.restype_io_equiv_classes)
    }

    for row in atom_records.itertuples(index=False):
        model = models[row.modeli]
        res = residues[model][row.chain, row.resi, row.insert]
        pos = (model, res)
        name = canonical_ordering.resolve_name3(row.resn)
        res_types[pos] = types.get(name, -1)
        chain_id[pos] = chains[model][row.chaini]
        chain_labels[pos] = row.chain
        res_labels[pos] = row.resi
        res_ins_codes[pos] = row.insert
        if res_types[pos] < 0:
            continue
        index = canonical_ordering.restypes_atom_index_mapping[name].get(
            row.atomn.strip()
        )
        if index is not None:
            coords[model, res, index] = (row.x, row.y, row.z)
            atom_occupancy[model, res, index] = row.occupancy
            atom_b_factor[model, res, index] = row.b

    def _ti32(x):
        return torch.tensor(x, dtype=torch.int32, device=device)

    def _tf32(x):
        return torch.tensor(x, dtype=torch.float32, device=device)

    if res_not_connected is not None:
        assert res_not_connected.shape == (n_models, n_res, 2)

    return CanonicalForm(
        chain_id=_ti32(chain_id),
        res_types=_ti32(res_types),
        coords=_tf32(coords),
        res_labels=res_labels,
        residue_insertion_codes=res_ins_codes,
        chain_labels=chain_labels,
        atom_occupancy=atom_occupancy,
        atom_b_factor=atom_b_factor,
        res_not_connected=res_not_connected,
        disulfides=None,
    )
