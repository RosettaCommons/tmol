"""AtomWorks encoding metadata shared by framework input/output adapters."""

import numpy

# Protein token/slot data from AtomWorks 3.0.4 UNIFIED_ATOM37_ENCODING:
# atomworks/ml/encoding_definitions.py (BSD-3-Clause).
# Kept as data because importing AtomWorks' ML encoding module registers optional
# Torch annotators. The compatibility test compares every token and slot.
# BSD 3-Clause License
#
# Copyright (c) 2025, Institute for Protein Design, University of Washington
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

_PROTEIN_SLOTS = {
    "<M>": "",
    "ALA": "0:N 1:CA 2:C 3:CB 4:O 36:OXT",
    "ARG": "0:N 1:CA 2:C 3:CB 4:O 5:CG 11:CD 23:NE 29:NH1 30:NH2 32:CZ 36:OXT",
    "ASN": "0:N 1:CA 2:C 3:CB 4:O 5:CG 15:ND2 16:OD1 36:OXT",
    "ASP": "0:N 1:CA 2:C 3:CB 4:O 5:CG 16:OD1 17:OD2 36:OXT",
    "CYS": "0:N 1:CA 2:C 3:CB 4:O 10:SG 36:OXT",
    "GLN": "0:N 1:CA 2:C 3:CB 4:O 5:CG 11:CD 25:NE2 26:OE1 36:OXT",
    "GLU": "0:N 1:CA 2:C 3:CB 4:O 5:CG 11:CD 26:OE1 27:OE2 36:OXT",
    "GLY": "0:N 1:CA 2:C 4:O 36:OXT",
    "HIS": "0:N 1:CA 2:C 3:CB 4:O 5:CG 13:CD2 14:ND1 20:CE1 25:NE2 36:OXT",
    "ILE": "0:N 1:CA 2:C 3:CB 4:O 6:CG1 7:CG2 12:CD1 36:OXT",
    "LEU": "0:N 1:CA 2:C 3:CB 4:O 5:CG 12:CD1 13:CD2 36:OXT",
    "LYS": "0:N 1:CA 2:C 3:CB 4:O 5:CG 11:CD 19:CE 35:NZ 36:OXT",
    "MET": "0:N 1:CA 2:C 3:CB 4:O 5:CG 18:SD 19:CE 36:OXT",
    "PHE": "0:N 1:CA 2:C 3:CB 4:O 5:CG 12:CD1 13:CD2 20:CE1 21:CE2 32:CZ 36:OXT",
    "PRO": "0:N 1:CA 2:C 3:CB 4:O 5:CG 11:CD 36:OXT",
    "SER": "0:N 1:CA 2:C 3:CB 4:O 8:OG 36:OXT",
    "THR": "0:N 1:CA 2:C 3:CB 4:O 7:CG2 9:OG1 36:OXT",
    "TRP": "0:N 1:CA 2:C 3:CB 4:O 5:CG 12:CD1 13:CD2 21:CE2 22:CE3 24:NE1 28:CH2 33:CZ2 34:CZ3 36:OXT",
    "TYR": "0:N 1:CA 2:C 3:CB 4:O 5:CG 12:CD1 13:CD2 20:CE1 21:CE2 31:OH 32:CZ 36:OXT",
    "VAL": "0:N 1:CA 2:C 3:CB 4:O 6:CG1 7:CG2 36:OXT",
    "UNK": "",
}
ATOMWORKS_NAME3S = list(_PROTEIN_SLOTS)
ATOMWORKS_ATOM37_NAMES = {name: [""] * 37 for name in ATOMWORKS_NAME3S}
for _name, _slots in _PROTEIN_SLOTS.items():
    for _entry in _slots.split():
        _slot, _atom = _entry.split(":")
        ATOMWORKS_ATOM37_NAMES[_name][int(_slot)] = _atom
_ATOMWORKS_MIN_PROTEIN_IDX = ATOMWORKS_NAME3S.index("ALA")
_ATOMWORKS_MAX_PROTEIN_IDX = ATOMWORKS_NAME3S.index("VAL")


def atomworks_restype_map(canonical_ordering):
    classes = canonical_ordering.restype_io_equiv_classes
    return numpy.asarray(
        [classes.index(n) if n in classes else -1 for n in ATOMWORKS_NAME3S],
        dtype=numpy.int64,
    )


def canonical_to_atomworks_mappings(co):
    """Canonical residue/atom identity to AtomWorks tokens and slots."""
    shape = (len(co.restype_io_equiv_classes), co.max_n_canonical_atoms)
    restype = numpy.zeros(shape[0], dtype=numpy.int64)
    atom = numpy.full(shape, -1, dtype=numpy.int64)
    real = numpy.zeros(shape, dtype=numpy.bool_)
    for aw_index, name3 in enumerate(ATOMWORKS_NAME3S):
        if name3 not in co.restype_io_equiv_classes:
            continue
        index = co.restype_io_equiv_classes.index(name3)
        restype[index] = aw_index
        mapping = co.restypes_atom_index_mapping[name3]
        for slot, name in enumerate(ATOMWORKS_ATOM37_NAMES[name3]):
            if name and name in mapping:
                atom[index, mapping[name]] = slot
                real[index, mapping[name]] = True
    return restype, atom, real
