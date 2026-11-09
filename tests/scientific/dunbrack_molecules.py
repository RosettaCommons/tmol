"""Rotameric and semirotameric chemistry, including D forms and empty domains."""

from .molecules import MolecularCase, CASES

DUNBRACK_CASES = {
    "l_peptide": MolecularCase((), (("ALA", "LEU", "ARG", "PHE"),), ((4,),)),
    "d_peptide": MolecularCase((), (("DALA", "DLEU", "DARG", "DPHE"),), ((4,),)),
    "variants": MolecularCase(
        (), (("ALA:nterm", "HIS_D", "CYD", "PHE:cterm"),), ((4,),)
    ),
    "mixed_batch": MolecularCase(
        CASES["mixed_batch"].parameter_files,
        (("ALA", "HYP", "PHE", "LG1"), ("DALA", "DLEU", "DARG", "DPHE")),
        ((3, 1), (4,)),
    ),
}
