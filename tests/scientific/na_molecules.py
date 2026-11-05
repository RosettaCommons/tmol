"""Connected DNA/RNA, real modified nucleotide and heterogeneous fixtures."""

from .molecules import MolecularCase, DOMAIN_CASES

NA_CASES = {
    "dna": MolecularCase((), (("DA", "DC", "DG", "DT"),), ((4,),)),
    "rna": MolecularCase((), (("RA", "RC", "RG", "RU"),), ((4,),)),
    "termini": MolecularCase((), (("DA:na5prime", "DC", "DT:na3prime"),), ((3,),)),
    "methylcytosine": DOMAIN_CASES["modified_dna"],
    "mixed_batch": MolecularCase(
        ("ligand_params/noncanonical_dna.tmol",),
        (("RA", "RU", "ALA"), ("5CM", "DA")),
        ((2, 1), (2,)),
    ),
}
