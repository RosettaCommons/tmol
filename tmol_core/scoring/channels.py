"""Stable score indices and native term lane order for all frontends."""

from types import MappingProxyType

# Append channels before the sentinel; existing indices are a public contract.
SCORE_TYPE_NAMES = (
    "fa_ljatr",
    "fa_ljrep",
    "fa_lk",
    "fa_elec",
    "hbond",
    "cart_lengths",
    "cart_angles",
    "cart_torsions",
    "cart_impropers",
    "cart_hxltorsions",
    "constraint",
    "disulfide",
    "omega",
    "rama",
    "dunbrack_rot",
    "dunbrack_rotdev",
    "dunbrack_semirot",
    "lk_ball_iso",
    "lk_ball",
    "lk_bridge",
    "lk_bridge_uncpl",
    "ref",
    "gen_torsions",
    "na_torsion",
    "na_torsion_well",
    "metal_coordination",
    "n_score_types",
)

# Term output lanes need not follow global indices (e.g. rama precedes omega).
TERM_CHANNELS = MappingProxyType(
    {
        "ljlk": ("fa_ljatr", "fa_ljrep", "fa_lk"),
        "elec": ("fa_elec",),
        "hbond": ("hbond",),
        "cartbonded": (
            "cart_lengths",
            "cart_angles",
            "cart_torsions",
            "cart_impropers",
            "cart_hxltorsions",
        ),
        "constraint": ("constraint",),
        "disulfide": ("disulfide",),
        "backbone_torsion": ("rama", "omega"),
        "dunbrack": ("dunbrack_rot", "dunbrack_rotdev", "dunbrack_semirot"),
        "lk_ball": ("lk_ball_iso", "lk_ball", "lk_bridge", "lk_bridge_uncpl"),
        "ref": ("ref",),
        "genbonded": ("gen_torsions",),
        "na_torsion": ("na_torsion", "na_torsion_well"),
        "metal": ("metal_coordination",),
    }
)


# Mature term arity used by packing and public score-function queries. A custom
# term supplies its own arity through the lifecycle adapter.
TERM_N_BODIES = MappingProxyType(
    {
        "ljlk": 2,
        "elec": 2,
        "hbond": 2,
        "cartbonded": 2,
        "constraint": 1,
        "disulfide": 2,
        "backbone_torsion": 2,
        "dunbrack": 1,
        "lk_ball": 2,
        "ref": 1,
        "genbonded": 2,
        "na_torsion": 2,
        "metal": 2,
    }
)
