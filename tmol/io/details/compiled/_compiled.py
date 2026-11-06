from tmol._load_ext import _load_ops

_ops = _load_ops(
    __name__,
    __file__,
    [
        "compiled.ops.cpp",
        "gen_pose_leaf_atoms.cpu.cpp",
        "gen_pose_leaf_atoms.cuda.cu",
        "resolve_his_taut.cpu.cpp",
        "resolve_his_taut.cuda.cu",
    ],
    "tmol_io",
)

_gen_pose_leaf_atoms = _ops.gen_pose_leaf_atoms
_resolve_his_taut = _ops.resolve_his_taut

_select_canonical_candidates = _ops.select_canonical_candidates

_canonical_chain_endings = _ops.canonical_chain_endings
_canonical_atom_termini = _ops.canonical_atom_termini

_canonical_variants = _ops.canonical_variants
_canonical_cycles = _ops.canonical_cycles
_canonical_wiring = _ops.canonical_wiring
_canonical_connection_errors = _ops.canonical_connection_errors
