from tmol._load_ext import _load_ops

_ops = _load_ops(
    __name__,
    __file__,
    [
        "compiled.ops.cpp",
        "na_torsion_pose_score.cpu.cpp",
        "na_torsion_pose_score.cuda.cu",
    ],
    "tmol_na_torsion",
)

_na_torsion_pose_score = _ops.na_torsion_pose_score

_na_rotamer_topology = _ops.na_rotamer_topology
_na_torsion_native_score = _ops.na_torsion_native_score

_na_sampling_pucker = _ops.na_sampling_pucker
_na_sampling_counts = _ops.na_sampling_counts
_na_sampling_fill = _ops.na_sampling_fill
