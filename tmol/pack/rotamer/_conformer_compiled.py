from tmol._load_ext import _load_ops

_ops = _load_ops(
    __name__,
    __file__,
    ["conformer.ops.cpp", "conformer.cpu.cpp", "conformer.cuda.cu"],
    "tmol_conformer_dofs",
)
copy_mainchain_dofs = _ops.copy_mainchain_dofs
assign_chi_dofs = _ops.assign_chi_dofs

copy_conformer_rows = _ops.copy_conformer_rows

correct_chi_dofs = _ops.correct_chi_dofs

select_pose_rotamers = _ops.select_pose_rotamers
plan_pose_imposition = _ops.plan_pose_imposition

opth_sampling_counts = _ops.opth_sampling_counts

opth_sampling_fill = _ops.opth_sampling_fill

opth_sampling_proton = _ops.opth_sampling_proton

conformer_row_indices = _ops.conformer_row_indices
