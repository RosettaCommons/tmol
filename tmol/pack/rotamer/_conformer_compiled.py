from tmol._load_ext import _load_ops

_ops = _load_ops(
    __name__,
    __file__,
    ["conformer.ops.cpp", "conformer.cpu.cpp", "conformer.cuda.cu"],
    "tmol_conformer_dofs",
)
copy_mainchain_dofs = _ops.copy_mainchain_dofs
assign_chi_dofs = _ops.assign_chi_dofs
