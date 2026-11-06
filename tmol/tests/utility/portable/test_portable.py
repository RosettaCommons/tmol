"""Guards for the portable math layer shared with the Metal backend.

These compile C++ in a subprocess rather than going through a pybind module:
what they assert is a property of the *headers* -- that tmol's kernel-evaluated
math neither reaches Eigen nor the C++ standard library's containers -- which a
compiled extension could not express. Whole native kernel guards additionally
verify independence from framework headers and allocation; kinematics retains
its established Eigen transforms.

Two tiers, because the math is in two states:

* Tier 1 (``portable_cone.cc``) compiles against nothing but the portable
  layer. No Eigen, no torch, no std containers: the state a Metal translation
  unit requires.
* Tier 2 (``portable_cone_tensor.cc``) covers math that still reads through
  TView/TensorAccessor. It cannot be freestanding until TView gains a Metal
  accessor, but it is already Eigen-free and must stay that way.
"""

import shutil
import subprocess
import sysconfig
from pathlib import Path

import pytest

HERE = Path(__file__).parent
ROOT = HERE.parents[3]
POISON = HERE / "poison"

requires_cxx = pytest.mark.skipif(
    shutil.which("c++") is None, reason="no host C++ compiler available"
)


def _torch_flags():
    """Tier 2 reaches the tensor layer, which needs torch's headers."""
    import torch

    inc = Path(torch.__file__).parent / "include"
    return [
        f"-I{inc}",
        f"-I{inc / 'torch' / 'csrc' / 'api' / 'include'}",
        f"-I{sysconfig.get_paths()['include']}",
        "-DTORCH_TARGET_VERSION=0x020b000000000000",
    ]


def _compile(source, out, includes=(), defines=(), run=False):
    cmd = [
        shutil.which("c++"),
        "-std=c++17",
        "-O1",
        *defines,
        *(f"-I{inc}" for inc in includes),
        f"-I{ROOT}",
        f"-I{ROOT / 'tmol' / 'extern'}",
        str(source),
    ]
    cmd += ["-o", str(out)] if run else ["-fsyntax-only"]
    return subprocess.run(cmd, capture_output=True, text=True)


@requires_cxx
def test_freestanding_math_needs_only_the_portable_layer(tmp_path):
    """Tier 1: compiles with no Eigen, no torch, nothing but the portable layer."""
    proc = _compile(
        HERE / "portable_cone.cc",
        tmp_path / "cone",
        includes=[POISON],
    )
    assert proc.returncode == 0, proc.stderr


@requires_cxx
def test_tensor_dependent_math_does_not_reach_eigen(tmp_path):
    """Tier 2: Eigen must not reach math that still uses the tensor layer."""
    proc = _compile(
        HERE / "portable_cone_tensor.cc",
        tmp_path / "cone_tensor",
        includes=[POISON],
        defines=_torch_flags(),
    )
    assert proc.returncode == 0, proc.stderr


@requires_cxx
def test_guard_trips_when_eigen_is_reintroduced(tmp_path):
    """The guard must actually fail -- a guard that cannot fail proves nothing."""
    src = tmp_path / "regressed.cc"
    src.write_text("#include <Eigen/Core>\n" + (HERE / "portable_cone.cc").read_text())
    proc = _compile(src, tmp_path / "regressed", includes=[POISON])
    assert proc.returncode != 0
    assert "must stay portable" in proc.stderr


@requires_cxx
def test_portable_headers_use_no_std_containers():
    """Metal has no standard library; the portable headers may not need one."""
    banned = (
        "#include <tuple>",
        "#include <vector>",
        "#include <algorithm>",
        "#include <type_traits>",
        "std::tuple",
        "std::vector",
    )
    for name in ("qualifiers.hh", "math.hh", "matrix.hh", "tuple.hh"):
        text = (ROOT / "tmol" / "utility" / "portable" / name).read_text()
        code = "\n".join(
            line for line in text.splitlines() if not line.lstrip().startswith("//")
        )
        for token in banned:
            assert token not in code, f"{name} uses {token}"


@requires_cxx
def test_portable_matrix_matches_eigen(tmp_path):
    """Byte layout identical to Eigen; arithmetic equal to within rounding."""
    binary = tmp_path / "equivalence"
    proc = _compile(
        HERE / "portable_equivalence.cc",
        binary,
        includes=[sysconfig.get_paths()["include"]],
        run=True,
    )
    assert proc.returncode == 0, proc.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr


@requires_cxx
def test_whole_pose_genbonded_needs_no_framework(tmp_path):
    """Instantiate full float/double score and derivative kernels without Torch."""
    proc = _compile(HERE / "native_genbonded.cc", tmp_path / "genbonded", run=True)
    assert proc.returncode == 0, proc.stderr


@requires_cxx
@pytest.mark.parametrize("stable", [False, True])
def test_native_gradient_copy_compiles_for_both_torch_abis(tmp_path, stable):
    flags = _torch_flags()
    if not stable:
        flags = [
            flag for flag in flags if not flag.startswith("-DTORCH_TARGET_VERSION")
        ]
        flags.append("-std=c++20")
    proc = _compile(HERE / "native_copy.cc", tmp_path / "copy", defines=flags)
    assert proc.returncode == 0, proc.stderr


@requires_cxx
def test_whole_pose_cartbonded_needs_no_framework(tmp_path):
    proc = _compile(HERE / "native_cartbonded.cc", tmp_path / "cartbonded", run=True)
    assert proc.returncode == 0, proc.stderr


@requires_cxx
def test_whole_pose_backbone_torsion_needs_no_framework(tmp_path):
    proc = _compile(
        HERE / "native_backbone_torsion.cc", tmp_path / "backbone", run=True
    )
    assert proc.returncode == 0, proc.stderr


@requires_cxx
def test_dihedral_precision_near_planar_geometry(tmp_path):
    binary = tmp_path / "dihedral"
    proc = _compile(HERE / "dihedral_precision.cc", binary, run=True)
    assert proc.returncode == 0, proc.stderr
    proc = subprocess.run([str(binary)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


@requires_cxx
def test_bspline_coefficient_preparation_needs_no_framework(tmp_path):
    proc = _compile(
        HERE / "native_bspline_coeffs.cc", tmp_path / "coefficients", run=True
    )
    assert proc.returncode == 0, proc.stderr


@requires_cxx
def test_whole_pose_metal_needs_no_framework(tmp_path):
    proc = _compile(HERE / "native_metal.cc", tmp_path / "metal", run=True)
    assert proc.returncode == 0, proc.stderr


@requires_cxx
def test_whole_pose_na_torsion_needs_no_framework(tmp_path):
    proc = _compile(HERE / "native_na_torsion.cc", tmp_path / "na", run=True)
    assert proc.returncode == 0, proc.stderr


@requires_cxx
def test_fused_nonbonded_needs_no_framework(tmp_path):
    proc = _compile(HERE / "native_fused_nonbonded.cc", tmp_path / "fused", run=True)
    assert proc.returncode == 0, proc.stderr


@requires_cxx
def test_inverse_kinematics_needs_no_framework(tmp_path):
    binary = tmp_path / "kinematics"
    proc = _compile(
        HERE / "native_kinematics.cc",
        binary,
        includes=(ROOT / "tmol/extern",),
        run=True,
    )
    assert proc.returncode == 0, proc.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@requires_cxx
def test_pose_kinematics_needs_no_framework(tmp_path):
    binary = tmp_path / "pose_kinematics"
    proc = _compile(
        HERE / "native_pose_kinematics.cc",
        binary,
        includes=(ROOT / "tmol/extern",),
        run=True,
    )
    assert proc.returncode == 0, proc.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@requires_cxx
def test_whole_pose_dunbrack_needs_no_framework(tmp_path):
    proc = _compile(HERE / "native_dunbrack.cc", tmp_path / "dunbrack", run=True)
    assert proc.returncode == 0, proc.stderr
    result = subprocess.run(
        [str(tmp_path / "dunbrack")], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


@requires_cxx
def test_native_neighbors_follow_coordinate_changes(tmp_path):
    proc = _compile(HERE / "native_neighbors.cc", tmp_path / "neighbors", run=True)
    assert proc.returncode == 0, proc.stderr
    result = subprocess.run(
        [str(tmp_path / "neighbors")], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


@requires_cxx
def test_whole_pose_lkball_needs_no_framework(tmp_path):
    proc = _compile(HERE / "native_lkball.cc", tmp_path / "lkball", run=True)
    assert proc.returncode == 0, proc.stderr
    result = subprocess.run([str(tmp_path / "lkball")], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@requires_cxx
def test_independent_nonbonded_needs_no_framework(tmp_path):
    proc = _compile(
        HERE / "native_independent_nonbonded.cc", tmp_path / "nonbonded", run=True
    )
    assert proc.returncode == 0, proc.stderr


@requires_cxx
def test_large_residue_connection_atoms(tmp_path):
    """A bond to atom 299 must retain its two-bond nonbonded exclusion."""
    proc = _compile(
        HERE / "native_connection_atoms.cc", tmp_path / "connections", run=True
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    result = subprocess.run(
        [str(tmp_path / "connections")], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


@requires_cxx
def test_scan_workspace_covers_large_final_tile(tmp_path):
    binary = tmp_path / "scan_workspace"
    proc = _compile(HERE / "native_scan_workspace.cc", binary, run=True)
    assert proc.returncode == 0, proc.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr


@requires_cxx
def test_hbond_native_launch_and_full_width_indices(tmp_path):
    binary = tmp_path / "hbond"
    proc = _compile(HERE / "native_hbond.cc", binary, run=True)
    assert proc.returncode == 0, proc.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr


@requires_cxx
def test_native_sampling_empty_segments(tmp_path):
    binary = tmp_path / "sampling"
    proc = _compile(HERE / "native_sampling.cc", binary, run=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr


@requires_cxx
def test_conformer_dofs_preserve_missing_geometry_and_ring_offsets(tmp_path):
    binary = tmp_path / "conformer_dofs"
    proc = _compile(HERE / "native_conformer_dofs.cc", binary, run=True)
    assert proc.returncode == 0, proc.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr


def test_native_cpu_annealer_without_torch(tmp_path):
    proc = _compile(HERE / "native_cpu_annealer.cc", tmp_path / "annealer", run=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_native_interaction_graph_without_torch(tmp_path):
    proc = _compile(
        HERE / "native_interaction_graph.cc", tmp_path / "interaction_graph", run=True
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_native_graph_retention_without_torch(tmp_path):
    proc = _compile(
        HERE / "native_graph_retention.cc", tmp_path / "graph_retention", run=True
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_native_rotamer_dispatch_without_torch(tmp_path):
    proc = _compile(
        HERE / "native_rotamer_dispatch.cc", tmp_path / "rotamer_dispatch", run=True
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


@requires_cxx
@pytest.mark.parametrize(
    "source",
    [
        "native_backbone_rotamer.cc",
        "native_connected_rotamers.cc",
        "native_constraint_topology.cc",
        "native_disulfide_rotamer.cc",
        "native_load_balance.cc",
        "native_metal_rotamer.cc",
        "native_na_rotamer_topology.cc",
        "native_na_sampling.cc",
        "native_opth_sampling.cc",
        "native_pose_imposition.cc",
    ],
)
def test_native_candidate_invariants_without_frameworks(tmp_path, source):
    binary = tmp_path / "native_candidates"
    compile_result = _compile(HERE / source, binary, run=True)
    assert compile_result.returncode == 0, compile_result.stdout + compile_result.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
