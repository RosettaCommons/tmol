from pathlib import Path

import pytest

from tmol.utility import _cpp_extension


@pytest.mark.parametrize(
    "module_dir, name, expected",
    [
        ("tests/numeric", "bspline.cuda.cu", "tests/numeric/bspline.cuda.cu"),
        (
            "tests/score/common",
            "warp_segreduce.cuda.cu",
            "tests/score/common/warp_segreduce.cuda.cu",
        ),
        (
            "tests/score/common",
            "warp_stride_reduce.cuda.cu",
            "tests/score/common/warp_stride_reduce.cuda.cu",
        ),
        (
            "kinematics/compiled",
            "compiled.cpu.cpp",
            "backends/cpu/kinematics/compiled/compiled.cpu.cpp",
        ),
        (
            "kinematics/compiled",
            "compiled.cuda.cu",
            "backends/cuda/kinematics/compiled/compiled.cuda.cu",
        ),
        (
            "score/ljlk/potentials",
            "../../common/whole_pose_scoring.cuda.cu",
            "backends/cuda/score/common/whole_pose_scoring.cuda.cu",
        ),
    ],
)
def test_source_paths(module_dir, name, expected):
    root = Path(_cpp_extension.__file__).resolve().parents[1]
    resolved = _cpp_extension._relpaths(root / module_dir / "_compiled.py", [name])
    assert Path(resolved[0]).resolve() == root / expected
    assert Path(resolved[0]).is_file()


def test_required_cxx_standard():
    assert _cpp_extension._required_cxx_standard("2", "12") == 17
    assert _cpp_extension._required_cxx_standard("2", "13") == 20
    assert _cpp_extension._required_cxx_standard("3", "0") == 20


def test_active_torch_cxx_standard_flags_match():
    expected = _cpp_extension._required_cxx_standard(
        *_cpp_extension._get_torch_version()
    )

    assert f"--std=c++{expected}" in _cpp_extension._required_flags
    assert "-fno-math-errno" in _cpp_extension._required_flags
    assert f"-std=c++{expected}" in _cpp_extension._required_cuda_flags


def test_jit_openmp_flags_match_torch_backend():
    select = _cpp_extension._jit_openmp_flags

    assert select("linux", "ATen parallel backend: OpenMP") == (
        ["-fopenmp"],
        ["-fopenmp"],
    )
    assert select("linux", "ATen parallel backend: native thread pool") == ([], [])
    assert select("darwin", "ATen parallel backend: OpenMP") == ([], [])


def test_select_cuda_architecture_prefers_active_device():
    select = _cpp_extension._select_cuda_architecture
    assert select(None, (9, 0)) == "9.0"
    assert select("   ", (9, 0)) == "9.0"
    assert select("8.0;9.0;12.0+PTX", (12, 0)) == "12.0"
    assert select("7.5;8.0;8.6;9.0", (8, 9)) == "8.9"
    assert select("8.6 9.0", (9, 0)) == "9.0"
    assert select("8.6", (9, 0)) == "8.6"


def test_custom_extension_flags_preserve_release_optimization():
    kwargs = _cpp_extension._augment_kwargs(
        "test_extension",
        ["test.cpp"],
        extra_cflags=["-DCUSTOM_CXX"],
        extra_cuda_cflags=["-DCUSTOM_CUDA"],
        with_cuda=False,
    )

    assert kwargs["extra_cflags"][:2] == ["-O3", "-DCUSTOM_CXX"]
    assert kwargs["extra_cuda_cflags"][:2] == ["-O3", "-DCUSTOM_CUDA"]


def test_custom_extension_linker_flags_are_preserved():
    kwargs = _cpp_extension._augment_kwargs(
        "test_extension",
        ["test.cpp"],
        extra_ldflags=["-Wl,--custom"],
        with_cuda=False,
    )

    assert kwargs["extra_ldflags"][0] == "-Wl,--custom"
    assert kwargs["extra_ldflags"][1:] == _cpp_extension._openmp_ldflags


def test_custom_extension_include_paths_are_preserved():
    kwargs = _cpp_extension._augment_kwargs(
        "test_extension",
        ["test.cpp"],
        extra_include_paths=["/custom/include"],
        with_cuda=False,
    )

    assert kwargs["extra_include_paths"][0] == "/custom/include"
