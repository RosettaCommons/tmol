"""What minimizing optH's proton chis adds to loading a structure."""

import biotite.structure as struc
import pytest
import torch

from tmol import beta2016_score_function, pose_stack_from_atom_array

_HYDROGENS = {
    "no_optH": dict(no_optH=True),
    "pack": dict(no_optH=False, optH_minimize=False),
    "pack_min": dict(no_optH=False, optH_minimize=True),
}


@pytest.mark.parametrize("n_poses", [1, 10])
@pytest.mark.parametrize("hydrogens", list(_HYDROGENS))
@pytest.mark.benchmark(group="load_with_optH")
def test_load_with_optH_benchmark(
    benchmark, biotite_1ubq, torch_device, n_poses, hydrogens
):
    """Time loading ubiquitin, its hydrogens optimized or not.

    The timed call is the whole load from an AtomArray(Stack): build context,
    pose construction, and hydrogen placement. Device work completes before the
    timed call returns. The resulting energy (mean per pose) and, on CUDA, the
    peak memory of one untimed load are recorded in extra_info, so the
    variants can be compared on cost and result together.
    """
    structure = biotite_1ubq
    if n_poses > 1:
        structure = struc.stack([biotite_1ubq] * n_poses)
    options = dict(packer_seed=0, **_HYDROGENS[hydrogens])

    def run():
        torch.manual_seed(0)
        pose = pose_stack_from_atom_array(structure, torch_device, **options)
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        return pose

    if torch_device.type == "cuda":
        run()
        torch.cuda.reset_peak_memory_stats(torch_device)
        run()
        benchmark.extra_info["peak_memory_mib"] = (
            torch.cuda.max_memory_allocated(torch_device) / 2**20
        )

    pose = benchmark.pedantic(run, rounds=5, iterations=1, warmup_rounds=1)

    assert pose.n_poses == n_poses
    assert torch.isfinite(pose.coords[pose.real_atoms]).all()
    sfxn = beta2016_score_function(torch_device)
    energy = sfxn.render_whole_pose_scoring_module(pose)(pose.coords.detach())
    benchmark.extra_info["mean_energy"] = float(energy.mean())
    benchmark.extra_info["n_poses"] = n_poses
    benchmark.extra_info["threads"] = torch.get_num_threads()
