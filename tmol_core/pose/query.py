"""Shared selection and result structure for named molecular torsion queries."""

import numpy


def _is_scalar(x):
    return isinstance(x, (int, numpy.integer, str))


def _as_list(x, default):
    if x is None:
        return list(default)
    return [x] if _is_scalar(x) else list(x)


def torsion_requests(
    active_types, block_type_ind, poses=None, blocks=None, names=None
):  # noqa: C901
    """Expand a selection into (pose, block, torsion index, name) tuples."""
    bti = numpy.asarray(block_type_ind)
    n_poses, max_n_blocks = bti.shape

    pose_list = _as_list(poses, range(n_poses))
    block_list = _as_list(blocks, range(max_n_blocks))
    name_list = None if names is None else _as_list(names, ())

    paired = poses is not None and blocks is not None
    if paired and not _is_scalar(poses) and not _is_scalar(blocks):
        if len(pose_list) != len(block_list):
            raise ValueError("poses and blocks must be the same length")
        pairs = list(zip(pose_list, block_list))
    else:
        pairs = [(p, b) for p in pose_list for b in block_list]

    if name_list is not None and len(name_list) == len(pairs) and len(pairs) > 1:
        paired_names = name_list
    else:
        paired_names = None

    requests = []
    for i, (pose, block) in enumerate(pairs):
        bt_ind = int(bti[pose, block])
        if bt_ind == -1:
            if blocks is None:
                continue
            raise ValueError(f"block {block} of pose {pose} is not a real block")
        bt = active_types[bt_ind]
        tor_inds = {tor.name: j for j, tor in enumerate(bt.torsions)}
        if paired_names is not None:
            wanted = [paired_names[i]]
        elif name_list is not None:
            wanted = name_list
        else:
            wanted = list(tor_inds)
        for name in wanted:
            if name not in tor_inds:
                if name_list is None or blocks is None:
                    continue
                raise ValueError(f"block type {bt.name} has no torsion {name!r}")
            requests.append((pose, block, tor_inds[name], name))
    return requests


def format_named_torsions(requests, values, shape, poses, blocks, names):
    """Preserve scalar, name-dictionary and pose/block nested output contracts."""
    if _is_scalar(poses) and _is_scalar(blocks):
        if _is_scalar(names):
            return float(values[0])
        return {
            name: value for (_, _, _, name), value in zip(requests, values.tolist())
        }
    out = [[{} for _ in range(shape[1])] for _ in range(shape[0])]
    for (pose, block, _, name), value in zip(requests, values.tolist()):
        out[pose][block][name] = value
    return out
