"""Torch execution of the common context-dependent missing-atom planner."""

import math
import torch
from tmol_core.io.context import build_missing_by_context
from tmol_core.io.context import (  # noqa: F401
    _dihedral,
    _normalize,
    _place_batch,
    _perpendicular_point,
    _BlockTypeGeometry,
    _geometry,
    _Step,
    _plan,
    _rotatable,
    _candidates,
    _Environment,
    _overlap,
    _clash,
    _waits_on,
    _build_together,
    _connection_nodes,
    _resolve,
    _BlockBuild,
    _rotate_about,
    _rotation_sign,
)


def _build_missing_by_context(
    pbt,
    pose_coords,
    missing,
    targets,
    offsets,
    block_types,
    connections,
    heavy_only=True,
):
    out = pose_coords.clone()

    def place(build):
        placed = _place_differentiably(
            build.steps,
            build.chosen,
            out,
            build.pose,
            build.s,
            build.known_atoms,
            build.node_global,
        )
        return {a: x.detach().cpu().double().numpy() for a, x in placed.items()}

    built = build_missing_by_context(
        pbt.active_block_types,
        {at.name: at for at in pbt.chem_db.atom_types},
        pbt.conn_atom.cpu().numpy(),
        pose_coords.detach().cpu().double().numpy(),
        missing.cpu().numpy(),
        targets.cpu().numpy(),
        offsets.cpu().numpy(),
        block_types.cpu().numpy(),
        connections.cpu().numpy(),
        place,
        heavy_only,
    )
    return out, torch.from_numpy(built).to(device=targets.device)


def _place_torch(a, b, c, dist, theta, phi):
    bc = torch.nn.functional.normalize(c - b, dim=-1)
    n = torch.nn.functional.normalize(torch.linalg.cross(b - a, bc), dim=-1)
    m = torch.linalg.cross(n, bc)
    return c + dist * (
        -math.cos(theta) * bc
        + math.sin(theta) * math.cos(phi) * m
        + math.sin(theta) * math.sin(phi) * n
    )


def _place_differentiably(steps, chosen, out, pose, s, known_atoms, node_global):
    """Rebuild the chosen placement in torch from the input coordinates."""
    local = {a: out[pose, s + a] for a in known_atoms}
    local.update({node: out[pose, g] for node, g in node_global.items()})

    def offset_point(o, through):
        axis = torch.nn.functional.normalize(through - o, dim=-1)
        other = torch.eye(3, dtype=o.dtype, device=o.device)[
            int(torch.argmin(axis.detach().abs()))
        ]
        return o + torch.linalg.cross(axis, other)

    def resolve(ref):
        if not isinstance(ref, tuple):
            return local[ref]
        kind, origin, through = ref
        o = local[origin]
        if kind == "perp" and through is not None:
            return offset_point(o, local[through])
        return o + o.new_tensor([1.0, 0.0, 0.0] if kind == "perp" else [0.0, 1.0, 0.0])

    for step in steps:
        offset = chosen.get(step.axis, 0.0) if step.axis is not None else 0.0
        gg, g, p = (resolve(r) for r in step.refs)
        local[step.atom] = _place_torch(
            gg, g, p, step.dist, step.theta, step.phi + offset
        )
        out[pose, s + step.atom] = local[step.atom]
    return {a: local[a] for a in (st.atom for st in steps)}
