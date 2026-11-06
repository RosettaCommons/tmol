"""JAX coordinate execution for the common context-dependent construction plan."""

from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np


def _plan(build):
    return (
        int(build.pose),
        int(build.s),
        tuple(map(int, build.known_atoms)),
        tuple((int(k), int(v)) for k, v in build.node_global.items()),
        tuple(
            (
                int(step.atom),
                step.refs,
                step.dist,
                step.theta,
                step.phi + build.chosen.get(step.axis, 0.0),
            )
            for step in build.steps
        ),
    )


@partial(jax.jit, static_argnames=("plans",))
def execute_context_plan(coords, plans):
    """Differentiate coordinates through fixed shared placement decisions.

    Topology, reference choices and clash-selected rotations are prepared on the
    host. This execution supports jit, vmap and coordinate derivatives; changing
    the discrete choices requires preparing a new plan.
    """

    def normalize(v):
        return v / jnp.sqrt(jnp.maximum(jnp.sum(v * v), jnp.asarray(1e-24, v.dtype)))

    for pose, start, known, nodes, steps in plans:
        local = {a: coords[pose, start + a] for a in known}
        local.update({node: coords[pose, index] for node, index in nodes})

        def resolve(ref):
            if not isinstance(ref, tuple):
                return local[ref]
            kind, origin, through = ref
            point = local[origin]
            if kind == "perp" and through is not None:
                axis = normalize(local[through] - point)
                other = jnp.eye(3, dtype=coords.dtype)[jnp.argmin(jnp.abs(axis))]
                return point + jnp.cross(axis, other)
            return point + jnp.asarray(
                [1.0, 0.0, 0.0] if kind == "perp" else [0.0, 1.0, 0.0], coords.dtype
            )

        for atom, refs, distance, theta, phi in steps:
            a, b, c = map(resolve, refs)
            bc = normalize(c - b)
            n = normalize(jnp.cross(b - a, bc))
            m = jnp.cross(n, bc)
            local[atom] = c + distance * (
                -math.cos(theta) * bc
                + math.sin(theta) * math.cos(phi) * m
                + math.sin(theta) * math.sin(phi) * n
            )
            coords = coords.at[pose, start + atom].set(local[atom])
    return coords


def prepare_context_completion(pose, missing, targets, *, heavy_only=True):
    """Run common planning, feeding actual JAX placement into each build wave."""
    from tmol_core.io.context import build_missing_by_context

    pbt = pose.packed_block_types
    out, plans = pose.coords, []

    def place(build):
        nonlocal out
        plan = _plan(build)
        plans.append(plan)
        out = execute_context_plan(out, (plan,))
        placed = np.asarray(out)[build.pose]
        return {a: placed[build.s + a].astype(np.float64) for a in build.built}

    built = build_missing_by_context(
        pbt.active_block_types,
        {at.name: at for at in pbt.chem_db.atom_types},
        np.asarray(pbt.conn_atoms),
        np.asarray(pose.coords, dtype=np.float64),
        np.asarray(missing),
        np.asarray(targets),
        np.asarray(pose.block_coord_offset),
        np.asarray(pose.block_type_ind),
        np.asarray(pose.inter_residue_connections),
        place,
        heavy_only,
    )
    return out, built, tuple(plans)
