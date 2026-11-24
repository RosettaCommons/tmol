"""Scoring of ligand pose batches against a rigid receptor.

The receptor enters through precomputed B-spline grids (``fa_atr``, ``fa_rep``,
``fa_sol``, ``fa_elec`` and, optionally, Rosetta's gridded LK-ball) plus live
hbond / LK-ball / ligand-intramolecular terms. :class:`GridScorer` builds every
per-ligand tensor once, so an evaluation allocates only its outputs and has no
host synchronisation or host-to-device copy: it can be captured in a CUDA graph.
"""

from __future__ import annotations

from typing import Sequence

import torch

from tmol.score import ScoreFunction, beta2016_score_function
from tmol.score._score_types import ScoreType

from .grid import (
    GRID_LK_BALL_FADE,
    GRID_LK_BALL_WATER_DISTANCE,
    GRID_SCORE_TYPES,
    LK_BALL_SCORE_TYPES,
    OUT_OF_BOUNDS_PENALTY,
    GridSet,
    ixform_rep,
)
from .live_hbond import LiveHBondData, score_ligand_hbond_with_gradients
from .live_ligand_intra import LiveLigandIntraData, score_ligand_intra_with_gradients
from .live_lk_ball import LiveLKBallData, score_ligand_lk_ball_with_gradients
from .potentials.compiled import (
    grid_score_atoms,
    grid_score_waters,
    interpolate_with_derivatives,
)


class _Interpolation(torch.autograd.Function):
    """Autograd bridge for the compiled value-and-spatial-derivative kernel."""

    @staticmethod
    def forward(ctx, coefficients, indices, grid_indices):
        values, derivatives, penalties, penalty_derivatives = (
            interpolate_with_derivatives(
                coefficients.contiguous(),
                indices.contiguous(),
                grid_indices.contiguous(),
                OUT_OF_BOUNDS_PENALTY,
            )
        )
        ctx.save_for_backward(derivatives, penalty_derivatives)
        return values, penalties

    @staticmethod
    def backward(ctx, grad_values, grad_penalties):
        derivatives, penalty_derivatives = ctx.saved_tensors
        grad = torch.zeros_like(derivatives)
        if grad_values is not None:
            grad.add_(derivatives * grad_values[:, None])
        if grad_penalties is not None:
            grad.add_(penalty_derivatives * grad_penalties[:, None])
        return None, grad, None


def interpolate_grid(
    coefficients: torch.Tensor,
    indices: torch.Tensor,
    grid_indices: torch.Tensor,
    penalty: bool = False,
):
    """Boundary-clamped tricubic B-spline values at zero-based ``indices``.

    Differentiable with respect to ``indices``. With ``penalty=True`` returns
    ``(values, out_of_bounds_penalties)``.
    """
    values, penalties = _Interpolation.apply(coefficients, indices, grid_indices)
    return (values, penalties) if penalty else values


class _Score(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coords, scorer):
        energies, gradients = scorer.value_and_grad(coords)
        ctx.save_for_backward(gradients)
        return energies

    @staticmethod
    def backward(ctx, grad_energies):
        (gradients,) = ctx.saved_tensors
        return gradients * grad_energies[:, None, None], None


class GridScorer:
    """Energy and analytic Cartesian gradient of ligand pose batches.

    Args:
        grids: Receptor grids (any device/dtype; copied to ``device``/``dtype``).
        ligand_atom_types: Grid atom type of each grid-scored ligand atom.
        ligand_charges: Partial charge of each grid-scored ligand atom.
        device, dtype: Where and in which precision poses are scored.
        score_function: Source of the term weights (default ``beta2016``).
        live_hbond, live_lk_ball, live_ligand_intra: Optional live terms.
        lk_ball_grid: Use the gridded Rosetta LK-ball total (needs
            ``live_lk_ball`` for the water topology) instead of the exact one.
        grid_atoms: Indices of the coordinate atoms scored on the grids, when
            the movable set is larger than the grid-scored set (flexible
            sidechains); default all atoms.
    """

    def __init__(
        self,
        grids: GridSet,
        ligand_atom_types: Sequence[str],
        ligand_charges: torch.Tensor,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        score_function: ScoreFunction | None = None,
        live_hbond: LiveHBondData | None = None,
        live_lk_ball: LiveLKBallData | None = None,
        live_ligand_intra: LiveLigandIntraData | None = None,
        lk_ball_grid: bool = False,
        grid_atoms: torch.Tensor | None = None,
    ) -> None:
        device = torch.empty(0, device=device).device
        self.grids = grids.on(device, dtype)
        self.live_hbond = live_hbond and live_hbond.to(device, dtype)
        self.live_lk_ball = live_lk_ball and live_lk_ball.to(device, dtype)
        self.live_ligand_intra = live_ligand_intra and live_ligand_intra.to(
            device, dtype
        )
        self.lk_ball_grid = lk_ball_grid
        if lk_ball_grid and self.live_lk_ball is None:
            raise ValueError("gridded LK-ball requires the ligand water topology")
        self._grid_atoms = None if grid_atoms is None else grid_atoms.to(device)

        to_grid = grids.atom_type_to_grid
        if any(name not in to_grid for name in ligand_atom_types):
            raise ValueError("ligand atom types are absent from the grid")
        self._grid_index = torch.tensor(
            [to_grid[name] for name in ligand_atom_types],
            dtype=torch.int64,
            device=device,
        )
        self._charges = ligand_charges.to(device=device, dtype=dtype)
        # Per-pose overrides, set when a population's poses hold different
        # ligand block types: [pose, atom] each, present=False zeroing a slot.
        self._pose_grid_index: torch.Tensor | None = None
        self._pose_charges: torch.Tensor | None = None
        self._pose_present: torch.Tensor | None = None
        #: (movable slot -> this pose's local index, slot is present) for the
        #: ligand-intra modules, or None while every pose is one molecule.
        self._intra_packing: tuple[torch.Tensor, torch.Tensor] | None = None
        self._grid_atom_index = (
            torch.arange(len(ligand_atom_types), device=device)
            if self._grid_atoms is None
            else self._grid_atoms
        )
        # Heavy-atom part of the gridded LK-ball total (empty: not gridded).
        gridded_lk = lk_ball_grid and self.grids.lk_ball_enabled
        self._lk_heavy_coeffs = (
            self.grids.lk_ball_coeffs if gridded_lk else self.grids.fa_atr_coeffs[:0]
        )

        sfxn = score_function or beta2016_score_function(device)

        def weights(score_types):
            return torch.stack([sfxn.get_weight(t) for t in score_types]).to(
                device=device, dtype=dtype
            )

        self._weights = weights(GRID_SCORE_TYPES)
        self._base_weights = self._weights.clone()
        self._hbond_weight = weights((ScoreType.hbond,))[0]
        self._lk_ball_weights = weights(LK_BALL_SCORE_TYPES)

        # Gridded LK-ball folds these weights into the grid, so they cannot be
        # reweighted at scoring time.
        if lk_ball_grid and self.grids.lk_ball_weights is not None:
            if not torch.allclose(
                self.grids.lk_ball_weights.to(device=device, dtype=dtype),
                self._lk_ball_weights,
            ):
                raise ValueError(
                    "lk_ball weights must match those used in grid construction"
                )

        # Movable slot -> its column in the grid-atom tables, -1 if not scored.
        self._grid_slot_of = None
        if lk_ball_grid:
            slot_of = torch.full(
                (self.live_lk_ball.n_ligand_atoms,),
                -1,
                dtype=torch.int64,
                device=device,
            )
            slot_of[self._grid_atom_index] = torch.arange(
                self._grid_atom_index.numel(), dtype=torch.int64, device=device
            )
            self._grid_slot_of = slot_of

        # Kernel tables, [table pose, ...]: batch pose p reads row p % rows. One
        # row while every pose holds the same ligand.
        self._type_table = self._grid_index[None].clone()
        self._charge_table = self._charges[None].clone()
        self._polar_type_table = None
        self._lk_tables = None
        self._refresh_polar_tables()

        self._mps = self._mps_grid_tensors()

    def _refresh_polar_tables(self) -> None:
        """Polar types and LK-ball water tables at the grid tables' row count.

        Every table one kernel reads must share its leading dimension. A polar
        slot is read from the row for that pose, so a pose whose ligand bound a
        metal takes its own sites, waters and retyped donor.
        """
        if self._grid_slot_of is None:
            return
        data = self.live_lk_ball
        rows = self._type_table.shape[0]
        tables = (
            data.ligand_polar_atoms,
            data.ligand_water_kinds,
            data.ligand_water_atoms,
            data.ligand_water_params,
            data.ligand_water_is_ring,
        )
        lk_rows = tables[0].shape[0]
        if rows == 1 and lk_rows > 1:
            # Construction order: the grid tables take their per-pose rows from
            # set_pose_atom_state, which runs next and refreshes these.
            return
        if rows % lk_rows:
            raise ValueError("LK-ball tables and grid tables disagree on rows")
        spread = torch.arange(rows, device=self.device) % lk_rows
        tables = tuple(t[spread] for t in tables)
        # Each polar atom's column in the grid-atom tables, then its type there.
        polar = tables[0]
        position = self._grid_slot_of[polar.clamp(min=0)]
        # No host sync here: this runs inside the captured objective. That
        # every polar atom is grid-scored is checked when the tables are set.
        real = (polar >= 0) & (position >= 0)
        types = self._type_table.gather(1, position.clamp(min=0))
        types = torch.where(real, types, types.new_full((), -1))
        if self._polar_type_table is None or self._polar_type_table.shape != (
            types.shape
        ):
            self._polar_type_table = types.contiguous()
            self._lk_tables = tuple(t.contiguous() for t in tables)
            return
        self._polar_type_table.copy_(types)
        for buffer, t in zip(self._lk_tables, tables):
            buffer.copy_(t)

    def _mps_grid_tensors(self) -> dict | None:
        """Metal copies of the per-atom grid kernel's inputs, or None.

        A CPU scorer runs its per-atom grid kernel on the Apple GPU when the
        Metal offload is enabled (see tmol.score._mps_offload); everything
        else, and every result handed back, stays on the host. Metal has no
        float64, so only float32 scorers offload.
        """
        from tmol.score._mps_offload import mps_offload_enabled

        if self.device.type != "cpu" or self._charges.dtype != torch.float32:
            return None
        if not mps_offload_enabled():
            return None
        mps = torch.device("mps")
        grids = self.grids
        return {
            "grid_atom_index": self._grid_atom_index.to(mps),
            "grid_index": self._type_table.to(mps),
            "charges": self._charge_table.to(mps),
            "atr": grids.fa_atr_coeffs.to(mps),
            "rep": grids.fa_rep_coeffs.to(mps),
            "sol": grids.fa_sol_coeffs.to(mps),
            "elec": grids.fa_elec_coeffs[None].to(mps),
            "lk": self._lk_heavy_coeffs.to(mps),
            "origin": grids.geometry.origin.to(mps),
            "weights": self._weights.to(mps),
        }

    @property
    def device(self) -> torch.device:
        return self._grid_index.device

    def set_rep_scale(self, scale: float) -> None:
        """Scale the repulsive weight, as Rosetta's packing ramp does.

        Weights are written in place so a captured CUDA graph keeps reading
        the tensor it was captured with.
        """
        self._weights.copy_(self._base_weights)
        self._weights[GRID_SCORE_TYPES.index(ScoreType.fa_ljrep)] *= scale
        if self._mps is not None:
            self._mps["weights"].copy_(self._weights)
        if self.live_ligand_intra is not None:
            self.live_ligand_intra.set_rep_scale(scale)

    def _grid_channels(self, grid_coords: torch.Tensor):
        """Per-atom ``fa_atr, fa_rep, fa_sol, fa_elec`` values and gradients.

        Returns ``(values [pose, atom, 4], gradients [pose, atom, 4, 3],
        penalty [pose, atom], penalty_gradient [pose, atom, 3])`` in Cartesian
        units, before term weights.
        """
        n_poses, n_atoms, _ = grid_coords.shape
        grids = self.grids
        indices = grids.geometry.zero_based_index_from_cartesian(
            grid_coords.reshape(-1, 3)
        ).contiguous()
        rows = self._type_table.shape[0]
        if n_poses % rows:
            raise ValueError(
                "grid tables need one row per pool pose, dividing the batch"
            )
        table_row = torch.arange(n_poses, device=self.device) % rows
        atom_type = self._type_table[table_row].reshape(-1)
        charge = self._charge_table[table_row].reshape(-1)
        keep = atom_type >= 0
        atom_type = atom_type.clamp(min=0)

        atr, d_atr, penalty, d_penalty = interpolate_with_derivatives(
            grids.fa_atr_coeffs, indices, atom_type, OUT_OF_BOUNDS_PENALTY
        )
        rep, d_rep, _, _ = interpolate_with_derivatives(
            grids.fa_rep_coeffs, indices, atom_type, 0.0
        )
        sol, d_sol, _, _ = interpolate_with_derivatives(
            grids.fa_sol_coeffs, indices, atom_type, 0.0
        )
        elec, d_elec, _, _ = interpolate_with_derivatives(
            grids.fa_elec_coeffs[None], indices, torch.zeros_like(atom_type), 0.0
        )
        scale = 1.0 / grids.geometry.voxel_spacing
        values = torch.stack((atr, ixform_rep(rep), sol, charge * elec), dim=1)
        # fa_rep is stored transformed; chain rule through its inverse.
        d_rep = d_rep * (-4.0 * rep.pow(-5))[:, None]
        gradients = torch.stack((d_atr, d_rep, d_sol, charge[:, None] * d_elec), dim=1)
        # A slot this pose's block type has no atom for scores nothing, whatever
        # its coordinate: replaced rather than scaled, since a coordinate off
        # the grid gives a non-finite fa_rep chain rule.
        zero = values.new_zeros(())
        values = torch.where(keep[:, None], values, zero)
        gradients = torch.where(keep[:, None, None], gradients, zero)
        penalty = torch.where(keep, penalty, zero)
        d_penalty = torch.where(keep[:, None], d_penalty, zero)
        return (
            values.reshape(n_poses, n_atoms, 4),
            (gradients * scale).reshape(n_poses, n_atoms, 4, 3),
            penalty.reshape(n_poses, n_atoms),
            (d_penalty * scale).reshape(n_poses, n_atoms, 3),
        )

    def set_pose_atom_state(
        self,
        grid_index: torch.Tensor | None,
        charges: torch.Tensor | None = None,
        present: torch.Tensor | None = None,
    ) -> None:
        """Give each pose its own ``[pose, atom]`` grid types, charges and mask.

        Deprotonating a donor to bind a metal retypes it, shifts its charges and
        drops a hydrogen, so a population mid-search is not one molecule. Passing
        ``None`` for ``grid_index`` restores the shared vectors.
        """
        if grid_index is None:
            self._pose_grid_index = None
            self._pose_charges = None
            self._pose_present = None
            self._set_tables(self._grid_index[None], self._charges[None])
            return
        if charges is None or present is None:
            raise ValueError("per-pose grid state needs charges and a presence mask")
        n_atoms = self._grid_index.shape[0]
        for name, tensor in (
            ("grid index", grid_index),
            ("charges", charges),
            ("presence", present),
        ):
            if tensor.dim() != 2 or tensor.shape[1] != n_atoms:
                raise ValueError(
                    f"per-pose {name} must be [pose, {n_atoms}], got "
                    f"{tuple(tensor.shape)}"
                )
        # A captured CUDA graph reads these by address, so a new population
        # writes into the same buffers. Only a changed pose count reallocates,
        # and that invalidates the capture through its own shape.
        shape = (grid_index.shape[0], n_atoms)
        if self._pose_grid_index is None or self._pose_grid_index.shape != shape:
            self._pose_grid_index = torch.empty(
                shape, dtype=torch.int64, device=self.device
            )
            self._pose_charges = torch.empty(
                shape, dtype=self._charges.dtype, device=self.device
            )
            self._pose_present = torch.empty(
                shape, dtype=torch.bool, device=self.device
            )
        self._pose_grid_index.copy_(grid_index)
        self._pose_charges.copy_(charges)
        self._pose_present.copy_(present)
        self._set_tables(
            torch.where(
                self._pose_present,
                self._pose_grid_index,
                self._pose_grid_index.new_full((), -1),
            ),
            self._pose_charges,
        )

    def set_intra_topology(
        self,
        block_types: torch.Tensor | None,
        connections: torch.Tensor | None = None,
        packed_index: torch.Tensor | None = None,
        present: torch.Tensor | None = None,
    ) -> None:
        """Give the ligand-intra modules this population's per-pose ligand
        block types, bonds and slot order. ``None`` restores one molecule."""
        if self.live_ligand_intra is None:
            return
        self.live_ligand_intra.set_pose_topology(block_types, connections)
        if block_types is None or packed_index is None:
            self._intra_packing = None
            return
        if present is None:
            raise ValueError("per-pose intra packing needs a presence mask")
        packed_index = packed_index.to(device=self.device, dtype=torch.int64)
        present = present.to(device=self.device, dtype=torch.bool)
        # Written in place: a captured graph reads these by address.
        if (
            self._intra_packing is None
            or self._intra_packing[0].shape != packed_index.shape
        ):
            self._intra_packing = (packed_index.clone(), present.clone())
        else:
            self._intra_packing[0].copy_(packed_index)
            self._intra_packing[1].copy_(present)

    def _set_tables(self, types: torch.Tensor, charges: torch.Tensor) -> None:
        """Write the grid kernel tables in place; only a row-count change
        reallocates, and that changes the batch a capture was taken for."""
        if self._type_table.shape != types.shape:
            self._type_table = types.contiguous().clone()
            self._charge_table = charges.contiguous().clone()
            self._polar_type_table = None
        else:
            self._type_table.copy_(types)
            self._charge_table.copy_(charges)
        self._refresh_polar_tables()
        if self._mps is not None:
            if self._mps["grid_index"].shape != self._type_table.shape:
                self._mps["grid_index"] = self._type_table.to("mps")
                self._mps["charges"] = self._charge_table.to("mps")
            else:
                self._mps["grid_index"].copy_(self._type_table)
                self._mps["charges"].copy_(self._charge_table)

    def _grid_coords(self, coords: torch.Tensor) -> torch.Tensor:
        if self._grid_atoms is None:
            return coords
        return coords.index_select(1, self._grid_atoms)

    def grid_channels(self, coords: torch.Tensor) -> torch.Tensor:
        """Weighted ``fa_atr, fa_rep, fa_sol, fa_elec`` per pose, without the
        out-of-bounds penalty: ``[pose, 4]``."""
        with torch.no_grad():
            values = self._grid_channels(self._grid_coords(coords.detach()))[0]
            return (values * self._weights).sum(dim=1)

    def value_and_grad(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-pose energy ``[pose]`` and gradient ``[pose, atom, 3]``, graph-free.

        ``coords`` are the movable ligand (and sidechain) atoms, ``[pose, atom, 3]``.
        """
        if coords.dtype != self._charges.dtype or coords.device != self.device:
            raise TypeError(
                f"scorer works on {self.device} {self._charges.dtype} coordinates"
            )
        with torch.no_grad():
            coords = coords.detach().contiguous()
            grids = self.grids
            if self._mps is not None:
                # Start the grid kernel on the GPU, score the live terms on the
                # host while it runs, then collect it.
                from tmol.score._mps_offload import _commit_mps

                m = self._mps
                pending = grid_score_atoms(
                    coords.to(m["atr"].device),
                    m["grid_atom_index"],
                    m["grid_index"],
                    m["charges"],
                    m["atr"],
                    m["rep"],
                    m["sol"],
                    m["elec"],
                    m["lk"],
                    m["origin"],
                    grids.geometry.voxel_spacing,
                    m["weights"],
                    OUT_OF_BOUNDS_PENALTY,
                )
                _commit_mps()
                live = self._live_terms(coords)
                atom_energies, grad = (t.cpu() for t in pending)
            else:
                atom_energies, grad = grid_score_atoms(
                    coords,
                    self._grid_atom_index,
                    self._type_table,
                    self._charge_table,
                    grids.fa_atr_coeffs,
                    grids.fa_rep_coeffs,
                    grids.fa_sol_coeffs,
                    grids.fa_elec_coeffs[None],
                    self._lk_heavy_coeffs,
                    grids.geometry.origin,
                    grids.geometry.voxel_spacing,
                    self._weights,
                    OUT_OF_BOUNDS_PENALTY,
                )
                live = None
            energies = atom_energies.sum(dim=1)
            if (
                self._lk_heavy_coeffs.shape[0]
                and self.live_lk_ball.ligand_polar_atoms.numel()
            ):
                grid_score_waters(
                    coords,
                    *self._lk_tables,
                    self._polar_type_table,
                    grids.lk_ball_water_coeffs,
                    grids.lk_bridge_coeffs,
                    grids.geometry.origin,
                    grids.geometry.voxel_spacing,
                    GRID_LK_BALL_WATER_DISTANCE,
                    GRID_LK_BALL_FADE,
                    grids.lk_bridge_enabled,
                    energies,
                    grad,
                )

            if live is None:
                live = self._live_terms(coords)
            for e, g in live:
                energies, grad = energies + e, grad + g
        return energies, grad

    def _intra_term(self, coords: torch.Tensor):
        """Ligand-intra energy and gradient, in movable-slot order.

        A pose whose ligand bound a metal may hold a block type with fewer
        atoms, which numbers the rest differently; the intra modules read a
        block's atoms at its own local indices, so the slots are gathered into
        that order and the gradient is taken back out of it. Index arithmetic
        only, so a captured graph keeps following it.
        """
        data = self.live_ligand_intra
        if self._intra_packing is None:
            return score_ligand_intra_with_gradients(data, coords)
        index, present = self._intra_packing
        n_poses = coords.shape[0]
        if n_poses % index.shape[0]:
            raise ValueError(
                "per-pose intra packing needs one row per pool pose, "
                "dividing the batch"
            )
        table_row = torch.arange(n_poses, device=coords.device) % index.shape[0]
        index, present = index[table_row], present[table_row]
        rows = torch.arange(n_poses, device=coords.device)[:, None]
        rows = rows.expand_as(index)
        safe = torch.where(present, index, index.new_zeros(()))
        packed = coords.new_zeros(coords.shape).index_put(
            (rows, safe),
            torch.where(present[..., None], coords, coords.new_zeros(())),
            accumulate=True,
        )
        energies, gradients = score_ligand_intra_with_gradients(data, packed)
        spread = gradients.gather(1, safe[..., None].expand(-1, -1, 3))
        return energies, torch.where(
            present[..., None], spread, gradients.new_zeros(())
        )

    def _live_terms(self, coords: torch.Tensor) -> list:
        """Weighted (energy, gradient) of each live term, in a fixed order."""
        terms = []
        if self.live_hbond is not None:
            e, g = score_ligand_hbond_with_gradients(self.live_hbond, coords)
            terms.append((self._hbond_weight * e, self._hbond_weight * g))
        if not self.lk_ball_grid and self.live_lk_ball is not None:
            e, g = score_ligand_lk_ball_with_gradients(self.live_lk_ball, coords)
            weights = self._lk_ball_weights
            terms.append(
                (
                    (e * weights).sum(dim=1),
                    (g * weights[None, :, None, None]).sum(dim=1),
                )
            )
        if self.live_ligand_intra is not None:
            terms.append(self._intra_term(coords))
        return terms

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        """Per-pose energy, differentiable with respect to ``coords``."""
        if torch.is_grad_enabled() and coords.requires_grad:
            return _Score.apply(coords, self)
        return self.value_and_grad(coords)[0]
