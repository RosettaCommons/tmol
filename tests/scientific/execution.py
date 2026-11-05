"""Small execution adapters for the common molecular scoring contract."""

import os
from unittest.mock import patch

import numpy as np
import pytest


class Execution:
    def __init__(self, name):
        self.framework, device = name.split("-")
        self.name = name
        self.requested_device = device
        self.capabilities = {"score", "coordinate_gradient"}
        if self.framework == "torch":
            self.module = pytest.importorskip("torch")
            torch = self.module
            available = (
                device == "cpu"
                or (device == "cuda" and torch.cuda.is_available())
                or (device == "mps" and torch.backends.mps.is_available())
            )
            if not available:
                pytest.skip(f"{name} device unavailable")
            if device == "mps":
                from tmol._cpp_lib import has_mps

                if not has_mps():
                    raise RuntimeError(
                        "torch-mps lane requires a native TMol Metal build"
                    )
            # Apple execution is the supported CPU-pose/Metal-offload contract.
            self.device = torch.device("cpu" if device == "mps" else device)
        elif self.framework == "jax":
            self.module = pytest.importorskip("jax")
            try:
                self.device = self.module.devices(device)[0]
            except RuntimeError:
                pytest.skip(f"{name} device unavailable")
            self.capabilities |= {"jit", "vmap", "reverse_jacobian"}
        else:
            raise ValueError(f"unknown scientific test backend: {name}")

    def load(self, path):
        if self.framework == "torch":
            from tmol import beta2016_score_function, pose_stack_from_file

            self.pose = pose_stack_from_file(path, self.device)
            self.score_function = beta2016_score_function(self.device)
            self._render_torch()
        else:
            from tmol_jax import beta2016_score_function, pose_stack_from_pdb

            with self.module.default_device(self.device):
                self.pose = pose_stack_from_pdb(path)
                self.score_function = beta2016_score_function()
            self.pose = self.pose.to(self.device)
        return self

    def _render_torch(self):
        from tmol.score._mps_offload import WHOLE_POSE_SCORE_TYPES

        # Pure CPU and Apple overlay are separate lanes even on a Mac. The
        # environment is read during render; each scorer retains its own plan.
        with patch.dict(
            os.environ,
            {"TMOL_MPS_OFFLOAD": "1" if self.requested_device == "mps" else "0"},
        ):
            self.scorer = self.score_function.render_whole_pose_scoring_module(
                self.pose
            )
        plan = getattr(self.scorer, "plan", None)
        active = self.score_function.all_score_types()
        gpu = (
            set(st for term in plan.gpu_terms for st in term.score_types())
            if plan
            else set()
        )
        self.execution_routes = {
            st.name: "mps" if st in gpu else self.device.type for st in active
        }
        if self.requested_device == "mps":
            expected = set(active) & WHOLE_POSE_SCORE_TYPES
            if expected != gpu:
                raise AssertionError(
                    f"requested Metal terms were not offloaded: {expected - gpu}"
                )

    def coords(self):
        return self.numpy(self.pose.coords)

    def molecule(self, case, terms=("gen_torsions",), parameter_database=None):
        if self.framework == "torch":
            from tmol import (
                ParameterDatabase,
                PackedBlockTypes,
                PoseStackBuilder,
                ScoreFunction,
                ScoreType,
            )
            from tmol.chemical._restypes import ResidueTypeSet

            db = (
                parameter_database
                if parameter_database is not None
                else case.database(ParameterDatabase.get_default())
            )
            restypes = ResidueTypeSet.from_database(db.chemical)
            by_name = {rt.name: rt for rt in restypes.residue_types}
            names = dict.fromkeys(name for row in case.sequences for name in row)
            pbt = PackedBlockTypes.from_restype_list(
                db.chemical, restypes, [by_name[name] for name in names], self.device
            )
            sequences = [list(row) for row in case.sequences]
            for row, edges in zip(sequences, case.extra_connections()):
                for index, (left, left_port, right, right_port) in enumerate(edges):
                    row[left] += f"--{left_port}-edge{index}"
                    row[right] += f"--{right_port}-edge{index}"
            self.pose = PoseStackBuilder.from_block_type_names(
                pbt,
                sequences,
                [list(row) for row in case.chain_lengths],
            )
            coordinates = np.zeros(self.pose.coords.shape, np.float32)
            for p, row in enumerate(case.blocks(db)):
                xyz = np.concatenate([r[2] for r in row])
                coordinates[p, : len(xyz)] = xyz
            self.pose.coords = self.module.as_tensor(coordinates, device=self.device)
            self.score_function = ScoreFunction(db, self.device)
        else:
            from tmol_jax.database import ParameterDatabase
            from tmol_jax import (
                ResidueBlock,
                pose_stack_from_residue_blocks,
                ScoreFunction,
                ScoreType,
            )

            db = (
                parameter_database
                if parameter_database is not None
                else case.database(ParameterDatabase.get_default())
            )
            with self.module.default_device(self.device):
                self.pose = pose_stack_from_residue_blocks(
                    [[ResidueBlock(*b) for b in row] for row in case.blocks(db)],
                    parameter_database=db,
                    connections=case.connections(),
                )
            # Location alone does not commit a JAX array: otherwise a later jit
            # can choose the process's default GPU for this explicit CPU lane.
            self.pose = self.pose.to(self.device)
            self.score_function = ScoreFunction(db)
        self.parameter_database = db
        for term in terms:
            self.score_function.set_weight(ScoreType[term], 1.0)
        if self.framework == "torch":
            self._render_torch()
        return self

    def palette_choices(self, *, noncanonical_design=False):
        if self.framework == "torch":
            from tmol.pack._packer_task import PackerPalette

            indices = self.pose.block_type_ind64
        else:
            from tmol_jax.pack import PackerPalette

            indices = self.pose.block_type_ind
        palette = PackerPalette(noncanonical_design=noncanonical_design)
        pbt = self.pose.packed_block_types
        counts, choices, original = palette.block_types_from_original(pbt, indices)
        repack = palette.create_restrict_to_repacking_mask(pbt, indices)
        return tuple(self.numpy(a) for a in (counts, choices, original, repack))

    def set_options(self, options):
        self.score_function.set_options(options)
        if self.framework == "torch":
            self._render_torch()

    def numpy(self, value):
        if self.framework == "torch":
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def scores(self, coords, *, sum_terms=True, apply_weights=True):
        if self.framework == "torch":
            x = self.module.as_tensor(coords, device=self.device)
            return self.numpy(
                self.scorer(x, sum_terms=sum_terms, apply_weights=apply_weights)
            )
        with self.module.default_device(self.device):
            x = self.module.device_put(coords, self.device)
            score = self.score_function.render_whole_pose_scoring_module(
                self.pose, sum_terms=sum_terms, apply_weights=apply_weights
            )
            return self.numpy(self.module.jit(score)(x))

    def float64_directional_derivative(self, coords, direction, step=1e-4):
        """Central energy difference at the exact input coordinates/direction.

        Metal analytic gradients stay float32. Its numerical oracle uses the
        matching native CPU float64 scorer because Metal has no float64 path.
        Other lanes evaluate the oracle on their requested device.
        """
        x = np.asarray(coords, dtype=np.float64)
        delta = step * np.asarray(direction, dtype=np.float64)

        def difference(score):
            return (score(x + delta) - score(x - delta)) / (2 * step)

        def score(value):
            return self.scores(value, sum_terms=False, apply_weights=False)

        if self.framework == "jax":
            with self.module.enable_x64():
                return difference(score)
        if self.requested_device == "mps":
            with patch.dict(os.environ, {"TMOL_MPS_OFFLOAD": "0"}):
                reference = self.score_function.render_whole_pose_scoring_module(
                    self.pose
                )

            def cpu_score(value):
                return self.numpy(
                    reference(
                        self.module.as_tensor(value, device="cpu"),
                        sum_terms=False,
                        apply_weights=False,
                    )
                )

            return difference(cpu_score)
        return difference(score)

    def value_and_grad(self, coords, *, coordinate_independent=False):
        if self.framework == "torch":
            x = self.module.tensor(coords, device=self.device, requires_grad=True)
            values = self.scorer(x)
            (gradient,) = self.module.autograd.grad(
                values.sum(), x, materialize_grads=coordinate_independent
            )
            return self.numpy(values), self.numpy(gradient)
        with self.module.default_device(self.device):
            x = self.module.device_put(coords, self.device)
            values, gradient = self.score_function.jitted_value_and_grad(self.pose)(x)
            return self.numpy(values), self.numpy(gradient)

    def channel_values_and_gradients(self, coords):
        """Unweighted channels and the derivative of each channel's pose sum."""
        if self.framework == "torch":
            x = self.module.tensor(coords, device=self.device, requires_grad=True)
            values = self.scorer(x, sum_terms=False, apply_weights=False)
            gradients = self.module.stack(
                [
                    self.module.autograd.grad(value.sum(), x, retain_graph=True)[0]
                    for value in values
                ]
            )
        else:
            with self.module.default_device(self.device):
                x = self.module.device_put(coords, self.device)
                score = self.score_function.render_whole_pose_scoring_module(
                    self.pose, sum_terms=False, apply_weights=False
                )
                values = self.module.jit(score)(x)
                gradients = self.module.jit(
                    self.module.jacrev(lambda x: score(x).sum(axis=1))
                )(x)
        return self.numpy(values), self.numpy(gradients)

    def minimize(self, move_mask, iterations=5):
        """Exercise each framework's public Cartesian minimizer on the same atoms."""
        if self.framework == "torch":
            from tmol import run_cart_min

            with patch.dict(
                os.environ,
                {"TMOL_MPS_OFFLOAD": "1" if self.requested_device == "mps" else "0"},
            ):
                pose = run_cart_min(
                    self.pose,
                    self.score_function,
                    coord_mask=self.module.as_tensor(move_mask, device=self.device),
                    optimizer_kwargs={"max_iter": iterations},
                )
        else:
            from tmol_jax import run_cart_min

            with self.module.default_device(self.device):
                pose = run_cart_min(
                    self.pose, self.score_function, move_mask, max_iter=iterations
                )
        return self.numpy(pose.coords)
