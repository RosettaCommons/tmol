"""Small execution adapters for the common canonical-protein scoring contract."""

import numpy as np
import pytest


class Execution:
    def __init__(self, name):
        self.framework, device = name.split("-")
        self.name = name
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
            self.device = torch.device(device)
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
            self.scorer = self.score_function.render_whole_pose_scoring_module(
                self.pose
            )
        else:
            from tmol_jax import beta2016_score_function, pose_stack_from_pdb

            with self.module.default_device(self.device):
                self.pose = pose_stack_from_pdb(path)
                self.score_function = beta2016_score_function()
        return self

    def coords(self):
        return self.numpy(self.pose.coords)

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

    def value_and_grad(self, coords):
        if self.framework == "torch":
            x = self.module.tensor(coords, device=self.device, requires_grad=True)
            values = self.scorer(x)
            (gradient,) = self.module.autograd.grad(values.sum(), x)
            return self.numpy(values), self.numpy(gradient)
        with self.module.default_device(self.device):
            x = self.module.device_put(coords, self.device)
            values, gradient = self.score_function.jitted_value_and_grad(self.pose)(x)
            return self.numpy(values), self.numpy(gradient)
