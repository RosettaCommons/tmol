"""Small execution adapters for common sparse conformer science checks."""

import numpy as np


class RotamerExecution:
    def __init__(self, backend, *, electrostatic=False, sampled=False):
        if backend.requested_device == "mps":
            raise NotImplementedError(
                "the sparse float64 science adapter requires CPU/CUDA; Metal needs a separate float32 validation"
            )
        self.backend = backend
        self.packed_block_types = backend.pose.packed_block_types
        if backend.framework == "jax":
            from tmol_jax.pack import (
                PackerTask,
                PackerPalette,
                SetPackerTask,
                IncludeCurrentSampler,
                build_rotamers,
            )
        else:
            from tmol import PackerTask, PackerPalette
            from tmol.pack import SetPackerTask, IncludeCurrentSampler, build_rotamers
        task = PackerTask(backend.pose, PackerPalette())
        task.restrict_to_repacking()
        if sampled:
            from .test_molecular_sampling import sampling_task

            task, _, _ = sampling_task(backend)
        task.add_conformer_sampler(IncludeCurrentSampler())
        _, self.rotamers = build_rotamers(
            backend.pose,
            SetPackerTask.from_packer_task(task),
            backend.parameter_database.chemical,
        )
        if backend.framework == "jax":
            from tmol_jax.pack.scoring import prepare_rotamer_nonbonded

            with backend.module.default_device(backend.device):
                self.state = prepare_rotamer_nonbonded(
                    backend.pose,
                    self.rotamers,
                    backend.parameter_database,
                    electrostatic=electrostatic,
                )
        else:
            from tmol.score import LJLKEnergyTerm, ElecEnergyTerm

            cls = ElecEnergyTerm if electrostatic else LJLKEnergyTerm
            term = cls(param_db=backend.parameter_database, device=backend.device)
            for block in backend.pose.packed_block_types.active_block_types:
                term.setup_block_type(block)
            term.setup_packed_block_types(backend.pose.packed_block_types)
            term.setup_poses(backend.pose)
            self.scorer = term.render_rotamer_scoring_module(
                backend.pose, self.rotamers
            )

    def array(self, value):
        b = self.backend
        if b.framework == "jax":
            return b.module.device_put(value, b.device)
        return b.module.as_tensor(value, device=b.device)

    def scores(self, coords):
        x = self.array(coords)
        if self.backend.framework == "torch":
            return self.scorer(x)
        from tmol_jax.pack.scoring import (
            nonbonded_candidate_pairs,
            sparse_nonbonded_values,
        )

        indices = nonbonded_candidate_pairs(
            self.rotamers,
            self.packed_block_types,
            reach=self.state[3],
            coords=x,
        )
        return sparse_nonbonded_values(x, self.state, indices), indices

    def value_and_grad(self, coords, weights):
        b = self.backend
        if b.framework == "torch":
            x = self.array(coords).detach().requires_grad_(True)
            values, indices = self.scorer(x)
            gradient = b.module.autograd.grad((values * self.array(weights)).sum(), x)[
                0
            ]
            return b.numpy(values), b.numpy(gradient), b.numpy(indices)
        from tmol_jax.pack.scoring import sparse_nonbonded_values

        _, indices = self.scores(coords)

        def weighted(x):
            values = sparse_nonbonded_values(x, self.state, indices)
            return (values * self.array(weights)).sum(), values

        (_, values), gradient = b.module.jit(
            b.module.value_and_grad(weighted, has_aux=True)
        )(self.array(coords))
        return np.asarray(values), np.asarray(gradient), np.asarray(indices)
