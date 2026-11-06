"""Small execution adapters for common sparse conformer science checks."""

import numpy as np


class RotamerExecution:
    def __init__(self, backend, *, electrostatic=False, sampled=False, family=None):
        if backend.requested_device == "mps":
            raise NotImplementedError(
                "the sparse float64 science adapter requires CPU/CUDA; Metal needs a separate float32 validation"
            )
        self.backend = backend
        self.family = family or ("elec" if electrostatic else "ljlk")
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
            from tmol_jax.pack.dunbrack_scoring import prepare_rotamer_dunbrack
            from tmol_jax.pack.lk_ball_scoring import prepare_rotamer_lk_ball
            from tmol_jax.pack.bonded_scoring import (
                prepare_rotamer_genbonded,
                prepare_rotamer_cartbonded,
            )
            from tmol_jax.pack.hbond_scoring import prepare_rotamer_hbond
            from tmol_jax.pack.na_scoring import prepare_rotamer_na_torsion
            from tmol_jax.pack.constraint_scoring import prepare_rotamer_constraints
            from tmol_jax.pack.backbone_scoring import prepare_rotamer_backbone
            from tmol_jax.pack.connection_scoring import (
                prepare_rotamer_disulfide,
                prepare_rotamer_metal,
            )

            with backend.module.default_device(backend.device):
                arguments = (backend.pose, self.rotamers, backend.parameter_database)
                if self.family in (
                    "dunbrack",
                    "lk_ball",
                    "genbonded",
                    "hbond",
                    "cartbonded",
                    "na_torsion",
                    "backbone",
                    "disulfide",
                    "metal",
                    "constraint",
                ):
                    prepare = {
                        "dunbrack": prepare_rotamer_dunbrack,
                        "lk_ball": prepare_rotamer_lk_ball,
                        "genbonded": prepare_rotamer_genbonded,
                        "cartbonded": prepare_rotamer_cartbonded,
                        "hbond": prepare_rotamer_hbond,
                        "na_torsion": prepare_rotamer_na_torsion,
                        "backbone": prepare_rotamer_backbone,
                        "disulfide": prepare_rotamer_disulfide,
                        "metal": prepare_rotamer_metal,
                        "constraint": prepare_rotamer_constraints,
                    }[self.family]
                    self.state = prepare(*arguments)
                else:
                    self.state = prepare_rotamer_nonbonded(
                        *arguments, electrostatic=self.family == "elec"
                    )
        else:
            from tmol.score import (
                LJLKEnergyTerm,
                ElecEnergyTerm,
                DunbrackEnergyTerm,
                LKBallEnergyTerm,
                GenBondedEnergyTerm,
                CartBondedEnergyTerm,
                HBondEnergyTerm,
                NaTorsionEnergyTerm,
                BackboneTorsionEnergyTerm,
                DisulfideEnergyTerm,
                ConstraintEnergyTerm,
            )

            from tmol.score.metal._metal_coordination_term import (
                MetalCoordinationEnergyTerm,
            )

            cls = {
                "elec": ElecEnergyTerm,
                "ljlk": LJLKEnergyTerm,
                "dunbrack": DunbrackEnergyTerm,
                "lk_ball": LKBallEnergyTerm,
                "genbonded": GenBondedEnergyTerm,
                "cartbonded": CartBondedEnergyTerm,
                "hbond": HBondEnergyTerm,
                "na_torsion": NaTorsionEnergyTerm,
                "backbone": BackboneTorsionEnergyTerm,
                "disulfide": DisulfideEnergyTerm,
                "metal": MetalCoordinationEnergyTerm,
                "constraint": ConstraintEnergyTerm,
            }[self.family]
            term = cls(param_db=backend.parameter_database, device=backend.device)
            for block in backend.pose.packed_block_types.active_block_types:
                term.setup_block_type(block)
            term.setup_packed_block_types(backend.pose.packed_block_types)
            term.setup_poses(backend.pose)
            self.term = term
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
        if self.family in ("genbonded", "cartbonded", "backbone"):
            from tmol_jax.pack.bonded_scoring import sparse_bonded_scores

            return sparse_bonded_scores(x, self.state)
        if self.family == "constraint":
            from tmol_jax.pack.constraint_scoring import sparse_constraint_scores

            return sparse_constraint_scores(x, self.state)
        if self.family in ("disulfide", "metal"):
            from tmol_jax.pack.connection_scoring import sparse_connection_scores

            return sparse_connection_scores(x, self.state)
        if self.family == "na_torsion":
            from tmol_jax.pack.na_scoring import sparse_na_torsion_scores

            return sparse_na_torsion_scores(x, self.state)
        if self.family == "dunbrack":
            from tmol_jax.pack.dunbrack_scoring import sparse_dunbrack_scores

            return sparse_dunbrack_scores(x, self.state)
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
        if self.family in ("lk_ball", "hbond"):
            from tmol_jax.pack.lk_ball_scoring import sparse_lk_ball_values
            from tmol_jax.pack.hbond_scoring import sparse_hbond_values

            evaluate = (
                sparse_lk_ball_values
                if self.family == "lk_ball"
                else sparse_hbond_values
            )
            return evaluate(x, self.state, indices), indices
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
            if self.family in ("genbonded", "cartbonded", "backbone"):
                from tmol_jax.pack.bonded_scoring import sparse_bonded_scores

                values = sparse_bonded_scores(x, self.state)[0]
            elif self.family == "constraint":
                from tmol_jax.pack.constraint_scoring import sparse_constraint_scores

                values = sparse_constraint_scores(x, self.state)[0]
            elif self.family in ("disulfide", "metal"):
                from tmol_jax.pack.connection_scoring import sparse_connection_scores

                values = sparse_connection_scores(x, self.state)[0]
            elif self.family == "na_torsion":
                from tmol_jax.pack.na_scoring import sparse_na_torsion_scores

                values = sparse_na_torsion_scores(x, self.state)[0]
            elif self.family == "dunbrack":
                from tmol_jax.pack.dunbrack_scoring import sparse_dunbrack_scores

                values = sparse_dunbrack_scores(x, self.state)[0]
            elif self.family in ("lk_ball", "hbond"):
                from tmol_jax.pack.lk_ball_scoring import sparse_lk_ball_values
                from tmol_jax.pack.hbond_scoring import sparse_hbond_values

                evaluate = (
                    sparse_lk_ball_values
                    if self.family == "lk_ball"
                    else sparse_hbond_values
                )
                values = evaluate(x, self.state, indices)
            else:
                values = sparse_nonbonded_values(x, self.state, indices)
            return (values * self.array(weights)).sum(), values

        (_, values), gradient = b.module.jit(
            b.module.value_and_grad(weighted, has_aux=True)
        )(self.array(coords))
        return np.asarray(values), np.asarray(gradient), np.asarray(indices)

    def constraint_values(self, coords, parameters):
        """Evaluate new floating parameters without rebuilding the molecule."""
        if self.backend.framework == "jax":
            from tmol_jax.pack.constraint_scoring import sparse_constraint_scores

            return sparse_constraint_scores(coords, self.state, parameters)[0]
        import attrs

        pose = self.backend.pose
        original = pose.constraint_set
        try:
            pose.constraint_set = attrs.evolve(original, constraint_params=parameters)
            scorer = self.term.render_rotamer_scoring_module(pose, self.rotamers)
            return scorer(coords)[0]
        finally:
            pose.constraint_set = original
