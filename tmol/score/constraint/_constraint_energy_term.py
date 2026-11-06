import torch
from tmol_core.scoring import constraints as constraint_physics

from .._energy_term import EnergyTerm

from tmol.database import ParameterDatabase
from tmol.chemical import RefinedResidueType
from tmol.pose._packed_block_types import PackedBlockTypes
from tmol.pose._pose_stack import PoseStack


def _torsion(atoms):
    from tmol.score.constraint.potentials._compiled import _get_torsion_angle

    return _get_torsion_angle(atoms)


_constraint_ops = constraint_physics.ConstraintOps(
    norm=lambda vectors: torch.linalg.norm(vectors, dim=-1),
    where=torch.where,
    trunc=torch.trunc,
    sign=torch.sign,
    torsion=_torsion,
)


class ConstraintEnergyTerm(EnergyTerm):
    """Score coordinate and geometric constraints attached to a pose stack."""

    device: torch.device  # = attr.ib()

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        super(ConstraintEnergyTerm, self).__init__(param_db=param_db, device=device)

        self.device = device

    @classmethod
    def class_name(cls):
        return "Constraint"

    @classmethod
    def score_types(cls):
        import tmol.score.terms._constraint_creator

        return tmol.score.terms._constraint_creator.ConstraintTermCreator.score_types()

    def n_bodies(self):
        return 1

    @classmethod
    def get_torsion_angle_test(cls, tensor):
        from tmol.score.constraint.potentials._compiled import _get_torsion_angle

        return _get_torsion_angle(tensor)

    @classmethod
    def harmonic(cls, atoms, params):
        return constraint_physics.harmonic(atoms, params, _constraint_ops)

    @classmethod
    def harmonic_coordinate(cls, atoms, params):
        return constraint_physics.harmonic_coordinate(atoms, params, _constraint_ops)

    @classmethod
    def bounded(cls, atoms, params):
        return constraint_physics.bounded(atoms, params, _constraint_ops)

    @classmethod
    def circularharmonic(cls, atoms, params):
        return constraint_physics.circularharmonic(atoms, params, _constraint_ops)

    def setup_block_type(self, block_type: RefinedResidueType):
        super(ConstraintEnergyTerm, self).setup_block_type(block_type)

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(ConstraintEnergyTerm, self).setup_packed_block_types(packed_block_types)

    def setup_poses(self, poses: PoseStack):
        super(ConstraintEnergyTerm, self).setup_poses(poses)

    def get_pose_score_term_function(self):
        return self.constraint_pose_scores

    def get_rotamer_score_term_function(self):
        return self.constraint_rotamer_scores

    def constraint_rotamer_scores(self, *args):
        """Score rotamers with the same routine the whole pose uses.

        The rotamer arg list carries one extra tensor -- which blocks sample in
        lockstep -- that only the pair terms' rotamer-pair enumerator reads.
        """
        return self.constraint_pose_scores(
            *args[:12], *args[13:], _sparse_rotamers=True
        )

    def get_score_term_attributes(self, pose_stack: PoseStack):
        return [
            pose_stack.get_constraint_set(),
        ]

    def constraint_pose_scores(  # noqa: C901
        self,
        coords,
        rot_coord_offset,  # rot coord offset
        pose_ind_for_atom,  # pose_ind_for_atom?? unused
        rot_offset_for_block,  # first rot for block
        first_rot_block_type,  # first rot block type
        block_ind_for_rot,
        pose_for_rot,
        block_type_ind_for_rot,
        n_rots_for_pose,
        rot_offset_for_pose,
        n_rots_for_block,
        rot_offset_for_block______2,  # three times?!
        max_n_rots_per_pose,
        constraint_set,
        output_block_pair_energies=False,
        *,
        _sparse_rotamers=False,
    ):
        device = coords.device

        # Early exit if we have no constraints
        if constraint_set is None:
            if _sparse_rotamers:
                return (
                    torch.zeros((1, 0), dtype=coords.dtype, device=device),
                    torch.zeros((3, 0), dtype=torch.int32, device=device),
                )
            n_poses = first_rot_block_type.shape[0]
            if not output_block_pair_energies:
                # Whole-pose scoring never represents multiple rotamers per
                # block. Handle it before inspecting the CUDA tensor so an
                # empty constraint set does not synchronize every evaluation.
                return (
                    torch.zeros((1, n_poses), dtype=coords.dtype, device=device),
                    torch.zeros((3, 0), dtype=torch.int32, device=device),
                )

            # Block-pair and rotamer scoring share this entry point. Only this
            # infrequent path needs to distinguish their output conventions.
            max_n_rots_per_block = n_rots_for_block.max().item()
            if max_n_rots_per_block > 1:
                return (
                    torch.zeros((1, 0), dtype=coords.dtype, device=device),
                    torch.zeros((3, 0), dtype=torch.int32, device=device),
                )

            max_n_blocks = first_rot_block_type.shape[1]
            return (
                torch.zeros(
                    (1, n_poses, max_n_blocks, max_n_blocks),
                    dtype=coords.dtype,
                    device=device,
                ),
                torch.zeros((3, 0), dtype=torch.int32, device=device),
            )

        # Whole-pose scoring has one conformer per block; its attribution does
        # not require a device reduction to detect sparse expansion.
        has_rotamers = _sparse_rotamers or (
            output_block_pair_energies and bool((n_rots_for_block.max() > 1).item())
        )
        from .potentials._compiled import _expand_constraint_candidates

        execution_device = (
            device if device.type in ("cpu", "cuda") else torch.device("cpu")
        )
        inputs = (
            constraint_set.constraint_atoms,
            constraint_set.constraint_num_unique_blocks,
            constraint_set.constraint_unique_blocks,
            rot_coord_offset,
            n_rots_for_block,
            rot_offset_for_block,
        )
        rotamerized_atoms, source_rows, rotamerized_inds = (
            value.to(device)
            for value in _expand_constraint_candidates(
                *(value.to(execution_device) for value in inputs), has_rotamers
            )
        )
        # Keep differentiable custom parameters on their original device.
        rotamerized_fn_inds = constraint_set.constraint_function_inds[source_rows]
        rotamerized_params = constraint_set.constraint_params[source_rows]

        def score_cnstrs(functions, types, atom_coords, params):
            cnstr_scores = torch.full(
                (len(types),), 0, dtype=coords.dtype, device=coords.device
            )
            for ind, fn in enumerate(functions):

                # get the constraints that match this constraint type
                c_inds = torch.nonzero(types == ind)

                # compute the scores for this function
                fn_coords = atom_coords[c_inds].squeeze(1)
                result = fn(fn_coords, params[c_inds].squeeze(1))
                result = result.unsqueeze(1)

                # add the result to the output tensor
                cnstr_scores[c_inds] += result

            return cnstr_scores

        # extract coordinates from the computed global atom indices
        atom_coords = coords[rotamerized_atoms.flatten()].view(
            rotamerized_atoms.size(0), rotamerized_atoms.size(1), 3
        )

        # score the constraints
        scores = score_cnstrs(
            constraint_set.constraint_functions,
            rotamerized_fn_inds,
            atom_coords,
            rotamerized_params,
        )

        if not has_rotamers:
            max_n_blocks = first_rot_block_type.shape[1]
            n_poses = first_rot_block_type.shape[0]
            sparse_scores = torch.sparse_coo_tensor(
                rotamerized_inds.transpose(0, 1),
                scores,
                size=(n_poses, max_n_blocks, max_n_blocks),
            )

            scores = sparse_scores.to_dense()
            if not output_block_pair_energies:
                scores = scores.sum(dim=(1, 2))

        return scores.unsqueeze(0), rotamerized_inds.transpose(0, 1)
