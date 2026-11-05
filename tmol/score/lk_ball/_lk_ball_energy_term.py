from tmol_core.scoring.lk_ball import (
    LK_BALL_TYPE_FIELDS,
    native_lk_ball_type_parameters,
    native_lk_ball_global_parameters,
    native_water_gen_global_parameters,
    prepare_lk_ball_block,
    pack_lk_ball_blocks,
)
import torch

from ._params import LKBallBlockTypeParams, LKBallPackedBlockTypesParams

from .._annotation_cache import AnnotationKey, _cached_annotation, _store_annotation
from .._atom_type_dependent_term import AtomTypeDependentTerm
from ..hbond._hbond_dependent_term import HBondDependentTerm
from ..ljlk._params import LJLKGlobalParams, LJLKParamResolver
from tmol.database import ParameterDatabase

from tmol.chemical import RefinedResidueType
from tmol.pose._packed_block_types import PackedBlockTypes
from tmol.pose._pose_stack import PoseStack
from tmol.score.common._convert_float64 import _convert_float64


class LKBallEnergyTerm(AtomTypeDependentTerm, HBondDependentTerm):
    tile_size: int = HBondDependentTerm.tile_size
    ljlk_global_params: LJLKGlobalParams
    ljlk_param_resolver: LJLKParamResolver

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        super(LKBallEnergyTerm, self).__init__(param_db=param_db, device=device)

        self.ljlk_param_resolver = LJLKParamResolver.from_database(
            param_db.chemical, param_db.scoring.ljlk, device=device
        )
        type_params = self.ljlk_param_resolver.type_params.to(torch.device("cpu"))
        self._lk_ball_type_params = native_lk_ball_type_parameters(
            {name: getattr(type_params, name).numpy() for name in LK_BALL_TYPE_FIELDS}
        )
        self.tile_size = LKBallEnergyTerm.tile_size
        self._lk_ball_block_key = AnnotationKey.from_sources(
            param_db.chemical,
            param_db.scoring.hbond,
            param_db.scoring.ljlk,
            settings=(self.tile_size,),
        )
        self._lk_ball_packed_key = AnnotationKey.from_sources(
            param_db.chemical,
            param_db.scoring.hbond,
            param_db.scoring.ljlk,
            settings=(
                self.ljlk_param_resolver.type_params.lj_radius.device,
                self.tile_size,
            ),
        )

        # Precompute the stacked lk-ball global-parameter tensors
        # These depend only on static values from ljlk_param_resolver globals,
        #   so they can be built once rather than on every forward
        gp = param_db.scoring.ljlk.global_parameters
        self._max_dis = float(gp.max_dis)
        self._lk_ball_global_params = torch.from_numpy(
            native_lk_ball_global_parameters(gp)
        ).to(device)
        self._lk_ball_water_gen_global_params = torch.from_numpy(
            native_water_gen_global_parameters(gp)
        ).to(device)

    @classmethod
    def class_name(cls):
        return "LKBall"

    @classmethod
    def score_types(cls):
        import tmol.score.terms._lk_ball_creator

        return tmol.score.terms._lk_ball_creator.LKBallTermCreator.score_types()

    def n_bodies(self):
        return 2

    def setup_block_type(self, block_type: RefinedResidueType):
        self.ljlk_param_resolver.validate_block_type(block_type)
        atom_params = super(LKBallEnergyTerm, self).setup_block_type(block_type)
        cached = _cached_annotation(
            block_type, "_lk_ball_annotation", self._lk_ball_block_key
        )
        if cached is not None:
            return cached

        hbbt_params = HBondDependentTerm.setup_block_type(self, block_type)
        prepared = prepare_lk_ball_block(
            block_type,
            hbbt_params,
            atom_params[0],
            self._lk_ball_type_params,
            self.tile_size,
        )
        bt_lk_ball_params = LKBallBlockTypeParams(**vars(prepared))
        setattr(block_type, "lk_ball_params", bt_lk_ball_params)
        return _store_annotation(
            block_type,
            "_lk_ball_annotation",
            self._lk_ball_block_key,
            bt_lk_ball_params,
            fields=("lk_ball_params",),
        )

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(LKBallEnergyTerm, self).setup_packed_block_types(packed_block_types)
        cached = _cached_annotation(
            packed_block_types, "_lk_ball_annotation", self._lk_ball_packed_key
        )
        if cached is not None:
            return cached
        blocks = [
            self.setup_block_type(bt) for bt in packed_block_types.active_block_types
        ]
        prepared = pack_lk_ball_blocks(
            blocks,
            packed_block_types.hbpbt_params.tile_donH_inds.shape[1],
            self.tile_size,
        )

        def _t(t):
            return torch.tensor(t, device=packed_block_types.device)

        lk_ball_params = LKBallPackedBlockTypesParams(
            **{name: _t(array) for name, array in prepared.items()}
        )
        setattr(packed_block_types, "lk_ball_params", lk_ball_params)
        return _store_annotation(
            packed_block_types,
            "_lk_ball_annotation",
            self._lk_ball_packed_key,
            lk_ball_params,
            fields=("lk_ball_params",),
            bindings=tuple(
                (block_type, "lk_ball_params")
                for block_type in packed_block_types.active_block_types
            ),
        )

    def setup_poses(self, pose_stack: PoseStack):
        super(LKBallEnergyTerm, self).setup_poses(pose_stack)
        # Whole-pose blocks move independently. Reuse this topology tensor
        # instead of allocating and filling it on every scoring call.
        pose_stack._lk_ball_no_lockstep = torch.full_like(pose_stack.block_type_ind, -1)
        # Plan once before graph capture; padding can greatly overstate the
        # useful work in the two specialized derivative kernels.
        blocks = pose_stack.max_n_blocks
        pose_stack._lk_ball_allow_split_backward = (
            pose_stack.device.type == "cuda"
            and pose_stack.block_type_ind.numel() >= 512
            and pose_stack.n_poses * blocks * (blocks + 1) // 2 >= 32768
            and int((pose_stack.block_type_ind >= 0).sum()) >= 512
        )

    def _gen_waters(self, common_args, pose_stack, hbond_params):
        from tmol.score.lk_ball.potentials._compiled import _gen_pose_waters

        args = [
            *common_args,
            pose_stack.inter_residue_connections,
            pose_stack.packed_block_types.n_atoms,
            pose_stack.packed_block_types.n_conn,
            pose_stack.packed_block_types.conn_atom,
            pose_stack.packed_block_types.n_all_bonds,
            pose_stack.packed_block_types.all_bonds,
            pose_stack.packed_block_types.atom_all_bond_ranges,
            hbond_params.tile_n_donH,
            hbond_params.tile_n_acc,
            hbond_params.tile_donH_inds,
            hbond_params.tile_donH_hvy_inds,
            hbond_params.tile_which_donH_of_donH_hvy,
            hbond_params.tile_acc_inds,
            hbond_params.tile_acceptor_hybridization,
            hbond_params.tile_acceptor_n_attached_H,
            hbond_params.is_hydrogen,
            self._lk_ball_water_gen_global_params,
            self.ljlk_param_resolver.global_params.lkb_water_tors_sp2,
            self.ljlk_param_resolver.global_params.lkb_water_tors_sp3,
            self.ljlk_param_resolver.global_params.lkb_water_tors_ring,
        ]
        if common_args[0].dtype == torch.float64:
            _convert_float64(args)

        return _gen_pose_waters(*args)

    def pose_score_lk_ball(self, *args):
        from tmol.score.lk_ball.potentials._compiled import _lk_ball_pose_score

        common_args = args[:-5]
        (
            pose_stack,
            hbond_params,
            lk_ball_params,
            block_pair_scoring,
            shared_block_neighbors,
        ) = args[-5:]

        water_coords = self._gen_waters(
            (*common_args[:12], pose_stack._lk_ball_no_lockstep, *common_args[12:]),
            pose_stack,
            hbond_params,
        )

        args = [
            *common_args,
            pose_stack.inter_residue_connections,
            pose_stack.inter_block_bondsep.near_blocks,
            pose_stack.inter_block_bondsep.bondsep,
            pose_stack.packed_block_types.n_atoms,
            pose_stack.packed_block_types.n_conn,
            pose_stack.packed_block_types.conn_atom,
            lk_ball_params.tile_n_polar_atoms,
            lk_ball_params.tile_n_occluder_atoms,
            lk_ball_params.tile_pol_occ_inds,
            lk_ball_params.tile_lk_ball_params,
            pose_stack.packed_block_types.bond_separation,
            self._lk_ball_global_params,
            # max_dis as host scalar for detect-neighbors call
            self._max_dis,
            water_coords,
            block_pair_scoring,
            shared_block_neighbors,
            getattr(pose_stack, "_lk_ball_allow_split_backward", False),
        ]
        if common_args[0].dtype == torch.float64:
            _convert_float64(args)

        return _lk_ball_pose_score(*args)

    def rotamer_score_lk_ball(self, *args):
        from tmol.score.lk_ball.potentials._compiled import (
            _lk_ball_rotamer_score_shared,
        )

        common_args = args[:-5]
        (
            pose_stack,
            hbond_params,
            lk_ball_params,
            block_pair_scoring,
            shared_dispatch_indices,
        ) = args[-5:]

        # water generation takes the same common args as the scoring call: a
        #    water can reach into a bonded neighbour, and in a group that
        #    neighbour's atoms differ from one conformer to the next
        water_coords = self._gen_waters(common_args, pose_stack, hbond_params)

        args = [
            *common_args,
            pose_stack.inter_residue_connections,
            pose_stack.inter_block_bondsep.near_blocks,
            pose_stack.inter_block_bondsep.bondsep,
            pose_stack.packed_block_types.n_atoms,
            pose_stack.packed_block_types.n_conn,
            pose_stack.packed_block_types.conn_atom,
            lk_ball_params.tile_n_polar_atoms,
            lk_ball_params.tile_n_occluder_atoms,
            lk_ball_params.tile_pol_occ_inds,
            lk_ball_params.tile_lk_ball_params,
            pose_stack.packed_block_types.bond_separation,
            self._lk_ball_global_params,
            # max_dis as host scalar for detect-neighbors call
            self._max_dis,
            water_coords,
            block_pair_scoring,
        ]
        if common_args[0].dtype == torch.float64:
            _convert_float64(args)

        return _lk_ball_rotamer_score_shared(*args, shared_dispatch_indices)

    def get_pose_score_term_function(self):
        return self.pose_score_lk_ball

    def get_rotamer_score_term_function(self):
        return self.rotamer_score_lk_ball

    def get_block_neighbor_cutoff(self):
        return self._max_dis

    def accepts_shared_rotamer_dispatch(self):
        return True

    def rotamer_dispatch_key(self):
        return "sphere_overlap"

    def get_score_term_attributes(self, pose_stack):
        annotation = self.setup_packed_block_types(pose_stack.packed_block_types)
        hbond = HBondDependentTerm.setup_packed_block_types(
            self, pose_stack.packed_block_types
        )
        return [pose_stack, hbond, annotation]
