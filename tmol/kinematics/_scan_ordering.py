import attr
import torch


from tmol_core.kinematics import scan_ordering as _core_scan_ordering
from tmol_core.kinematics import chemical as _core_chemical_kinematics


from tmol.types._torch import Tensor
from tmol.types._attrs import ValidateAttrs
from tmol.types._functional import _validate_args
from tmol.pose._pose_stack import PoseStack

from ._datatypes import (
    KinForest,
    KinForestScanData,
    KinematicModuleData,
    PBTGenerationalSegScanPathSegs,
)

_handle_case_on_primary_exit_path = (
    _core_chemical_kinematics._handle_case_on_primary_exit_path
)
_handle_case_of_no_exit_scan_path_segment = (
    _core_chemical_kinematics._handle_case_of_no_exit_scan_path_segment
)
_record_scan_path_segments = _core_chemical_kinematics._record_scan_path_segments
_annotate_block_type_for_in_and_out_pair = (
    _core_chemical_kinematics._annotate_block_type_for_in_and_out_pair
)
_mark_kin_uaids_controlling_torsions = (
    _core_chemical_kinematics._mark_kin_uaids_controlling_torsions
)
_gather_scan_path_data = _core_chemical_kinematics._gather_scan_path_data
_bond_csgraph = _core_chemical_kinematics._bond_csgraph
ResidueKinforestData = _core_chemical_kinematics.ResidueKinforestData
_tree_bond_indices = _core_chemical_kinematics._tree_bond_indices
_block_group_kinforest_data = _core_chemical_kinematics._block_group_kinforest_data
_annotate_block_type_with_residue_kinforest_data = (
    _core_chemical_kinematics._annotate_block_type_with_residue_kinforest_data
)
_annotate_block_type_with_gen_scan_path_segs = (
    _core_chemical_kinematics._annotate_block_type_with_gen_scan_path_segs
)
BTGenerationalSegScanPathSegs = _core_chemical_kinematics.BTGenerationalSegScanPathSegs

_get_children = _core_scan_ordering._get_children
_get_scans = _core_scan_ordering._get_scans


@attr.s(auto_attribs=True, frozen=True)
class KinForestScanOrdering(ValidateAttrs):
    """Scan plans for parallel kinematic operations.

    The KinForestScanOrdering class divides the tree into a set of paths. Along
    each path is a continuous chain of atoms that either (1) require their
    coordinate frames computed as a cumulative product of homogeneous
    transforms for the coordinate update algorithm, or (2) require the
    cumulative sum of their f1f2 vectors for the derivative calculation
    algorithm. In both cases, these paths can be processed efficiently
    on the GPU using an algorithm called "scan" and batches of these paths
    can be processed at once in a variant called "segmented scan."

    The same set of paths is used for both the refold algorithm and
    the derivative summation; the refold algorithm starts at path
    roots and multiplies homogeneous transforms towards the leaves.
    The derivative summation algorithm starts at the leaves and sums
    upwards towards the roots.

    To accomplish this, the GPUKinForestReordering class reorders the atoms from
    the original KinForest order ("ko") where atoms are known by their
    kinforest-index ("ki") into 1) their refold order ("ro") where atoms are
    known by their refold index ("ri") and 2) their deriv-sum order ("dso")
    where atoms are known by their deriv-sum index.

    Each scan operation is performed as a series of depth-based "generations",
    in which the scans at generation n are *only* dependent on the results of
    scans in generations [0...n-1] inclusive. In the forward-scan generations
    are ordered from the kinematic root, and scan segments begin with a value
    derived from the result of a generation [0...n-1] scan. In the
    backward-scan generations are ordered from kinematic leaves, and scan
    segments pull summation results from multiple generation [0...n-1] scans.

    For further details on parallel segmented scan operations see:

    * Mark Harris, "Parallel Prefix Sum with CUDA."
      GPU Gems 3, Nvidia Corporation
      https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html
      http://developer.download.nvidia.com/
      compute/cuda/2_2/sdk/website/projects/scan/doc/scan.pdf

    * Sengupta, Shubhabrata, et al. "Scan primitives for GPU computing."
      Graphics hardware. Vol. 2007. 2007.
      http://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Sengupta07.pdf

    * Sean Baxter, "moderngpu"
      http://moderngpu.github.io/moderngpu
    """

    kinforest_cache_key = "__KinForestScanOrdering_cache__"

    forward_scan_paths: KinForestScanData
    backward_scan_paths: KinForestScanData

    @classmethod
    @_validate_args
    def for_kinforest(cls, kinforest):
        """Calculate and cache refold ordering over kinforest

        KinForest data structure is frozen; so it is safe to cache the gpu scan
        ordering for a single object. Store as a private property of the input
        kinforest, lifetime of the cache will then be managed via the target
        object.
        ."""

        if not hasattr(kinforest, cls.kinforest_cache_key):
            object.__setattr__(
                kinforest,
                cls.kinforest_cache_key,
                cls.calculate_from_kinforest(kinforest),
            )

        return getattr(kinforest, cls.kinforest_cache_key)

    @classmethod
    @_validate_args
    def calculate_from_kinforest(cls, kinforest: KinForest):
        """Setup for operations over KinForest.
        ``device`` is inferred from kinforest tensor device.
        """

        parent = kinforest.parent.cpu().numpy()
        try:
            # Preserve the mature planner's ordered-parent checks and
            # zero-copy ownership for existing Torch forests.
            ordering = _core_scan_ordering._scan_ordering_for_ordered_parent(parent)
        except AssertionError:
            # Pose-native forests can use another storage order. The public
            # common preparation validates and remaps those before planning.
            ordering = _core_scan_ordering.prepare_scan_ordering(parent)

        def materialize(plan):
            def tensor(array):
                return torch.from_numpy(
                    array if array.flags.writeable else array.copy()
                )

            return KinForestScanData(
                nodes=tensor(plan.nodes).to(kinforest.parent.device),
                scans=tensor(plan.scans).to(kinforest.parent.device),
                gens=tensor(plan.gens),  # keep generations on CPU
            )

        return cls(
            forward_scan_paths=materialize(ordering.forward),
            backward_scan_paths=materialize(ordering.backward),
        )


@_validate_args
def _construct_kin_module_data_for_pose(
    pose_stack: PoseStack,
    fold_forest_edges: Tensor[torch.int32][:, :, 4],
):
    """Construct the indexed forest and scan order for batched poses.

    Args:
        pose_stack: Poses whose atoms will form the kinematic forest.
        fold_forest_edges: Fold-forest edges shaped ``[pose, edge, 4]``.

    Returns:
        Immutable topology and traversal data for forward and backward scans.
    """
    from tmol.kinematics.compiled._compiled_ops import (
        _calculate_ff_edge_delays,
        _get_children_op,
        _get_block_parent_connectivity_from_toposort,
        _get_id_and_frame_xyz,
        _get_jump_atom_indices,
        _get_kinforest_scans_from_stencils2,
        _get_kfo_indices_for_atoms,
        _get_kfo_atom_parents,
    )

    device = pose_stack.device
    pbt = pose_stack.packed_block_types
    _annotate_packed_block_type_with_gen_scan_path_segs(pbt)
    pbt_gssps = pbt.gen_seg_scan_path_segs

    ff_edges_cpu = fold_forest_edges.cpu()
    ff_edges_device = fold_forest_edges.to(device)

    cpu = torch.device("cpu")
    result = _calculate_ff_edge_delays(
        pose_stack.block_coord_offset,
        pose_stack.block_type_ind,
        ff_edges_cpu,
        pbt_gssps.scan_path_seg_that_builds_output_conn,
        pbt_gssps.nodes_for_gen,
        pbt_gssps.scan_path_seg_starts,
        pose_stack.block_type_ind.to(cpu),
        pose_stack.inter_residue_connections.to(cpu).to(torch.int32),
        pbt_gssps.scan_path_seg_that_builds_output_conn.to(cpu),
        pbt.polymeric_conn_inds.to(cpu).to(torch.int32),
        pbt.n_conn.to(cpu).to(torch.int32),
    )

    (
        dfs_order_of_ff_edges,
        n_ff_edges,
        ff_edge_parent,
        first_ff_edge_for_block,
        pose_stack_ff_parent,
        max_gen_depth_of_ff_edge,
        first_child_of_ff_edge,
        delay_for_edge,
        toposort_index_for_edge,
    ) = tuple(x.to(device) for x in result)

    pose_stack_atom_for_jump, pose_stack_atom_for_root_jump = _get_jump_atom_indices(
        ff_edges_device, pose_stack.block_type_ind, pbt_gssps.jump_atom
    )

    pose_stack_block_in_and_first_out = _get_block_parent_connectivity_from_toposort(
        pose_stack.block_type_ind,
        pose_stack.inter_residue_connections,
        pose_stack_ff_parent,
        dfs_order_of_ff_edges,
        n_ff_edges,
        ff_edges_device,
        first_ff_edge_for_block,
        first_child_of_ff_edge,
        delay_for_edge,
        toposort_index_for_edge,
        pbt.n_conn,
        pbt.polymeric_conn_inds,
    )

    block_kfo_offset, kfo_2_orig_mapping, atom_kfo_index = _get_kfo_indices_for_atoms(
        pose_stack.block_coord_offset,
        pose_stack.block_type_ind,
        pbt.n_atoms,
        pbt.atom_is_real,
    )

    kfo_atom_parents, kfo_atom_grandparents = _get_kfo_atom_parents(
        pose_stack.block_type_ind,
        pose_stack.inter_residue_connections,
        pose_stack_ff_parent,
        pose_stack_block_in_and_first_out,
        pbt_gssps.parents,
        kfo_2_orig_mapping,
        atom_kfo_index,
        pbt_gssps.jump_atom,
        pbt.n_conn,
        pbt.conn_atom,
    )

    n_children, child_list_span, child_list, is_atom_jump = _get_children_op(
        pose_stack.block_type_ind,
        pose_stack_block_in_and_first_out,
        kfo_2_orig_mapping,
        kfo_atom_parents,
        pbt.n_conn,
    )

    id, frame_x, frame_y, frame_z, keep_atom_fixed = _get_id_and_frame_xyz(
        pose_stack.coords.shape[1],
        pose_stack.block_coord_offset,
        kfo_2_orig_mapping,
        kfo_atom_parents,
        child_list_span,
        child_list,
        is_atom_jump,
    )

    nodes_fw, scans_fw, gens_fw, nodes_bw, scans_bw, gens_bw = (
        _get_kinforest_scans_from_stencils2(
            pose_stack.max_n_atoms,
            pose_stack.block_coord_offset,
            pose_stack.block_type_ind,
            pose_stack.inter_residue_connections,
            ff_edges_device,
            torch.max(delay_for_edge).item(),
            delay_for_edge,
            toposort_index_for_edge,
            first_ff_edge_for_block,
            pose_stack_ff_parent,
            pose_stack_block_in_and_first_out,
            pbt_gssps.parents,
            kfo_2_orig_mapping,
            atom_kfo_index,
            pbt_gssps.jump_atom,
            pbt.n_conn,
            pbt.polymeric_conn_inds,
            pbt_gssps.n_gens,
            pbt_gssps.scan_path_seg_that_builds_output_conn,
            pbt_gssps.nodes_for_gen,
            pbt_gssps.n_scan_path_segs,
            pbt_gssps.scan_path_seg_starts,
            pbt_gssps.scan_path_seg_is_real,
            pbt_gssps.scan_path_seg_is_inter_block,
            pbt_gssps.scan_path_seg_lengths,
        )
    )

    # This feels so clunky after all that slick C++
    is_res_real = pose_stack.block_type_ind != -1
    is_atom_real = pbt.atom_is_real[pose_stack.block_type_ind[is_res_real]]

    block_atom_dof_type = pbt_gssps.dof_type[
        pose_stack.block_type_ind[is_res_real],
        pose_stack_block_in_and_first_out[is_res_real][:, 0],
    ]
    doftype = torch.zeros((id.shape[0],), dtype=torch.int32, device=id.device)
    doftype[1:] = block_atom_dof_type[is_atom_real]

    return KinematicModuleData(
        forest=KinForest(
            id=id,
            doftype=doftype,
            parent=kfo_atom_parents,
            frame_x=frame_x,
            frame_y=frame_y,
            frame_z=frame_z,
        ),
        scan_data_fw=KinForestScanData(
            nodes=nodes_fw,
            scans=scans_fw,
            gens=gens_fw.cpu(),
        ),
        scan_data_bw=KinForestScanData(
            nodes=nodes_bw,
            scans=scans_bw,
            gens=gens_bw.cpu(),
        ),
        block_in_and_first_out=pose_stack_block_in_and_first_out,
        keep_atom_fixed=keep_atom_fixed,
        pose_stack_atom_for_jump=pose_stack_atom_for_jump,
        pose_stack_atom_for_root_jump=pose_stack_atom_for_root_jump,
    )


def _annotate_packed_block_type_with_gen_scan_path_segs(pbt):
    if hasattr(pbt, "gen_seg_scan_path_segs"):
        return
    arrays = _core_chemical_kinematics.prepare_packed_kinematic_stencils(
        pbt.active_block_types,
        max_n_atoms=pbt.max_n_atoms,
        max_n_conn=pbt.max_n_conn,
        max_n_torsions=pbt.max_n_torsions,
    )
    pbt.gen_seg_scan_path_segs = PBTGenerationalSegScanPathSegs.from_numpy(
        arrays, pbt.device
    )
