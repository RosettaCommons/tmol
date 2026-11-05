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
    from tmol.kinematics.compiled._compiled_ops import _assemble_pose
    from tmol_core.kinematics.pose import validate_pose_kinematics_inputs

    device = pose_stack.device
    pbt = pose_stack.packed_block_types
    _annotate_packed_block_type_with_gen_scan_path_segs(pbt)
    stencils = pbt.gen_seg_scan_path_segs
    ff_edges_cpu = fold_forest_edges.cpu()
    block_type_cpu = pose_stack.block_type_ind.cpu()
    connections_cpu = pose_stack.inter_residue_connections.cpu().to(torch.int32)
    n_conn_cpu = pbt.n_conn.cpu().to(torch.int32)
    polymeric_conn_cpu = pbt.polymeric_conn_inds.cpu().to(torch.int32)
    validate_pose_kinematics_inputs(
        block_type_cpu.numpy(),
        pose_stack.block_coord_offset.cpu().numpy(),
        connections_cpu.numpy(),
        ff_edges_cpu.numpy(),
        pbt.n_atoms.cpu().numpy(),
        n_conn_cpu.numpy(),
        polymeric_conn_cpu.numpy(),
        pose_stack.max_n_pose_atoms,
    )
    (
        id,
        doftype,
        kfo_atom_parents,
        frame_x,
        frame_y,
        frame_z,
        nodes_fw,
        scans_fw,
        gens_fw,
        nodes_bw,
        scans_bw,
        gens_bw,
        pose_stack_block_in_and_first_out,
        keep_atom_fixed,
        pose_stack_atom_for_jump,
        pose_stack_atom_for_root_jump,
    ) = _assemble_pose(
        pose_stack.max_n_pose_atoms,
        pose_stack.block_coord_offset,
        pose_stack.block_type_ind,
        pose_stack.inter_residue_connections,
        fold_forest_edges.to(device),
        pbt.n_atoms,
        pbt.atom_is_real,
        pbt.n_conn,
        pbt.conn_atom,
        pbt.polymeric_conn_inds,
        stencils.jump_atom,
        stencils.parents,
        stencils.dof_type,
        stencils.n_gens,
        stencils.scan_path_seg_that_builds_output_conn,
        stencils.nodes_for_gen,
        stencils.n_scan_path_segs,
        stencils.scan_path_seg_starts,
        stencils.scan_path_seg_is_real,
        stencils.scan_path_seg_is_inter_block,
        stencils.scan_path_seg_lengths,
        ff_edges_cpu,
        block_type_cpu,
        connections_cpu,
        stencils.scan_path_seg_that_builds_output_conn.cpu(),
        polymeric_conn_cpu,
        n_conn_cpu,
    )

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
