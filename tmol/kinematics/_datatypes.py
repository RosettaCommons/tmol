import numpy
import torch
import attrs

from tmol.types._torch import Tensor
from tmol.types._tensor import TensorGroup
from tmol.types._array import NDArray
from tmol.types._attrs import ConvertAttrs
from tmol.types._functional import _convert_args
from tmol_core import kinematics as _core_kinematics
from tmol_core.kinematics import chemical as _core_chemical_kinematics
from tmol_core.kinematics import NodeType, prepare_kinematic_forest

BTGenerationalSegScanPathSegs = _core_chemical_kinematics.BTGenerationalSegScanPathSegs
BondDOFTypes = _core_kinematics.BondDOFTypes
JumpDOFTypes = _core_kinematics.JumpDOFTypes
n_movable_bond_dof_types = _core_kinematics.n_movable_bond_dof_types
n_movable_jump_dof_types = _core_kinematics.n_movable_jump_dof_types


@attrs.define(auto_attribs=True, frozen=True)
class KinForest(TensorGroup, ConvertAttrs):
    """A collection of atom-level kinematic trees, each of which can be processed
    in parallel.

    A kinematic description of a collection of atom locations, each atom location
    corresponding to a node within a tree. The root of each tree in this forest
    is built from a jump from the global reference frame at the origin. (The global
    reference frame will later be treated as a node in the forest, effectively
    linking all the trees, but this is a minor technical detail; best to think
    of this as several independent trees than a single tree). Every other node
    corresponds to a derived orientation, with an atomic coordinate at the
    center of the frame.

    Each node in the tree is connected by one of two "node types":

    1) Jump nodes, representing an arbitrary rigid body transform between two
    reference frames via six degrees of freedom, 3 translational and
    3 rotational.

    2) Bond nodes, representing the relationships between two atom reference
    frames via three bond degrees of freedom: the translation from the parent
    to the child along the bond axis (bond length, d), the rotation from the
    grand-parent-to-parent bond axis to the bond axis (an improper bond
    angle, theta), and the rotation about the grand-parent-to-parent bond axis
    (bond torsion, phi). Bond nodes include an additional, redundent,
    degree of freedom representing concerted rotation of all downstream atoms
    about the parent-to-self bond. These DOFs are used to represent
    the torsions that alter the location of several children. For example,
    chi1 is represented as the 4th DOF for the CB atom of LEU. A rotation
    about the CA-->CB bond axis will spin CG, HB1 and HB2. In this scheme,
    the phi DOF would be 0 for CG, 120 for HB1 and 240 for HB2. This differs
    from the Rosetta3 implementation of downstream-dihedral propagation
    where Chi1 would live as the phi DOF of CG, and CG's rotation would
    carry forward to HB1 and HB2 (requiring that CG be the first child of
    CB).

    The atoms in the `KinForest` have their own order that is distinct
    from the ordering in the target (e.g. a PoseStack) where there might
    be gaps between sets of atoms (e.g. because each Pose in the stack
    has a different number of atoms, so a contiguous block of atom indices
    from 0-100 might have a gap before the next contiguous block begins
    at 150). When working with a `KinForest`, remembering what order
    and array's indices is in (the kin-forest order (KFO) or the target
    order (TO)) and what a value/index read out of an array represents (is
    the index an index in KFO or TO?) is *very* challenging. The documentation
    for these arrays includes whether the arrays are indexed in KFO or TO
    and whether the values they hold are KFO or TO indices.

    The `KinForest` data structure itself is frozen and can not be modified post
    construction. `KinForests` are not typically built directly by users,
    but rather are constructed as part of building the `KinModuleData` class
    by `construct_kin_module_data` that lives in scan_ordering.py.
    The `_KinematicBuilder` factory class is now deprecated but use to be
    responsible for construction of `KinForests`

    Indices::
        id = the TO index in KFO; i.e. kin_forest_order_2_target_order
        parent = KFO index of the parent, in KFO
        frame_x = KFO index of self, in KFO
        frame_y = KFO index of parent, in KFO
        frame_z = KFO index of grandparent, in KFO
    """

    id: Tensor[torch.int32][...]
    doftype: Tensor[torch.int32][...]
    parent: Tensor[torch.int32][...]
    frame_x: Tensor[torch.int32][...]
    frame_y: Tensor[torch.int32][...]
    frame_z: Tensor[torch.int32][...]

    def to_host_indices(self):
        """Snapshot and validate native indices outside tensor execution."""
        return prepare_kinematic_forest(
            *(
                getattr(self, name).detach().cpu().numpy()
                for name in ("parent", "frame_x", "frame_y", "frame_z", "doftype")
            )
        )

    @classmethod
    def from_host_indices(cls, id, indices, device=None):
        """Create a Torch forest from common host metadata and target atom IDs."""
        names = ("parent", "frame_x", "frame_y", "frame_z", "doftype")
        indices = prepare_kinematic_forest(*(getattr(indices, name) for name in names))
        if id.dtype != torch.int32 or id.ndim != 1 or len(id) != indices.n_nodes:
            raise ValueError("id must be an int32 vector with one entry per node")
        device = id.device if device is None else device
        return cls(
            id=id.to(device),
            **{
                name: torch.from_numpy(getattr(indices, name).copy()).to(device)
                for name in names
            },
        )

    @classmethod
    @_convert_args
    def node(
        cls,
        id: int,
        doftype: NodeType,
        parent: int,
        frame_x: int,
        frame_y: int,
        frame_z: int,
    ):
        """Construct a single node from element values."""
        return cls(
            id=torch.Tensor([id]),
            doftype=torch.Tensor([doftype]),
            parent=torch.Tensor([parent]),
            frame_x=torch.Tensor([frame_x]),
            frame_y=torch.Tensor([frame_y]),
            frame_z=torch.Tensor([frame_z]),
        )

    @classmethod
    def root_node(cls):
        """The global/root kinematic node at KinForest[0]."""
        return cls.node(
            id=-1, doftype=NodeType.root, parent=0, frame_x=0, frame_y=0, frame_z=0
        )


@attrs.define(auto_attribs=True, frozen=True)
class KinForestScanData(TensorGroup, ConvertAttrs):
    """Node, scan-start, and generation-start tensors for a kinematic scan."""

    nodes: Tensor[torch.int]
    scans: Tensor[torch.int]
    gens: Tensor[torch.int]


@attrs.define(auto_attribs=True, frozen=True)
class KinematicModuleData:
    """Topology and traversal tensors required by a pose kinematics module."""

    forest: KinForest
    scan_data_fw: KinForestScanData
    scan_data_bw: KinForestScanData

    # some extra tensors we need to describe
    # the fold forest in its entirety
    block_in_and_first_out: Tensor[torch.int][:, :]
    keep_atom_fixed: Tensor[torch.bool][:, :]
    pose_stack_atom_for_jump: Tensor[torch.int][:, :, 2]
    pose_stack_atom_for_root_jump: Tensor[torch.int][:, 2]


@attrs.define(auto_attribs=True, slots=True, frozen=True)
class KinDOF(TensorGroup, ConvertAttrs):
    """Internal coordinate data.

    The KinDOF data structure holds two logical views: the "raw" view a
    sparsely populated [n,9] tensor of DOF values and a set of named property
    accessors providing access to specific entries within this array. This is
    logically equivalent a C union datatype, the interpretation of an entry in
    the DOF buffer depends on the type of the corresponding KinForest entry.
    """

    raw: Tensor[torch.double][..., 9]

    @property
    def bond(self):
        return BondDOF(raw=self.raw[..., :4])

    @property
    def jump(self):
        return JumpDOF(raw=self.raw[..., :9])

    def clone(self):
        return KinDOF(raw=self.raw.clone())


@attrs.define(auto_attribs=True, slots=True, frozen=True)
class BondDOF(TensorGroup, ConvertAttrs):
    """A bond dof view of KinDOF."""

    raw: Tensor[torch.double][..., 4]

    @property
    def phi_p(self):
        return self.raw[..., BondDOFTypes.phi_p]

    @property
    def theta(self):
        return self.raw[..., BondDOFTypes.theta]

    @property
    def d(self):
        return self.raw[..., BondDOFTypes.d]

    @property
    def phi_c(self):
        return self.raw[..., BondDOFTypes.phi_c]


@attrs.define(auto_attribs=True, slots=True, frozen=True)
class JumpDOF(TensorGroup, ConvertAttrs):
    """A jump dof view of KinDOF."""

    raw: Tensor[torch.double][..., 9]

    @property
    def RBx(self):
        return self.raw[..., JumpDOFTypes.RBx]

    @property
    def RBy(self):
        return self.raw[..., JumpDOFTypes.RBy]

    @property
    def RBz(self):
        return self.raw[..., JumpDOFTypes.RBz]

    @property
    def RBdel_alpha(self):
        return self.raw[..., JumpDOFTypes.RBdel_alpha]

    @property
    def RBdel_beta(self):
        return self.raw[..., JumpDOFTypes.RBdel_beta]

    @property
    def RBdel_gamma(self):
        return self.raw[..., JumpDOFTypes.RBdel_gamma]

    @property
    def RBalpha(self):
        return self.raw[..., JumpDOFTypes.RBalpha]

    @property
    def RBbeta(self):
        return self.raw[..., JumpDOFTypes.RBbeta]

    @property
    def RBgamma(self):
        return self.raw[..., JumpDOFTypes.RBgamma]


@attrs.define
class PBTGenerationalSegScanPathSegs:
    jump_atom: NDArray[numpy.int64][:]  # n-bt
    parents: Tensor[torch.int32][:, :, :]  # n-bt x n-input x n-atoms
    dof_type: Tensor[torch.int32][:, :, :]  # n-bt x n-input x n-atoms
    input_conn_atom: Tensor[torch.int32][:, :]  # n-bt x n-input
    n_gens: Tensor[torch.int32][:, :, :]  # n-bt x n-input x n-output
    n_nodes_for_gen: Tensor[torch.int32][:, :, :, :]
    nodes_for_gen: Tensor[torch.int32][
        :, :, :, :, :
    ]  # n-input x n-output x max-n-gen x max-n-nodes-per-gen
    n_scan_path_segs: Tensor[torch.int32][
        :, :, :, :
    ]  # n-bt x n-input x n-output x n-gen
    scan_path_seg_that_builds_output_conn: NDArray[numpy.int64][
        :, :, :, :, 2
    ]  # n-bt x n-input x n-output x n-conn x 2
    scan_path_seg_starts: Tensor[torch.int32][:, :, :, :, :]
    scan_path_seg_is_real: Tensor[bool][:, :, :, :, :]
    scan_path_seg_is_inter_block: Tensor[bool][:, :, :, :, :]
    scan_path_seg_lengths: Tensor[torch.int32][:, :, :, :, :]
    uaid_for_torsion_by_inconn: NDArray[numpy.int64][
        :, :, :, 3
    ]  # n-bt x n-input x n-torsions x 3
    torsion_direction: NDArray[numpy.int64][:, :, :]  # n-bt x n-input x n-torsions

    @classmethod
    def from_numpy(cls, arrays, device):
        def convert(array):
            # Empty NumPy arrays have zero strides; native tensor views retain
            # the normal contiguous Torch layout even when an axis is empty.
            if array.size == 0:
                return torch.empty(
                    array.shape, dtype=torch.from_numpy(array).dtype, device=device
                )
            return torch.from_numpy(array).to(device)

        return cls(**{name: convert(array) for name, array in arrays.items()})

    @classmethod
    def empty(
        cls,
        device,
        n_bt: int,
        max_n_input_types: int,
        max_n_output_types: int,
        max_n_atoms: int,
        max_n_conn: int,
        max_n_gens: int,
        max_n_scan_path_segs_per_gen: int,
        max_n_nodes_per_gen: int,
        max_n_torsions: int,
    ):
        arrays = _core_chemical_kinematics.empty_packed_kinematic_stencils(
            n_bt,
            max_n_input_types,
            max_n_output_types,
            max_n_atoms,
            max_n_conn,
            max_n_gens,
            max_n_scan_path_segs_per_gen,
            max_n_nodes_per_gen,
            max_n_torsions,
        )
        return cls.from_numpy(arrays, device)
