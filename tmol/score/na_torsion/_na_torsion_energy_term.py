import numpy
from tmol_core.scoring.na_torsion_topology import prepare_na_torsion_topology
import torch

from .._energy_term import EnergyTerm
from .._annotation_cache import AnnotationKey, _cached_annotation, _store_annotation

from tmol.database import ParameterDatabase
from tmol.chemical import RefinedResidueType
from tmol.pose._packed_block_types import PackedBlockTypes
from tmol.pose._pose_stack import PoseStack
from ._params import (
    NaTorsionParams,
    _block_type_params,
    _polymer_index,
    TORSION_NAMES,
    TORSION_IND,
    SYN_MEAN,
)
from ._potentials import (
    _bi_bii_weight,
    _blended_devsq,
    _dihedral,
    pucker_weights,
    _syn_weight,
    _triple_bin_weights,
    _wrap_degrees,
)

N_TORSION = len(TORSION_NAMES)
ALPHA, BETA, GAMMA, DELTA, EPSILON, ZETA = 0, 1, 2, 3, 4, 5
CHI = TORSION_IND["chi1"]
SUGAR_SLOTS = (DELTA, TORSION_IND["nu4"], TORSION_IND["nu0"], TORSION_IND["nu1"])


class NaTorsionEnergyTerm(EnergyTerm):
    """Score nucleic-acid backbone, sugar-pucker, and glycosidic torsions."""

    device: torch.device

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        super(NaTorsionEnergyTerm, self).__init__(param_db=param_db, device=device)
        self.params = NaTorsionParams.from_database(param_db.scoring.na_torsion, device)
        self.element_for_atom_type = {
            at.name: at.element for at in param_db.chemical.atom_types
        }
        self.device = device
        self._block_annotation_key = AnnotationKey.from_sources(param_db.chemical)
        self._packed_annotation_key = AnnotationKey.from_sources(
            param_db.chemical, settings=(self.device,)
        )
        self._pose_annotation_key = AnnotationKey.from_sources(
            param_db.chemical, settings=(self.device,)
        )

    @classmethod
    def class_name(cls):
        return "NaTorsion"

    @classmethod
    def score_types(cls):
        import tmol.score.terms._na_torsion_creator

        return tmol.score.terms._na_torsion_creator.NaTorsionTermCreator.score_types()

    def n_bodies(self):
        return 2

    def setup_block_type(self, block_type: RefinedResidueType):
        super(NaTorsionEnergyTerm, self).setup_block_type(block_type)
        cached = _cached_annotation(
            block_type, "_na_torsion_annotation", self._block_annotation_key
        )
        if cached is not None:
            return cached
        params = _block_type_params(block_type, self.element_for_atom_type)
        setattr(block_type, "na_torsion_params", params)
        return _store_annotation(
            block_type,
            "_na_torsion_annotation",
            self._block_annotation_key,
            params,
            fields=("na_torsion_params",),
        )

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(NaTorsionEnergyTerm, self).setup_packed_block_types(packed_block_types)
        cached = _cached_annotation(
            packed_block_types,
            "_na_torsion_annotation",
            self._packed_annotation_key,
        )
        if cached is not None:
            return cached

        bts = packed_block_types.active_block_types
        block_params = [self.setup_block_type(bt) for bt in bts]
        stack = lambda key: numpy.stack(  # noqa: E731
            [params[key] for params in block_params]
        )

        def to_t(arr):
            return torch.tensor(arr, dtype=torch.int32, device=self.device)

        setattr(
            packed_block_types,
            "na_torsion_base",
            to_t(numpy.array([params["base"] for params in block_params])),
        )
        setattr(packed_block_types, "na_torsion_uaids", to_t(stack("uaids")))
        setattr(packed_block_types, "na_torsion_ring", to_t(stack("ring")))
        setattr(
            packed_block_types,
            "na_torsion_down",
            to_t(numpy.array([params["down"] for params in block_params])),
        )
        return _store_annotation(
            packed_block_types,
            "_na_torsion_annotation",
            self._packed_annotation_key,
            (
                packed_block_types.na_torsion_base,
                packed_block_types.na_torsion_uaids,
                packed_block_types.na_torsion_ring,
                packed_block_types.na_torsion_down,
            ),
            fields=(
                "na_torsion_base",
                "na_torsion_uaids",
                "na_torsion_ring",
                "na_torsion_down",
            ),
            bindings=tuple(
                (block_type, "na_torsion_params")
                for block_type in packed_block_types.active_block_types
            ),
        )

    def setup_poses(self, poses: PoseStack):
        super(NaTorsionEnergyTerm, self).setup_poses(poses)
        cached = _cached_annotation(
            poses, "_na_torsion_annotation", self._pose_annotation_key
        )
        if cached is not None:
            return cached

        pbt = poses.packed_block_types
        params = _pose_indices(
            poses.block_type_ind_for_rot,
            pbt.na_torsion_base,
            pbt.na_torsion_uaids,
            pbt.na_torsion_ring,
            pbt.na_torsion_down,
            pbt.atom_downstream_of_conn,
            poses.block_coord_offset,
            poses.inter_residue_connections,
            poses.max_n_pose_atoms,
        )
        poses.na_torsion_pose_params = params
        poses.na_torsion_has_na = bool(params[1].any())
        return _store_annotation(
            poses,
            "_na_torsion_annotation",
            self._pose_annotation_key,
            (params, poses.na_torsion_has_na),
            fields=("na_torsion_pose_params", "na_torsion_has_na"),
        )

    def get_pose_score_term_function(self):
        return _eval_na_torsion_for_pose

    def get_rotamer_score_term_function(self):
        return _eval_na_torsion_for_rotamers

    def _pose_has_na(self, pose_stack):
        if hasattr(pose_stack, "na_torsion_has_na"):
            return pose_stack.na_torsion_has_na
        pbt = pose_stack.packed_block_types
        bt = pose_stack.block_type_ind64
        return bool(
            (pbt.na_torsion_base.to(torch.int64)[bt.clamp_min(0)] >= 0)[bt >= 0].any()
        )

    def pose_score_term_is_invariant_zero(self, pose_stack):
        return not self._pose_has_na(pose_stack)

    def block_pair_score_term_is_invariant_zero_off_diagonal(self, pose_stack):
        # The torsions may read atoms from a bonded neighbour, but their energy
        # is attributed to the nucleotide's diagonal block-pair entry.
        return True

    def get_score_term_attributes(self, pose_stack):
        p = self.params
        has_na = self._pose_has_na(pose_stack)
        return [
            has_na,
            *pose_stack.na_torsion_pose_params,
            *self.subterm_attributes(pose_stack),
            p.weight_bb,
            p.weight_chi,
            p.weight_sugar,
        ]

    def subterm_attributes(self, pose_stack):
        """Arguments of na_torsion_subterms, which the eval functions take
        first and then follow with the harmonic subterm weights."""
        pbt = pose_stack.packed_block_types
        p = self.params
        return [
            pbt.na_torsion_base,
            pbt.na_torsion_uaids,
            pbt.na_torsion_ring,
            pbt.na_torsion_down,
            pbt.atom_downstream_of_conn,
            pose_stack.block_coord_offset,
            pose_stack.inter_residue_connections,
            p.backbone_means,
            p.backbone_sdev,
            p.sugar_means,
            p.chi_means,
            p.sdev_sugar,
            p.sdev_chi,
            p.well_pucker,
            p.well_alpha_gamma,
            p.well_bibii_pucker,
            p.well_alphanext_bibii,
            p.well_chi_syn,
            p.is_north,
            p.pucker_temperature,
            p.bin_blend_sdev,
        ]

    def get_rotamer_score_term_attributes(self, pose_stack, rotamer_set):
        """Resolve candidate torsion atom indices once when rendering."""
        pbt = pose_stack.packed_block_types
        indices = _rotamer_indices(
            rotamer_set.block_type_ind_for_rot,
            rotamer_set.pose_for_rot,
            rotamer_set.block_ind_for_rot,
            rotamer_set.coord_offset_for_rot,
            rotamer_set.rot_offset_for_block,
            rotamer_set.first_rot_block_type,
            pose_stack.inter_residue_connections,
            pbt.atom_downstream_of_conn,
            pbt.na_torsion_base,
            pbt.na_torsion_uaids,
            pbt.na_torsion_ring,
            pbt.na_torsion_down,
        )
        return [*self.get_score_term_attributes(pose_stack), *indices]


def _eval_na_torsion_for_pose(
    # common args
    rot_coords,
    _rot_coord_offset,
    _pose_ind_for_atom,
    _first_rot_for_block,
    _first_rot_block_type,
    _block_ind_for_rot,
    _pose_ind_for_rot,
    block_type_ind_for_rot,
    _n_rots_for_pose,
    _rot_offset_for_pose,
    _n_rots_for_block,
    _rot_offset_for_block,
    _max_n_rots_per_pose,
    # term args
    has_na,
    pose_base,
    pose_is_na,
    pose_torsion_indices,
    pose_torsion_ok,
    pose_ring_indices,
    pose_ring_ok,
    pose_prev,
    bt_base,
    bt_uaids,
    bt_ring,
    bt_down,
    atom_downstream_of_conn,
    block_coord_offset,
    inter_residue_connections,
    backbone_means,
    backbone_sdev,
    sugar_means,
    chi_means,
    sdev_sugar,
    sdev_chi,
    well_pucker,
    well_alpha_gamma,
    well_bibii_pucker,
    well_alphanext_bibii,
    well_chi_syn,
    is_north,
    pucker_temperature,
    bin_blend_sdev,
    weight_bb,
    weight_chi,
    weight_sugar,
    output_block_pair_energies: bool,
):
    """Evaluate nucleic-acid torsion energies for packed rotamer coordinates."""
    if not has_na:
        n_poses, max_n_blocks = block_coord_offset.shape
        zero = torch.zeros(
            (n_poses, max_n_blocks), dtype=rot_coords.dtype, device=rot_coords.device
        )
        return _finish((zero, zero), output_block_pair_energies)

    if rot_coords.device.type in ("cpu", "cuda") and not output_block_pair_energies:
        from .potentials._compiled import _na_torsion_pose_score

        return _na_torsion_pose_score(
            rot_coords,
            pose_base,
            pose_is_na,
            pose_torsion_indices,
            pose_torsion_ok,
            pose_ring_indices,
            pose_ring_ok,
            pose_prev,
            backbone_means,
            backbone_sdev,
            sugar_means,
            chi_means,
            sdev_sugar,
            sdev_chi,
            well_pucker,
            well_alpha_gamma,
            well_bibii_pucker,
            well_alphanext_bibii,
            well_chi_syn,
            is_north,
            weight_bb,
            weight_chi,
            weight_sugar,
            pucker_temperature,
            bin_blend_sdev,
        )

    e_bb, e_chi, e_sugar, e_well = _pose_subterms(
        rot_coords,
        pose_base,
        pose_is_na,
        pose_torsion_indices,
        pose_torsion_ok,
        pose_ring_indices,
        pose_ring_ok,
        pose_prev,
        backbone_means,
        backbone_sdev,
        sugar_means,
        chi_means,
        sdev_sugar,
        sdev_chi,
        well_pucker,
        well_alpha_gamma,
        well_bibii_pucker,
        well_alphanext_bibii,
        well_chi_syn,
        is_north,
        pucker_temperature,
        bin_blend_sdev,
    )
    poly = _polymer_index(pose_base)
    harmonic = (
        weight_bb[poly] * e_bb + weight_chi[poly] * e_chi + weight_sugar[poly] * e_sugar
    )
    well = e_well
    zero = torch.zeros_like(harmonic)
    return _finish(
        (
            torch.where(pose_is_na, harmonic, zero),
            torch.where(pose_is_na, well, zero),
        ),
        output_block_pair_energies,
    )


def _pose_indices(*args):
    # Static topology is prepared once on the host. Coordinate execution stays
    # on the requested device and uses these immutable index arrays.
    device = args[0].device
    arrays = [a.detach().cpu().numpy() for a in args[:-1]]
    return tuple(
        torch.as_tensor(value, device=device)
        for value in prepare_na_torsion_topology(*arrays, args[-1])
    )


def _pose_subterms(
    rot_coords,
    base,
    is_na,
    torsion_indices,
    torsion_ok,
    ring_indices,
    ring_ok,
    prev,
    *energy_params,
):
    """Evaluate coordinate-dependent NA terms using pre-resolved topology."""
    xyz = rot_coords[torsion_indices.reshape(-1)].reshape(*torsion_indices.shape, 3)
    xyz = torch.where(
        torsion_ok.unsqueeze(-1).unsqueeze(-1), xyz, torch.zeros_like(xyz)
    )
    ring_xyz = rot_coords[ring_indices.reshape(-1)].reshape(*ring_indices.shape, 3)
    ring_xyz = torch.where(ring_ok.unsqueeze(-1), ring_xyz, torch.zeros_like(ring_xyz))

    n_poses, max_n_blocks = base.shape

    def flat(tensor):
        return tensor.reshape(n_poses * max_n_blocks, *tensor.shape[2:])

    energies = _subterm_energies(
        flat(xyz),
        flat(torsion_ok),
        flat(ring_xyz),
        flat(base),
        flat(prev),
        *energy_params,
    )
    shape = (n_poses, max_n_blocks)
    return tuple(energy.reshape(shape) for energy in energies)


def _na_torsion_subterms(
    rot_coords,
    block_type_ind_for_rot,
    bt_base,
    bt_uaids,
    bt_ring,
    bt_down,
    atom_downstream_of_conn,
    block_coord_offset,
    inter_residue_connections,
    backbone_means,
    backbone_sdev,
    sugar_means,
    chi_means,
    sdev_sugar,
    sdev_chi,
    well_pucker,
    well_alpha_gamma,
    well_bibii_pucker,
    well_alphanext_bibii,
    well_chi_syn,
    is_north,
    pucker_temperature,
    bin_blend_sdev,
):
    """Per-block (bb, chi, sugar, well) energies, the mask, and the base index.

    The base index carries the polymer, which the caller needs in order to pick
    the per-polymer subterm weights.
    """
    n_poses = block_coord_offset.shape[0]
    indices = _pose_indices(
        block_type_ind_for_rot,
        bt_base,
        bt_uaids,
        bt_ring,
        bt_down,
        atom_downstream_of_conn,
        block_coord_offset,
        inter_residue_connections,
        rot_coords.shape[0] // n_poses,
    )
    base, is_na = indices[:2]
    energies = _pose_subterms(
        rot_coords,
        *indices,
        backbone_means,
        backbone_sdev,
        sugar_means,
        chi_means,
        sdev_sugar,
        sdev_chi,
        well_pucker,
        well_alpha_gamma,
        well_bibii_pucker,
        well_alphanext_bibii,
        well_chi_syn,
        is_north,
        pucker_temperature,
        bin_blend_sdev,
    )
    return (*energies, is_na, base)


def _subterm_energies(
    xyz,
    tor_ok,
    ring_xyz,
    base,
    prev,
    backbone_means,
    backbone_sdev,
    sugar_means,
    chi_means,
    sdev_sugar,
    sdev_chi,
    well_pucker,
    well_alpha_gamma,
    well_bibii_pucker,
    well_alphanext_bibii,
    well_chi_syn,
    is_north,
    pucker_temperature,
    bin_blend_sdev,
):
    """(bb, chi, sugar, well) energies for a flat list of nucleotides.

    xyz: (N, n_torsion, 4, 3) torsion atoms, tor_ok: (N, n_torsion),
    ring_xyz: (N, 5, 3), base: (N), prev: (N) index into N of the preceding
    nucleotide, -1 if there is none.
    """
    dtype = xyz.dtype
    is_na = base >= 0
    # DNA and RNA share the functional form and nothing else; every table below
    # that is not per-base is gathered on this index
    poly = _polymer_index(base)
    tors = _dihedral(xyz[..., 0, :], xyz[..., 1, :], xyz[..., 2, :], xyz[..., 3, :])
    tors = torch.where(tor_ok, tors, torch.zeros_like(tors)) % 360.0
    zero = torch.zeros(tors.shape[:-1], dtype=dtype, device=tors.device)

    # a residue whose sugar ring did not resolve has no pucker: its ring
    #    coordinates are zeros, and a distribution over them means nothing
    has_ring = is_na & (ring_xyz.abs().sum((-1, -2)) > 0)
    pucker = pucker_weights(ring_xyz, pucker_temperature).to(dtype)
    pucker = torch.where(has_ring.unsqueeze(-1), pucker, torch.zeros_like(pucker))

    # --- backbone -----------------------------------------------------------
    means = backbone_means.to(dtype)[poly]
    sdev_bb = backbone_sdev.to(dtype)[poly]
    e_bb = zero

    bin_w = {}
    for tor in (ALPHA, GAMMA):
        ang = tors[..., tor]
        w = _triple_bin_weights(ang, means[:, tor, :3], bin_blend_sdev)
        bin_w[tor] = w
        e_bb = e_bb + torch.where(
            tor_ok[..., tor],
            _blended_devsq(ang, means[:, tor, :3], w) / sdev_bb[:, tor] ** 2,
            zero,
        )

    w_bi = _bi_bii_weight(tors[..., EPSILON], tors[..., ZETA])
    both = tor_ok[..., EPSILON] & tor_ok[..., ZETA]
    for tor in (EPSILON, ZETA):
        ang = tors[..., tor]
        w = torch.stack([w_bi, 1.0 - w_bi], dim=-1)
        e_bb = e_bb + torch.where(
            both,
            _blended_devsq(ang, means[:, tor, :2], w) / sdev_bb[:, tor] ** 2,
            zero,
        )

    # beta is binned on the previous residue's BI/BII state;
    # fall back to BI mean when there is no preceding nucleotide
    prev_safe = prev.clamp_min(0)
    prev_ok = (prev >= 0) & both[prev_safe]
    w_beta = torch.where(prev_ok, w_bi[prev_safe], torch.ones_like(w_bi))
    ang = tors[..., BETA]
    e_bb = e_bb + torch.where(
        tor_ok[..., BETA],
        _blended_devsq(
            ang, means[:, BETA, :2], torch.stack([w_beta, 1.0 - w_beta], dim=-1)
        )
        / sdev_bb[:, BETA] ** 2,
        zero,
    )

    # --- chi ----------------------------------------------------------------
    chi = tors[..., CHI]
    w_syn = _syn_weight(chi)
    dev_pucker = _wrap_degrees(
        chi.unsqueeze(-1) - chi_means.to(dtype)[base.clamp_min(0)]
    )
    dev_syn = _wrap_degrees(chi - SYN_MEAN).unsqueeze(-1)
    w_syn1 = w_syn.unsqueeze(-1)
    e_chi = (pucker * ((1.0 - w_syn1) * dev_pucker**2 + w_syn1 * dev_syn**2)).sum(
        -1
    ) / sdev_chi.to(dtype)[poly] ** 2
    e_chi = torch.where(tor_ok[..., CHI], e_chi, zero)

    # --- sugar --------------------------------------------------------------
    e_sugar = zero
    for slot, tor in enumerate(SUGAR_SLOTS):
        dev = _wrap_degrees(
            tors[..., tor].unsqueeze(-1) - sugar_means.to(dtype)[poly][:, :, slot]
        )
        e_sugar = e_sugar + torch.where(
            tor_ok[..., tor], (pucker * dev**2).sum(-1), zero
        )
    e_sugar = e_sugar / sdev_sugar.to(dtype)[poly] ** 2

    # --- wells --------------------------------------------------------------
    # -ln P of each bin assignment
    e_well = torch.where(has_ring, (pucker * well_pucker.to(dtype)[poly]).sum(-1), zero)

    w_a, w_g = bin_w[ALPHA], bin_w[GAMMA]
    e_well = e_well + torch.where(
        tor_ok[..., ALPHA] & tor_ok[..., GAMMA],
        torch.einsum("ni,nij,nj->n", w_a, well_alpha_gamma.to(dtype)[poly], w_g),
        zero,
    )

    w_bibii = torch.stack([w_bi, 1.0 - w_bi], dim=-1)
    w_north = (pucker * is_north.to(dtype)).sum(-1)
    w_ns = torch.stack([w_north, 1.0 - w_north], dim=-1)
    e_well = e_well + torch.where(
        both,
        torch.einsum("nb,nbs,ns->n", w_bibii, well_bibii_pucker.to(dtype)[poly], w_ns),
        zero,
    )

    w_bprev = torch.stack([w_beta, 1.0 - w_beta], dim=-1)
    e_well = e_well + torch.where(
        prev_ok & tor_ok[..., ALPHA],
        torch.einsum(
            "ni,nib,nb->n", w_a, well_alphanext_bibii.to(dtype)[poly], w_bprev
        ),
        zero,
    )

    syn_table = well_chi_syn.to(dtype)[..., base.clamp_min(0)].permute(2, 0, 1)
    e_well = e_well + torch.where(
        tor_ok[..., CHI],
        torch.einsum(
            "ns,nsp,np->n",
            torch.stack([1.0 - w_syn, w_syn], dim=-1),
            syn_table,
            pucker,
        ),
        zero,
    )

    return e_bb, e_chi, e_sugar, e_well


def _rotamer_indices(
    block_type_ind_for_rot,
    pose_ind_for_rot,
    block_ind_for_rot,
    rot_coord_offset,
    first_rot_for_block,
    first_rot_block_type,
    inter_residue_connections,
    atom_downstream_of_conn,
    bt_base,
    bt_uaids,
    bt_ring,
    bt_down,
):
    """Resolve candidate topology with the common native background convention.

    Inactive atom indices use -1; valid indices and masks retain their meaning.
    Host metadata preparation supports devices without a native topology launch.
    """
    from .potentials._compiled import _na_rotamer_topology

    device = block_type_ind_for_rot.device
    execution_device = device if device.type in ("cpu", "cuda") else torch.device("cpu")
    inputs = (
        block_type_ind_for_rot,
        pose_ind_for_rot,
        block_ind_for_rot,
        rot_coord_offset,
        first_rot_for_block,
        first_rot_block_type,
        inter_residue_connections,
        atom_downstream_of_conn,
        bt_base,
        bt_uaids,
        bt_ring,
        bt_down,
    )
    result = _na_rotamer_topology(
        *(value.to(device=execution_device, dtype=torch.int64) for value in inputs)
    )
    return tuple(value.to(device=device) for value in result)


def _eval_na_torsion_for_rotamers(
    # common args
    rot_coords,
    rot_coord_offset,
    _pose_ind_for_atom,
    first_rot_for_block,
    first_rot_block_type,
    block_ind_for_rot,
    pose_ind_for_rot,
    block_type_ind_for_rot,
    n_rots_for_pose,
    _rot_offset_for_pose,
    _n_rots_for_block,
    _rot_offset_for_block,
    _lockstep_group_for_block,
    _max_n_rots_per_pose,
    # term args
    has_na,
    _pose_base,
    _pose_is_na,
    _pose_torsion_indices,
    _pose_torsion_ok,
    _pose_ring_indices,
    _pose_ring_ok,
    _pose_prev,
    bt_base,
    bt_uaids,
    bt_ring,
    bt_down,
    atom_downstream_of_conn,
    _block_coord_offset,
    inter_residue_connections,
    backbone_means,
    backbone_sdev,
    sugar_means,
    chi_means,
    sdev_sugar,
    sdev_chi,
    well_pucker,
    well_alpha_gamma,
    well_bibii_pucker,
    well_alphanext_bibii,
    well_chi_syn,
    is_north,
    pucker_temperature,
    bin_blend_sdev,
    weight_bb,
    weight_chi,
    weight_sugar,
    rotamer_base,
    rotamer_is_na,
    rotamer_torsion_indices,
    rotamer_atom_ok,
    rotamer_torsion_ok,
    rotamer_ring_indices,
    rotamer_ring_ok,
    rotamer_prev,
    output_block_pair_energies: bool,
):
    device = rot_coords.device
    dtype = rot_coords.dtype
    n_rots = block_type_ind_for_rot.shape[0]
    n_poses = n_rots_for_pose.shape[0]

    if not has_na:
        score = torch.zeros((2, n_rots), dtype=dtype, device=device)
    else:
        score = _na_rotamer_score(
            rot_coords,
            rotamer_base.unsqueeze(1),
            rotamer_is_na.unsqueeze(1),
            rotamer_torsion_indices.unsqueeze(1),
            rotamer_torsion_ok.unsqueeze(1),
            rotamer_ring_indices.unsqueeze(1),
            rotamer_ring_ok.unsqueeze(1),
            rotamer_prev.unsqueeze(1),
            backbone_means,
            backbone_sdev,
            sugar_means,
            chi_means,
            sdev_sugar,
            sdev_chi,
            well_pucker,
            well_alpha_gamma,
            well_bibii_pucker,
            well_alphanext_bibii,
            well_chi_syn,
            is_north,
            weight_bb,
            weight_chi,
            weight_sugar,
            pucker_temperature,
            bin_blend_sdev,
        )

    if output_block_pair_energies:
        # a one-body term: each energy sits on the diagonal of the rotamer pair
        rot_ind = torch.arange(n_rots, dtype=torch.int32, device=device)
        indices = torch.stack([pose_ind_for_rot.to(torch.int32), rot_ind, rot_ind])
        output_scores = score
    else:
        output_scores = torch.zeros((2, n_poses), dtype=dtype, device=device)
        output_scores.index_add_(1, pose_ind_for_rot.to(torch.int64), score)
        indices = torch.zeros((0,), dtype=torch.int32, device=device)

    return output_scores, indices


def _finish(scores, output_block_pair_energies):
    stacked = torch.stack(scores)  # (score type, pose, block)
    if output_block_pair_energies:
        return torch.diag_embed(stacked), None
    return torch.sum(stacked, dim=2), None


def _na_rotamer_score_torch(
    coords,
    base,
    is_na,
    torsion_indices,
    torsion_ok,
    ring_indices,
    ring_ok,
    prev,
    *parameters,
):
    # Reuse the whole-pose gather and portable equations for higher derivatives
    # and devices without a native NA kernel. One row represents one candidate.
    (
        *energy_params,
        weight_bb,
        weight_chi,
        weight_sugar,
        pucker_temperature,
        bin_blend_sdev,
    ) = parameters
    bb, chi, sugar, well = _pose_subterms(
        coords,
        base,
        is_na,
        torsion_indices,
        torsion_ok,
        ring_indices,
        ring_ok,
        prev,
        *energy_params,
        pucker_temperature,
        bin_blend_sdev,
    )
    poly = _polymer_index(base)
    harmonic = (
        weight_bb[poly] * bb + weight_chi[poly] * chi + weight_sugar[poly] * sugar
    )
    score = torch.stack((harmonic, well))
    return torch.where(is_na.unsqueeze(0), score, torch.zeros_like(score)).squeeze(-1)


class _NativeNaRotamerScore(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        from .potentials._compiled import _na_torsion_native_score

        ctx.save_for_backward(*args[:-2])
        ctx.constants = args[-2:]
        # The unused cotangent input avoids an allocation on the forward path.
        return _na_torsion_native_score(*args, args[0], False)[0]

    @staticmethod
    def backward(ctx, grad_output):
        args = (*ctx.saved_tensors, *ctx.constants)
        if torch.is_grad_enabled():
            # Preserve higher coordinate derivatives through the portable path.
            score = _na_rotamer_score_torch(*args)
            gradient = torch.autograd.grad(
                score, args[0], grad_output, create_graph=True
            )[0]
        else:
            from .potentials._compiled import _na_torsion_native_score

            gradient = _na_torsion_native_score(*args, grad_output, True)[1]
        return (gradient,) + (None,) * (len(args) - 1)


def _na_rotamer_score(*args):
    if args[0].device.type in ("cpu", "cuda") and not any(
        value.requires_grad for value in args[1:-2] if value.is_floating_point()
    ):
        return _NativeNaRotamerScore.apply(*args)
    return _na_rotamer_score_torch(*args)
