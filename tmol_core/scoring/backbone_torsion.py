"""Backbone table identity, reflection and topology preparation."""

import attr
import numpy
from tmol_core.chemical.restypes import uaid_t
from tmol_core.database.mirrored_dunbrack import _reflect


def _mirror_grid(table, bbstart, bbstep):
    return _reflect(table, (0, 1), bbstart, bbstep)


def invert_flags(phi, psi, block_type_name):
    phi, psi = numpy.asarray(phi, dtype=bool), numpy.asarray(psi, dtype=bool)
    if (phi != psi).any():
        raise ValueError(
            f"{block_type_name}: a rama/omega lookup row inverts only one of "
            "phi and psi; the mirrored tables invert both together"
        )
    return phi


def table_lookup(rows, names):
    """Resolve row identities once; unassigned table IDs retain the -1 mask."""
    indices = {name: i for i, name in enumerate(names)}
    if len(indices) != len(names):
        raise ValueError("Backbone table names must be unique")
    result = {}
    for row in rows:
        key = (row.res_middle, row.res_upper)
        if key in result:
            raise ValueError(f"Duplicate backbone lookup row {key}")
        result[key] = (indices.get(row.table_id, -1), row.invert_phi, row.invert_psi)
    return result


@attr.s(auto_attribs=True, frozen=True, slots=True)
class BackboneTableData:
    tables: numpy.ndarray
    bbsteps: numpy.ndarray
    bbstarts: numpy.ndarray


def prepare_backbone_tables(rama_tables, omega_tables, fit_coefficients):
    """Fit the common periodic cubic model through a native CPU binding.

    Inputs contain NumPy tables. fit_coefficients accepts a float32 array and
    returns same-shaped coefficients using the shared native spline prefilter.
    Original and mirrored tables occupy [0,n) and [n,2*n), respectively.
    """

    def family_shape(rows, fields, name):
        if not rows:
            raise ValueError(f"{name} requires at least one parameter table")
        shape = numpy.asarray(getattr(rows[0], fields[0])).shape
        if len(shape) != 2 or min(shape) == 0:
            raise ValueError(f"{name} tables must be nonempty two-dimensional grids")
        if any(
            numpy.asarray(getattr(row, field)).shape != shape
            for row in rows
            for field in fields
        ):
            raise ValueError(f"All {name} tables must share one grid shape")
        return shape

    shape = family_shape(rama_tables, ("table",), "Ramachandran")
    omega_shape = family_shape(omega_tables, ("mu", "sigma"), "omega")
    rama = numpy.empty((2 * len(rama_tables), *shape), dtype=numpy.float32)
    for i, row in enumerate(rama_tables):
        raw = numpy.asarray(row.table, dtype=numpy.float32)
        rama[i] = fit_coefficients(raw)
        rama[i + len(rama_tables)] = fit_coefficients(
            _mirror_grid(raw, row.bbstart, row.bbstep)
        )
    omega = numpy.empty((2 * len(omega_tables), 2, *omega_shape), dtype=numpy.float32)
    for i, row in enumerate(omega_tables):
        mu, sigma = (numpy.asarray(v, dtype=numpy.float32) for v in (row.mu, row.sigma))
        omega[i, 0] = fit_coefficients(mu)
        omega[i, 1] = fit_coefficients(sigma)
        omega[i + len(omega_tables), 0] = fit_coefficients(
            360.0 - _mirror_grid(mu, row.bbstart, row.bbstep)
        )
        omega[i + len(omega_tables), 1] = fit_coefficients(
            _mirror_grid(sigma, row.bbstart, row.bbstep)
        )

    def packed(tables, rows):
        return BackboneTableData(
            tables=tables,
            bbsteps=numpy.asarray([r.bbstep for r in rows] * 2, dtype=numpy.float32),
            bbstarts=numpy.asarray([r.bbstart for r in rows] * 2, dtype=numpy.float32),
        )

    return packed(rama, rama_tables), packed(omega, omega_tables)


@attr.s(frozen=True, slots=True, auto_attribs=True)
class BackboneTorsionBlockTypeParams:
    rama_table_inds: numpy.ndarray  # non-prepro/prepro
    omega_table_inds: numpy.ndarray  # non-prepro/prepro
    lower_conn_ind: numpy.ndarray
    upper_conn_ind: numpy.ndarray
    is_pro: numpy.ndarray
    backbone_torsion_atoms: numpy.ndarray


class BackboneTorsionPreparation:
    def __init__(self, rama_rows, omega_rows, rama_names, omega_names):
        self.rama_lookup = table_lookup(rama_rows, rama_names)
        self.omega_lookup = table_lookup(omega_rows, omega_names)
        self.n_rama_tables = len(rama_names)
        self.n_omega_tables = len(omega_names)

    def prepare_block(self, block_type):
        def uaids_to_np(tor_uaids):
            uaids = numpy.full((4,), -1, dtype=uaid_t)
            for i in range(4):
                uaids[i] = numpy.array(tor_uaids[i], dtype=uaid_t)
            return uaids

        rname = block_type.rama_reference

        def table_inds(lookup, n_tables):
            inds = numpy.full(2, -1, dtype=numpy.int64)
            if rname is None:
                return inds
            for i, upper in enumerate(("_", "PRO")):
                row = lookup.get((rname, upper))
                if row is None:
                    continue
                index, phi, psi = row
                if index < 0:
                    raise ValueError(
                        f"{block_type.name}: backbone lookup {rname}/{upper} names a missing table"
                    )
                mirrored = bool(invert_flags(phi, psi, block_type.name)) ^ bool(
                    block_type.reference_mirrored
                )
                inds[i] = index + n_tables if mirrored else index
            return inds

        rama_table_inds = table_inds(self.rama_lookup, self.n_rama_tables)
        omega_table_inds = table_inds(self.omega_lookup, self.n_omega_tables)

        backbone_torsion_atoms = numpy.full((3, 4), -1, dtype=uaid_t)
        if rama_table_inds[0] != -1 or rama_table_inds[1] != -1:
            for i, tor in enumerate(["phi", "psi", "omega"]):
                if tor in block_type.torsion_to_uaids:
                    backbone_torsion_atoms[i] = uaids_to_np(
                        block_type.torsion_to_uaids[tor]
                    )
        backbone_torsion_atoms = backbone_torsion_atoms.reshape(-1)

        lower_conn_id = numpy.full((1,), -1, dtype=numpy.int32)
        if "down" in block_type.connection_to_cidx:
            lower_conn_id[0] = block_type.connection_to_cidx["down"]
        upper_conn_id = numpy.full((1,), -1, dtype=numpy.int32)
        if "up" in block_type.connection_to_cidx:
            upper_conn_id[0] = block_type.connection_to_cidx["up"]
        is_pro = numpy.full((1,), 0, dtype=numpy.int32)

        # is_pro selects the *previous* residue's prepro table, so it follows
        #    the tables the residue reads rather than what it is called
        if block_type.rama_reference == "PRO":
            is_pro[0] = 1

        return BackboneTorsionBlockTypeParams(
            rama_table_inds=rama_table_inds,
            omega_table_inds=omega_table_inds,
            lower_conn_ind=lower_conn_id,
            upper_conn_ind=upper_conn_id,
            is_pro=is_pro,
            backbone_torsion_atoms=backbone_torsion_atoms,
        )

    @staticmethod
    def prepare_packed(block_params):
        n_types = len(block_params)
        bt_rama_table = numpy.full((n_types, 2), -1, dtype=numpy.int32)
        bt_omega_table = numpy.full((n_types, 2), -1, dtype=numpy.int32)
        bt_lower_conn_ind = numpy.full((n_types,), -1, dtype=numpy.int32)
        bt_upper_conn_ind = numpy.full((n_types,), -1, dtype=numpy.int32)
        bt_is_pro = numpy.full((n_types,), -1, dtype=numpy.int32)
        bt_backbone_torsion_atoms = numpy.full((n_types, 12, 3), -1, dtype=numpy.int32)

        for i, i_bbtors_params in enumerate(block_params):
            bt_rama_table[i] = i_bbtors_params.rama_table_inds
            bt_omega_table[i] = i_bbtors_params.omega_table_inds
            bt_lower_conn_ind[i] = i_bbtors_params.lower_conn_ind[0]
            bt_upper_conn_ind[i] = i_bbtors_params.upper_conn_ind[0]
            bt_is_pro[i] = i_bbtors_params.is_pro[0]
            bt_backbone_torsion_atoms[i] = i_bbtors_params.backbone_torsion_atoms.view(
                numpy.int32
            ).reshape(12, 3)

        return dict(
            bt_rama_table=bt_rama_table,
            bt_omega_table=bt_omega_table,
            bt_lower_conn_ind=bt_lower_conn_ind,
            bt_upper_conn_ind=bt_upper_conn_ind,
            bt_is_pro=bt_is_pro,
            bt_backbone_torsion_atoms=bt_backbone_torsion_atoms,
        )
