"""Metal site, donor, fan and bridge preparation shared by execution adapters."""

from itertools import combinations
import numpy
from tmol_core.chemical import _ideal_distances, _metal_table

# kernel sentinels for metal_conn_virt
NOT_A_SITE = -2
NO_VIRTUAL = -1
# most in-residue metals one donor atom can satisfy
MAX_INTERNAL_METALS = 4


class MetalCoordinationPreparation:
    def __init__(self, params, *, metal_table=None):
        self.params = params
        table = _metal_table() if metal_table is None else metal_table
        self.ion_for_atom_type = {ion["atom_type"]: ion for ion in table["ions"]}
        self.donor_radii = table["donor_radii"]
        self.donor_keys = tuple(self.donor_radii)
        self.vertices_for = {g["name"]: g["vertices"] for g in table["geometries"]}

    def prepare_packed(self, block_types, atom_types, max_n_conn=None):
        block_types = tuple(block_types)
        n_types = len(block_types)
        required = max((len(bt.connections) for bt in block_types), default=0)
        if max_n_conn is None:
            max_n_conn = max(1, required)
        if max_n_conn < required:
            raise ValueError("Metal connection layout truncates active connections")
        atom_types = {at.name: at for at in atom_types}
        n_keys = len(self.donor_keys)
        n_conn = max_n_conn
        conn_metal = numpy.full((n_types, n_conn), -1, dtype=numpy.int32)
        conn_virt = numpy.full((n_types, n_conn), NOT_A_SITE, numpy.int32)
        conn_key = numpy.full((n_types, n_conn), -1, dtype=numpy.int32)
        site_params = numpy.zeros((n_types, n_conn, n_keys, 4), dtype=numpy.float32)
        fans = [[] for _ in range(n_types)]

        internal_metal = numpy.full(
            (n_types, n_conn, MAX_INTERNAL_METALS), -1, dtype=numpy.int32
        )
        internal_d0 = numpy.zeros(
            (n_types, n_conn, MAX_INTERNAL_METALS), dtype=numpy.float32
        )

        for i, bt in enumerate(block_types):
            for c, conn in enumerate(bt.connections):
                atom = bt.atoms[bt.atom_to_idx[conn.atom]]
                conn_key[i, c] = self.donor_key(atom_types[atom.atom_type])
            for site in bt.metal_sites:
                self.setup_site(bt, i, site, conn_metal, conn_virt, site_params, fans)
            self.setup_internal_bridges(bt, i, internal_metal, internal_d0)

        max_fan = max(1, max((len(f) for f in fans), default=0))
        fan_atoms = numpy.full((n_types, max_fan, 2), -1, dtype=numpy.int32)
        fan_params = numpy.zeros((n_types, max_fan, 2), dtype=numpy.float32)
        for i, rows in enumerate(fans):
            for j, (a, b, l0) in enumerate(rows):
                fan_atoms[i, j] = (a, b)
                fan_params[i, j] = (l0, self.params.global_parameters.fan_sd)

        bridge_params = numpy.zeros((n_keys, 2), dtype=numpy.float32)
        width = numpy.radians(self.params.global_parameters.bridge_width)
        for k, key in enumerate(self.donor_keys):
            floor = self.params.bridge_floor(key)
            bridge_params[k] = (numpy.radians(floor) if floor else 0.0, width)

        return {
            "metal_coordination_conn_metal": conn_metal,
            "metal_coordination_conn_virt": conn_virt,
            "metal_coordination_conn_key": conn_key,
            "metal_coordination_site_params": site_params,
            "metal_coordination_fan_atoms": fan_atoms,
            "metal_coordination_fan_params": fan_params,
            "metal_coordination_bridge_internal_metal": internal_metal,
            "metal_coordination_bridge_internal_d0": internal_d0,
            "metal_coordination_bridge_params": bridge_params,
        }

    def setup_site(self, bt, i, site, conn_metal, conn_virt, site_params, fans):
        """Fill one metal's site connections, parameters and fan for block type i."""
        metal = bt.atom_to_idx[site.metal_atom]
        metal_type = bt.atoms[metal].atom_type
        virts = [bt.atom_to_idx[v] for v in site.site_virts]
        dist = _ideal_distances(self.ion_for_atom_type[metal_type], self.donor_radii)
        radial_sd, lateral_sd = self.params.widths(metal_type)
        params = [
            (dist[key], self.params.well_depth(metal_type, key), radial_sd, lateral_sd)
            for key in self.donor_keys
        ]
        for k, name in enumerate(site.site_connections):
            c = bt.connection_to_cidx[name]
            conn_metal[i, c] = metal
            conn_virt[i, c] = virts[k] if virts else NO_VIRTUAL
            site_params[i, c] = params
        if not virts:
            return
        if not site.internal_satisfiers:
            ideal = self.ideal_fan(bt, site)
            for a, b in combinations(list(ideal), 2):
                fans[i].append((a, b, numpy.linalg.norm(ideal[a] - ideal[b])))
            return
        # a cofactor's free sites hold their direction against its own atoms
        ideal = bt.ideal_coords  # in icoor order
        coords = {
            j: ideal[bt.icoors_index[bt.atoms[j].name]] for j in range(len(bt.atoms))
        }
        anchors = [metal, *(bt.atom_to_idx[a] for a in site.internal_satisfiers)]
        for n, v in enumerate(virts):
            for other in anchors + virts[:n]:
                fans[i].append((v, other, numpy.linalg.norm(coords[v] - coords[other])))

    def setup_internal_bridges(self, bt, i, internal_metal, internal_d0):
        """Metals of bt its own donating atoms satisfy, and their ideal distances."""
        if not bt.metal_sites:
            return
        ideal = bt.ideal_coords  # in icoor order

        def at(name):
            return ideal[bt.icoors_index[name]]

        site_names = {n for site in bt.metal_sites for n in site.site_connections}
        for c, conn in enumerate(bt.connections):
            if conn.name in site_names:
                continue
            metals = [
                site.metal_atom
                for site in bt.metal_sites
                if conn.atom in site.internal_satisfiers
            ][:MAX_INTERNAL_METALS]
            for k, metal in enumerate(metals):
                internal_metal[i, c, k] = bt.atom_to_idx[metal]
                internal_d0[i, c, k] = numpy.linalg.norm(at(metal) - at(conn.atom))

    def donor_key(self, atom_type):
        """Index of an atom type's metal-donor distance key, or -1."""
        for name in (atom_type.name, atom_type.element):
            if name in self.donor_keys:
                return self.donor_keys.index(name)
        return -1

    def ideal_fan(self, bt, site):
        """Ideal positions of the metal and its site virtuals, by atom index."""
        d = {ic.name: ic.d for ic in bt.icoors}
        verts = numpy.asarray(self.vertices_for[site.geometry], dtype=numpy.float64)
        verts /= numpy.linalg.norm(verts, axis=1, keepdims=True)
        out = {bt.atom_to_idx[site.metal_atom]: numpy.zeros(3)}
        for vertex, name in zip(verts, site.site_virts):
            out[bt.atom_to_idx[name]] = vertex * d[name]
        return out


def validate_donors(block_type_ind, connections, conn_virt, conn_key):
    """Every occupied metal site must connect to a parameterized donor atom."""
    bt, irc = numpy.asarray(block_type_ind), numpy.asarray(connections)
    if not (bt >= 0).any():
        return
    is_site = (conn_virt[bt.clip(min=0)] != NOT_A_SITE) & (bt >= 0)[..., None]
    pose, block, conn = numpy.nonzero(is_site & (irc[..., 0] >= 0))
    if len(pose) == 0:
        return
    partners = irc[pose, block, conn]
    key = conn_key[bt[pose, partners[:, 0]], partners[:, 1]]
    if (key < 0).any():
        raise ValueError("a metal site is bonded to an atom that is not a donor")
