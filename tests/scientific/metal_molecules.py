"""Explicit coordination systems using shared chemistry and physical atom names."""

from dataclasses import dataclass

import attrs
import numpy as np

from tmol_core.chemical import _site_metal, _site_connections
from tmol_core.chemical.restypes import refined_from_raw
from .molecules import MolecularCase


@dataclass(frozen=True)
class MetalCase(MolecularCase):
    donor_patches: tuple = ()
    coordination: tuple = ()

    def database(self, base):
        from tmol_core.chemical.metal_donors import _metal_donor_patch

        database = super().database(base)
        patches = [_metal_donor_patch(*row) for row in self.donor_patches]
        return attrs.evolve(
            database, chemical=database.chemical.with_variants_applied(patches)
        )

    def extra_connections(self):
        return self.coordination

    def blocks(self, database, dtype=np.float32):
        blocks = super().blocks(database, dtype)
        raw = {r.name: r for r in database.chemical.residues}
        for row, edges in zip(blocks, self.coordination):
            # Place each shared donor at the origin, then metal centers on
            # separated rays. Keep full molecular geometry by rigid translation.
            donors = sorted({edge[2] for edge in edges})
            for donor in donors:
                name, names, coords = row[donor]
                bt = refined_from_raw(raw[name])
                port = next(edge[3] for edge in edges if edge[2] == donor)
                atom = bt.connections[bt.connection_to_cidx[port]].atom
                coords -= coords[names.index(atom)].copy()
            for index, (metal, port, donor, _) in enumerate(edges):
                name, names, coords = row[metal]
                bt = refined_from_raw(raw[name])
                site_index = _site_connections(bt).index(port)
                site, _ = _site_metal(bt, site_index)
                angle = 0.45 * index
                target = np.array(
                    [-2.25 * np.cos(angle), -2.25 * np.sin(angle), 0.10], dtype=dtype
                )
                coords += target - coords[names.index(site.metal_atom)]
        return blocks


METAL_CASES = {
    "zinc_site": MetalCase(
        (),
        (("ZN_tetrahedral", "ASP:metal_OD1"),),
        ((1, 1),),
        (("ASP", "OD1", 1),),
        (((0, "site1", 1, "metal_OD1"),),),
    ),
    "calcium_irregular": MetalCase(
        (),
        (("CA_irregular", "ASP:metal_OD1"),),
        ((1, 1),),
        (("ASP", "OD1", 1),),
        (((0, "site1", 1, "metal_OD1"),),),
    ),
    "bridging_donor": MetalCase(
        (),
        (("ZN_tetrahedral", "ZN_tetrahedral", "ASP:metal2_OD1"),),
        ((1, 1, 1),),
        (("ASP", "OD1", 2),),
        (((0, "site1", 2, "metal_OD1"), (1, "site1", 2, "metal2_OD1")),),
    ),
    "cluster_bridge": MetalCase(
        (),
        (("ZN_tetrahedral", "SF4:metal_S1"),),
        ((1, 1),),
        (("SF4", "S1", 1),),
        (((0, "site1", 1, "metal_S1"),),),
    ),
    "mixed_batch": MetalCase(
        (),
        (("ZN_tetrahedral", "ASP:metal_OD1"), ("CA_irregular", "ASP:metal_OD1", "GLY")),
        ((1, 1), (1, 1, 1)),
        (("ASP", "OD1", 1),),
        (((0, "site1", 1, "metal_OD1"),), ((0, "site1", 1, "metal_OD1"),)),
    ),
}
