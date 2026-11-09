"""Shared chemical spanning trees and generational kinematic stencils.

Inputs are refined structural residue types. Native execution frontends own
device conversion; chemistry, torsion priorities, fragment joining, scan paths
and padding conventions are prepared once here.
"""

from collections import defaultdict
import attr
import numpy
import scipy.sparse as sparse
import scipy.sparse.csgraph as csgraph

from . import NodeType


@attr.define
class BTGenerationalSegScanPathSegs:
    jump_atom: int
    parents: numpy.ndarray  # n-input x n-atoms
    dof_type: numpy.ndarray  # n-input x n-atoms
    input_conn_atom: numpy.ndarray  # n-input
    n_gens: numpy.ndarray  # n-input x n-output
    n_nodes_for_gen: numpy.ndarray
    nodes_for_gen: numpy.ndarray  # n-input x n-output x max-n-gen x max-n-nodes-per-gen
    n_scan_path_segs: numpy.ndarray  # n-input x n-output x n-gen
    scan_path_seg_that_builds_output_conn: (
        numpy.ndarray
    )  # n-input x n-output x n-conn x 2
    scan_path_seg_starts: numpy.ndarray
    scan_path_seg_is_real: numpy.ndarray
    scan_path_seg_is_inter_block: numpy.ndarray
    scan_path_seg_lengths: numpy.ndarray
    uaid_for_torsion_by_inconn: numpy.ndarray  # n-input x n-torsions x 3
    torsion_direction: numpy.ndarray  # n-input x n-torsions

    @classmethod
    def empty(
        cls,
        n_input_types: int,
        n_output_types: int,
        n_atoms: int,
        n_conn: int,
        max_n_gens: int,
        max_n_scan_path_segs_per_gen: int,
        max_n_nodes_per_gen: int,
        n_torsions: int,
    ):
        io = (n_input_types, n_output_types)
        return cls(
            jump_atom=-1,
            parents=numpy.full(
                (n_input_types, n_atoms), -1, dtype=int
            ),  # independent of primary output
            dof_type=numpy.full(
                (n_input_types, n_atoms), -1, dtype=int
            ),  # independent of primary output
            input_conn_atom=numpy.full(n_input_types, -1, dtype=int),
            n_gens=numpy.zeros(io, dtype=int),
            n_nodes_for_gen=numpy.zeros(io + (max_n_gens,), dtype=int),
            nodes_for_gen=numpy.full(
                io + (max_n_gens, max_n_nodes_per_gen), -1, dtype=int
            ),
            n_scan_path_segs=numpy.zeros(io + (max_n_gens,), dtype=int),
            scan_path_seg_that_builds_output_conn=numpy.full(
                io + (n_conn, 2), -1, dtype=int
            ),
            scan_path_seg_starts=numpy.full(
                io + (max_n_gens, max_n_scan_path_segs_per_gen), -1, dtype=int
            ),
            scan_path_seg_is_real=numpy.zeros(
                io + (max_n_gens, max_n_scan_path_segs_per_gen), dtype=bool
            ),
            scan_path_seg_is_inter_block=numpy.zeros(
                io + (max_n_gens, max_n_scan_path_segs_per_gen), dtype=bool
            ),
            scan_path_seg_lengths=numpy.zeros(
                io + (max_n_gens, max_n_scan_path_segs_per_gen), dtype=int
            ),
            uaid_for_torsion_by_inconn=numpy.full(
                (n_input_types, n_torsions, 3), -1, dtype=int
            ),
            torsion_direction=numpy.full((n_input_types, n_torsions), 1, dtype=int),
        )


def _handle_case_on_primary_exit_path(
    k_atom_ind,
    j_conn_atom,
    k_kids,
    gen_depth_given_first_descendant,
    gen_depth,
):
    # in this case, the first_descendant for this atom
    # has already been decided
    if k_atom_ind == j_conn_atom:
        # this atom's first descendent is the atom on the next residue
        # to which this residue is connected
        gen_depth[k_atom_ind] = max([gen_depth[l] for l in k_kids]) + 1
    else:
        # first_descendant is already determined for this atom
        gen_depth[k_atom_ind] = gen_depth_given_first_descendant()


def _handle_case_of_no_exit_scan_path_segment(
    bt,
    n_conn,
    i,
    j,
    k,
    k_kids,
    k_atom_ind,
    is_on_exit_sp_segment,
    first_descendant,
    interres_conn_scan_path_segment_rooted_by_atom,
    atom_rooting_scan_path_segment_for_interres_conn,
    gen_depth_given_first_descendant,
    gen_depth,
):
    # most-common case: an atom not on the primary-exit sp seg, and that isn't
    # itself a connection atom.
    # First we ask: are we on one or more exit scan path segments?
    # NOTE: this just chooses the first exit spseg atom it encounters
    # as the first descendant and so I pause and think: if we have
    # a block type with 4 inter-residue connections where the fold
    # forest branches at this residue, then the algorithm for constructing
    # the fewest-number-of-generations KinForest here is going
    # to fail: we are treating all exit paths out of this residue
    # as interchangable and we might say connection c should be
    # ahead of connection c' in a case where c' has a greater gen_depth
    # than c. We will still get a valid KinForest, but it will lack
    # the "fewest number of generations possible" property.
    #
    # The case I am designing for here is: there's a jump that has
    # landed at a beta-amino acid's CA atom and there are exit paths
    # through the N- and C-terminal ends of the residue and if the
    # primary exit path is the C-term, then the N-term exit path should
    # still have priority over the side-chain path.
    #
    #         R
    #         |
    # ...     CB    C
    #     \ /   \  / \
    #      N      CA   ...
    #
    # The path starting at CB should go towards N and not towards R.
    # If we are only dealing with polymeric residues that have an
    # up- and a down connection and that's it (e.g. nucleic acids),
    # then this algorithm will still produce optimal KinForests.
    # (I have to use a beta-amino acid as an example here because if
    # we consider the case of an alpha-amino acid, then the exit path
    # at N is already the root of a new scan path and there's no decision
    # making that has to be made.)
    #
    # A case that this would fail to deliver the optimally-efficient
    # (fewest number of generations) KinForest would be if this R group
    # also contained an inter-residue connection and there were an
    # edge in the FoldForest (a "chemical edge") leaving from that
    # connection to some further chain, e.g., it could be a sugar
    # group attached to a beta-ASN. Now if the path (CA->CB->N) takes
    # precedence over the path (CA->CB->R), then everything down-
    # stream of the R would have a generation-delay one greater than
    # it would otherwise. Again, a KinForest produced by this algorithm
    # is still valid, it could just be slightly slower to fold through
    # than it would be otherwise.
    if j != n_conn + 1:
        for kid in k_kids:
            if is_on_exit_sp_segment[kid]:
                first_descendant[k_atom_ind] = kid
                is_on_exit_sp_segment[k_atom_ind] = True
                assert interres_conn_scan_path_segment_rooted_by_atom[kid] >= 0
                kid_conn_ind = interres_conn_scan_path_segment_rooted_by_atom[kid]
                # k_atom_ind becomes the new root of the scan path
                # building to the kid_conn_ind interresidue connection
                interres_conn_scan_path_segment_rooted_by_atom[k_atom_ind] = (
                    kid_conn_ind
                )
                interres_conn_scan_path_segment_rooted_by_atom[kid] = -1
                atom_rooting_scan_path_segment_for_interres_conn[kid_conn_ind] = (
                    k_atom_ind
                )
                # stop now to ensure that we do not ovewrite the first_descendant
                # of k_atom_ind if it should happen to have two kids that
                # are on exit paths!
                break

    if not is_on_exit_sp_segment[k_atom_ind]:
        # which should be the first descendant? the one with the greatest gen depth
        first_descendant[k_atom_ind] = k_kids[
            numpy.argmax(numpy.array([gen_depth[kid] for kid in k_kids]))
        ]
    gen_depth[k_atom_ind] = gen_depth_given_first_descendant()


def _record_scan_path_segments(
    bt,
    bfto_2_orig,
    processed_node_into_scan_path_segment,
    i,
    j,
    i_conn_atom,
    n_conn,
    preds,
    gen_to_build_atom,
    first_descendant,
    is_on_exit_sp_segment,
    gen_scan_path_segments,
):
    for k in range(bt.n_atoms):
        k_atom_ind = bfto_2_orig[k]
        if processed_node_into_scan_path_segment[k_atom_ind]:
            # we have already added this atom and its first
            # descendant (and their first descendant and so on)
            # to a scan path segment, so we can continue
            continue

        # if we arrive here, that means k_atom_ind is the root of a
        # new scan path segment
        path = []
        # we have already processed the first scan path segment
        # from the entrace-point atom to the first exit-point atom
        # unless we are process the "is-a-leaf-node" case
        assert k_atom_ind != i_conn_atom or j == n_conn + 1
        # put the _parent_ of this new root at the beginning of
        # the scan path segment since we build the root's coordinate
        # frame from its parent's coordinate frame
        if k_atom_ind != i_conn_atom:
            path.append(preds[k_atom_ind])
            focused_atom = k_atom_ind

            gen_to_build_atom[focused_atom] = gen_to_build_atom[preds[focused_atom]] + 1
        else:
            focused_atom = k_atom_ind
            gen_to_build_atom[focused_atom] = 0

        # now we traverse the path along each atom's first descendant
        while focused_atom >= 0:
            path.append(focused_atom)
            processed_node_into_scan_path_segment[focused_atom] = True
            focused_atom = first_descendant[focused_atom]
            if focused_atom >= 0:
                gen_to_build_atom[focused_atom] = gen_to_build_atom[preds[focused_atom]]

        if is_on_exit_sp_segment[k_atom_ind]:
            # we will go ahead and put exit sp segs at the beginning of the
            # list of scan path segs for a generation, however, there is no
            # demand that we must do so.
            gen_scan_path_segments[gen_to_build_atom[k_atom_ind]].insert(0, path)
        else:
            gen_scan_path_segments[gen_to_build_atom[k_atom_ind]].append(path)


def _annotate_block_type_for_in_and_out_pair(  # noqa: C901
    i,
    i_conn_atom,
    j,
    bt,
    n_conn,
    preds,
    bfto_2_orig,
    is_conn_atom,
    scan_path_segment_data,
):
    # we will generate a list of scan-path segments for each generation
    # and as part of this building process, we will track which scan-
    # path segments are exit paths to other blocks.
    gen_scan_path_segments = defaultdict(list)
    atom_rooting_scan_path_segment_for_interres_conn = numpy.full(
        (n_conn,), -1, dtype=numpy.int64
    )
    interres_conn_scan_path_segment_rooted_by_atom = numpy.full(
        (bt.n_atoms,), -1, dtype=numpy.int64
    )
    scan_path_segment_building_interres_conn = numpy.full(
        (n_conn,), -1, dtype=numpy.int64
    )
    gen_of_scan_path_segment_building_interres_conn = numpy.full(
        (n_conn,), -1, dtype=numpy.int64
    )

    is_on_primary_exit_sp_seg = numpy.zeros((bt.n_atoms,), dtype=bool)
    first_descendant = numpy.full((bt.n_atoms,), -9999, dtype=numpy.int64)
    if j <= n_conn:
        # Case 1: we have a designated exit from this block type to
        # the next block in the kinematic tree.
        #
        # Start at the j_conn_atom and work backwards toward the root,
        # which marks the first scan-path segment for this block type:
        # the "primary exit scan-path segment"
        j_conn_atom = (
            bt.ordered_connection_atoms[j]
            if j < n_conn
            else bt.default_jump_connection_atom_index
        )

        is_on_primary_exit_sp_seg[i_conn_atom] = True

        focused_atom = j_conn_atom
        primary_exit_scan_path_segment = []
        while focused_atom != i_conn_atom:
            is_on_primary_exit_sp_seg[focused_atom] = True
            primary_exit_scan_path_segment.append(focused_atom)
            pred = preds[focused_atom]
            first_descendant[pred] = focused_atom
            focused_atom = pred
        primary_exit_scan_path_segment.append(i_conn_atom)
        primary_exit_scan_path_segment.reverse()
        # we need to prioritize exit scan-path segments of all stripes
        # in constructing the trees
        is_on_exit_sp_segment = is_on_primary_exit_sp_seg.copy()
        for k in range(n_conn):
            if k == i or k == j:
                continue  # truly unnecessary; nothing changes if I remove these two lines
            k_conn_atom = bt.ordered_connection_atoms[k]
            is_on_exit_sp_segment[k_conn_atom] = True
            atom_rooting_scan_path_segment_for_interres_conn[k] = k_conn_atom
            interres_conn_scan_path_segment_rooted_by_atom[k_conn_atom] = k
        gen_scan_path_segments[0].append(primary_exit_scan_path_segment)
        # our first exit scan path segment: keep track of the gen/scan-path-seg indices
        # for exit scan-path segments using inter-residue connections. We don't have
        # to worry about scan paths that exit by jump or that dont exit.
        if j < n_conn:
            gen_of_scan_path_segment_building_interres_conn[j] = 0
            scan_path_segment_building_interres_conn[j] = 0
    else:
        # Case 2: A leaf node of the kinematic tree.
        # we will not be exiting from any connection point.
        # NOTE: this is an inter-block segment
        primary_exit_scan_path_segment = []
        is_on_exit_sp_segment = numpy.zeros((bt.n_atoms,), dtype=bool)
        pass

    # Create a list of children for each atom.
    n_kids = numpy.zeros((bt.n_atoms,), dtype=numpy.int64)
    atom_kids = [[] for _ in range(bt.n_atoms)]
    for k in range(bt.n_atoms):
        if preds[k] < 0:
            assert (
                k == i_conn_atom
            ), f"bad predecesor for atom {k} in {bt.name}, {preds[k]}"
            continue  # the root
        n_kids[preds[k]] += 1
        atom_kids[preds[k]].append(k)

    # now we label each node with its "generation depth" using a
    # leaf-to-root traversal perscribed by the original DFS, taking
    # into account the fact that priority must be given to
    # exit scan-path segments -- that is, we must describe exit spsegs
    # being the first children of their parents and the other children
    # as being younger siblings.
    gen_depth = numpy.ones((bt.n_atoms,), dtype=numpy.int64)
    for k in range(bt.n_atoms - 1, -1, -1):
        k_atom_ind = bfto_2_orig[k]
        k_kids = atom_kids[k_atom_ind]
        if len(k_kids) == 0:
            continue
        # from here forward, we know that k_atom_ind has > 0 children

        def gen_depth_given_first_descendant():
            # First, set the first_descendant for k_atom_ind.
            # Then, the logic is: we have to add one to the
            # gen-depth of every child except the first descendant
            # which we get "for free" since it will be built
            # along the same-scan path segment as k_atom_ind
            return max(
                [
                    (
                        gen_depth[k_kid] + 1
                        if k_kid != first_descendant[k_atom_ind]
                        else gen_depth[k_kid]
                    )
                    for k_kid in k_kids
                ]
            )

        if is_on_primary_exit_sp_seg[k_atom_ind]:
            _handle_case_on_primary_exit_path(
                k_atom_ind,
                j_conn_atom,
                k_kids,
                gen_depth_given_first_descendant,
                gen_depth,
            )
        else:

            if j != n_conn + 1 and is_conn_atom[k_atom_ind]:
                # In this case, "the" connection (there can possibly be more than one!)
                # will be the first child and the other descendants will be second children.
                # We save the gen depth, but when calculating the gen depth of the
                # fold-forest, if this residue is at the upstream end of an edge, then
                # its depth will have to be calculated as the min gen-depth of the
                # intra-residue bits and the gen-depth of the nodes downstream of it.
                # TO DO: This case needs to be properly handled when calculating the
                # maximum number of generations to run gen-seg-scan.
                gen_depth[k_atom_ind] = max([gen_depth[l] for l in k_kids]) + 1
            else:
                _handle_case_of_no_exit_scan_path_segment(
                    bt,
                    n_conn,
                    i,
                    j,
                    k,
                    k_kids,
                    k_atom_ind,
                    is_on_exit_sp_segment,
                    first_descendant,
                    interres_conn_scan_path_segment_rooted_by_atom,
                    atom_rooting_scan_path_segment_for_interres_conn,
                    gen_depth_given_first_descendant,
                    gen_depth,
                )
    # OKAY!
    # now we have paths rooted at each node up to the root
    # we need to turn these paths into scan paths
    # Let's now traverse the atoms in bfs order and build the scan paths
    # along the way
    processed_node_into_scan_path_segment = is_on_primary_exit_sp_seg.copy()
    gen_to_build_atom = numpy.full((bt.n_atoms,), -1, dtype=numpy.int64)
    gen_to_build_atom[is_on_primary_exit_sp_seg] = 0

    _record_scan_path_segments(
        bt,
        bfto_2_orig,
        processed_node_into_scan_path_segment,
        i,
        j,
        i_conn_atom,
        n_conn,
        preds,
        gen_to_build_atom,
        first_descendant,
        is_on_exit_sp_segment,
        gen_scan_path_segments,
    )

    # Now we need to assemble the scan path segments in a compact way:
    ij_n_gens = gen_depth[i_conn_atom]
    ij_n_scan_path_segments = numpy.array(
        [len(gen_scan_path_segments[k]) for k in range(ij_n_gens)], dtype=int
    )
    ij_scan_path_segment_starts = [
        numpy.zeros((ij_n_scan_path_segments[k],), dtype=int) for k in range(ij_n_gens)
    ]
    ij_scan_path_segment_lengths = [
        numpy.array(
            [
                len(gen_scan_path_segments[k][l])
                for l in range(len(gen_scan_path_segments[k]))
            ],
            dtype=int,
        )
        for k in range(ij_n_gens)
    ]
    for k in range(ij_n_gens):
        offset = 0
        for l in range(ij_n_scan_path_segments[k]):
            ij_scan_path_segment_starts[k][l] = offset
            offset += ij_scan_path_segment_lengths[k][l]
    ij_scan_path_segment_is_inter_block = [
        numpy.zeros((ij_n_scan_path_segments[k],), dtype=bool) for k in range(ij_n_gens)
    ]

    for k in range(ij_n_gens):
        for l in range(ij_n_scan_path_segments[k]):
            l_first_at = gen_scan_path_segments[k][l][0 if k == 0 else 1]
            # "interblock" is really asking "does this scan path segment
            # exit to a different block?". This is "answered" by whether the
            # last atom in the scan path segment is a connection atom.
            # The SPSs that are inter-block are going to be roots of SPs
            # in the forward pass, and they are likely to not be roots
            # of SPs in the backward pass as long as there are edges leaving
            # from the connection atoms.
            kl_last_atom = gen_scan_path_segments[k][l][-1]
            ij_scan_path_segment_is_inter_block[k][l] = (
                is_conn_atom[kl_last_atom] and j != n_conn + 1
            ) or (  # is the last atom in the path a connection atom?
                k == 0 and l == 0
            )  # the first scan path segment is always inter-block
            conn_for_path = interres_conn_scan_path_segment_rooted_by_atom[l_first_at]
            if conn_for_path != -1:
                gen_of_scan_path_segment_building_interres_conn[conn_for_path] = k
                scan_path_segment_building_interres_conn[conn_for_path] = l

    ij_n_nodes_for_gen = numpy.array(
        [
            sum(len(path) for path in gen_scan_path_segments[k])
            for k in range(ij_n_gens)
        ],
        dtype=int,
    )
    scan_path_segment_data[(i, j)] = dict(
        n_gens=ij_n_gens,
        n_nodes_for_gen=ij_n_nodes_for_gen,
        nodes_for_gen=gen_scan_path_segments,
        n_scan_path_segs=ij_n_scan_path_segments,
        gen_building_output_conn=gen_of_scan_path_segment_building_interres_conn,
        scan_path_seg_building_output_conn=scan_path_segment_building_interres_conn,
        scan_path_seg_starts=ij_scan_path_segment_starts,
        scan_path_seg_is_inter_block=ij_scan_path_segment_is_inter_block,
        scan_path_seg_lengths=ij_scan_path_segment_lengths,
    )


def _mark_kin_uaids_controlling_torsions(
    uaid_for_torsion, torsion_direction, i, iconn_at, bt, preds
):
    for j in range(bt.n_torsions):
        at2 = bt.ordered_torsions[j][1][0]
        at3 = bt.ordered_torsions[j][2][0]
        if at2 == -1 and at3 == -1:
            # okay, both atoms live on other residues
            # thus, we have to make some guesses.
            # At least atom1 has to be in this residue
            # so if atom1 is the connection atom for
            # the input connection i, then we will reason
            # atom 3 must be the partent of atom 2
            # on the other residue that we do not have
            # access to; otherwise, we will conclude
            # that we will leave this residue through
            # atom 1 and arrive at atom 2 first and
            # therefore atom 2 is the parent of atom 3
            #
            # case 1:
            # atom3
            #   |
            #   v
            # atom2 (torsion lives in phi_c of this atom)
            #   |
            #   v
            # atom1 (input connection atom)
            #
            # case 2:
            # atom1 (some-non-input-connection-connection atom)
            #   |
            #   v
            # atom2 (the input connectio atom on the downstream residue)
            #   |
            #   v
            # atom3 (torsion lives in phi_c of this atom)

            at1 = bt.ordered_torsions[j][0][0]
            assert at1 != -1
            if at1 == iconn_at:
                # case 1
                # atom 3 is the parent of atom 2
                uaid_for_torsion[i, j, :] = bt.ordered_torsions[j][1]
                torsion_direction[i, j] = -1
            else:
                # case 2
                # atom 2 is the parent of atom 3
                uaid_for_torsion[i, j, :] = bt.ordered_torsions[j][2]
                torsion_direction[i, j] = 1
        else:
            # at least one of atoms 2 and 3 are in this residue
            if at2 == -1 or at3 == -1:
                # but one of them is not
                in_res_at = at2 if at2 != -1 else at3
                if in_res_at == iconn_at:
                    # then atom 3 is the parent of atom 2
                    # and the torsion lives in phi_c of atom 2
                    uaid_for_torsion[i, j, :] = bt.ordered_torsions[j][1]
                    torsion_direction[i, j] = -1
                else:
                    # then atom 2 is the parent of atom 3
                    # and the torsion lives in phi_c of atom 3
                    uaid_for_torsion[i, j, :] = bt.ordered_torsions[j][2]
                    torsion_direction[i, j] = 1
            else:
                # easiest case! both atoms 2 and 3 are intra-residue
                # torsion lives in phi_c of the child atom
                assert preds[at2] == at3 or preds[at3] == at2
                at2_is_parent = preds[at3] == at2
                uaid_for_torsion[i, j, :] = (
                    bt.ordered_torsions[j][2]
                    if at2_is_parent
                    else bt.ordered_torsions[j][1]
                )
                torsion_direction[i, j] = 1 if at2_is_parent else -1


def _gather_scan_path_data(scan_path_segment_data, i, j, bt_gen_seg_scan_path_segments):
    ij_n_gens = scan_path_segment_data[(i, j)]["n_gens"]
    bt_gen_seg_scan_path_segments.n_gens[i, j] = ij_n_gens
    bt_gen_seg_scan_path_segments.scan_path_seg_that_builds_output_conn[i, j, :, 0] = (
        scan_path_segment_data[(i, j)]["gen_building_output_conn"]
    )
    bt_gen_seg_scan_path_segments.scan_path_seg_that_builds_output_conn[i, j, :, 1] = (
        scan_path_segment_data[(i, j)]["scan_path_seg_building_output_conn"]
    )
    for k in range(ij_n_gens):
        bt_gen_seg_scan_path_segments.n_nodes_for_gen[i, j, k] = scan_path_segment_data[
            (i, j)
        ]["n_nodes_for_gen"][k]
        bt_gen_seg_scan_path_segments.n_scan_path_segs[i, j, k] = (
            scan_path_segment_data[(i, j)]["n_scan_path_segs"][k]
        )
        bt_gen_seg_scan_path_segments.scan_path_seg_is_real[
            i, j, k, : bt_gen_seg_scan_path_segments.n_scan_path_segs[i, j, k]
        ] = True

        ijk_n_scan_path_segs = scan_path_segment_data[(i, j)]["n_scan_path_segs"][k]
        bt_gen_seg_scan_path_segments.scan_path_seg_starts[
            i, j, k, :ijk_n_scan_path_segs
        ] = scan_path_segment_data[(i, j)]["scan_path_seg_starts"][k]
        bt_gen_seg_scan_path_segments.scan_path_seg_is_inter_block[
            i, j, k, :ijk_n_scan_path_segs
        ] = scan_path_segment_data[(i, j)]["scan_path_seg_is_inter_block"][k]
        bt_gen_seg_scan_path_segments.scan_path_seg_lengths[
            i, j, k, :ijk_n_scan_path_segs
        ] = scan_path_segment_data[(i, j)]["scan_path_seg_lengths"][k]
        for l in range(ijk_n_scan_path_segs):
            m_offset = scan_path_segment_data[(i, j)]["scan_path_seg_starts"][k][l]
            for m in range(len(scan_path_segment_data[(i, j)]["nodes_for_gen"][k][l])):
                bt_gen_seg_scan_path_segments.nodes_for_gen[i, j, k, m_offset + m] = (
                    scan_path_segment_data[(i, j)]["nodes_for_gen"][k][l][m]
                )


def _bond_csgraph(bonds, weight, n_atoms):
    """A sparse n_atoms x n_atoms graph with one weighted edge per bond row."""
    weights = numpy.broadcast_to(
        numpy.full((1,), weight, dtype=numpy.float32), bonds[:, 0].shape
    )
    return sparse.csr_matrix(
        (weights, (bonds[:, 0], bonds[:, 1])), shape=(n_atoms, n_atoms)
    )


@attr.s(auto_attribs=True)
class ResidueKinforestData:
    """Per-block-type kinforest data rooted at the default jump connection
    atom (e.g. CA for amino acids, O for water).

    Produced by annotate_block_type_with_residue_kinforest_data and consumed
    by construct_single_residue_kinforest to build the rotamer kinforest
    without the deprecated _KinematicBuilder.
    """

    jump_atom: int  # root atom index in original (TO) atom order
    preds: numpy.ndarray  # BFS predecessors in TO; -9999 for root
    bfto_2_orig: numpy.ndarray  # BFS traversal order → TO (= KFO ordering)
    dof_type: numpy.ndarray  # NodeType per atom in TO


def _tree_bond_indices(bt):
    """The block type's bonds plus one tree-only edge into each unbonded fragment.

    A block type may hold several molecules, such as a ligand and a counter-ion.
    Its icoors place each further fragment from an atom of another; that pair is
    the edge the kinematic tree follows, though it is not a bond.
    """
    bonds = numpy.asarray(bt.bond_indices, dtype=numpy.int64).reshape(-1, 2)

    def components(edges):
        graph = sparse.csr_matrix(
            (numpy.ones(len(edges)), (edges[:, 0], edges[:, 1])),
            shape=(bt.n_atoms, bt.n_atoms),
        )
        return csgraph.connected_components(graph, directed=False)

    n_fragments, fragment = components(bonds)
    if n_fragments == 1:
        return bonds
    joins = []
    for i in range(bt.n_atoms):
        parent_icoor = bt.icoors_ancestors[bt.at_to_icoor_ind[i], 0]
        parent = bt.atom_to_idx.get(bt.icoors[parent_icoor].name)
        if parent is not None and fragment[parent] != fragment[i]:
            joins += [(i, parent), (parent, i)]
    edges = numpy.concatenate([bonds, numpy.array(joins, dtype=numpy.int64)])
    if components(edges)[0] != 1:
        raise ValueError(
            f"{bt.name}: its atoms form {n_fragments} unbonded fragments and its "
            "icoors do not place each one from another, so no kinematic tree "
            "reaches them all"
        )
    return edges


def _block_group_kinforest_data(block_types, links, anchor: int = 0):
    """Spanning tree over a group of blocks joined by inter-block bonds.

    The single-block case of annotate_block_type_with_residue_kinforest_data,
    generalized: atoms of the group's blocks are concatenated, each link adds
    one bond between the two blocks' connection atoms, and the tree is rooted
    at the anchor block's jump atom. Torsion-defining bonds -- the linkage
    torsions among them -- are prioritized so they become tree edges and so
    carry a degree of freedom.

    ``links`` are (parent, parent_conn, child, child_conn) tuples, where the
    block entries index ``block_types`` and the conn entries are connection
    indices on those blocks. Returns the ResidueKinforestData for the
    concatenated group together with each block's atom offset.
    """
    offsets = numpy.cumsum([0] + [bt.n_atoms for bt in block_types])
    n_atoms = int(offsets[-1])

    all_bonds, tor_bonds = [], []
    for i, bt in enumerate(block_types):
        off = int(offsets[i])
        all_bonds.append(_tree_bond_indices(bt) + off)
        tor_bonds.extend(
            (uaids[1][0] + off, uaids[2][0] + off)
            for uaids in bt.torsion_to_uaids.values()
            if uaids[1][0] >= 0 and uaids[2][0] >= 0
        )

    # each link is a bond, and it is a torsion axis: the chi built on it must
    #    be a tree edge or the linkage has no degree of freedom
    for parent, parent_conn, child, child_conn in links:
        a = int(offsets[parent]) + int(
            block_types[parent].ordered_connection_atoms[parent_conn]
        )
        b = int(offsets[child]) + int(
            block_types[child].ordered_connection_atoms[child_conn]
        )
        all_bonds.append(numpy.array([[a, b], [b, a]], dtype=numpy.int64))
        tor_bonds.extend([(a, b), (b, a)])

    all_bonds = numpy.concatenate(all_bonds)
    tor_bonds = (
        numpy.array(tor_bonds, dtype=numpy.int64)
        if tor_bonds
        else numpy.zeros((0, 2), dtype=numpy.int64)
    )

    bond_graph = _bond_csgraph(all_bonds, -1, n_atoms) + _bond_csgraph(
        tor_bonds, -0.125, n_atoms
    )
    spanning_tree = csgraph.minimum_spanning_tree(bond_graph.tocsr())

    jump_atom = int(offsets[anchor]) + (
        block_types[anchor].default_jump_connection_atom_index
    )
    bfto_2_orig, preds = csgraph.breadth_first_order(
        spanning_tree, jump_atom, directed=False, return_predecessors=True
    )

    dof_type = numpy.full((n_atoms,), NodeType.bond, dtype=numpy.int64)
    dof_type[jump_atom] = NodeType.jump

    return (
        ResidueKinforestData(
            jump_atom=jump_atom,
            preds=preds,
            bfto_2_orig=bfto_2_orig,
            dof_type=dof_type,
        ),
        offsets.astype(numpy.int64),
    )


def _annotate_block_type_with_residue_kinforest_data(bt):
    """Annotate a block type with the single-residue kinforest spanning tree,
    rooted at its default jump connection atom (CA for amino acids, O for water).

    Caches:
      bt.residue_kinforest_data  -- ResidueKinforestData (public, used by rotamer packing)
      bt._bond_graph_mst         -- scipy MST (private, reused by
                                    _annotate_block_type_with_gen_scan_path_segs)
    """
    if hasattr(bt, "residue_kinforest_data"):
        return

    potential_bonds = _bond_csgraph(_tree_bond_indices(bt), -1, bt.n_atoms)
    tor_atoms = [
        (uaids[1][0], uaids[2][0])
        for tor, uaids in bt.torsion_to_uaids.items()
        if uaids[1][0] >= 0 and uaids[2][0] >= 0
    ]
    if len(tor_atoms) == 0:
        tor_atoms = numpy.zeros((0, 2), dtype=numpy.int64)
    else:
        tor_atoms = numpy.array(tor_atoms)

    prioritized_bonds = _bond_csgraph(tor_atoms, -0.125, bt.n_atoms)
    bond_graph = potential_bonds + prioritized_bonds
    bond_graph_spanning_tree = csgraph.minimum_spanning_tree(bond_graph.tocsr())

    # Cache MST so _annotate_block_type_with_gen_scan_path_segs can reuse it
    # without recomputing the (identical) spanning tree.
    setattr(bt, "_bond_graph_mst", bond_graph_spanning_tree)

    jump_atom = bt.default_jump_connection_atom_index
    bfto_2_orig, preds = csgraph.breadth_first_order(
        bond_graph_spanning_tree,
        jump_atom,
        directed=False,
        return_predecessors=True,
    )

    dof_type = numpy.full((bt.n_atoms,), NodeType.bond, dtype=numpy.int64)
    dof_type[jump_atom] = NodeType.jump

    setattr(
        bt,
        "residue_kinforest_data",
        ResidueKinforestData(
            jump_atom=jump_atom,
            preds=preds,
            bfto_2_orig=bfto_2_orig,
            dof_type=dof_type,
        ),
    )


# TO DO: jit this!
def _annotate_block_type_with_gen_scan_path_segs(bt):  # noqa: C901
    if hasattr(bt, "gen_seg_scan_path_segs"):
        return
    n_conn = len(bt.connections)

    n_input_types = n_conn + 2  # n_conn + jump input + root "input"
    n_output_types = n_conn + 2  # n_conn + jump output + no output at all

    # Compute (and cache) the MST and jump-rooted BFS once, reused below for
    # the i >= n_conn input types and by construct_single_residue_kinforest.
    _annotate_block_type_with_residue_kinforest_data(bt)
    bond_graph_spanning_tree = bt._bond_graph_mst

    # As we are iterating across atoms, we need to keep track of which atoms
    # are bridges to other resiudes, so write down the reverse mapping from
    # atom index to the inter-residue connection index
    is_conn_atom = numpy.zeros((bt.n_atoms,), dtype=bool)
    conn_ind_for_atom = numpy.full((bt.n_atoms,), -1, dtype=numpy.int64)
    for i in range(n_conn):
        is_conn_atom[bt.ordered_connection_atoms[i]] = True
        conn_ind_for_atom[bt.ordered_connection_atoms[i]] = i

    scan_path_segment_data = {}
    parents = numpy.full((n_input_types, bt.n_atoms), -1, dtype=numpy.int64)
    input_conn_atom = numpy.zeros((n_input_types,), dtype=numpy.int64)
    dof_type = numpy.full(
        (
            n_input_types,
            bt.n_atoms,
        ),
        NodeType.bond,
        dtype=numpy.int64,
    )

    uaid_for_torsion = numpy.full(
        (n_input_types, bt.n_torsions, 3), -1, dtype=numpy.int64
    )
    torsion_direction = numpy.full((n_input_types, bt.n_torsions), 1, dtype=numpy.int64)

    for i in range(n_input_types):

        i_conn_atom = (
            bt.ordered_connection_atoms[i]
            if i < n_conn
            else bt.default_jump_connection_atom_index
        )
        input_conn_atom[i] = i_conn_atom
        if i < n_conn:
            bfto_2_orig, preds = csgraph.breadth_first_order(
                bond_graph_spanning_tree,
                i_conn_atom,
                directed=False,
                return_predecessors=True,
            )
        else:
            # Both i == n_conn and i == n_conn+1 are rooted at
            # default_jump_connection_atom_index; reuse the cached BFS result.
            bfto_2_orig = bt.residue_kinforest_data.bfto_2_orig
            preds = bt.residue_kinforest_data.preds
        parents[i, :] = preds
        if i >= n_conn:
            dof_type[i, i_conn_atom] = NodeType.jump

        _mark_kin_uaids_controlling_torsions(
            uaid_for_torsion, torsion_direction, i, i_conn_atom, bt, preds
        )

        # Now, the parent of the i_conn_atom comes from the previous residue, so we will
        # need to fix this atom when we are hooking the blocks together. For now, leave
        # it as -9999 (which is what csgraph labels it as) so that we can tell if we have
        # not corrected this parent index later on.
        for j in range(n_output_types):
            if i == j and i < n_conn:
                # we cannot enter from one inter-residue connection point and then
                # leave by that same inter-residue connection point unless we are
                # building a jump
                continue
            _annotate_block_type_for_in_and_out_pair(
                i,
                i_conn_atom,
                j,
                bt,
                n_conn,
                preds,
                bfto_2_orig,
                is_conn_atom,
                scan_path_segment_data,
            )

        # end for j
    # end for i

    # Now let's count out the maximum number of generations, scans, and nodes-per-gen
    # so we can create the BTGenerationalSegScanPaths object
    max_n_gens = max(
        scan_path_segment_data[(i, j)]["n_gens"]
        for i in range(n_input_types)
        for j in range(n_output_types)
        if (i, j) in scan_path_segment_data
    )
    max_n_scan_path_segments = max(
        max(
            scan_path_segment_data[(i, j)]["n_scan_path_segs"][k]
            for k in range(scan_path_segment_data[(i, j)]["n_gens"])
        )
        for i in range(n_input_types)
        for j in range(n_output_types)
        if (i, j) in scan_path_segment_data
    )
    max_n_nodes_per_gen = max(
        max(
            scan_path_segment_data[(i, j)]["n_nodes_for_gen"][k]
            for k in range(scan_path_segment_data[(i, j)]["n_gens"])
        )
        for i in range(n_input_types)
        for j in range(n_output_types)
        if (i, j) in scan_path_segment_data
    )
    bt_gen_seg_scan_path_segments = BTGenerationalSegScanPathSegs.empty(
        n_input_types,
        n_output_types,
        bt.n_atoms,
        n_conn,
        max_n_gens,
        max_n_scan_path_segments,
        max_n_nodes_per_gen,
        bt.n_torsions,
    )
    bt_gen_seg_scan_path_segments.jump_atom = bt.default_jump_connection_atom_index
    bt_gen_seg_scan_path_segments.parents = parents
    bt_gen_seg_scan_path_segments.dof_type[:] = dof_type
    bt_gen_seg_scan_path_segments.input_conn_atom = input_conn_atom
    bt_gen_seg_scan_path_segments.uaid_for_torsion_by_inconn = uaid_for_torsion
    bt_gen_seg_scan_path_segments.torsion_direction = torsion_direction
    # Finally, we populate the BTGenerationalSegScanPathSegs object
    for i in range(n_input_types):
        for j in range(n_output_types):
            if (i, j) not in scan_path_segment_data:
                continue
            _gather_scan_path_data(
                scan_path_segment_data, i, j, bt_gen_seg_scan_path_segments
            )

    setattr(bt, "gen_seg_scan_path_segs", bt_gen_seg_scan_path_segments)


def empty_packed_kinematic_stencils(
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
    io = (n_bt, max_n_input_types, max_n_output_types)
    return dict(
        jump_atom=numpy.full((n_bt,), -1, dtype=numpy.int32),
        parents=numpy.full(
            (n_bt, max_n_input_types, max_n_atoms),
            -1,
            dtype=numpy.int32,
        ),  # independent of primary output
        dof_type=numpy.full(
            (n_bt, max_n_input_types, max_n_atoms),
            -1,
            dtype=numpy.int32,
        ),  # independent of primary output
        input_conn_atom=numpy.full((n_bt, max_n_input_types), -1, dtype=numpy.int32),
        n_gens=numpy.zeros(io, dtype=numpy.int32),
        n_nodes_for_gen=numpy.zeros(io + (max_n_gens,), dtype=numpy.int32),
        nodes_for_gen=numpy.full(
            io + (max_n_gens, max_n_nodes_per_gen),
            -1,
            dtype=numpy.int32,
        ),
        n_scan_path_segs=numpy.zeros(io + (max_n_gens,), dtype=numpy.int32),
        scan_path_seg_that_builds_output_conn=numpy.full(
            io + (max_n_conn, 2), -1, dtype=numpy.int32
        ),
        scan_path_seg_starts=numpy.full(
            io + (max_n_gens, max_n_scan_path_segs_per_gen),
            -1,
            dtype=numpy.int32,
        ),
        scan_path_seg_is_real=numpy.zeros(
            io + (max_n_gens, max_n_scan_path_segs_per_gen),
            dtype=bool,
        ),
        scan_path_seg_is_inter_block=numpy.zeros(
            io + (max_n_gens, max_n_scan_path_segs_per_gen),
            dtype=bool,
        ),
        scan_path_seg_lengths=numpy.zeros(
            io + (max_n_gens, max_n_scan_path_segs_per_gen),
            dtype=numpy.int32,
        ),
        uaid_for_torsion_by_inconn=numpy.full(
            (n_bt, max_n_input_types, max_n_torsions, 3),
            -1,
            dtype=numpy.int32,
        ),
        torsion_direction=numpy.full(
            (n_bt, max_n_input_types, max_n_torsions),
            1,
            dtype=numpy.int32,
        ),
    )


def prepare_packed_kinematic_stencils(
    block_types, *, max_n_atoms, max_n_conn, max_n_torsions
):
    """Pack all residue scan metadata into native int32/bool host arrays.

    Every array is named like the existing per-type stencil field, with a
    leading residue-type axis. The caller supplies physical palette capacities;
    scan capacities are derived from the prepared residue graphs. Unused entries
    retain the canonical zero/-1/+1 padding of the native ABI.
    """
    block_types = tuple(block_types)
    for bt in block_types:
        _annotate_block_type_with_gen_scan_path_segs(bt)
    data = [bt.gen_seg_scan_path_segs for bt in block_types]

    def maximum(field, axis):
        return max((getattr(value, field).shape[axis] for value in data), default=0)

    packed = empty_packed_kinematic_stencils(
        len(data),
        maximum("n_gens", 0),
        maximum("n_gens", 1),
        max_n_atoms,
        max_n_conn,
        maximum("n_nodes_for_gen", 2),
        maximum("scan_path_seg_starts", 3),
        maximum("nodes_for_gen", 3),
        max_n_torsions,
    )
    for i, value in enumerate(data):
        for field in attr.fields(BTGenerationalSegScanPathSegs):
            src = numpy.asarray(getattr(value, field.name))
            packed[field.name][(i, *(slice(0, extent) for extent in src.shape))] = src
    return packed
