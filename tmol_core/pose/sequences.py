"""Logical chain labels for ragged, optionally multichain sequences."""

import copy

import numpy


def chain_ids_from_lengths(residue_counts, max_n_residues, chain_lengths=None):
    if chain_lengths is not None and len(chain_lengths) == 0:
        chain_lengths = None
    counts = numpy.asarray(residue_counts)
    result = numpy.full((len(counts), max_n_residues), -1, dtype=numpy.int32)
    if numpy.any(counts < 0) or numpy.any(counts > max_n_residues):
        raise ValueError("Sequence residue counts exceed the padded extent")
    if chain_lengths is not None and len(chain_lengths) != len(counts):
        raise ValueError("Provide chain lengths for every pose")
    for pose, count in enumerate(counts):
        lengths = [int(count)] if chain_lengths is None else chain_lengths[pose]
        if any(length < 0 for length in lengths) or sum(lengths) != count:
            raise ValueError("Chain lengths must partition each pose's residues")
        start = 0
        for chain, length in enumerate(lengths):
            result[pose, start : start + length] = chain
            start += length
    return result


def find_connections_in_sequences(sequences):
    """Trim paired NAME--connection-label annotations in encounter order.

    Unpaired labels retain the mature behavior: they create no extra bond. A
    third use of a label is an error. Input sequences are never modified.
    """
    ps_conns = []
    trimmed_seqs = copy.deepcopy(sequences)
    for i in range(len(sequences)):
        labels = {}
        p_conns = []
        completed_labels = {}
        for j, resname in enumerate(sequences[i]):
            if len(resname) < 6:
                # X--C-I
                # is the shortest possible string containing
                # an inter-residue connection
                continue
            connections = resname.split("--")
            if len(connections) < 2:
                # no inter-residue connections specified here,
                # just a long name for the residue type
                continue
            trimmed_seqs[i][j] = connections[0]
            for conn in connections[1:]:
                conn_name, conn_label = conn.split("-")
                if conn_label in labels:
                    partner, partner_conn_name = labels[conn_label]
                    if partner == -1:
                        # error: more than two inter-residue connections have
                        # been given the same connection label
                        prev_conn = completed_labels[conn_label]
                        err_msg = (
                            "Fatal error: found more than two "
                            + "residue-connections with the "
                            + 'same connection label: "'
                            + conn_label
                            + '"'
                            + "\nPreviously encountered between residues "
                            + str(prev_conn[0])
                            + ", conn "
                            + prev_conn[1]
                            + " and "
                            + str(prev_conn[2])
                            + ", conn "
                            + prev_conn[3]
                            + " and "
                            + "now found again for "
                            + str(j)
                            + " as part of the residue "
                            + resname
                            + "\n"
                        )
                        raise ValueError(err_msg)
                    conn = (partner, partner_conn_name, j, conn_name)
                    completed_labels[conn_label] = conn
                    labels[conn_label] = (-1, None)
                    p_conns.append(conn)
                else:
                    labels[conn_label] = (j, conn_name)
        ps_conns.append(p_conns)
    return trimmed_seqs, ps_conns


def resolve_connection_pairs(
    active_types, sequences, block_type_indices, residue_connections
):
    """When there are only a handful of inter-residue connections that
    must be resolved by name, such as disulfides, then handle these
    few connections one-by-one.
    """
    ps_conn_inds = []
    bt_inds = block_type_indices

    def conn_ind_for_bt(bt_ind, c_name):
        bt = active_types[bt_ind]
        return bt.connection_to_cidx[c_name]

    for i, pose_conns in enumerate(residue_connections):
        pose_conn_inds = []
        for r1, c_name1, r2, c_name2 in pose_conns:
            c1, c2 = None, None
            try:
                bt1_ind = bt_inds[i, r1]
                bt2_ind = bt_inds[i, r2]
                c1 = conn_ind_for_bt(bt1_ind, c_name1)
                c2 = conn_ind_for_bt(bt2_ind, c_name2)
                pose_conn_inds.append((r1, c1, r2, c2))
            except KeyError:
                if c1 is None:
                    missing = (c_name1, r1, bt1_ind, c_name2, r2)
                else:
                    missing = (c_name2, r2, bt2_ind, c_name1, r1)

                new_err_msg = (
                    "Failed to find connection '"
                    + missing[0]
                    + "' on residue type '"
                    + sequences[i][missing[1]]
                    + "' which is listed as forming a chemical"
                    + " bond to connection '"
                    + missing[3]
                    + "' on residue type '"
                    + sequences[i][missing[4]]
                    + "'\n"
                    + "Valid connection names on '"
                    + sequences[i][missing[1]]
                    + "' are: "
                    + "'"
                    + "', '".join(
                        x
                        for x in active_types[missing[2]].connection_to_cidx.keys()
                        if x is not None
                    )
                    + "'"
                )
                raise ValueError(new_err_msg)

        ps_conn_inds.append(pose_conn_inds)
    return ps_conn_inds
