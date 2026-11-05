"""Shared Dunbrack scoring and sampling table preparation.

Inputs are structural records containing NumPy arrays. The fitting callback
accepts a sequence of rank-2 or rank-3 float32 tables and returns coefficients
from the canonical native periodic spline prefilter. Framework adapters own
only storage conversion and the native binding.
"""

import itertools
import attr
import numpy


@attr.s(auto_attribs=True, frozen=True, slots=True)
class DunbrackTableData:
    scoring_db: dict
    scoring_db_aux: dict
    sampling_db: dict
    all_table_indices: dict
    rotameric_table_indices: dict
    semirotameric_table_indices: dict


def _exclusive_cumsum1d(values):
    return numpy.cumsum(values, dtype=values.dtype) - values


def _concatenate_padded(arrays):
    shape = [max(a.shape[i] for a in arrays) for i in range(arrays[0].ndim)]
    shape[0] = sum(a.shape[0] for a in arrays)
    result = numpy.zeros(shape, dtype=arrays[0].dtype)
    offset = 0
    for array in arrays:
        result[
            (
                slice(offset, offset + array.shape[0]),
                *(slice(0, s) for s in array.shape[1:]),
            )
        ] = array
        offset += array.shape[0]
    return result


def prepare_dunbrack_tables(database, fit_coefficients):
    """Prepare scoring, sampling, topology and validated residue lookup tables."""
    return _DunbrackBuilder(fit_coefficients).prepare(database)


class _DunbrackBuilder:
    def __init__(self, fit_coefficients):
        self.fit_coefficients = fit_coefficients

    def _pack_coefficients(self, tables, ndim=2):
        if not tables:
            metadata = numpy.empty((0, ndim), dtype=numpy.int64)
            return (
                numpy.empty((0,) * (ndim + 1), dtype=numpy.float32),
                metadata,
                metadata,
            )
        coefficients = self.fit_coefficients(tables)
        shape = [max(a.shape[i] for a in coefficients) for i in range(ndim)]
        result = numpy.zeros((len(coefficients), *shape), dtype=coefficients[0].dtype)
        for i, array in enumerate(coefficients):
            result[(i, *(slice(0, s) for s in array.shape))] = array
        sizes = numpy.asarray([a.shape for a in coefficients], dtype=numpy.int64)
        strides = numpy.tile(
            numpy.asarray(result.strides[1:]) // result.itemsize, (len(coefficients), 1)
        )
        return result, sizes, strides

    def prepare(self, dun_database):
        all_rotlibs = [
            rotlib
            for rotlib in itertools.chain(
                dun_database.rotameric_libraries, dun_database.semi_rotameric_libraries
            )
        ]

        (
            all_table_indices,
            rotameric_table_indices,
            semirotameric_table_indices,
        ) = self._create_table_indices(
            [x.table_name for x in all_rotlibs],
            dun_database.dun_lookup,
            len(dun_database.rotameric_libraries),
        )
        nchi_for_table_set = self._create_nchi_for_table_set(all_rotlibs)

        prob_table_nrots, prob_table_offsets = self._create_prob_table_offsets(
            all_rotlibs
        )
        p_coeffs, pc_sizes, pc_strides, nlp_coeffs = self._compute_rotprob_coeffs(
            all_rotlibs
        )

        rotameric_mean_offsets = self._create_rot_mean_offsets(all_rotlibs)

        mean_coeffs, mc_sizes, mc_strides = self._calculate_rot_mean_coeffs(all_rotlibs)
        sdev_coeffs = self._calculate_rot_sdev_coeffs(all_rotlibs)

        rot_bb_start, rot_bb_step, rot_bb_per = self._create_rot_periodicities(
            all_rotlibs
        )

        rot_ri2ti, semirot_ri2ti, allchi_ri2ti = self._create_rotind2tableinds(
            dun_database
        )

        rotameric_chi_ri2ti_offsets = self._create_rotameric_rotind2tableind_offsets(
            dun_database
        )
        all_chi_rotind2tableind_offsets = self._create_all_chi_rotind2tableind_offsets(
            dun_database
        )

        sr_coeffs, sr_sizes, sr_strides = self._calc_semirot_coeffs(dun_database)

        sr_start, sr_step, sr_periodicity = self._create_semirot_periodicity(
            dun_database,
            rot_bb_start[len(dun_database.rotameric_libraries) :],
            rot_bb_step[len(dun_database.rotameric_libraries) :],
        )
        sr_tableset_offsets = self._create_semirot_offsets(dun_database)

        sorted_rotamer_2_rotamer = self.create_sorted_rot_2_rot(all_rotlibs)

        rotwells = self.create_rotamer_well_table(all_rotlibs)

        scoring_db = dict(
            rotameric_neglnprob_tables=nlp_coeffs,
            rotprob_table_sizes=pc_sizes,
            rotprob_table_strides=pc_strides,
            rotameric_mean_tables=mean_coeffs,
            rotameric_sdev_tables=sdev_coeffs,
            rotmean_table_sizes=mc_sizes,
            rotmean_table_strides=mc_strides,
            rotameric_bb_start=rot_bb_start,
            rotameric_bb_step=rot_bb_step,
            rotameric_bb_periodicity=rot_bb_per,
            rotameric_rotind2tableind=rot_ri2ti,
            semirotameric_rotind2tableind=semirot_ri2ti,
            semirotameric_tables=sr_coeffs,
            semirot_table_sizes=sr_sizes,
            semirot_table_strides=sr_strides,
            semirot_start=sr_start,
            semirot_step=sr_step,
            semirot_periodicity=sr_periodicity,
        )
        scoring_db_aux = dict(
            rotameric_prob_tableset_offsets=prob_table_offsets,
            rotameric_meansdev_tableset_offsets=rotameric_mean_offsets,
            nchi_for_table_set=nchi_for_table_set,
            rotameric_chi_ri2ti_offsets=rotameric_chi_ri2ti_offsets,
            semirotameric_tableset_offsets=sr_tableset_offsets,
        )

        sampling_db = dict(
            rotameric_prob_tables=p_coeffs,
            rotprob_table_sizes=pc_sizes,
            rotprob_table_strides=pc_strides,
            rotameric_mean_tables=mean_coeffs,
            rotameric_sdev_tables=sdev_coeffs,
            rotmean_table_sizes=mc_sizes,
            rotmean_table_strides=mc_strides,
            rotameric_meansdev_tableset_offsets=rotameric_mean_offsets,
            n_rotamers_for_tableset=prob_table_nrots,
            n_rotamers_for_tableset_offsets=prob_table_offsets,
            sorted_rotamer_2_rotamer=sorted_rotamer_2_rotamer,
            rotameric_bb_start=rot_bb_start,
            rotameric_bb_step=rot_bb_step,
            rotameric_bb_periodicity=rot_bb_per,
            rotameric_bb_source_start=self._create_rot_source_starts(
                all_rotlibs, rot_bb_start
            ),
            rotameric_bb_is_mirrored=numpy.asarray(
                [
                    getattr(lib.rotameric_data, "backbone_is_mirrored", False)
                    for lib in all_rotlibs
                ],
                dtype=numpy.bool_,
            ),
            rotameric_rotind2tableind=rot_ri2ti,
            semirotameric_rotind2tableind=semirot_ri2ti,
            all_chi_rotind2tableind=allchi_ri2ti,
            all_chi_rotind2tableind_offsets=all_chi_rotind2tableind_offsets,
            semirotameric_tables=sr_coeffs,
            semirot_table_sizes=sr_sizes,
            semirot_table_strides=sr_strides,
            semirot_start=sr_start,
            semirot_step=sr_step,
            semirot_periodicity=sr_periodicity,
            nchi_for_table_set=nchi_for_table_set,
            rotwells=rotwells,
        )

        return DunbrackTableData(
            scoring_db=scoring_db,
            scoring_db_aux=scoring_db_aux,
            sampling_db=sampling_db,
            all_table_indices=all_table_indices,
            rotameric_table_indices=rotameric_table_indices,
            semirotameric_table_indices=semirotameric_table_indices,
        )

    def _create_table_indices(self, all_table_names, dun_lookup, n_rotameric):
        """Validate declared references once, then derive family-local indices."""
        indices = {name: i for i, name in enumerate(all_table_names)}
        if len(indices) != len(all_table_names):
            raise ValueError("Dunbrack table names must be unique across families")
        names = [row.residue_name for row in dun_lookup]
        if len(set(names)) != len(names):
            raise ValueError("Dunbrack residue lookup names must be unique")
        missing = {
            row.residue_name: row.dun_table_name
            for row in dun_lookup
            if row.dun_table_name not in indices
        }
        if missing:
            raise ValueError(f"Dunbrack lookup references missing tables: {missing}")
        all_tables = {
            row.residue_name: indices[row.dun_table_name] for row in dun_lookup
        }
        return (
            all_tables,
            {name: i if i < n_rotameric else -1 for name, i in all_tables.items()},
            {
                name: i - n_rotameric if i >= n_rotameric else -1
                for name, i in all_tables.items()
            },
        )

    def _create_nchi_for_table_set(self, all_rotlibs):
        return numpy.asarray(
            [rotlib.rotameric_data.rotamers.shape[1] for rotlib in all_rotlibs],
            dtype=numpy.int32,
        )

    def _create_prob_table_offsets(self, all_rotlibs):
        prob_table_nrots = numpy.asarray(
            [
                rotlib.rotameric_data.rotamer_probabilities.shape[0]
                for rotlib in all_rotlibs
            ],
            dtype=numpy.int32,
        )
        return prob_table_nrots, _exclusive_cumsum1d(prob_table_nrots)

    def _compute_rotprob_coeffs(self, all_rotlibs):
        rotameric_prob_tables = [
            rotlib.rotameric_data.rotamer_probabilities[i, :, :].copy()
            for rotlib in all_rotlibs
            for i in range(rotlib.rotameric_data.rotamer_probabilities.shape[0])
        ]
        for table in rotameric_prob_tables:
            table[table == 0] = 1e-6
        rotameric_neglnprob_tables = [
            -1 * numpy.log(table) for table in rotameric_prob_tables
        ]

        prob_coeffs, prob_coeffs_sizes, prob_coeffs_strides = self._pack_coefficients(
            rotameric_prob_tables
        )

        neglnprob_coeffs, _, _2 = self._pack_coefficients(rotameric_neglnprob_tables)
        return prob_coeffs, prob_coeffs_sizes, prob_coeffs_strides, neglnprob_coeffs

    def _create_rot_mean_offsets(self, all_rotlibs):
        mean_table_n_entries = [
            rotlib.rotameric_data.rotamer_means.shape[0]
            * rotlib.rotameric_data.rotamer_means.shape[3]
            for rotlib in all_rotlibs
        ]
        return _exclusive_cumsum1d(
            numpy.asarray(mean_table_n_entries, dtype=numpy.int32)
        )

    def _calculate_rot_mean_coeffs(self, all_rotlibs):
        rotameric_mean_tables = []
        for rotlib in all_rotlibs:
            data = rotlib.rotameric_data
            for i in range(data.rotamer_means.shape[0]):
                for j in range(data.rotamer_means.shape[3]):
                    x = data.rotamer_means[i, :, :, j].copy()
                    # Choose the source library's angular branch before
                    # interpolation. Its reflection uses [-240, 120], not
                    # the original [-120, 240] branch: otherwise neighboring
                    # mirrored means can acquire different extra full turns.
                    if getattr(data, "backbone_is_mirrored", False):
                        x[x > 120] -= 360
                    else:
                        x[x < -120] += 360
                    x *= numpy.pi / 180
                    rotameric_mean_tables.append(x)

        mean_coeffs, mean_coeffs_sizes, mean_coeffs_strides = self._pack_coefficients(
            rotameric_mean_tables
        )

        return mean_coeffs, mean_coeffs_sizes, mean_coeffs_strides

    def _calculate_rot_sdev_coeffs(self, all_rotlibs):
        rotameric_sdev_tables = [
            rotlib.rotameric_data.rotamer_stdvs[i, :, :, j].copy() * numpy.pi / 180
            for rotlib in all_rotlibs
            for i in range(rotlib.rotameric_data.rotamer_stdvs.shape[0])
            for j in range(rotlib.rotameric_data.rotamer_stdvs.shape[3])
        ]

        sdev_coeffs, _, _2 = self._pack_coefficients(rotameric_sdev_tables)
        return sdev_coeffs

    def _create_rot_periodicities(self, all_rotlibs):
        if not all_rotlibs:
            empty = numpy.empty((0, 2), dtype=numpy.float32)
            return empty, empty, empty
        rotameric_bb_start = numpy.asarray(
            [
                list(rotlib.rotameric_data.backbone_dihedral_start)
                for rotlib in all_rotlibs
            ],
            dtype=numpy.float32,
        )
        rotameric_bb_start *= numpy.pi / 180
        rotameric_bb_step = numpy.asarray(
            [
                list(rotlib.rotameric_data.backbone_dihedral_step)
                for rotlib in all_rotlibs
            ],
            dtype=numpy.float32,
        )
        rotameric_bb_step *= numpy.pi / 180
        rotameric_bb_periodicity = numpy.ones_like(rotameric_bb_step) * 2 * numpy.pi
        return rotameric_bb_start, rotameric_bb_step, rotameric_bb_periodicity

    def _create_rot_source_starts(self, all_rotlibs, target_starts):
        data = [lib.rotameric_data for lib in all_rotlibs]
        sources = [getattr(d, "backbone_source_start", None) for d in data]
        if all(
            s is None or s is d.backbone_dihedral_start for s, d in zip(sources, data)
        ):
            return target_starts
        return numpy.asarray(
            [
                list(d.backbone_dihedral_start if s is None else s)
                for s, d in zip(sources, data)
            ],
            dtype=numpy.float32,
        ) * (numpy.pi / 180)

    def _create_rotind2tableinds(self, dun_database):
        """
        rotameric_rotind2tableind: a mapping based on the rotameric chi for a residue
           to the table index for that rotamer; the table index can then be used to
           access the rotamer probability, rotamer chi mean, and rotamer sdev
           tables
        semirotameric_rotind2tableind: a mapping based on the rotameric chi for a
           residue to the table index for that rotamer; the table index can then be
           used to access the semirotameric_tables table. Only valid for
           semirotameric residue types
        all_chi_rotind2tableind: a mapping based on both rotameric and non-rotameric
           chi for a residue to the table index for that rotamer; the table index
           can then be used to access the rotamer probability, rotamer chi mean,
           and rotamer sdev tables
        """

        rotameric_rotind2tableind = []
        semirotameric_rotind2tableind = []
        all_chi_rotind2tableind = []
        ntablerots = [0]
        for rotlib in dun_database.rotameric_libraries:
            rotamers = rotlib.rotameric_data.rotamers
            exponents = [x for x in range(rotamers.shape[1])]
            exponents.reverse()
            exponents = numpy.asarray(exponents, dtype=numpy.int32)
            prods = numpy.power(3, exponents)
            rotinds = numpy.sum((rotamers - 1) * prods, 1)
            ri2ti = -1 * numpy.ones([3 ** rotamers.shape[1]], dtype=numpy.int32)
            ri2ti[rotinds] = numpy.arange(rotamers.shape[0], dtype=numpy.int32)

            if (
                len(rotlib.rotameric_data.rotamer_alias.shape) == 2
                and rotlib.rotameric_data.rotamer_alias.shape[0] > 0
            ):
                orig_rotids = rotlib.rotameric_data.rotamer_alias[
                    :, 0 : rotamers.shape[1]
                ].copy()
                alt_rotids = rotlib.rotameric_data.rotamer_alias[
                    :, rotamers.shape[1] :
                ].copy()
                orig_inds = numpy.sum((orig_rotids - 1) * prods, 1)
                alt_inds = numpy.sum((alt_rotids - 1) * prods, 1)
                ri2ti[orig_inds.astype(numpy.int64)] = ri2ti[
                    alt_inds.astype(numpy.int64)
                ]

            # Wells absent from the library take Rosetta's representative: the
            # first defined well differing in one chi, scanning from the last
            # chi inward, bins in order; else the most common rotamer.
            # A -1 entry would index into the previous library's tables.
            # Mirror the tie-break as well as the wells: reflected rows must
            # choose the same source rotamer as their L counterparts.
            bins = (
                range(2, -1, -1)
                if rotlib.rotameric_data.backbone_is_mirrored
                else range(3)
            )
            defined = ri2ti.tolist()
            strides = prods.tolist()[::-1]
            most_common = None
            for well, table in enumerate(defined):
                if table >= 0:
                    continue
                for stride in strides:
                    bin_ = (well // stride) % 3
                    alts = (well + (b - bin_) * stride for b in bins if b != bin_)
                    alt = next((a for a in alts if defined[a] >= 0), None)
                    if alt is not None:
                        ri2ti[well] = defined[alt]
                        break
                else:
                    if most_common is None:
                        probs = rotlib.rotameric_data.rotamer_probabilities
                        most_common = int(
                            probs.reshape(probs.shape[0], -1).mean(1).argmax()
                        )
                    ri2ti[well] = most_common

            rotameric_rotind2tableind.extend(list(ri2ti))
            semirotameric_rotind2tableind.extend([0] * len(ri2ti))
            all_chi_rotind2tableind.extend(list(ri2ti))
            ntablerots.append(rotinds.shape[0])

        for rotlib in dun_database.semi_rotameric_libraries:
            rotameric_rotamers = rotlib.rotameric_data.rotamers
            semirotameric_rotamers = rotlib.rotameric_chi_rotamers
            exponents = [x for x in range(semirotameric_rotamers.shape[1])]
            exponents.reverse()
            exponents = numpy.asarray(exponents, dtype=numpy.int32)
            prods = numpy.power(3, exponents)
            rotinds = numpy.sum((semirotameric_rotamers - 1) * prods, 1)
            sr_ri2ti = -1 * numpy.ones(
                [3 ** semirotameric_rotamers.shape[1]], dtype=numpy.int32
            )
            sr_ri2ti[rotinds] = numpy.arange(
                semirotameric_rotamers.shape[0], dtype=numpy.int32
            )
            semirotameric_rotind2tableind.extend(list(sr_ri2ti))

            # OK: this code assumes that a) for the rotameric data, all combinations
            # of rotameric-chi rotamers + binned-non-rotameric-chi are defined
            # (an assumption not needed for rotameric residues) so that
            # 3^(n-rotameric-chi) divides n-rotameric-rotamers cleanly , and
            # b) the rotamers stored in the rotameric_data are in sorted order

            r_ri2ti = -1 * numpy.ones(
                [3 ** semirotameric_rotamers.shape[1]], dtype=numpy.int32
            )
            n_nonrotameric_chi_rotamers = (
                rotameric_rotamers.shape[0] / 3 ** semirotameric_rotamers.shape[1]
            )
            r_ri2ti[rotinds] = (
                n_nonrotameric_chi_rotamers
                * numpy.arange(semirotameric_rotamers.shape[0])
            ).astype(numpy.int32)
            rotameric_rotind2tableind.extend(list(r_ri2ti))

            ac_ri2ti = numpy.arange(rotameric_rotamers.shape[0], dtype=numpy.int32)
            all_chi_rotind2tableind.extend(list(ac_ri2ti))

        rotameric_rotind2tableind = numpy.asarray(
            rotameric_rotind2tableind, dtype=numpy.int32
        ).reshape((-1,))

        semirotameric_rotind2tableind = numpy.asarray(
            semirotameric_rotind2tableind, dtype=numpy.int32
        ).reshape((-1,))

        all_chi_rotind2tableind = numpy.asarray(
            all_chi_rotind2tableind, dtype=numpy.int32
        )
        return (
            rotameric_rotind2tableind,
            semirotameric_rotind2tableind,
            all_chi_rotind2tableind,
        )

    def _create_rotameric_rotind2tableind_offsets(self, dun_database):
        rotamer_sets = [
            rotlib.rotameric_data.rotamers
            for rotlib in dun_database.rotameric_libraries
        ] + [
            rotlib.rotameric_chi_rotamers
            for rotlib in dun_database.semi_rotameric_libraries
        ]

        # same for both rotameric and semi-rotameric rotind2tableind tables
        return _exclusive_cumsum1d(
            numpy.asarray(
                [3 ** rotamers.shape[1] for rotamers in rotamer_sets],
                dtype=numpy.int32,
            ),
        )

    def _create_all_chi_rotind2tableind_offsets(self, dun_database):
        """
        Offsets into the all-chi rotind2tableind table
        """
        rotamer_counts_sets = [
            3 ** rotlib.rotameric_data.rotamers.shape[1]
            for rotlib in dun_database.rotameric_libraries
        ] + [
            rotlib.rotameric_data.rotamers.shape[0]
            for rotlib in dun_database.semi_rotameric_libraries
        ]

        return _exclusive_cumsum1d(
            numpy.asarray(rotamer_counts_sets, dtype=numpy.int32),
        )

    def _calc_semirot_coeffs(self, dun_database):
        semirotameric_prob_tables = [
            rotlib.nonrotameric_chi_probabilities[i, :, :, :].copy()
            for rotlib in dun_database.semi_rotameric_libraries
            for i in range(rotlib.nonrotameric_chi_probabilities.shape[0])
        ]
        # these aren't used for rotamer building, so we'll just use this for
        # the neglnprobs
        for table in semirotameric_prob_tables:
            table[table == 0] = 1e-6
            table[:] = -1 * numpy.log(table)

        return self._pack_coefficients(semirotameric_prob_tables, ndim=3)

    def _create_semirot_periodicity(self, dun_database, backbone_start, backbone_step):
        semirot_start = numpy.zeros(
            (len(dun_database.semi_rotameric_libraries), 3),
            dtype=numpy.float32,
        )
        semirot_start[:, :2] = backbone_start
        semirot_start[:, 2] = (
            numpy.asarray(
                [x.non_rot_chi_start for x in dun_database.semi_rotameric_libraries],
                dtype=numpy.float32,
            )
            * numpy.pi
            / 180
        )

        semirot_step = numpy.zeros(
            (len(dun_database.semi_rotameric_libraries), 3),
            dtype=numpy.float32,
        )
        semirot_step[:, :2] = backbone_step
        semirot_step[:, 2] = (
            numpy.asarray(
                [x.non_rot_chi_step for x in dun_database.semi_rotameric_libraries],
                dtype=numpy.float32,
            )
            * numpy.pi
            / 180
        )

        semirot_periodicity = numpy.zeros(
            (len(dun_database.semi_rotameric_libraries), 3),
            dtype=numpy.float32,
        )
        semirot_periodicity[:, 0] = 2 * numpy.pi
        semirot_periodicity[:, 1] = 2 * numpy.pi
        semirot_periodicity[:, 2] = (
            numpy.asarray(
                [x.non_rot_chi_period for x in dun_database.semi_rotameric_libraries],
                dtype=numpy.float32,
            )
            * numpy.pi
            / 180
        )
        return semirot_start, semirot_step, semirot_periodicity

    def _create_semirot_offsets(self, dun_database):
        nsemirot_rotamers = [
            rotlib.nonrotameric_chi_probabilities.shape[0]
            for rotlib in dun_database.semi_rotameric_libraries
        ]
        return _exclusive_cumsum1d(numpy.asarray(nsemirot_rotamers, dtype=numpy.int32))

    def create_sorted_rot_2_rot(self, all_rotlibs):
        if not all_rotlibs:
            return numpy.empty((0, 0, 0), dtype=numpy.int64)
        sorted_2_rotinds = _concatenate_padded(
            [
                rot.rotameric_data.prob_sorted_rot_inds.transpose(2, 0, 1)
                for rot in all_rotlibs
            ]
        )

        return sorted_2_rotinds.transpose(1, 2, 0)

    def create_rotamer_well_table(self, all_rotlibs):
        if not all_rotlibs:
            return numpy.empty((0, 0), dtype=numpy.int32)
        rotwells = _concatenate_padded(
            [rot.rotameric_data.rotamers for rot in all_rotlibs]
        )
        return rotwells
