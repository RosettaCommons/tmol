"""Compatibility labels over the shared host structure metadata model."""

import attrs
import numpy as np

from tmol_core.io.pdb_info import (
    PDBInfo as CommonPDBInfo,
    DEFAULT_ATOM_OCCUPANCY,
    DEFAULT_ATOM_B_FACTOR,
)


_UNSPECIFIED = object()


@attrs.define(init=False, slots=False)
class PDBInfo(CommonPDBInfo):
    """Shared metadata with the original JAX label aliases and constructor."""

    res_name: object = None
    atom_record_type: object = None

    def __init__(
        self,
        chain_id=_UNSPECIFIED,
        res_num=_UNSPECIFIED,
        res_name=None,
        atom_record_type=None,
        **metadata,
    ):
        if chain_id is not _UNSPECIFIED:
            metadata["chain_labels"] = chain_id
        if res_num is not _UNSPECIFIED:
            metadata["residue_labels"] = res_num
        super().__init__(
            **{
                name: metadata.pop(name, None)
                for name in attrs.fields_dict(CommonPDBInfo)
            }
        )
        if metadata:
            raise TypeError("unknown metadata fields: " + ", ".join(metadata))
        self.res_name = res_name
        self.atom_record_type = atom_record_type

    def _as_common(self):
        """Rectangular host metadata for shared planners; accept legacy ragged labels."""
        fields = {f.name: getattr(self, f.name) for f in attrs.fields(CommonPDBInfo)}
        if self.chain_labels is not None:
            labels = np.full(self.residue_labels.shape, "", object)
            for pose, row in enumerate(self.chain_labels):
                labels[pose, : len(row)] = row
            fields["chain_labels"] = labels
        return CommonPDBInfo(**fields)

    @property
    def chain_id(self):
        return self.chain_labels

    @chain_id.setter
    def chain_id(self, value):
        self.chain_labels = value

    @property
    def res_num(self):
        return self.residue_labels

    @res_num.setter
    def res_num(self, value):
        self.residue_labels = value

    @property
    def n_poses(self):
        for value in (
            self.res_num,
            self.chain_id,
            self.res_name,
            self.atom_record_type,
        ):
            if value is not None and len(value):
                return len(value)
        return 0

    @property
    def max_n_blocks(self):
        return self.res_num.shape[1] if self.res_num is not None else 0

    def __getitem__(self, index):
        return self.take(np.atleast_1d(np.arange(self.n_poses)[index]))

    @classmethod
    def empty(cls, n_poses, max_n_blocks):
        return cls(
            chain_id=[[] for _ in range(n_poses)],
            res_num=np.full((n_poses, max_n_blocks), -1, np.int32),
            res_name=[[] for _ in range(n_poses)],
            atom_record_type=[[] for _ in range(n_poses)],
        )


__all__ = ["PDBInfo", "DEFAULT_ATOM_OCCUPANCY", "DEFAULT_ATOM_B_FACTOR"]
