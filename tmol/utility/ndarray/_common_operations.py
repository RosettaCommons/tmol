import numpy
from tmol_core.array import invert_mapping as _invert_mapping  # noqa: F401
from typing import Union
from tmol.types._array import NDArray
from tmol.types._functional import _validate_args


@_validate_args
def _exclusive_cumsum1d(
    inds: Union[NDArray[numpy.int32][:], NDArray[numpy.int64][:]],
) -> Union[NDArray[numpy.int32][:], NDArray[numpy.int64][:]]:
    return numpy.concatenate(
        (
            numpy.array([0], dtype=inds.dtype),
            numpy.cumsum(inds)[:-1],
        ),
        axis=0,
    )


@_validate_args
def _exclusive_cumsum2d(
    inds: Union[NDArray[numpy.int32][:, :], NDArray[numpy.int64][:, :]],
) -> Union[NDArray[numpy.int32][:, :], NDArray[numpy.int64][:, :]]:
    return numpy.concatenate(
        (
            numpy.zeros((inds.shape[0], 1), dtype=inds.dtype),
            numpy.cumsum(inds, axis=1)[:, :-1],
        ),
        axis=1,
    )
