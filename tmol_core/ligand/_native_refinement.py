"""Validated host ABI shared by framework-specific native library loaders."""

import ctypes

import numpy as np


def _input_type(real):
    real_pointer = ctypes.POINTER(real)
    int_pointer = ctypes.POINTER(ctypes.c_int32)

    class Input(ctypes.Structure):
        _fields_ = (
            [("coordinates", real_pointer)]
            + [
                (key, ctypes.c_int64)
                for key in (
                    "n_atoms",
                    "n_dims",
                    "n_exact",
                    "n_lower",
                    "n_upper",
                    "n_components",
                    "n_component_atoms",
                    "n_chiral",
                )
            ]
            + [
                ("pairs", int_pointer),
                ("distances", real_pointer),
                ("component_offsets", int_pointer),
                ("component_atoms", int_pointer),
                ("chiral_atoms", int_pointer),
                ("chiral_sign", real_pointer),
                ("weights", real_pointer),
                ("four_dimensional", ctypes.c_bool),
                ("fourth_weight", real),
            ]
        )

    return Input, real_pointer, int_pointer


def _validate_arrays(x, p):
    if not isinstance(x, np.ndarray) or x.ndim != 2:
        raise TypeError("expected a NumPy coordinate matrix")
    if x.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError("coordinates must have float32 or float64 dtype")
    if not x.flags.c_contiguous or not x.flags.aligned:
        raise TypeError("coordinates must be contiguous and aligned")
    counts = (p.n_exact, p.n_lower, p.n_upper)
    if any(not isinstance(n, (int, np.integer)) or n < 0 for n in counts):
        raise ValueError("pair counts must be nonnegative integers")
    arrays = (
        (p.pairs, np.int32, (sum(counts), 2)),
        (p.distances, x.dtype, (sum(counts),)),
        (p.component_offsets, np.int32, None),
        (p.component_atoms, np.int32, None),
        (p.chiral_atoms, np.int32, (len(p.chiral_sign), 4)),
        (p.chiral_sign, x.dtype, (len(p.chiral_sign),)),
        (p.weights, x.dtype, (5,)),
    )
    for array, dtype, shape in arrays:
        if not isinstance(array, np.ndarray) or array.dtype != dtype:
            raise TypeError("native metadata has an invalid array dtype")
        if not array.flags.c_contiguous or not array.flags.aligned:
            raise TypeError("native metadata must be contiguous and aligned")
        if (shape is None and array.ndim != 1) or (
            shape is not None and array.shape != shape
        ):
            raise ValueError("native metadata has an invalid array shape")
    if len(p.component_offsets) == 0:
        raise ValueError("component offsets require the final sentinel")
    if p.component_offsets[0] != 0 or p.component_offsets[-1] != len(p.component_atoms):
        raise ValueError("component offsets must span the component atom array")


class NativeRefinementBackend:
    """Load one common CPU ABI; the frontend supplies only its library path.

    Calls return newly owned NumPy gradients. No Torch or JAX imports are used.
    The native kernel validates indices before reading coordinate storage.
    """

    def __init__(self, library_path):
        self.library = ctypes.CDLL(str(library_path))
        self.types = {}
        self.functions = {}
        if self._function("schema_version", [], ctypes.c_int)() != 1:
            raise RuntimeError("unsupported native ligand refinement ABI")
        for dtype, real, suffix in (
            (np.dtype("float32"), ctypes.c_float, "f32"),
            (np.dtype("float64"), ctypes.c_double, "f64"),
        ):
            input_type, real_pointer, int_pointer = _input_type(real)
            size = self._function("size_" + suffix, [], ctypes.c_int64)()
            if size != ctypes.sizeof(input_type):
                raise RuntimeError("native ligand refinement input layout mismatch")
            self.types[dtype] = input_type, real_pointer, int_pointer
            self.functions[dtype] = self._function(
                suffix,
                [ctypes.POINTER(input_type), real_pointer, real_pointer],
                ctypes.c_int,
            )
        self._denormal_mode = self._function("denormal_mode", [], ctypes.c_int)
        self._set_denormal_mode = self._function(
            "set_denormal_mode", [ctypes.c_int], ctypes.c_int
        )

    def _function(self, name, arguments, result):
        function = getattr(self.library, "tmol_ligand_refinement_" + name)
        function.argtypes = arguments
        function.restype = result
        return function

    def denormal_mode(self):
        """Return output/input flush bits, or -1 when unsupported."""
        return self._denormal_mode()

    def set_denormal_mode(self, mode):
        status = self._set_denormal_mode(mode)
        if status == 1:
            raise ValueError("denormal mode must be in [0, 3]")
        return status == 0

    def value_and_grad(self, x, p, *, four_dimensional=False, fourth_weight=0):
        _validate_arrays(x, p)
        input_type, real_pointer, int_pointer = self.types[x.dtype]

        def integers(a):
            return a.ctypes.data_as(int_pointer)

        def reals(a):
            return a.ctypes.data_as(real_pointer)

        inputs = input_type(
            reals(x),
            *x.shape,
            p.n_exact,
            p.n_lower,
            p.n_upper,
            len(p.component_offsets) - 1,
            len(p.component_atoms),
            len(p.chiral_sign),
            integers(p.pairs),
            reals(p.distances),
            integers(p.component_offsets),
            integers(p.component_atoms),
            integers(p.chiral_atoms),
            reals(p.chiral_sign),
            reals(p.weights),
            four_dimensional,
            fourth_weight,
        )
        value = np.empty((), dtype=x.dtype)
        gradient = np.empty_like(x)
        status = self.functions[x.dtype](
            ctypes.byref(inputs), reals(value), reals(gradient)
        )
        if status:
            error = FloatingPointError if status == 3 else ValueError
            raise error(f"native ligand refinement failed with status {status}")
        return value[()], gradient
