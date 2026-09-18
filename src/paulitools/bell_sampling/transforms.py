"""Transforms of functions indexed by phase-free Pauli labels."""

from __future__ import annotations

import numpy as np
from numba import njit

from .._numba import NUMBA_CACHE


@njit(cache=NUMBA_CACHE, nogil=True)
def _symplectic_fwht_inplace(values, n_qubits, inverse):
    """Apply ordinary butterflies, then exchange the output Z/X planes."""
    size = values.size
    if inverse:
        # Scaling first avoids unnecessary overflow in an otherwise finite
        # inverse transform. The butterfly is linear over real/complex data.
        values /= size
    stride = 1
    while stride < size:
        for start in range(0, size, 2 * stride):
            for offset in range(stride):
                left = start + offset
                right = left + stride
                a, b = values[left], values[right]
                values[left], values[right] = a + b, a - b
        stride *= 2

    mask = (1 << n_qubits) - 1
    for index in range(size):
        swapped = ((index & mask) << n_qubits) | (index >> n_qubits)
        if index < swapped:
            values[index], values[swapped] = values[swapped], values[index]


def symplectic_fwht(values, *, inverse=False):
    r"""Return the symplectic Walsh-Hadamard transform of a Pauli function.

    ``values`` is a one-dimensional vector of length ``4**n`` (including
    length one for zero qubits). Its index is ``z | (x << n)``, where bit
    ``j`` of each plane denotes qubit ``j``. Thus the one-qubit order is
    ``I, Z, X, Y``; qubit zero is the leftmost Pauli label character.

    The forward transform is

    .. math::

       \widehat f(z,x) = \sum_{z',x'}
         (-1)^{z\cdot x' + x\cdot z'} f(z',x').

    ``inverse=True`` applies the same transform divided by ``4**n``.
    No probability normalization or Bell-distribution physics is inferred.
    The input is never mutated. Real data is converted to ``float64`` and
    complex data to ``complex128``; integers beyond the exact floating-point
    range can therefore be rounded. NaN/infinite input raises ``ValueError``;
    nonfinite arithmetic results raise ``FloatingPointError``.

    Compiled butterflies require O(n * 4**n) work and O(4**n) output storage.
    This reduces dense transform costs but still enumerates all Pauli labels.
    """
    if not isinstance(inverse, (bool, np.bool_)):
        raise TypeError("inverse must be a boolean")
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("values must be a one-dimensional vector")
    size = array.size
    if size == 0 or size & (size - 1) or (size.bit_length() - 1) % 2:
        raise ValueError("values must have length 4**n for a nonnegative integer n")
    if array.dtype.kind not in "biufc":
        raise TypeError("values must contain real or complex numbers")
    dtype = np.complex128 if array.dtype.kind == "c" else np.float64
    with np.errstate(over="ignore", invalid="ignore"):
        output = np.array(array, dtype=dtype, order="C", copy=True)
    if not np.all(np.isfinite(output)):
        raise ValueError("values must be finite and representable in float64/complex128")
    _symplectic_fwht_inplace(output, (size.bit_length() - 1) // 2, bool(inverse))
    if not np.all(np.isfinite(output)):
        raise FloatingPointError("symplectic transform overflowed float64/complex128")
    return output
