"""Support XOR of Bell labels; no phase multiplication or independence inference."""

import numpy as np
from numba import njit, prange

from .._numba import NUMBA_CACHE
from ..zx_array import ZXArray
from ._inputs import as_collection


@njit(cache=NUMBA_CACHE, nogil=True)
def _paired_xor(z_left, x_left, z_right, x_right):
    z, x = np.empty_like(z_left), np.empty_like(x_left)
    for row in range(z.shape[0]):
        for chunk in range(z.shape[1]):
            z[row, chunk] = z_left[row, chunk] ^ z_right[row, chunk]
            x[row, chunk] = x_left[row, chunk] ^ x_right[row, chunk]
    return z, x


@njit(cache=NUMBA_CACHE, nogil=True, parallel=True)
def _paired_xor_parallel(z_left, x_left, z_right, x_right):
    z, x = np.empty_like(z_left), np.empty_like(x_left)
    for row in prange(z.shape[0]):
        for chunk in range(z.shape[1]):
            z[row, chunk] = z_left[row, chunk] ^ z_right[row, chunk]
            x[row, chunk] = x_left[row, chunk] ^ x_right[row, chunk]
    return z, x


def bell_differences(left, right, *, parallel=False):
    """Pairwise XOR two equal-length, equal-width collections of Bell labels.

    Global phases are ignored and output phases are zero. The caller must
    establish the independence of the two streams when its estimator needs
    independent Bell differences. No rows are broadcast, shuffled, or reused.
    Numeric inputs are bit matrices; wrap legacy arrays with ZXArray.from_raw.
    """
    left = as_collection(left)
    right = as_collection(right, n_qubits=left.n_qubits)
    if len(left) != len(right):
        raise ValueError("Bell streams must contain the same number of samples")
    width, zl, xl, _ = left.kernel_args()
    _, zr, xr, _ = right.kernel_args()
    kernel = _paired_xor_parallel if parallel else _paired_xor
    z, x = kernel(zl, xl, zr, xr)
    return ZXArray._from_chunks(width, z, x, np.zeros(len(left), dtype=np.uint8),
                                force_large=left.is_large or right.is_large)


def cyclic_bell_differences(samples, *, parallel=False):
    """XOR each label with its successor, wrapping the final row to the first.

    This diagnostic deliberately reuses samples and produces dependent rows;
    it is not a substitute for two independent streams. Empty input remains
    empty and a one-row input produces the identity.
    """
    samples = as_collection(samples)
    width, z, x, _ = samples.kernel_args()
    kernel = _paired_xor_parallel if parallel else _paired_xor
    out_z, out_x = kernel(z, x, np.roll(z, -1, axis=0), np.roll(x, -1, axis=0))
    return ZXArray._from_chunks(width, out_z, out_x,
                                np.zeros(len(samples), dtype=np.uint8),
                                force_large=samples.is_large)
