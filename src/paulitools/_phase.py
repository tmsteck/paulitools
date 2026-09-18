"""Compiled algebra for i**q times ordinary Hermitian Pauli strings."""

import numpy as np
from numba import njit

from ._numba import NUMBA_CACHE
from .large_pauli import _popcount_uint64


@njit(cache=NUMBA_CACHE, nogil=True)
def pack_bit_matrices(z_bits, x_bits):
    rows, width = z_bits.shape
    chunks = (width + 63) // 64
    z = np.zeros((rows, chunks), dtype=np.uint64)
    x = np.zeros((rows, chunks), dtype=np.uint64)
    for row in range(rows):
        for qubit in range(width):
            mask = np.uint64(1) << np.uint64(qubit % 64)
            if z_bits[row, qubit]:
                z[row, qubit // 64] |= mask
            if x_bits[row, qubit]:
                x[row, qubit // 64] |= mask
    return z, x


@njit(cache=NUMBA_CACHE, nogil=True)
def unpack_chunk_matrices(z, x, width):
    rows = z.shape[0]
    z_bits = np.empty((rows, width), dtype=np.uint8)
    x_bits = np.empty((rows, width), dtype=np.uint8)
    for row in range(rows):
        for qubit in range(width):
            shift = np.uint64(qubit % 64)
            z_bits[row, qubit] = (z[row, qubit // 64] >> shift) & np.uint64(1)
            x_bits[row, qubit] = (x[row, qubit // 64] >> shift) & np.uint64(1)
    return z_bits, x_bits


@njit(cache=NUMBA_CACHE, nogil=True)
def multiply_chunks(z_a, x_a, phase_a, z_b, x_b, phase_b):
    """Multiply corresponding rows, broadcasting a single row on either side.

    Each canonical Hermitian string is i**popcount(x&z) X**x Z**z.
    Moving Z from the left factor past X from the right contributes -1.
    """
    count_a, chunks = z_a.shape
    count_b = z_b.shape[0]
    if count_a == count_b:
        count = count_a
    elif count_a == 1:
        count = count_b
    elif count_b == 1:
        count = count_a
    else:
        raise ValueError("Multiply requires equal row counts or a single-Pauli operand")
    z_out = np.empty((count, chunks), dtype=np.uint64)
    x_out = np.empty((count, chunks), dtype=np.uint64)
    phases = np.empty(count, dtype=np.uint8)
    for row in range(count):
        a = 0 if count_a == 1 else row
        b = 0 if count_b == 1 else row
        # Numba preserves unsigned scalars through int(uint8); mixing that
        # accumulator with signed popcounts would promote it to float64.
        phase = np.int64(phase_a[a]) + np.int64(phase_b[b])
        for chunk in range(chunks):
            za, xa = z_a[a, chunk], x_a[a, chunk]
            zb, xb = z_b[b, chunk], x_b[b, chunk]
            zc, xc = za ^ zb, xa ^ xb
            z_out[row, chunk], x_out[row, chunk] = zc, xc
            phase += (_popcount_uint64(xa & za) + _popcount_uint64(xb & zb)
                      - _popcount_uint64(xc & zc)
                      + 2 * _popcount_uint64(za & xb))
        phases[row] = phase & 3
    return z_out, x_out, phases


@njit(cache=NUMBA_CACHE, nogil=True)
def row_basis_bits(matrix):
    """Independent RREF rows over GF(2), without phase information."""
    data = matrix.copy()
    rows, columns = data.shape
    rank = 0
    for column in range(columns):
        pivot = -1
        for row in range(rank, rows):
            if data[row, column]:
                pivot = row
                break
        if pivot < 0:
            continue
        tmp = data[rank].copy()
        data[rank] = data[pivot]
        data[pivot] = tmp
        for row in range(rows):
            if row != rank and data[row, column]:
                data[row] ^= data[rank]
        rank += 1
        if rank == rows:
            break
    return data[:rank].copy()
