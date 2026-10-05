"""Incremental centers and target intersections for batches sharing prefixes."""
import numpy as np
from numba import njit, prange
from .._numba import NUMBA_CACHE
from ..large_pauli import _popcount_uint64
from ._inputs import as_collection, nonnegative_count
from .subspace import SupportBasis, _reduce_chunks


@njit(cache=NUMBA_CACHE, nogil=True, inline="always")
def _pair(a, b, words):
    value = 0
    for w in range(words):
        value ^= _popcount_uint64((a[w] & b[w+words]) ^ (a[w+words] & b[w])) & 1
    return value


@njit(cache=NUMBA_CACHE, nogil=True)
def _insert(row, basis, occupied):
    value = row.copy()
    for w in range(value.size):
        for bit in range(64):
            mask = np.uint64(1) << np.uint64(bit)
            if value[w] & mask:
                pivot = w*64 + bit
                if occupied[pivot]:
                    value ^= basis[pivot]
                else:
                    occupied[pivot] = True
                    basis[pivot] = value
                    return True
    return False


@njit(cache=NUMBA_CACHE, nogil=True)
def _rank(rows, count):
    # Compact elimination: target residues of centers are usually low rank.
    basis = np.empty((count, rows.shape[1]), dtype=np.uint64)
    pivot_words = np.empty(count, dtype=np.int64)
    masks = np.empty(count, dtype=np.uint64)
    rank = 0
    for i in range(count):
        row = rows[i].copy()
        for j in range(rank):
            if row[pivot_words[j]] & masks[j]:
                row ^= basis[j]
        found = False
        for w in range(row.size):
            if row[w]:
                for b in range(64):
                    mask = np.uint64(1) << np.uint64(b)
                    if row[w] & mask:
                        pivot_words[rank] = w
                        masks[rank] = mask
                        basis[rank] = row
                        rank += 1
                        found = True
                        break
            if found:
                break
    return rank


@njit(cache=NUMBA_CACHE, nogil=True)
def _one(rows, residues, sizes, n):
    width = rows.shape[1]
    words = width // 2
    full_basis = np.zeros((width*64, width), dtype=np.uint64)
    occupied = np.zeros(width*64, dtype=np.bool_)
    a = np.empty((n, width), dtype=np.uint64)
    b = np.empty_like(a)
    ar = np.empty_like(a)
    br = np.empty_like(a)
    center = np.empty_like(a)
    cr = np.empty_like(a)
    pairs = 0
    dim = 0
    span_rank = 0
    out = np.zeros((sizes.size, 2), dtype=np.int32)
    next_size = 0
    for i in range(rows.shape[0]):
        if _insert(rows[i], full_basis, occupied):
            span_rank += 1
            v = rows[i].copy()
            vr = residues[i].copy()
            for j in range(pairs):
                with_b = _pair(v, b[j], words)
                with_a = _pair(v, a[j], words)
                if with_b:
                    v ^= a[j]
                    vr ^= ar[j]
                if with_a:
                    v ^= b[j]
                    vr ^= br[j]
            partner = -1
            for j in range(dim):
                if _pair(v, center[j], words):
                    partner = j
                    break
            if partner == -1:
                center[dim] = v
                cr[dim] = vr
                dim += 1
            else:
                a[pairs] = center[partner]
                ar[pairs] = cr[partner]
                b[pairs] = v
                br[pairs] = vr
                for j in range(dim):
                    if j != partner and _pair(v, center[j], words):
                        center[j] ^= a[pairs]
                        cr[j] ^= ar[pairs]
                dim -= 1
                center[partner] = center[dim]
                cr[partner] = cr[dim]
                pairs += 1
        if i+1 == sizes[next_size]:
            out[next_size, 0] = dim
            out[next_size, 1] = dim - _rank(cr, dim)
            next_size += 1
            if next_size == sizes.size:
                break
        if span_rank == 2*n:
            break  # Full ambient span has zero center forever.
    return out


@njit(cache=NUMBA_CACHE, nogil=True)
def _serial(rows, residues, sizes, n):
    maximum = sizes[-1]
    count = rows.shape[0] // maximum
    result = np.empty((count, sizes.size, 2), dtype=np.int32)
    for i in range(count):
        result[i] = _one(rows[i*maximum:(i+1)*maximum], residues[i*maximum:(i+1)*maximum], sizes, n)
    return result


@njit(cache=NUMBA_CACHE, nogil=True, parallel=True)
def _parallel(rows, residues, sizes, n):
    maximum = sizes[-1]
    count = rows.shape[0] // maximum
    result = np.empty((count, sizes.size, 2), dtype=np.int32)
    for i in prange(count):
        result[i] = _one(rows[i*maximum:(i+1)*maximum], residues[i*maximum:(i+1)*maximum], sizes, n)
    return result


def prefix_center_intersection_ranks(samples, target, batch_sizes, *, parallel=False):
    """Return center and target-intersection ranks for repeated prefix batches.

    samples is a ZXArray (or ordinary object-API Pauli input), with consecutive
    blocks of max(batch_sizes) rows. target is a same-width SupportBasis or
    ZXArray. Phases are ignored. Positive strictly increasing batch_sizes
    select prefixes within each block. Returns int32 shape (blocks, sizes, 2):
    dim(center(span(prefix))) and dim(center(span(prefix)) intersect target).
    No ambient centralizer is computed. n must be positive; empty samples
    return zero blocks. Different prefix sizes intentionally reuse rows;
    independence of blocks or input Bell differences belongs to the caller.

    Packed incremental symplectic Gram-Schmidt updates a hyperbolic basis
    plus its center. Canonical target coset reduction is linear, so target
    membership constraints can be updated alongside this basis. The center's
    intersection dimension is its dimension minus the rank of its residues.
    All transforms are phase-free GF(2) support operations.
    """
    data = as_collection(samples)
    if data.n_qubits < 1:
        raise ValueError("positive Pauli width required")
    sizes = np.asarray([nonnegative_count(v, name="batch_size") for v in batch_sizes], dtype=np.int64)
    if sizes.size == 0 or sizes[0] < 1 or np.any(np.diff(sizes) <= 0):
        raise ValueError("batch_sizes must be positive and strictly increasing")
    if data.n_paulis % sizes[-1]:
        raise ValueError("sample count must be a multiple of the maximum batch size")
    prepared = target if isinstance(target, SupportBasis) else SupportBasis(target)
    if prepared.n_qubits != data.n_qubits:
        raise ValueError("Pauli widths differ")
    _, z, x, _ = data.kernel_args()
    rows = np.ascontiguousarray(np.concatenate((z, x), axis=1))
    residues = _reduce_chunks(rows, prepared._chunks, prepared._pivot_words, prepared._pivot_masks)
    kernel = _parallel if parallel else _serial
    return kernel(rows, residues, sizes, data.n_qubits)
