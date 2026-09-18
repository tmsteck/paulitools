"""Prepared, phase-free Pauli subspaces for Bell sampling.

A support subspace is a vector space over GF(2). Its elements are ordinary
Pauli labels modulo *all* phases, not signed stabilizer generators. Preparation
uses canonical left-to-right ``Z|X`` pivots; repeated membership, coset
reduction, and combinations use packed, compiled XOR kernels.
"""

import numpy as np
from numba import njit

from .._numba import NUMBA_CACHE
from .._phase import pack_bit_matrices, row_basis_bits, unpack_chunk_matrices
from ..group import null_space
from ..zx_array import ZXArray
from ._inputs import as_collection, nonnegative_count

__all__ = ["SupportBasis", "SupportSampler"]


@njit(cache=NUMBA_CACHE, nogil=True)
def _reduce_chunks(rows, basis, pivot_words, pivot_masks):
    """Reduce rows using exactly the pivot order used to prepare the basis."""
    result = rows.copy()
    for row in range(result.shape[0]):
        for pivot in range(basis.shape[0]):
            if result[row, pivot_words[pivot]] & pivot_masks[pivot]:
                for word in range(result.shape[1]):
                    result[row, word] ^= basis[pivot, word]
    return result


@njit(cache=NUMBA_CACHE, nogil=True)
def _contains_chunks(rows, basis, pivot_words, pivot_masks):
    reduced = _reduce_chunks(rows, basis, pivot_words, pivot_masks)
    result = np.ones(rows.shape[0], dtype=np.bool_)
    for row in range(reduced.shape[0]):
        for word in range(reduced.shape[1]):
            if reduced[row, word]:
                result[row] = False
                break
    return result


@njit(cache=NUMBA_CACHE, nogil=True)
def _combine_chunks(coefficients, basis):
    result = np.zeros((coefficients.shape[0], basis.shape[1]), dtype=np.uint64)
    for row in range(coefficients.shape[0]):
        for generator in range(basis.shape[0]):
            if coefficients[row, generator]:
                for word in range(basis.shape[1]):
                    result[row, word] ^= basis[generator, word]
    return result


@njit(cache=NUMBA_CACHE, nogil=True)
def _combine_bits(coefficients, basis):
    result = np.zeros((coefficients.shape[0], basis.shape[1]), dtype=np.uint8)
    for row in range(coefficients.shape[0]):
        for generator in range(basis.shape[0]):
            if coefficients[row, generator]:
                for column in range(basis.shape[1]):
                    result[row, column] ^= basis[generator, column]
    return result


def _pivots(matrix):
    # Every supplied row is a nonzero canonical RREF row.
    return np.argmax(matrix, axis=1).astype(np.intp) if len(matrix) else np.empty(0, dtype=np.intp)


class SupportBasis:
    """Own a canonical prepared basis of a phase-free Pauli support span.

    ``data`` follows the :class:`ZXArray` object input conventions: labels,
    Pauli objects, or binary ``Z|X`` matrices. Wrap legacy packed vectors with
    ``ZXArray.from_raw`` explicitly. Empty input needs ``n_qubits`` or an empty
    width-aware ZXArray. Inputs are copied; public basis and buffer accessors
    return independent copies. Both real and imaginary input phases are
    deliberately discarded.

    This class does not test commutativity or signed stabilizer consistency.
    Use ``ZXArray.stabilizer_basis()`` for that different operation.
    """

    __slots__ = ("_n_qubits", "_bits", "_chunks", "_pivots", "_pivot_words", "_pivot_masks")

    def __init__(self, data, *, n_qubits=None):
        if isinstance(data, SupportBasis):
            collection = data.to_zxarray()
        else:
            collection = data
        collection = as_collection(collection, n_qubits=n_qubits)
        self._n_qubits = collection.n_qubits
        self._bits = row_basis_bits(collection.binary())
        self._pivots = _pivots(self._bits)
        z, x = pack_bit_matrices(self._bits[:, :self.n_qubits], self._bits[:, self.n_qubits:])
        self._chunks = np.ascontiguousarray(np.concatenate((z, x), axis=1))
        n_chunks = z.shape[1]
        qubits = self._pivots % self.n_qubits if self.n_qubits else self._pivots.copy()
        self._pivot_words = np.ascontiguousarray(qubits // 64 + (self._pivots >= self.n_qubits) * n_chunks)
        self._pivot_masks = np.left_shift(np.uint64(1), (qubits % 64).astype(np.uint64))
        for buffer in (self._bits, self._chunks, self._pivots, self._pivot_words, self._pivot_masks):
            buffer.flags.writeable = False

    @property
    def n_qubits(self):
        """Width shared by every element of the span."""
        return self._n_qubits

    @property
    def rank(self):
        """Dimension over GF(2), so the span contains ``2**rank`` elements."""
        return self._bits.shape[0]

    @property
    def basis(self):
        """Independent positive-phase ZXArray of canonical generators."""
        return self.to_zxarray()

    def __repr__(self):
        return f"SupportBasis(n_qubits={self.n_qubits}, rank={self.rank})"

    def binary(self):
        """Return an independent binary ``(rank, 2*n_qubits)`` Z|X array."""
        return self._bits.copy()

    def to_zxarray(self):
        """Return independent positive-phase canonical generators."""
        return self._from_chunks(self._chunks)

    def _from_chunks(self, chunks):
        n_chunks = (self.n_qubits + 63) // 64
        return ZXArray._from_chunks(self.n_qubits, chunks[:, :n_chunks].copy(),
                                    chunks[:, n_chunks:].copy(),
                                    np.zeros(chunks.shape[0], dtype=np.uint8))

    def _candidate_chunks(self, candidates):
        collection = as_collection(candidates, n_qubits=self.n_qubits)
        _, z, x, _ = collection.kernel_args()
        return np.ascontiguousarray(np.concatenate((z, x), axis=1))

    def _other_basis(self, other):
        if isinstance(other, SupportBasis):
            if other.n_qubits != self.n_qubits:
                raise ValueError("Pauli widths differ; pad inputs explicitly")
            return other
        return SupportBasis(other, n_qubits=self.n_qubits)

    def contains(self, candidates):
        """Return one boolean per candidate indicating membership modulo phase."""
        rows = self._candidate_chunks(candidates)
        return _contains_chunks(rows, self._chunks, self._pivot_words, self._pivot_masks)

    def coset_reduce(self, candidates):
        """Return canonical positive-phase representatives modulo this span.

        Two inputs have the same output exactly when their XOR difference is
        in this span. In particular, every member reduces to identity.
        """
        rows = self._candidate_chunks(candidates)
        return self._from_chunks(_reduce_chunks(rows, self._chunks, self._pivot_words, self._pivot_masks))

    def linear_combinations(self, coefficients):
        """Batch XOR canonical generators with binary coefficients.

        A vector of length ``rank`` requests one row. A matrix has shape
        ``(count, rank)``. Outputs are positive-phase support representatives;
        this is not phase-aware multiplication of signed Pauli generators.
        """
        coefficients = np.asarray(coefficients)
        if coefficients.ndim == 1:
            coefficients = coefficients.reshape(1, -1)
        if coefficients.ndim != 2 or coefficients.shape[1] != self.rank:
            raise ValueError("Coefficients must have shape (count, rank) or (rank,)")
        if not np.all(np.isin(coefficients, (0, 1))):
            raise ValueError("Coefficients must contain only binary 0/1 values")
        coefficients = np.ascontiguousarray(coefficients, dtype=np.uint8)
        return self._from_chunks(_combine_chunks(coefficients, self._chunks))

    def union(self, other):
        """Return the linear span of both subspaces, not a set union."""
        other = self._other_basis(other)
        return SupportBasis(np.concatenate((self._bits, other._bits), axis=0))

    def _intersection_coefficients(self, other):
        # A combination of our rows is in the other span iff the same
        # combination of their canonical coset representatives vanishes.
        reduced = _reduce_chunks(self._chunks, other._chunks, other._pivot_words, other._pivot_masks)
        n_chunks = (self.n_qubits + 63) // 64
        z, x = unpack_chunk_matrices(reduced[:, :n_chunks], reduced[:, n_chunks:], self.n_qubits)
        residues = np.concatenate((z, x), axis=1)
        return row_basis_bits(null_space(np.ascontiguousarray(residues.T)))

    def intersection(self, other):
        """Return the common support subspace."""
        other = self._other_basis(other)
        coefficients = self._intersection_coefficients(other)
        return SupportBasis(self.linear_combinations(coefficients))

    def quotient_dimension(self, other):
        """Return ``dim((self + other) / other)`` without a containment assumption."""
        other = self._other_basis(other)
        return self.rank - self._intersection_coefficients(other).shape[0]

    def sampler(self, *, exclude_identity=False, exclude_span=None):
        """Prepare repeated sampling with fixed exclusions; no RNG is retained."""
        return SupportSampler(self, exclude_identity=exclude_identity, exclude_span=exclude_span)

    def sample(self, count, *, rng, exclude_identity=False, exclude_span=None):
        """Uniform draws with replacement, preparing exclusions once for this call.

        Use ``sampler(...).sample(count, rng=rng)`` to reuse an exclusion
        decomposition across repeated calls. See :class:`SupportSampler` for
        exclusion, RNG, and empty-support contracts.
        """
        return self.sampler(exclude_identity=exclude_identity, exclude_span=exclude_span).sample(count, rng=rng)


class SupportSampler:
    """Prepared uniform support sampling, including efficient span exclusion.

    Construct with ``SupportBasis.sampler(...)`` or a SupportBasis directly.
    ``exclude_span`` restricts draws to ``basis.span \\ exclude_span.span``
    even when the excluded span is not a subset. Any excluded span contains
    identity, so ``exclude_identity`` is redundant when a span is supplied.
    With neither option, the whole span is allowed.

    Preparation constructs a coefficient-space intersection and complement.
    Only an all-zero complement draw is retried, with acceptance probability
    at least one half. The decomposition is reused across ``sample`` calls;
    random state always comes from the explicit caller-provided Generator.
    All input data is snapshotted via the immutable prepared SupportBasis.

    An empty allowed set is permitted at construction: ``sample(0, rng=...)``
    returns an empty collection; a positive count raises ValueError.
    """

    __slots__ = ("_basis", "_intersection", "_free_columns")

    def __init__(self, basis, *, exclude_identity=False, exclude_span=None):
        if not isinstance(basis, SupportBasis):
            raise TypeError("basis must be a prepared SupportBasis")
        if not isinstance(exclude_identity, (bool, np.bool_)):
            raise TypeError("exclude_identity must be bool")
        excluded = basis._other_basis(exclude_span) if exclude_span is not None else None
        self._basis = basis
        if excluded is None and not exclude_identity:
            self._intersection = None
            self._free_columns = None
        else:
            self._intersection = (np.empty((0, basis.rank), dtype=np.uint8) if excluded is None
                                  else basis._intersection_coefficients(excluded))
            pivots = _pivots(self._intersection)
            self._free_columns = np.setdiff1d(np.arange(basis.rank), pivots, assume_unique=True)
            self._intersection.flags.writeable = False
            self._free_columns.flags.writeable = False

    @property
    def n_qubits(self):
        """Width of all sampled Paulis."""
        return self._basis.n_qubits

    def sample(self, count, *, rng):
        """Return independent uniform positive-phase draws with replacement."""
        count = nonnegative_count(count)
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be an explicit numpy.random.Generator")
        if count == 0:
            return ZXArray.empty(self.n_qubits)
        if self._intersection is None:
            coefficients = rng.integers(0, 2, size=(count, self._basis.rank), dtype=np.uint8)
        else:
            if len(self._free_columns) == 0:
                raise ValueError("No support elements remain after excluding the requested span")
            intersection_coefficients = rng.integers(0, 2, size=(count, len(self._intersection)), dtype=np.uint8)
            coefficients = _combine_bits(intersection_coefficients, self._intersection)
            nonzero = rng.integers(0, 2, size=(count, len(self._free_columns)), dtype=np.uint8)
            missing = np.flatnonzero(~np.any(nonzero, axis=1))
            while len(missing):
                nonzero[missing] = rng.integers(0, 2, size=(len(missing), len(self._free_columns)), dtype=np.uint8)
                missing = missing[~np.any(nonzero[missing], axis=1)]
            coefficients[:, self._free_columns] ^= nonzero
        return self._basis._from_chunks(_combine_chunks(coefficients, self._basis._chunks))
