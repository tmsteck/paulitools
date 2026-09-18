"""Compiled support-only Bell masks and reusable numerical sample snapshots.

Two filters are deliberately distinct: ``commuting_mask`` requires <s,g>=0,
whereas ``bell_filter_mask`` requires <s,g>=Y(g), with Y the number of local
Y factors modulo two. Overall Pauli phases are ignored throughout this module.
For commuting generators Y is linear on their support span, so the Bell mask
depends only on that span. Noncommuting Bell generators are rejected.

These functions define numerical statistics, not sample-independence or
experimental-validity policies. A filtered score divides by ALL original shots;
it is not the conditional mean over surviving shots.
"""

from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange

from .._numba import NUMBA_CACHE
from ..large_pauli import _popcount_uint64
from ..zx_array import ZXArray
from ._inputs import as_collection


@njit(cache=NUMBA_CACHE, nogil=True, inline="always")
def _row_y_parity(z, x, row):
    parity = 0
    for chunk in range(z.shape[1]):
        parity ^= _popcount_uint64(z[row, chunk] & x[row, chunk]) & 1
    return parity


@njit(cache=NUMBA_CACHE, nogil=True)
def _y_parities(z, x):
    result = np.empty(z.shape[0], dtype=np.uint8)
    for row in range(z.shape[0]):
        result[row] = _row_y_parity(z, x, row)
    return result


@njit(cache=NUMBA_CACHE, nogil=True, parallel=True)
def _y_parities_parallel(z, x):
    result = np.empty(z.shape[0], dtype=np.uint8)
    for row in prange(z.shape[0]):
        result[row] = _row_y_parity(z, x, row)
    return result


@njit(cache=NUMBA_CACHE, nogil=True, inline="always")
def _row_pairing(z_a, x_a, row_a, z_b, x_b, row_b):
    parity = 0
    for chunk in range(z_a.shape[1]):
        paired = ((z_a[row_a, chunk] & x_b[row_b, chunk])
                  ^ (x_a[row_a, chunk] & z_b[row_b, chunk]))
        parity ^= _popcount_uint64(paired) & 1
    return parity


@njit(cache=NUMBA_CACHE, nogil=True)
def _mutually_commuting(z, x):
    for row in range(z.shape[0]):
        for other in range(row):
            if _row_pairing(z, x, row, z, x, other):
                return False
    return True


@njit(cache=NUMBA_CACHE, nogil=True)
def _cross_commuting(z_a, x_a, z_b, x_b):
    for row in range(z_a.shape[0]):
        for other in range(z_b.shape[0]):
            if _row_pairing(z_a, x_a, row, z_b, x_b, other):
                return False
    return True


@njit(cache=NUMBA_CACHE, nogil=True, inline="always")
def _row_passes(z, x, row, gen_z, gen_x, targets):
    for other in range(gen_z.shape[0]):
        if _row_pairing(z, x, row, gen_z, gen_x, other) != targets[other]:
            return False
    return True


@njit(cache=NUMBA_CACHE, nogil=True)
def _filter_mask(z, x, gen_z, gen_x, targets, initial_mask):
    result = initial_mask.copy()
    for row in range(z.shape[0]):
        if result[row]:
            result[row] = _row_passes(z, x, row, gen_z, gen_x, targets)
    return result


@njit(cache=NUMBA_CACHE, nogil=True, parallel=True)
def _filter_mask_parallel(z, x, gen_z, gen_x, targets, initial_mask):
    result = initial_mask.copy()
    for row in prange(z.shape[0]):
        if result[row]:
            result[row] = _row_passes(z, x, row, gen_z, gen_x, targets)
    return result


@njit(cache=NUMBA_CACHE, nogil=True)
def _signed_sum(parities, mask):
    total = 0
    for row in range(parities.size):
        if mask[row]:
            total += 1 if parities[row] == 0 else -1
    return total


def _chunks(collection):
    width, z, x, _ = collection.kernel_args()
    return width, np.ascontiguousarray(z), np.ascontiguousarray(x)


def _readonly(array):
    array.flags.writeable = False
    return array


def _generator_chunks(generators, width):
    collection = as_collection(generators, n_qubits=width)
    _, z, x = _chunks(collection)
    return collection, z, x


def _require_commuting(z, x):
    if not _mutually_commuting(z, x):
        raise ValueError("Bell filter generators must mutually commute")


def _apply_mask(z, x, gen_z, gen_x, targets, initial_mask, parallel):
    kernel = _filter_mask_parallel if parallel else _filter_mask
    return kernel(z, x, gen_z, gen_x, targets, initial_mask)


def y_parities(paulis, *, parallel=False):
    """Return one uint8 parity of local Y factors per Pauli, ignoring phases."""
    _, z, x = _chunks(as_collection(paulis))
    return (_y_parities_parallel if parallel else _y_parities)(z, x)


def commuting_mask(samples, generators, *, parallel=False):
    """Return bool mask selecting samples commuting with every generator.

    Generators need not commute with one another. Empty generators accept every
    sample; empty samples produce an empty mask. Widths must match explicitly.
    No sample-by-generator matrix is allocated, and each row exits on failure.
    """
    width, z, x = _chunks(as_collection(samples))
    _, gen_z, gen_x = _generator_chunks(generators, width)
    return _apply_mask(z, x, gen_z, gen_x, np.zeros(gen_z.shape[0], dtype=np.uint8),
                       np.ones(z.shape[0], dtype=np.bool_), parallel)


def bell_filter_mask(samples, generators, *, parallel=False):
    """Return bool mask for <sample,g>=Y(g) for every commuting generator g.

    Unlike ``commuting_mask``, an odd-Y generator selects anticommuting samples.
    Overall phases are ignored, including generator phases. Noncommuting
    generators raise ValueError; empty generators accept all samples.
    """
    width, z, x = _chunks(as_collection(samples))
    _, gen_z, gen_x = _generator_chunks(generators, width)
    _require_commuting(gen_z, gen_x)
    return _apply_mask(z, x, gen_z, gen_x, _y_parities(gen_z, gen_x),
                       np.ones(z.shape[0], dtype=np.bool_), parallel)


def bell_purity(samples):
    """Return mean((-1)**Y(sample)) over nonempty samples, ignoring phases."""
    return BellSamplePool(samples).purity


def bell_filtered_purity(samples, generators, *, parallel=False):
    """Return sum(mask[s] * (-1)**Y(s)) / total original sample count.

    The mask is the shifted ``bell_filter_mask``. Empty samples raise ValueError.
    Use BellSamplePool for repeated proposals to avoid rebuilding sample chunks.
    """
    return BellSamplePool(samples).filter(generators, parallel=parallel).score


@dataclass(frozen=True, init=False, eq=False)
class BellSamplePool:
    """Immutable numerical snapshot of samples for repeated Bell filtering.

    Input chunks and Y parities are copied and cached. Later input mutation
    cannot change this pool. Construct a fresh pool for a different sample set;
    this class does not enforce independent draws or any verifier policy.
    """

    _n_qubits: int
    _z: np.ndarray = field(repr=False)
    _x: np.ndarray = field(repr=False)
    _parities: np.ndarray = field(repr=False)
    _total: int = field(repr=False)

    def __init__(self, samples):
        width, z, x = _chunks(as_collection(samples))
        parities = _y_parities(z, x)
        total = _signed_sum(parities, np.ones(z.shape[0], dtype=np.bool_))
        object.__setattr__(self, "_n_qubits", width)
        object.__setattr__(self, "_z", _readonly(z))
        object.__setattr__(self, "_x", _readonly(x))
        object.__setattr__(self, "_parities", _readonly(parities))
        object.__setattr__(self, "_total", int(total))

    @property
    def n_qubits(self):
        return self._n_qubits

    @property
    def n_samples(self):
        return self._z.shape[0]

    @property
    def purity(self):
        """Unfiltered Bell score; an empty pool has no defined mean."""
        if self.n_samples == 0:
            raise ValueError("At least one sample is required to estimate purity")
        return self._total / self.n_samples

    def filter(self, generators, *, parallel=False):
        """Create an immutable BellFilterState without committing any policy."""
        collection, z, x = _generator_chunks(generators, self.n_qubits)
        _require_commuting(z, x)
        mask = _apply_mask(self._z, self._x, z, x, _y_parities(z, x),
                           np.ones(self.n_samples, dtype=np.bool_), parallel)
        return BellFilterState._from_parts(self, collection, z, x, mask)


@dataclass(frozen=True, init=False, eq=False)
class BellFilterState:
    """Immutable filter proposal tied to a numerical BellSamplePool snapshot.

    ``extend`` returns a new state and only evaluates added constraints against
    still-surviving samples. Callers decide whether to retain that proposal.
    Public ``mask`` and ``generators`` accessors return independent copies.
    """

    _pool: BellSamplePool = field(repr=False)
    _generators: ZXArray = field(repr=False)
    _z: np.ndarray = field(repr=False)
    _x: np.ndarray = field(repr=False)
    _mask: np.ndarray = field(repr=False)
    _total: int = field(repr=False)

    def __init__(self):
        raise TypeError("Create BellFilterState with BellSamplePool.filter")

    @classmethod
    def _from_parts(cls, pool, generators, z, x, mask):
        result = object.__new__(cls)
        object.__setattr__(result, "_pool", pool)
        object.__setattr__(result, "_generators", generators.copy())
        object.__setattr__(result, "_z", _readonly(z))
        object.__setattr__(result, "_x", _readonly(x))
        object.__setattr__(result, "_mask", _readonly(mask))
        object.__setattr__(result, "_total", int(_signed_sum(pool._parities, mask)))
        return result

    @property
    def score(self):
        """Signed surviving total divided by ALL samples in the original pool."""
        if self._pool.n_samples == 0:
            raise ValueError("At least one sample is required to estimate purity")
        return self._total / self._pool.n_samples

    @property
    def mask(self):
        """Independent boolean mask in the original sample order."""
        return self._mask.copy()

    @property
    def generators(self):
        """Independent copy of the supplied generators, including their phases."""
        return self._generators.copy()

    def extend(self, additions, *, parallel=False):
        """Return a proposal using added constraints; this state is unchanged.

        Added generators must commute mutually and with all current generators.
        Duplicates and dependent rows are allowed. Phases do not affect filters.
        """
        incoming, z, x = _generator_chunks(additions, self._pool.n_qubits)
        _require_commuting(z, x)
        if not _cross_commuting(self._z, self._x, z, x):
            raise ValueError("Added Bell filter generators must commute with current generators")
        mask = _apply_mask(self._pool._z, self._pool._x, z, x, _y_parities(z, x),
                           self._mask, parallel)
        combined = self._generators.copy()
        combined.extend(incoming)
        return self._from_parts(self._pool, combined,
                                np.concatenate((self._z, z), axis=0),
                                np.concatenate((self._x, x), axis=0), mask)
