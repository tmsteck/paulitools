"""Packed-Pauli conversions and Bell-outcome estimators.

The estimators below operate on the explicitly documented Bell-outcome sign
functions. They do not infer a generic state expectation from arbitrary samples.
"""

from collections.abc import Mapping
from numbers import Integral

import numpy as np
from numba import njit, prange

from ._numba import NUMBA_CACHE
from .core import _validate_legacy_array_nb, symplectic_inner_product_int, toZX


@njit(cache=NUMBA_CACHE)
def toBinary(pauli):
    """Return a row-major ``(number_of_operators, 2*k)`` Z|X bit matrix."""
    if pauli.ndim != 1:
        raise ValueError("Pauli input must be a one-dimensional packed array")
    _validate_legacy_array_nb(pauli)
    k = pauli[0]
    output = np.empty((len(pauli) - 1, 2 * k), dtype=np.int8)
    for i in range(len(pauli) - 1):
        value = pauli[i + 1] >> 1
        for j in range(2 * k):
            output[i, j] = (value >> j) & 1
    return output


@njit(cache=NUMBA_CACHE)
def convert_array_type(arr, dtype):
    new_arr = np.empty(arr.shape, dtype=dtype)
    new_arr[:] = arr
    return new_arr


@njit(cache=NUMBA_CACHE)
def popcount(n):
    """Counts the number of set bits in an integer (Hamming weight)."""
    count = 0
    while n > 0:
        # This efficiently removes the rightmost set bit
        n &= (n - 1)
        count += 1
    return count

@njit(cache=NUMBA_CACHE)
def y_parity_int(int_rep, k):
    """Return the Y-basis parity of a packed Pauli integer."""
    mask = (1 << k) - 1
    z_bits = (int_rep >> 1) & mask
    x_bits = (int_rep >> (k + 1)) & mask
    return popcount(x_bits & z_bits) & 1


@njit(cache=NUMBA_CACHE)
def x_parity_int(int_rep, k):
    """Return the X-only parity of a packed Pauli integer."""
    mask = (1 << k) - 1
    z_bits = (int_rep >> 1) & mask
    x_bits = (int_rep >> (k + 1)) & mask
    return popcount(x_bits & (mask ^ z_bits)) & 1


@njit(cache=NUMBA_CACHE)
def z_parity_int(int_rep, k):
    """Return the Z-only parity of a packed Pauli integer."""
    mask = (1 << k) - 1
    z_bits = (int_rep >> 1) & mask
    x_bits = (int_rep >> (k + 1)) & mask
    return popcount(z_bits & (mask ^ x_bits)) & 1


@njit(cache=NUMBA_CACHE)
def getParity(pauli, basis='Y'):
    """
    Calculates the parity of a specific Pauli operator ('X', 'Y', or 'Z')
    in a Pauli string using efficient bitwise operations.
    
    Args:
        pauli (np.ndarray): Pauli operator in ZX format, where pauli[0] is the
                            number of qubits (k) and pauli[1] is the integer
                            representation.
        basis (str): The basis to count ('X', 'Y', or 'Z').
    
    Returns:
        int: 0 if the count of the basis operators is even, 1 if it is odd.
    """
    if pauli.ndim != 1:
        raise ValueError("Pauli input must be a one-dimensional packed array")
    _validate_legacy_array_nb(pauli)
    if len(pauli) != 2:
        raise ValueError("getParity requires exactly one packed Pauli")
    k = pauli[0]
    int_rep = pauli[1]

    if basis == 'Y':
        return y_parity_int(int_rep, k)
    elif basis == 'X':
        return x_parity_int(int_rep, k)
    elif basis == 'Z':
        return z_parity_int(int_rep, k)

    raise ValueError("basis must be X, Y, or Z")


def _as_legacy_paulis(data):
    """Normalize supported Python inputs before entering packed kernels."""
    from .pauli import Pauli
    from .zx_array import ZXArray

    if isinstance(data, (Pauli, ZXArray)):
        data = data.legacy_array(copy=False)
    if isinstance(data, np.ndarray) and data.ndim == 1:
        if not np.issubdtype(data.dtype, np.integer):
            raise TypeError("Packed Pauli arrays must have integer dtype")
        if data.dtype.kind == "u" and np.any(data > np.iinfo(np.int64).max):
            raise ValueError("Packed Pauli integers must fit signed int64")
        result = np.ascontiguousarray(data, dtype=np.int64)
    else:
        result = toZX(data)
    _validate_legacy_array_nb(result)
    return result


def _probability_weights(values):
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise ValueError("Probabilities must be real")
    weights = np.asarray(values, dtype=np.float64)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("A nonempty one-dimensional probability array is required")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Probabilities must be finite and nonnegative")
    total = float(weights.sum())
    if not np.isfinite(total) or not np.isclose(total, 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("Probabilities must sum to one")
    return np.ascontiguousarray(weights / total)


def _outcomes_from_keys(keys, k=None):
    values = np.empty(len(keys), dtype=np.int64)
    if len(keys) == 0:
        raise ValueError("At least one outcome is required")
    for i, key in enumerate(keys):
        if not isinstance(key, str):
            raise TypeError("Outcome keys must be Pauli strings or Z|X bit strings")
        outcome = toZX(key)
        _validate_legacy_array_nb(outcome)
        if k is None:
            k = int(outcome[0])
        if outcome[0] != k:
            raise ValueError("All outcomes and observables must have the same qubit count")
        values[i] = outcome[1]
    return k, values


@njit(cache=NUMBA_CACHE)
def _signed_shot_weights(shots, weights, k, include_shot_y):
    result = weights.copy()
    if include_shot_y:
        for j in range(len(shots)):
            if y_parity_int(shots[j], k):
                result[j] = -result[j]
    return result


@njit(cache=NUMBA_CACHE)
def _weighted_pauli_row(pauli, shots, weights, k):
    pauli_parity = y_parity_int(pauli, k) ^ (pauli & 1)
    total = 0.0
    for j in range(len(shots)):
        exponent = pauli_parity ^ symplectic_inner_product_int(pauli, shots[j], k)
        total += weights[j] if exponent == 0 else -weights[j]
    return total


@njit(cache=NUMBA_CACHE, nogil=True)
def _weighted_pauli_estimates(paulis, shots, weights, k, include_shot_y):
    signed_weights = _signed_shot_weights(shots, weights, k, include_shot_y)
    result = np.empty(len(paulis), dtype=np.float64)
    for i in range(len(paulis)):
        result[i] = _weighted_pauli_row(paulis[i], shots, signed_weights, k)
    return result


@njit(cache=NUMBA_CACHE, nogil=True, parallel=True)
def _weighted_pauli_estimates_parallel(paulis, shots, weights, k, include_shot_y):
    signed_weights = _signed_shot_weights(shots, weights, k, include_shot_y)
    result = np.empty(len(paulis), dtype=np.float64)
    for i in prange(len(paulis)):
        result[i] = _weighted_pauli_row(paulis[i], shots, signed_weights, k)
    return result


def _dictionary_expectations(pauli_input, probs, parallel, include_shot_y):
    if not isinstance(probs, Mapping):
        raise TypeError("probs must map outcome strings to probabilities")
    paulis = _as_legacy_paulis(pauli_input)
    weights = _probability_weights(list(probs.values()))
    k, shots = _outcomes_from_keys(list(probs), int(paulis[0]))
    kernel = _weighted_pauli_estimates_parallel if parallel else _weighted_pauli_estimates
    return kernel(paulis[1:], shots, weights, k, include_shot_y)


def get_pauli_obs(pauli_input, probs, parallel=False):
    r"""Evaluate the Bell-outcome estimator ``sum_s p(s) (-1)^(sign(P)+Y(P)+<P,s>)``.

    ``Y`` is the number of Y factors modulo two and ``<P,s>`` is the binary
    symplectic product. ``pauli_input`` accepts a Pauli string, a list of strings,
    a ``Pauli``/``ZXArray`` object, or a one-dimensional packed integer array.
    These legacy estimators support up to 31 qubits and real observable phases.
    Outcome keys are Pauli strings or
    Z|X bit strings of the same width. Outcome phase signs are ignored; an
    observable's minus sign negates its estimate. This explicitly chosen sign
    convention is not a generic state-expectation reconstruction.

    Probabilities must be nonempty, finite, nonnegative, and sum to one within
    floating-point tolerance; accepted rounding error is normalized away.
    ``parallel=True`` uses compiled threads across observables. Returns a float
    array, including for a single observable.
    """
    return _dictionary_expectations(pauli_input, probs, parallel, False)


def get_pauli_pauli_obs(pauli_input, probs, parallel=False):
    r"""Evaluate ``sum_s p(s) (-1)^(sign(P)+Y(P)+<P,s>+Y(s))``.

    This is the Bell-outcome convention with the additional outcome Y parity.
    Input, sign, probability, and parallelism contracts match ``get_pauli_obs``.
    """
    return _dictionary_expectations(pauli_input, probs, parallel, True)


def getCentralizer(counts, return_generators=False):
    """Return the historical *center within the span* of outcome differences.

    Keys are equally wide Pauli strings or Z|X bit strings; count values are
    ignored. Cyclic differences span all pairwise differences without allocating
    a quadratic matrix. The returned center is a binary Z|X row matrix, matching
    ``group.centralizer``'s historical contract, not the full ambient centralizer.
    With ``return_generators=True``, also return its reduced packed generators.
    """
    from .group import centralizer, differences, row_reduce

    if not isinstance(counts, Mapping):
        raise TypeError("counts must map outcome strings to counts or probabilities")
    k, values = _outcomes_from_keys(list(counts))
    paulis = np.empty(len(values) + 1, dtype=np.int64)
    paulis[0], paulis[1:] = k, values
    generators = row_reduce(differences(paulis))
    center = centralizer(generators, reduced=True)
    return (center, generators) if return_generators else center


def _packed_shot_scalar(value, max_value, float_limit):
    if isinstance(value, Integral):
        packed = int(value)
    elif isinstance(value, (float, np.floating)):
        limit = min(float_limit, 2 ** (np.finfo(type(value)).nmant + 1))
        if not np.isfinite(value) or value != np.floor(value):
            raise ValueError("Packed shots must be finite integers")
        if abs(value) >= limit:
            raise ValueError("Floating packed shots may have lost integer precision; use integer/object storage")
        packed = int(value)
    else:
        raise TypeError("Packed shots must be integer scalars")
    if packed < 0 or packed > max_value:
        raise ValueError("Packed shot has bits outside the declared qubit width")
    return packed


def Pauli_expectation(shots, pauli):
    """Return the same scalar Bell-outcome estimator as ``get_pauli_obs``.

    ``pauli`` is one observable, usually the packed array ``[k, value]``.
    ``shots`` has shape ``(N, 2)``: packed outcome integer and probability.
    The formula is ``sum_s p(s) (-1)^(sign(P)+Y(P)+<P,s>)``. Outcome signs
    are ignored. Probabilities follow ``get_pauli_obs``'s normalized contract.

    Float-stored outcome values must be integral and strictly below the dtype's
    consecutive-integer limit (2**53 for float64). At that boundary a rounded
    input cannot be distinguished from an exact integer. For larger values use
    an object array or nested rows preserving integer outcomes and float weights.
    """
    observable = _as_legacy_paulis(pauli)
    if len(observable) != 2:
        raise ValueError("Pauli_expectation requires exactly one observable")
    float_limit = 2 ** 53
    if isinstance(shots, np.ndarray) and shots.dtype.kind == "f":
        float_limit = 2 ** (np.finfo(shots.dtype).nmant + 1)
    rows = np.asarray(shots, dtype=object)
    if rows.ndim != 2 or rows.shape[1] != 2 or rows.shape[0] == 0:
        raise ValueError("shots must be a nonempty array of (packed outcome, probability) rows")
    k = int(observable[0])
    max_value = (1 << (2 * k + 1)) - 1
    values = np.array(
        [_packed_shot_scalar(value, max_value, float_limit) for value in rows[:, 0]],
        dtype=np.int64,
    )
    weights = _probability_weights(rows[:, 1].tolist())
    return float(_weighted_pauli_estimates(observable[1:], values, weights, k, False)[0])


@njit(cache=NUMBA_CACHE, nogil=True)
def filtered_purity(generators, shots, shot_parities=None):
    r"""Average ``(-1)^Y(s)`` over shots passing every generator's Bell filter.

    A shot passes generator g when ``Y(g)+<s,g>`` is even. Packed generator
    phase signs are ignored, as in the original filter definition. An empty
    generator set applies no filter. Shots must be nonempty and have the same
    declared width as the generators. Optional parities are a one-dimensional
    0/1 array with exactly one entry per shot.
    """
    if generators.ndim != 1 or shots.ndim != 1:
        raise ValueError("Generators and shots must be one-dimensional packed arrays")
    _validate_legacy_array_nb(generators)
    _validate_legacy_array_nb(shots)
    if generators[0] != shots[0]:
        raise ValueError("Generators and shots must have the same qubit count")
    if len(shots) <= 1:
        raise ValueError("At least one shot is required")
    k, num_shots = shots[0], len(shots) - 1
    if shot_parities is not None:
        if shot_parities.ndim != 1 or len(shot_parities) != num_shots:
            raise ValueError("shot_parities must contain exactly one parity per shot")
        for parity in shot_parities.flat:
            if parity != 0 and parity != 1:
                raise ValueError("shot_parities entries must be zero or one")
    gen_parities = np.empty(len(generators) - 1, dtype=np.int8)
    for j in range(len(gen_parities)):
        gen_parities[j] = y_parity_int(generators[j + 1], k)
    total = 0.0
    for i in range(num_shots):
        passes = True
        for j in range(len(gen_parities)):
            if gen_parities[j] ^ symplectic_inner_product_int(shots[i + 1], generators[j + 1], k):
                passes = False
                break
        if passes:
            parity = y_parity_int(shots[i + 1], k) if shot_parities is None else shot_parities.flat[i]
            total += 1.0 if parity == 0 else -1.0
    return total / num_shots


@njit(cache=NUMBA_CACHE, nogil=True)
def get_purity(shots):
    """Return the Bell-shot mean of ``(-1)^Y(s)`` for nonempty packed shots."""
    if shots.ndim != 1:
        raise ValueError("Shots must be a one-dimensional packed array")
    _validate_legacy_array_nb(shots)
    if len(shots) <= 1:
        raise ValueError("At least one shot is required")
    total = 0.0
    for i in range(1, len(shots)):
        total += 1.0 if y_parity_int(shots[i], shots[0]) == 0 else -1.0
    return total / (len(shots) - 1)


def filtered_purity_reference(generators, shots, shot_parities=None):
    """Quiet Python reference for the exact contract of ``filtered_purity``."""
    if not isinstance(generators, np.ndarray) or not isinstance(shots, np.ndarray):
        raise TypeError("Generators and shots must be one-dimensional packed arrays")
    if generators.ndim != 1 or shots.ndim != 1:
        raise ValueError("Generators and shots must be one-dimensional packed arrays")
    generators = _as_legacy_paulis(generators)
    shots = _as_legacy_paulis(shots)
    if generators[0] != shots[0]:
        raise ValueError("Generators and shots must have the same qubit count")
    if len(shots) <= 1:
        raise ValueError("At least one shot is required")
    k, num_shots = int(shots[0]), len(shots) - 1
    mask = (1 << k) - 1

    def y_parity(value):
        return bin((int(value) >> 1) & (int(value) >> (k + 1)) & mask).count("1") & 1

    def inner(a, b):
        za, xa = (int(a) >> 1) & mask, (int(a) >> (k + 1)) & mask
        zb, xb = (int(b) >> 1) & mask, (int(b) >> (k + 1)) & mask
        return bin((za & xb) ^ (xa & zb)).count("1") & 1

    if shot_parities is None:
        parities = [y_parity(s) for s in shots[1:]]
    else:
        parities = np.asarray(shot_parities)
        if parities.ndim != 1 or len(parities) != num_shots:
            raise ValueError("shot_parities must contain exactly one parity per shot")
        if np.any((parities != 0) & (parities != 1)):
            raise ValueError("shot_parities entries must be zero or one")
    return sum(
        1 if parity == 0 else -1
        for shot, parity in zip(shots[1:], parities)
        if all((y_parity(g) ^ inner(shot, g)) == 0 for g in generators[1:])
    ) / num_shots
