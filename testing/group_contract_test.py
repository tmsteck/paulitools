"""Independent finite checks of binary group and signed stabilizer contracts."""

import itertools

import numpy as np
import pytest
from numba import njit

from paulitools.group import (
    ambient_centralizer,
    center,
    centralizer,
    radical,
    row_reduce,
    stabilizer_reduce,
    stabilizer_reduce_bits,
)


def _binary_rows(values, width):
    return np.asarray(
        [[(int(value) >> bit) & 1 for bit in range(width)] for value in values],
        dtype=np.uint8,
    ).reshape(len(values), width)


def _span(rows):
    width = rows.shape[1]
    result = {tuple([0] * width)}
    for row in rows:
        result |= {tuple((np.asarray(value) + row) % 2) for value in result}
    return result


def _packed(qubits, values):
    return np.asarray([qubits] + [2 * int(value) for value in values], dtype=np.int64)


def _rank_mod2(rows):
    """Independent Python rank oracle; real-valued rank is inappropriate."""
    matrix = np.array(rows, dtype=np.uint8, copy=True)
    pivot = 0
    for col in range(matrix.shape[1]):
        candidates = np.flatnonzero(matrix[pivot:, col])
        if candidates.size == 0:
            continue
        selected = pivot + int(candidates[0])
        matrix[[pivot, selected]] = matrix[[selected, pivot]]
        for row in range(pivot + 1, len(matrix)):
            if matrix[row, col]:
                matrix[row] ^= matrix[pivot]
        pivot += 1
    return pivot


@pytest.mark.parametrize("qubits", [1, 2])
def test_centers_and_ambient_commutants_match_exhaustive_binary_sets(qubits):
    width = 2 * qubits
    vectors = _binary_rows(range(1 << width), width)
    cases = [()] + [(value,) for value in range(1 << width)]
    cases += list(itertools.combinations_with_replacement(range(1 << width), 2))
    for values in cases:
        rows = _binary_rows(values, width)
        original = _packed(qubits, values)
        products = vectors[:, :qubits] @ rows[:, qubits:].T
        products += vectors[:, qubits:] @ rows[:, :qubits].T
        expected_ambient = {tuple(row) for row in vectors[np.all(products % 2 == 0, axis=1)]}
        expected_center = _span(rows) & expected_ambient
        result_center = center(original)
        result_ambient = ambient_centralizer(original)
        assert _span(result_center) == expected_center
        assert _span(result_ambient) == expected_ambient
        assert result_ambient.shape == (width - _rank_mod2(rows), width)
        assert _rank_mod2(result_center) == result_center.shape[0]
        assert _rank_mod2(result_ambient) == result_ambient.shape[0]
        np.testing.assert_array_equal(centralizer(original), result_center)
        np.testing.assert_array_equal(original, _packed(qubits, values))


def test_center_and_radical_keep_the_same_supplied_basis():
    # XI and ZI anticommute; IZ generates the center. Reordering this
    # independent basis must reorder radical coefficients, not operators.
    qubits = 2
    for values in itertools.permutations([4, 1, 2]):
        packed = _packed(qubits, values)
        binary = _binary_rows(values, 4)
        coefficients = radical(packed, reduced=True)
        result = center(packed, reduced=True)
        np.testing.assert_array_equal((coefficients @ binary) % 2, result)
        assert _span(result) == {(0, 0, 0, 0), (0, 1, 0, 0)}
        assert _span(result) == _span(center(packed))
        assert _span(ambient_centralizer(packed, True)) == _span(ambient_centralizer(packed))


def test_random_group_dimension_identities():
    rng = np.random.default_rng(913)
    for qubits in [3, 5, 12, 31]:
        for count in [0, 1, 2 * qubits + 3]:
            binary = rng.integers(0, 2, (count, 2 * qubits), dtype=np.uint8)
            values = [sum(int(bit) << col for col, bit in enumerate(row)) for row in binary]
            packed = _packed(qubits, values)
            basis_packed = row_reduce(packed)
            basis = _binary_rows([int(value) >> 1 for value in basis_packed[1:]], 2 * qubits)
            gram = (basis[:, :qubits] @ basis[:, qubits:].T
                    + basis[:, qubits:] @ basis[:, :qubits].T) % 2
            expected_dimension = _rank_mod2(binary) - _rank_mod2(gram)
            assert center(packed).shape == (expected_dimension, 2 * qubits)
            assert radical(packed).shape == (expected_dimension, len(basis))
            assert ambient_centralizer(packed).shape == (2 * qubits - len(basis), 2 * qubits)


_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.diag([1, -1]).astype(np.complex128)
_LOCAL = { (0, 0): _I, (0, 1): _X, (1, 0): _Z, (1, 1): _Y }


def _dense_rows(z, x, phases):
    matrices = []
    for zr, xr, phase in zip(z, x, phases):
        matrix = np.ones((1, 1), dtype=np.complex128)
        for zbit, xbit in zip(zr, xr):
            matrix = np.kron(matrix, _LOCAL[int(zbit), int(xbit)])
        matrices.append((1j ** int(phase)) * matrix)
    return matrices


def _matrix_key(matrix):
    return tuple(matrix.reshape(-1))


def _dense_group(generators, qubits):
    identity = np.eye(1 << qubits, dtype=np.complex128)
    matrices = [identity]
    for generator in generators:
        matrices += [matrix @ generator for matrix in matrices]
    return {_matrix_key(matrix) for matrix in matrices}


@pytest.mark.parametrize("qubits", [1, 2])
def test_signed_stabilizers_against_exhaustive_dense_matrix_products(qubits):
    # Dense matrix multiplication is independent of the kernel's phase formula.
    for values in itertools.product(range(1 << (2 * qubits)), repeat=2):
        binary = _binary_rows(values, 2 * qubits)
        z, x = binary[:, :qubits], binary[:, qubits:]
        source_z, source_x = z.copy(), x.copy()
        for signs in itertools.product([0, 2], repeat=2):
            phases = np.asarray(signs, dtype=np.uint8)
            generators = _dense_rows(z, x, phases)
            commuting = np.array_equal(generators[0] @ generators[1], generators[1] @ generators[0])
            original_group = _dense_group(generators, qubits)
            contradiction = _matrix_key(-np.eye(1 << qubits)) in original_group
            if not commuting or contradiction:
                with pytest.raises(ValueError):
                    stabilizer_reduce_bits(z, x, phases)
                continue
            rz, rx, rp = stabilizer_reduce_bits(z, x, phases)
            assert _dense_group(_dense_rows(rz, rx, rp), qubits) == original_group
            assert _rank_mod2(np.concatenate([rz, rx], axis=1)) == len(rp)
            assert np.all((rp == 0) | (rp == 2))
            np.testing.assert_array_equal(z, source_z)
            np.testing.assert_array_equal(x, source_x)
            np.testing.assert_array_equal(phases, signs)


def test_bell_generator_relation_preserves_minus_yy():
    # XX * ZZ = -YY. The third row is redundant only with its negative sign.
    z = np.array([[0, 0], [1, 1], [1, 1]], dtype=np.uint8)
    x = np.array([[1, 1], [0, 0], [1, 1]], dtype=np.uint8)
    phases = np.array([0, 0, 2], dtype=np.uint8)
    rz, rx, rp = stabilizer_reduce_bits(z, x, phases)
    assert len(rp) == 2
    assert _dense_group(_dense_rows(rz, rx, rp), 2) == _dense_group(_dense_rows(z, x, phases), 2)
    with pytest.raises(ValueError, match="-I"):
        stabilizer_reduce_bits(z, x, np.zeros(3, dtype=np.uint8))


@pytest.mark.parametrize("qubits", [0, 1, 31, 32, 64, 65, 129])
def test_stabilizer_bit_reduction_has_no_word_boundary_limit(qubits):
    z = np.zeros((3, qubits), dtype=np.uint8)
    x = np.zeros_like(z)
    phases = np.zeros(3, dtype=np.uint8)
    if qubits:
        # Equal signed generators with support on the highest qubit.
        z[:2, -1] = 1
        x[:2, -1] = 1
        phases[:2] = 2
    rz, rx, rp = stabilizer_reduce_bits(z, x, phases)
    assert len(rp) == int(qubits > 0)
    if qubits:
        assert rp[0] == 2
        np.testing.assert_array_equal(rz[0], z[0])
        np.testing.assert_array_equal(rx[0], x[0])
    ez, ex, ep = stabilizer_reduce_bits(z[:0], x[:0], phases[:0])
    assert ez.shape == ex.shape == (0, qubits)
    assert ep.shape == (0,)


def test_stabilizer_validation():
    z = np.zeros((1, 2), dtype=np.uint8)
    for phase in [-1, 1, 3, 4, 0.5]:
        with pytest.raises(ValueError, match="Hermitian"):
            stabilizer_reduce_bits(z, z, np.array([phase]))
    with pytest.raises(ValueError, match="binary"):
        stabilizer_reduce_bits(np.array([[2, 0]]), z, np.zeros(1, dtype=np.uint8))
    with pytest.raises(ValueError, match="match"):
        stabilizer_reduce_bits(z, z[:, :1], np.zeros(1, dtype=np.uint8))
    with pytest.raises(ValueError, match="one entry"):
        stabilizer_reduce_bits(z, z, np.zeros(0, dtype=np.uint8))
    with pytest.raises(ValueError, match="-I"):
        stabilizer_reduce_bits(z, z, np.array([2], dtype=np.uint8))


@njit
def _compiled_stabilizer_workflow(packed):
    return stabilizer_reduce(packed), center(packed), ambient_centralizer(packed)


def test_packed_stabilizer_adapter_and_nopython_composability():
    # Legacy integer words for +XX, +ZZ, -YY.
    packed = np.array([2, 24, 6, 31], dtype=np.int64)
    original = packed.copy()
    reduced, result_center, result_ambient = _compiled_stabilizer_workflow(packed)
    binary = _binary_rows([int(value) >> 1 for value in reduced[1:]], 4)
    phases = (reduced[1:] & 1) * 2
    expected = _dense_group([np.kron(_X, _X), np.kron(_Z, _Z)], 2)
    assert _dense_group(_dense_rows(binary[:, :2], binary[:, 2:], phases), 2) == expected
    np.testing.assert_array_equal(packed, original)
    assert result_center.shape == result_ambient.shape == (2, 4)
    assert _compiled_stabilizer_workflow.nopython_signatures
    for qubits in [0, 1, 31]:
        identity = np.array([qubits, 0, 0], dtype=np.int64)
        np.testing.assert_array_equal(stabilizer_reduce(identity), [qubits])
    highest_x = np.array([31, (1 << 62) | 1, (1 << 62) | 1], dtype=np.int64)
    np.testing.assert_array_equal(stabilizer_reduce(highest_x), highest_x[:2])
    for invalid in [[], [32, 0], [1, 8], [1, -1], [1, 1], [1, 2, 4]]:
        with pytest.raises(ValueError):
            stabilizer_reduce(np.asarray(invalid, dtype=np.int64))
