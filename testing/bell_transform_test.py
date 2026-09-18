"""Independent dense-matrix checks of the symplectic Fourier convention."""

import numpy as np
import pytest

from paulitools.bell_sampling.transforms import symplectic_fwht


def _dense_symplectic_matrix(n_qubits):
    size = 4 ** n_qubits
    mask = (1 << n_qubits) - 1
    output = np.empty((size, size), dtype=np.int8)
    for row in range(size):
        z, x = row & mask, row >> n_qubits
        for column in range(size):
            zp, xp = column & mask, column >> n_qubits
            parity = (bin(z & xp).count("1") + bin(x & zp).count("1")) % 2
            output[row, column] = 1 - 2 * parity
    return output


@pytest.mark.parametrize("n_qubits", range(5))
@pytest.mark.parametrize("complex_input", (False, True))
def test_transform_matches_independent_dense_matrix(n_qubits, complex_input):
    rng = np.random.default_rng(159 + n_qubits)
    values = rng.normal(size=4 ** n_qubits)
    if complex_input:
        values = values + 1j * rng.normal(size=values.size)
    original = values.copy()
    dense = _dense_symplectic_matrix(n_qubits)
    transformed = symplectic_fwht(values)
    np.testing.assert_allclose(transformed, dense @ values, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(symplectic_fwht(values, inverse=True), dense @ values / values.size,
                               atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(symplectic_fwht(transformed, inverse=True), values,
                               atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(symplectic_fwht(transformed), values.size * values,
                               atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(values, original)
    assert transformed.dtype == (np.complex128 if complex_input else np.float64)


def test_one_qubit_order_and_strided_input():
    # I,Z,X,Y: Z commutes with I,Z and anticommutes with X,Y.
    np.testing.assert_array_equal(symplectic_fwht([0, 1, 0, 0]), [1, 1, -1, -1])
    values = np.arange(8, dtype=np.int64)[::2]
    np.testing.assert_array_equal(symplectic_fwht(values), _dense_symplectic_matrix(1) @ values)
    np.testing.assert_array_equal(values, [0, 2, 4, 6])


@pytest.mark.parametrize("values", ([], [1, 2], np.ones(8), np.ones(5), 1.0, np.ones((4, 1))))
def test_bad_shape_or_size_rejected(values):
    with pytest.raises(ValueError):
        symplectic_fwht(values)


@pytest.mark.parametrize("values", ([np.nan], [np.inf], [complex(0, np.inf)]))
def test_nonfinite_input_rejected(values):
    with pytest.raises(ValueError, match="finite"):
        symplectic_fwht(values)


@pytest.mark.parametrize("values", (["1"], np.array([1], dtype=object)))
def test_non_numeric_dtype_rejected(values):
    with pytest.raises(TypeError):
        symplectic_fwht(values)


def test_overflow_rejected_and_inverse_scaling_avoids_spurious_overflow():
    largest = np.finfo(np.float64).max
    values = np.full(4, largest)
    with pytest.raises(FloatingPointError, match="overflowed"):
        symplectic_fwht(values)
    np.testing.assert_array_equal(symplectic_fwht(values, inverse=True), [largest, 0, 0, 0])
    with pytest.raises(TypeError, match="boolean"):
        symplectic_fwht([1], inverse="yes")
