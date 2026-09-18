"""Parser and packed-kernel regression tests with independent finite oracles."""

from itertools import product

import numpy as np
import pytest

from paulitools import (
    PauliInt, bsip_array, commute_array_fast, commutation_matrix,
    concatenate_ZX, left_pad, right_pad, symplectic_inner_product_extended,
    symplectic_inner_product_int, toString, toString_extended, toZX,
    toZX_extended, toZX_large,
)
from paulitools.core import _pack_zx_bitplanes, _pack_pauli_char_matrix
from paulitools.large_pauli import (
    _symplectic_matrix_chunks, _symplectic_matrix_chunks_parallel,
)


@pytest.mark.parametrize("text", ["-X", "Y", "-XZ", "+ZY"])
def test_padding_preserves_operator_and_phase(text):
    signed = text if text[0] in "+-" else "+" + text
    pauli = toZX(text)
    width = len(signed) - 1
    assert toString(right_pad(pauli, width + 2)) == signed + "II"
    assert toString(left_pad(pauli, width + 2)) == signed[0] + "II" + signed[1:]
    assert toString(concatenate_ZX([pauli, toZX("I" * (width + 2))])).split(", ")[0] == signed + "II"


@pytest.mark.parametrize("width", [32, 64, 65])
def test_oversized_legacy_inputs_raise_before_packing(width):
    with pytest.raises(ValueError, match="31 qubits"):
        toZX("I" * (width - 1) + "X")
    with pytest.raises(ValueError, match="31 qubits"):
        toZX(np.zeros((2, 2 * width), dtype=np.uint8))
    with pytest.raises(ValueError, match="31 qubits"):
        toZX("0" * (2 * width))
    with pytest.raises(ValueError, match="31 qubits"):
        toZX(np.ones(width), fast_input_type="eigen_z")
    for pad in (left_pad, right_pad):
        with pytest.raises(ValueError, match="31 qubits"):
            pad(toZX("X"), width)
    # Internal compiled packers also guard direct callers.
    with pytest.raises(ValueError, match="31 qubits"):
        _pack_zx_bitplanes(np.zeros((1, width), dtype=np.uint8), np.zeros((1, width), dtype=np.uint8))
    with pytest.raises(ValueError, match="31 qubits"):
        _pack_pauli_char_matrix(np.zeros((1, width), dtype=np.uint8), np.array([width]), np.array([0]), width)


@pytest.mark.parametrize("width", [1, 31, 32, 64, 65])
def test_explicit_numeric_encodings_are_batch_independent(width):
    bits = np.ones((1, 2 * width), dtype=np.int8)
    assert toString_extended(toZX_extended(bits, encoding="bits")) == "+" + "Y" * width
    assert toString_extended(toZX_extended(bits, encoding="eigenvalues")) == "+" + "I" * width
    eigen_batch = np.r_[bits, -bits]
    out = toZX_extended(eigen_batch, encoding="eigenvalues")
    assert toString_extended(out) == "+" + "I" * width + ", +" + "Y" * width
    assert toString_extended(toZX_large(bits, encoding="eigenvalues")) == "+" + "I" * width
    with pytest.raises(ValueError):
        toZX_extended(-bits, encoding="bits")
    with pytest.raises(ValueError):
        toZX_extended(bits * 0, encoding="eigenvalues")
    with pytest.raises(ValueError):
        toZX_extended(bits * 2, encoding="auto")


@pytest.mark.parametrize("width", [31, 32, 64, 65])
def test_binary_lowercase_and_mixed_width_dispatch(width):
    source = ["-x", "y" + "i" * (width - 1)]
    expected = "-X" + "I" * (width - 1) + ", +Y" + "I" * (width - 1)
    assert toString_extended(toZX_extended(source)) == expected
    assert toString_extended(toZX_large(source)) == expected
    binary = "1" + "0" * (width - 1) + "0" * width
    assert toString_extended(toZX_extended(binary)) == "+Z" + "I" * (width - 1)
    assert toString_extended(toZX_large([binary, binary])) == ", ".join(["+Z" + "I" * (width - 1)] * 2)


def test_legacy_auto_encoding_remains_compatible():
    assert toString(toZX(np.array([1, 1]))) == "+Y"
    assert toString(toZX(np.array([[1, 1], [-1, 1]]))) == "+I, +Z"
    with pytest.raises(ValueError):
        toZX("X", encoding="unknown")


def test_extended_small_symplectic_default():
    assert symplectic_inner_product_extended(toZX("X"), toZX("Z")) == 1


def test_all_two_qubit_matrix_commutators_match_packed_kernels():
    matrices = {
        "I": np.eye(2, dtype=complex), "X": np.array([[0, 1], [1, 0]]),
        "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1]),
    }
    strings = ["".join(s) for s in product("IXYZ", repeat=2)]
    dense = [np.kron(matrices[s[0]], matrices[s[1]]) for s in strings]
    expected = np.array([[not np.array_equal(a @ b, b @ a) for b in dense] for a in dense], dtype=np.int8)
    packed = toZX(strings)
    for parallel in (False, True):
        np.testing.assert_array_equal(bsip_array(packed, parallel), expected)
        np.testing.assert_array_equal(commute_array_fast(packed, parallel), 1 - expected)
        np.testing.assert_array_equal(commutation_matrix(toZX_large(strings), parallel), expected)


def test_word_and_chunk_boundaries_against_character_oracle():
    rng = np.random.default_rng(20260916)
    for width in (31, 32, 64, 65, 129):
        letters = rng.integers(0, 4, (11, width))
        strings = ["".join("IXYZ"[i] for i in row) for row in letters]
        expected = np.array([[np.count_nonzero((a != 0) & (b != 0) & (a != b)) % 2 for b in letters] for a in letters], dtype=np.int8)
        large = toZX_large(strings)
        np.testing.assert_array_equal(commutation_matrix(large), expected)
        z = np.ascontiguousarray([p.z_chunks for p in large.paulis])
        x = np.ascontiguousarray([p.x_chunks for p in large.paulis])
        for kernel in (_symplectic_matrix_chunks, _symplectic_matrix_chunks_parallel):
            np.testing.assert_array_equal(kernel(z[:3], x[:3], z[3:], x[3:]), expected[:3, 3:])
        if width == 31:
            packed = toZX(strings)
            np.testing.assert_array_equal(bsip_array(packed), expected)
            for i, a in enumerate(packed[1:]):
                for j, b in enumerate(packed[1:]):
                    assert symplectic_inner_product_int(a, b, width) == expected[i, j]


def test_noncanonical_large_tail_bits_are_rejected():
    for width in (1, 31, 32, 65):
        chunks = (width + 63) // 64
        bad = np.zeros(chunks, dtype=np.uint64)
        bad[-1] = np.uint64(1) << np.uint64(width % 64)
        good = np.zeros_like(bad)
        with pytest.raises(ValueError, match="beyond"):
            PauliInt(width, 0, bad, good)
        with pytest.raises(ValueError, match="beyond"):
            PauliInt(width, 0, good, bad)


def test_empty_batch_and_zero_qubit_large_conversion():
    np.testing.assert_array_equal(bsip_array(np.array([3], dtype=np.int64)), np.zeros((0, 0), dtype=np.int8))
    assert toString_extended(toZX_extended("", force_large=True)) == "+"
    assert commutation_matrix(toZX_large(np.zeros((0, 130), dtype=np.uint8))).shape == (0, 0)
